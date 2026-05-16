from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass (frozen = True)
class PRPConfig:
    samplerate: int = 400
    smooth_ms: float = 15.0
    pre_window_ms: float = 40.0
    post_window_ms: float = 40.0
    noise_window_ms: float = 40.0
    future_confirm_ms: float = 150.0

    min_change_multiplier: float = 3.3  # 3.3
    min_absolute_change: float = 8e-6   # 8e-6
    min_future_net_change: float = 7e-6 # 7e-6 <-- update these if you find better.

    inner_region_ratio: float = 0.7


    @property
    def smooth_samples(self) -> int:
        return max(3, int(round(self.smooth_ms * self.samplerate / 1000)))

    @property
    def pre_window_samples(self) -> int:
        return max(1, int(round(self.pre_window_ms * self.samplerate / 1000)))

    @property
    def post_window_samples(self) -> int:
        return max(1, int(round(self.post_window_ms * self.samplerate / 1000)))
    
    @property
    def noise_window_samples(self) -> int:
        return max(3, int(round(self.noise_window_ms * self.samplerate / 1000)))
    
    @property 
    def future_confirm_samples(self) -> int:
        return max(1, int(round(self.future_confirm_ms * self.samplerate / 1000)))


@dataclass (frozen = True)
class PRPResult:
    start_index: int | None
    end_index: int | None
    prp_index: int | None
    confidence: float | None
    reason: str

@dataclass (frozen = True)
class PRPRegion:
    start_index: int
    end_index: int
    peak_strength: float
    mean_strength: float
    area_strength: float

    @property 
    def width(self) -> int:
        return self.end_index - self.start_index + 1

    @property
    def midpoint_index(self) -> int:
        return (self.start_index + self.end_index) // 2

@dataclass(frozen = True)
class BPConfig:
    smooth_ms: float = 10.0
    search_start_offset_ms: float = 250.0
    search_end_ratio: float = 0.75
    peak_sigma_mult: float = 3.0
    local_window_ms: float = 25.0
    min_index: int = 0
    min_region_width: int = 3
    inner_region_ratio: float = 0.7
    motion_threshold_ratio: float = 0.35
    motion_reference_window_ms: float = 500.0


@dataclass(frozen = True)
class BPResult:
    start_index: int | None
    end_index: int | None
    bp_index: int | None
    confidence: float
    reason: str


@dataclass(frozen = True)
class SPConfig:
    smooth_ms: float = 10.0
    min_peak_motion: float = 1e-6
    min_sp_delay_ms: float = 100.0


@dataclass(frozen = True)
class SPResult:
    start_index: int | None
    end_index: int | None
    sp_index: int | None
    confidence: float
    reason: str


@dataclass(frozen = True)
class SharedSignalFeatures:
    signal: np.ndarray
    samplerate: int

    smooth_10: np.ndarray
    d1_10: np.ndarray
    d2_10: np.ndarray
    abs_d1: np.ndarray
    abs_d2: np.ndarray
    
    smooth_15: np.ndarray   


@dataclass(frozen=True)
class MultivariatConfig:
    prp_baseline_ms: float = 500.0
    prp_smooth_ms: float = 35.0
    prp_sustain_ms: float = 120.0
    prp_future_confirm_ms: float = 180.0
    prp_threshold_std_mult: float = 4.5
    prp_threshold_range_ratio: float = 0.015
    prp_min_abs_change: float = 1e-8

    bp_smooth_ms: float = 75.0
    bp_sustain_ms: float = 150.0
    bp_future_confirm_ms: float = 700.0
    bp_slope_threshold_ratio: float = 0.18
    bp_min_progress_ratio: float = 0.04
    bp_pre_prp_tolerance_ms: float = 300.0
    bp_search_start_ms: float = 50.0
    bp_consensus_tolerance_ms: float = 200.0
    candidate_score_ratio: float = 0.90
    pressure_peak_threshold_ratio: float = 0.18
    pressure_sp_peak_threshold_ratio: float = 0.08
    pressure_peak_min_gap_ms: float = 150.0

    sp_smooth_ms: float = 100.0
    sp_min_after_bp_ms: float = 120.0
    sp_consensus_tolerance_ms: float = 1000.0
    sp_plateau_ms: float = 600.0
    sp_plateau_ratio: float = 0.03
    sp_plateau_min_ms: float = 400.0
    sp_plateau_max_ms: float = 10000.0
    sp_final_tolerance_ratio: float = 0.08
    sp_progress_ratio: float = 0.92
    sp_future_verify_ms: float = 4000.0
    sp_future_motion_ratio: float = 0.06
    sp_min_motion_floor: float = 0.02


def ms_to_samples(ms: float, samplerate: int, minimum: int = 1) -> int: 
    n = int(round((ms / 1000.0) * samplerate))
    return max(minimum, n)


def adaptive_ms_for_signal(
    base_ms: float,
    n_samples: int,
    samplerate: int,
    ratio: float,
    min_ms: float,
    max_ms: float,
) -> float:
    if n_samples <= 0 or samplerate <= 0:
        return base_ms

    duration_ms = (float(n_samples) / float(samplerate)) * 1000.0
    adaptive_ms = duration_ms * ratio
    return float(np.clip(max(base_ms, adaptive_ms, min_ms), min_ms, max_ms))


def validate_signal_1d(signal: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)

    if x.ndim != 1:
        raise ValueError("signal must be 1d")
    
    if x.size == 0:
        raise ValueError("signal must be non-empty")
    
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or INF")
    
    return x

def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    
    padd_left = window // 2 
    pad_right = window - 1 - padd_left
    
    padded = np.pad(x, (padd_left, pad_right), mode = "edge")
    kernel = np.ones(window, dtype = np.float32) / float(window)

    return np.convolve(padded, kernel, mode = "valid").astype(np.float32)

def moving_avg_guassian(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    
    padd_left = window // 2 
    pad_right = window - 1 - padd_left
    
    padded = np.pad(x, (padd_left, pad_right), mode = "edge")

    sigma = max(window / 6.0, 1.0)
    idx = np.arange(window, dtype=np.float32) - (window - 1) / 2.0
    kernel = np.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= np.sum(kernel)

    return np.convolve(padded, kernel, mode = "valid").astype(np.float32)


def gaussian_smooth_signal(
    signal: np.ndarray | list[float],
    samplerate: int,
    window_ms: float,
    polyorder: int = 2,
) -> np.ndarray:
    x = validate_signal_1d(signal)
    if samplerate <= 0:
        raise ValueError("samplerate must be more than 0")

    window = ms_to_samples(window_ms, samplerate, minimum=polyorder + 2)
    if window % 2 == 0:
        window += 1

    return moving_avg_guassian(x, min(window, max(1, x.size)))

def safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    std = float(np.std(x))
    return max(std, eps)

    # Gaussian bump to get a soft location target (trying to start with a "general" location for PRP)
def gaussian(
    n_samples: int,
    center: int,
    samplerate: int,
    sigma_ms: float,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be more than 0")
    
    if center < 0 or center >= n_samples:
        raise ValueError("center is out of bounds")
    
    sigma_samples = max((sigma_ms / 1000.0) * samplerate, 1.0)

    idx = np.arange(n_samples, dtype = np.float32)
    bump = np.exp(-0.5 * ((idx - float(center)) / float(sigma_samples)) ** 2)
    return bump.astype(np.float32)


def compute_shared_features(signal: np.ndarray | list[float], samplerate: int) -> SharedSignalFeatures:
    x = validate_signal_1d(signal)

    if samplerate <= 0:
        raise ValueError("samplerate must be more than 0")
    
    smooth_10_n = ms_to_samples(10.0, samplerate, minimum=3)
    smooth_15_n = ms_to_samples(15.0, samplerate, minimum=3)

    smooth_10 = moving_avg_guassian(x, smooth_10_n)
    d1_10 = np.gradient(smooth_10).astype(np.float32)
    d2_10 = np.gradient(d1_10).astype(np.float32)

    abs_d1 = np.abs(d1_10).astype(np.float32)
    abs_d2 = np.abs(d2_10).astype(np.float32)
    
    smooth_15 = moving_avg_guassian(x, smooth_15_n)

    return SharedSignalFeatures(
        signal=x,
        samplerate=samplerate,
        smooth_10=smooth_10,
        d1_10=d1_10,
        d2_10=d2_10,
        abs_d1=abs_d1,
        abs_d2=abs_d2,
        smooth_15=smooth_15,
    )

def smooth_signal(signal: np.ndarray, config: PRPConfig) -> np.ndarray:
    return moving_avg(signal.astype(np.float32), config.smooth_samples)

def sliding_window(signal: np.ndarray, window_size: int, mode: str) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    n = x.size

    if window_size <= 0:
        raise ValueError("window size must be above 0")
    
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min' ")
    
    out = np.empty(n, dtype=np.float32)
    dq: deque[int] = deque()

    def is_better(new_idx: int, old_idx: int) -> bool:
        if mode =="max":
            return x[new_idx] >= x[old_idx]
        return x[new_idx] <= x[old_idx]
    
    for i in range(n):
        window_start = i - window_size + 1

        while dq and dq[0] < window_start:
            dq.popleft()

        while dq and is_better(i, dq[-1]):
            dq.pop()

        dq.append(i)
        out[i] = x[dq[0]]
    
    return out


def compute_local_std(signal: np.ndarray, window_samples: int, min_std: float = 1e-12) -> np.ndarray:

    x = np.asarray(signal, dtype=np.float32)
    n = x.size

    half = window_samples // 2

    cumulative_sum = np.zeros(n + 1, dtype=np.float32)
    cumulative_sum[1:] = np.cumsum(x)

    cumulative_sum_sq = np.zeros(n + 1, dtype=np.float32)
    cumulative_sum_sq[1:] = np.cumsum(x * x)

    local_std = np.zeros(n, dtype=np.float32)

    for i in range(n):
        start = max(0, i - half)
        stop = min(n, i + half + 1)

        length = stop - start

        if length < 2:
            local_std[i] = min_std
            continue

        sum_x = cumulative_sum[stop] - cumulative_sum[start]
        sum_x2 = cumulative_sum_sq[stop] - cumulative_sum_sq[start]

        mean = sum_x / length
        mean_sq = sum_x2 / length

        var = mean_sq - mean * mean
        var = max(var, 0.0)

        std = np.sqrt(var)
        local_std[i] = max(std, min_std)
    
    return local_std

def compute_local_change(signal: np.ndarray, pre_window_samples: int, post_window_samples: int) -> np.ndarray:

    x = np.asarray(signal, dtype=np.float32)
    n = x.size

    cumulative_sum = np.zeros(n + 1, dtype=np.float32)
    cumulative_sum[1:] = np.cumsum(x)

    local_change = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pre_start = max(0, i - pre_window_samples)
        pre_stop = i

        post_start = i
        post_stop = min(n, i + post_window_samples)

        pre_len = pre_stop - pre_start
        post_len = post_stop - post_start

        if pre_len == 0 or post_len == 0:
            continue

        pre_sum = cumulative_sum[pre_stop] - cumulative_sum[pre_start]
        post_sum = cumulative_sum[post_stop] - cumulative_sum[post_start]

        pre_mean = pre_sum / pre_len
        post_mean = post_sum / post_len

        local_change[i] = post_mean - pre_mean

    return local_change

def compute_future_net_rise(signal: np.ndarray, future_samples: int,) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    n = x.size

    if n == 0:
        return np.zeros(0, dtype=np.float32)
    
    if future_samples < 0:
        raise ValueError("future samples be >= 0")
    
    window_size = future_samples + 1

    x_rev = x[::-1].copy()
    future_max_rev = sliding_window(x_rev, window_size=window_size, mode="max")
    future_max = future_max_rev[::-1]

    return (future_max - x).astype(np.float32)

def compute_future_net_drop(signal: np.ndarray, future_samples: int,) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    n = x.size

    if n == 0:
        return np.zeros(0, dtype=np.float32)
    
    if future_samples < 0:
        raise ValueError("future samples must be >= 0")
    
    window_size = future_samples + 1

    x_rev = x[::-1].copy()
    future_min_rev = sliding_window(x_rev, window_size=window_size, mode="min")
    future_min = future_min_rev[::-1]

    return (x - future_min).astype(np.float32)
    

def compute_change_strength(local_change: np.ndarray, local_std: np.ndarray, min_std: float = 1e-12) -> np.ndarray:
    safe_std = np.maximum(local_std, min_std)
    return np.abs(local_change) / safe_std

def compute_change_candidates(
        local_change: np.ndarray, 
        change_strength: np.ndarray,
        future_net_rise: np.ndarray,
        future_net_drop: np.ndarray,
        min_change_multiplier: float,
        min_absolute_change: float,
        min_future_net_change: float,
) -> np.ndarray:
    
    is_rise = local_change > 0
    future_confirm = np.where(is_rise, future_net_rise, future_net_drop)

    return(change_strength >= min_change_multiplier) & (np.abs(local_change) >= min_absolute_change) & (future_confirm >= min_future_net_change)


# ----------- Regional helpers --------------- #

def build_regions_with_stats(mask: np.ndarray, strength_signal: np.ndarray) -> list [PRPRegion]:
    regions: list[PRPRegion] = []
    start: int | None = None

    for i, is_candidate in enumerate(mask):
        if is_candidate and start is None:
            start = i

        elif not is_candidate and start is not None:
            end = i - 1

            region_strength = strength_signal[start:end + 1]
            regions.append(
                PRPRegion(
                start_index = start, 
                end_index = end, 
                peak_strength = float(np.max(region_strength)), 
                mean_strength = float(np.mean(region_strength)),
                area_strength = float(np.sum(region_strength)),
                )
            )
            start = None

    if start is not None:
        end = len(mask) - 1 
        region_strength = strength_signal[start:end + 1]
        regions.append(
            PRPRegion(
                start_index=start,
                end_index=end,
                peak_strength=float(np.max(region_strength)),
                mean_strength=float(np.mean(region_strength)),
                area_strength=float(np.sum(region_strength)),
            )
        )

    return regions

def select_earliest_strong_region(regions: list[PRPRegion]) -> PRPRegion | None: 
    
    if not regions:
        return None
    return regions[0]


def tighten_region(region: PRPRegion, strength_signal: np.ndarray, inner_region_ratio: float) -> PRPRegion:
    region_strength = strength_signal[region.start_index:region.end_index + 1]
    inner_threshold = inner_region_ratio * region.peak_strength

    keep_indices = np.where(region_strength >= inner_threshold)[0]

    if len(keep_indices) == 0:
        return region
    
    new_start = region.start_index + int(keep_indices[0])
    new_end = region.start_index + int(keep_indices[-1])
    new_strength = strength_signal[new_start:new_end + 1]

    return PRPRegion(
        start_index=new_start,
        end_index=new_end,
        peak_strength=float(np.max(new_strength)),
        mean_strength=float(np.mean(new_strength)),
        area_strength=float(np.sum(new_strength)),
    )

def build_prp_result(selected_regions: PRPRegion | None) -> PRPResult:
    if selected_regions is None:
        return PRPResult(
            start_index=None,
            end_index=None,
            prp_index=None,
            confidence=0.0,
            reason="No candidate region passed the minimum peak strength"
        )
    
    return PRPResult(
        start_index=selected_regions.start_index,
        end_index=selected_regions.end_index,
        prp_index=selected_regions.midpoint_index,
        confidence=float(selected_regions.peak_strength),
        reason=" Select midpoint of earliest strong candidate region",
    )

# -------- PRP detection ------ #

def detect_prp(signal: np.ndarray, config: PRPConfig, shared: SharedSignalFeatures | None = None) -> PRPResult:
    x = validate_signal_1d(signal)

    shared_features = shared or compute_shared_features(x, config.samplerate)
    smoothed = shared_features.smooth_15

    local_std = compute_local_std(
        smoothed, 
        config.noise_window_samples
    )

    local_change = compute_local_change(
        smoothed, 
        config.pre_window_samples, 
        config.post_window_samples
    )
    
    change_strength = compute_change_strength(
        local_change=local_change,
        local_std = local_std,
    )
    
    future_net_rise = compute_future_net_rise(
        smoothed,
        config.future_confirm_samples,
    )

    future_net_drop = compute_future_net_drop(
        smoothed,
        config.future_confirm_samples
    )

    candidate_mask = compute_change_candidates(
        change_strength=change_strength,
        local_change=local_change,
        future_net_drop=future_net_drop,
        future_net_rise=future_net_rise,
        min_change_multiplier=config.min_change_multiplier,
        min_absolute_change=config.min_absolute_change,
        min_future_net_change=config.min_future_net_change
    )

    regions = build_regions_with_stats(
        mask=candidate_mask,
        strength_signal=change_strength,
    )

    selected_region = select_earliest_strong_region(regions=regions)

    if selected_region is not None:
        selected_region = tighten_region(
            region=selected_region,
            strength_signal=change_strength,
            inner_region_ratio=config.inner_region_ratio,
        )

    return build_prp_result(selected_region)


def strength_signal(future_drop: np.ndarray, local_std: np.ndarray, min_std: float = 1e-12) -> np.ndarray:
    safe_std = np.maximum(local_std, min_std)
    return future_drop / safe_std

# ------ BP Detection ------ #

def detect_bp(
    signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    config: BPConfig | None = None,
    shared: SharedSignalFeatures | None = None,
) -> BPResult:
    cfg = config or BPConfig()
    x = validate_signal_1d(signal)

    if prp_index is None:
        return BPResult(start_index=None, end_index=None, bp_index=None, confidence=0.0, reason="missing prp")

    shared_features = shared or compute_shared_features(x, samplerate)

    abs_d1 = shared_features.abs_d1
    abs_d2 = shared_features.abs_d2

    offset_n = ms_to_samples(cfg.search_start_offset_ms, samplerate)
    local_window_n = ms_to_samples(cfg.local_window_ms, samplerate)

    search_end = min(int(round(x.size * cfg.search_end_ratio)), x.size)
    primary_start = max(cfg.min_index, prp_index + offset_n)
    fallback_start = max(cfg.min_index, prp_index + 1)

    if search_end <= fallback_start + local_window_n:
        return BPResult(start_index=None, end_index=None, bp_index=None, confidence=0.0, reason="search window too short")

    def _select_region(window_start: int) -> BPResult | None:
        if window_start + local_window_n >= search_end:
            return None

        baseline_start = max(0, prp_index - local_window_n)
        baseline_kink = abs_d2[baseline_start:prp_index]
        if baseline_kink.size == 0:
            baseline_kink = abs_d2[:local_window_n]

        kink_threshold = float(np.mean(baseline_kink) + cfg.peak_sigma_mult * safe_std(baseline_kink))

        motion_window = abs_d1[window_start:search_end]
        kink_window = abs_d2[window_start:search_end]
        if motion_window.size == 0 or kink_window.size == 0:
            return None

        max_motion = float(np.max(motion_window))
        max_kink = float(np.max(kink_window))
        if max_motion <= 0.0 or max_kink <= 0.0:
            return None

        norm_motion = motion_window / (max_motion + 1e-8)
        norm_kink = kink_window / (max_kink + 1e-8)
        bp_strength = 0.5 * norm_motion + 0.5 * norm_kink

        reference_window_n = ms_to_samples(cfg.motion_reference_window_ms, samplerate)
        reference_end = min(search_end, window_start + reference_window_n)
        reference_motion = abs_d1[window_start:reference_end]
        if reference_motion.size == 0:
            reference_motion = motion_window

        reference_peak_motion = float(np.max(reference_motion))
        motion_threshold = cfg.motion_threshold_ratio * reference_peak_motion
        motion_mask = motion_window >= motion_threshold
        regions = build_regions_with_stats(motion_mask, bp_strength)

        selected_region: PRPRegion | None = None
        fallback_region: PRPRegion | None = None

        for region in regions:
            if fallback_region is None:
                fallback_region = region
            if region.width >= cfg.min_region_width:
                selected_region = region
                break

        if selected_region is None:
            selected_region = fallback_region

        if selected_region is None:
            return None

        selected_region = tighten_region(
            region=selected_region,
            strength_signal=bp_strength,
            inner_region_ratio=cfg.inner_region_ratio,
        )

        region_start = window_start + selected_region.start_index
        region_end = window_start + selected_region.end_index

        bp_index: int | None = None
        point_search_start = region_start + max(1, (region_end - region_start) // 4)

        upper_bound = min(region_end, len(abs_d2) - 2)
        lower_bound = max(point_search_start, 1)

        for i in range(lower_bound, upper_bound + 1):
            value = float(abs_d2[i])
            if value < kink_threshold:
                continue
            if value >= float(abs_d2[i - 1]) and value >= float(abs_d2[i + 1]):
                bp_index = i
                break

        if bp_index is None:
            bp_index = selected_region.midpoint_index + window_start

        confidence = float(np.clip(selected_region.peak_strength, 0.0, 1.0))
        return BPResult(
            start_index=region_start,
            end_index=region_end,
            bp_index=bp_index,
            confidence=confidence,
            reason="detected from earliest strong bp region",
        )

    primary_result = _select_region(primary_start)
    if primary_result is not None:
        return primary_result

    fallback_result = _select_region(fallback_start)
    if fallback_result is not None:
        return fallback_result

    return BPResult(start_index=None, end_index=None, bp_index=None, confidence=0.0, reason="no bp found")


# -------- SP Detection ------- # 

def detect_sp(
    signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    config: SPConfig | None = None,
    shared: SharedSignalFeatures | None = None,
) -> SPResult:
    cfg = config or SPConfig()
    x = validate_signal_1d(signal)

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    if prp_index is None:
        return SPResult(
            start_index=None,
            end_index=None,
            sp_index=None,
            confidence=0.0,
            reason="missing prp",
        )

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate, minimum=3)
    x_smooth = moving_avg(x.astype(np.float32), smooth_n)
    d1 = np.gradient(x_smooth).astype(np.float32)

    global_smooth_n = ms_to_samples(40.0, samplerate, minimum=5)
    d1_trend = moving_avg(d1, global_smooth_n).astype(np.float32)

    deviation = np.abs(d1 - d1_trend).astype(np.float32)

    n = x.size
    edge_guard = max(5, global_smooth_n)

    min_sp_delay = ms_to_samples(cfg.min_sp_delay_ms, samplerate, minimum=1)
    search_start = max(prp_index + min_sp_delay, edge_guard)
    late_cutoff = int(0.95 * n)
    search_end = min(late_cutoff, n - edge_guard)

    if search_end <= search_start + 5:
        return SPResult(
            start_index=None,
            end_index=None,
            sp_index=None,
            confidence=0.0,
            reason="no valid sp search window",
        )
    
    net_change = float(x_smooth[search_start - 1] - x_smooth[search_start])
    pressure_decreasing = net_change > 0.0

    if pressure_decreasing:
        search_dev = deviation[search_start:search_end]

        if search_dev.size == 0:
            return SPResult(
                start_index=None,
                end_index=None,
                sp_index=None,
                confidence=0.0,
                reason="empty sp deviation window",
            )
        
        dev_max = float(np.max(search_dev))
        threshold = max(0.05 * dev_max, cfg.min_peak_motion)

        mask = search_dev > threshold
        regions = build_regions_with_stats(mask, search_dev)

        if not regions:
            return SPResult(
                start_index=None,
                end_index=None,
                sp_index=None,
                confidence=0.0,
                reason="no significant sp deviation found",
            )

        selected_region = max(regions, key=lambda r: r.area_strength)

        region_start = search_start + selected_region.start_index
        region_end = search_start + selected_region.end_index

        local_dev = search_dev[selected_region.start_index:selected_region.end_index + 1]
        peak_offset = int(np.argmax(local_dev))
        sp_index = region_start + peak_offset

        confidence = float(np.clip(selected_region.area_strength / (sum(r.area_strength for r in regions) + 1e-8), 0.0, 1.0))

        return SPResult(
            start_index=region_start,
            end_index=region_end,
            sp_index=sp_index,
            confidence=confidence,
            reason="selected peak of last significant deviation before settling",
        )
    
    # increasing pressure logic
    else:
        search_d1 = d1_trend[search_start:search_end]

        if search_d1.size < 10:
            return SPResult(None, None, None, 0.0, "empty sp increasing window")

        # ignore early noise. We know SP is the latest significant event.
        noise_start = int(0.25 * search_d1.size)
        noise_part = search_d1[noise_start:] if search_d1.size > 20 else search_d1

        baseline = float(np.median(noise_part))
        mad = float(np.median(np.abs(noise_part - baseline))) + 1e-12

        late_peak = (
            float(np.max(search_d1[noise_start:]))
            if search_d1[noise_start:].size
            else float(np.max(search_d1))
        )

        step_n = ms_to_samples(300.0, samplerate, minimum=10)
        plateau_n = ms_to_samples(700.0, samplerate, minimum=20)

        step_candidates = []
        
        stride = ms_to_samples(25.0, samplerate, minimum=1)
        if search_d1.size > step_n + plateau_n:
            for i in range(step_n, search_d1.size - plateau_n, stride):
                before = search_d1[i - step_n:i]
                after = search_d1[i:i + plateau_n]

                before_level = float(np.median(before))
                after_level = float(np.median(after))

                step_height = after_level - before_level
                after_noise = float(np.median(np.abs(after - after_level))) + 1e-12

                min_step_height = max(4.0 * mad, 0.02 * abs(late_peak))

                if step_height > min_step_height:
                    if after_noise < 0.50 * abs(step_height):
                        step_candidates.append(i)

        if step_candidates:
            sp_local = step_candidates[-1]
            sp_index = search_start + sp_local

            max_region_n = ms_to_samples(500.0, samplerate, minimum=5)

            event_end = min(search_d1.size, sp_local + max_region_n)
            region = search_d1[sp_local:event_end]

            peak = float(np.max(region))
            confidence = float(np.clip((peak - baseline) / (0.3 * mad + 1e-8), 0.0, 1.0))

            return SPResult(
                start_index=sp_index,
                end_index=min(search_end - 1, sp_index + max_region_n),
                sp_index=sp_index,
                confidence=confidence,
                reason="selected rightmost positive step into stable plateau",
            )

        # bump & hump catcher
        threshold = baseline + 5.0 * mad

        mask = search_d1 > threshold
        regions = build_regions_with_stats(mask, search_d1)

        if not regions:
            return SPResult(None, None, None, 0.0, "no positive sp bumps found")

        min_width = ms_to_samples(10.0, samplerate, minimum=1)
        min_height = max(baseline + 4.0 * mad, 0.08 * abs(late_peak))

        min_area = min_height * min_width

        valid_regions = []

        for r in regions:
            width = r.end_index - r.start_index + 1
            region = search_d1[r.start_index:r.end_index + 1]

            peak = float(np.max(region))
            area = float(np.sum(np.maximum(region - baseline, 0.0)))

            if width < min_width:
                continue

            if peak < min_height:
                continue

            if area < min_area:
                continue

            valid_regions.append(r)

        if not valid_regions:
            return SPResult(None, None, None, 0.0, "no valid positive sp bumps found")

        selected_region = valid_regions[-1]
        region_start = search_start + selected_region.start_index

        region = search_d1[selected_region.start_index:selected_region.end_index + 1]
        peak = float(np.max(region))

        confidence = float(np.clip((peak - baseline) / (0.3 * mad + 1e-8), 0.0, 1.0))

        max_region_n = ms_to_samples(250.0, samplerate, minimum=5)
        region_end = min(search_start + selected_region.end_index, region_start + max_region_n, search_end - 1)

        sp_index = region_start

        return SPResult(
            start_index=region_start,
            end_index=region_end,
            sp_index=sp_index,
            confidence=confidence,
            reason="selected rightmost significant positive bump",
        )


def _baseline_stats(
    signal: np.ndarray,
    samplerate: int,
    baseline_ms: float,
) -> tuple[float, float, int]:
    baseline_n = min(signal.size, ms_to_samples(baseline_ms, samplerate, minimum=3))
    baseline = signal[:baseline_n]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    robust_std = max(1.4826 * mad, safe_std(baseline, eps=1e-12))
    return center, robust_std, baseline_n


def detect_PRP_multivariat(
    pressure_signal: np.ndarray,
    samplerate: int,
    config: MultivariatConfig | None = None,
    shared_pressure: SharedSignalFeatures | None = None,
) -> PRPResult:
    cfg = config or MultivariatConfig()
    pressure = gaussian_smooth_signal(
        pressure_signal,
        samplerate,
        cfg.prp_smooth_ms,
        polyorder=2,
    )
    baseline, baseline_std, baseline_n = _baseline_stats(
        pressure,
        samplerate,
        cfg.prp_baseline_ms,
    )
    signal_range = max(float(np.max(pressure) - np.min(pressure)), baseline_std, 1e-12)
    sustain_n = ms_to_samples(cfg.prp_sustain_ms, samplerate, minimum=3)
    future_n = ms_to_samples(cfg.prp_future_confirm_ms, samplerate, minimum=sustain_n)
    threshold = max(
        cfg.prp_threshold_std_mult * baseline_std,
        cfg.prp_threshold_range_ratio * signal_range,
        cfg.prp_min_abs_change,
    )

    upper = max(baseline_n, pressure.size - max(sustain_n, future_n))
    for index in range(baseline_n, upper + 1):
        sustain_window = pressure[index:index + sustain_n]
        if sustain_window.size < sustain_n:
            continue
        mean_dev = float(np.mean(np.abs(sustain_window - baseline)))
        if mean_dev < threshold:
            continue

        future_window = pressure[index:index + future_n]
        if future_window.size < future_n:
            continue
        future_change = abs(float(np.median(future_window)) - baseline)
        if future_change < 0.8 * threshold:
            continue

        confidence = float(
            np.clip(
                0.5 * min(mean_dev / (threshold + 1e-12), 1.0)
                + 0.5 * min(future_change / (threshold + 1e-12), 1.0),
                0.0,
                1.0,
            )
        )
        return PRPResult(
            start_index=index,
            end_index=min(pressure.size - 1, index + sustain_n - 1),
            prp_index=index,
            confidence=confidence,
            reason="multivariat pressure sustained baseline departure used for prp",
        )

    return PRPResult(None, None, None, 0.0, "no multivariat prp found")


def _indices_within_tolerance(
    index_a: int | None,
    index_b: int | None,
    tolerance_samples: int,
) -> bool:
    if index_a is None or index_b is None:
        return False
    return abs(index_a - index_b) <= tolerance_samples


def _pressure_derivative_peaks(
    *,
    shared_features: SharedSignalFeatures,
    samplerate: int,
    prp_index: int,
    start_ms: float,
    min_gap_ms: float,
    threshold_ratio: float,
) -> list[tuple[int, int, int, float]]:
    derivative = np.asarray(shared_features.abs_d1, dtype=np.float32)
    if derivative.size < 3:
        return []

    search_start = max(1, prp_index + ms_to_samples(start_ms, samplerate, minimum=1))
    search_end = min(derivative.size - 1, int(round(derivative.size * 0.95)))
    if search_end <= search_start + 1:
        return []

    window = derivative[search_start:search_end]
    if window.size < 3:
        return []

    peak_threshold = max(float(np.max(window)) * threshold_ratio, 1e-8)
    min_gap = ms_to_samples(min_gap_ms, samplerate, minimum=1)
    peaks: list[tuple[int, int, int, float]] = []

    index = search_start
    while index < search_end:
        value = float(derivative[index])
        if value < peak_threshold:
            index += 1
            continue

        region_start = index
        region_end = index
        peak_index = index
        peak_value = value

        while region_end + 1 < search_end and float(derivative[region_end + 1]) >= peak_threshold:
            region_end += 1
            current_value = float(derivative[region_end])
            if current_value > peak_value:
                peak_value = current_value
                peak_index = region_end

        if peaks and region_start - peaks[-1][2] < min_gap:
            prev_start, prev_peak, prev_end, prev_score = peaks[-1]
            if peak_value > prev_score:
                peaks[-1] = (prev_start, peak_index, region_end, peak_value)
            else:
                peaks[-1] = (prev_start, prev_peak, region_end, prev_score)
        else:
            peaks.append((region_start, peak_index, region_end, peak_value))

        index = region_end + 1

    if not peaks:
        return []

    max_peak = max(score for _start, _peak, _end, score in peaks)
    return [
        (start, peak, end, float(np.clip(score / (max_peak + 1e-12), 0.0, 1.0)))
        for start, peak, end, score in peaks
    ]


def _bp_index_from_region(
    region: tuple[int, int, int, float],
    derivative: np.ndarray,
    *,
    use_settling_point: bool,
) -> int:
    region_start, peak_index, region_end, _score = region
    if not use_settling_point:
        return peak_index

    peak_value = float(derivative[peak_index])
    settle_threshold = 0.45 * peak_value
    sustain_n = max(2, min(6, region_end - peak_index + 1))

    for index in range(peak_index + 1, region_end + 1):
        window_end = min(region_end + 1, index + sustain_n)
        if window_end <= index:
            continue
        mean_level = float(np.mean(derivative[index:window_end]))
        if mean_level <= settle_threshold:
            return index

    if region_end > peak_index:
        return peak_index + max(1, (region_end - peak_index) // 3)

    return peak_index


def _pressure_bp_local_minimum(
    pressure_signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    *,
    config: MultivariatConfig,
    shared_pressure: SharedSignalFeatures | None = None,
) -> BPResult:
    x = validate_signal_1d(pressure_signal)
    if prp_index is None:
        return BPResult(None, None, None, 0.0, "missing prp")

    shared_features = shared_pressure or compute_shared_features(x, samplerate)
    peaks = _pressure_derivative_peaks(
        shared_features=shared_features,
        samplerate=samplerate,
        prp_index=prp_index,
        start_ms=config.bp_search_start_ms,
        min_gap_ms=config.pressure_peak_min_gap_ms,
        threshold_ratio=config.pressure_peak_threshold_ratio,
    )
    if not peaks:
        return BPResult(None, None, None, 0.0, "no pressure derivative bp peak found")

    derivative = np.asarray(shared_features.abs_d1, dtype=np.float32)
    region_count = len(peaks)
    if region_count == 1:
        bp_region = peaks[0]
        best_index = _bp_index_from_region(bp_region, derivative, use_settling_point=True)
        reason = "single clear pressure derivative region after prp; early settling point used for bp"
    elif region_count == 2:
        bp_region = peaks[0]
        best_index = _bp_index_from_region(bp_region, derivative, use_settling_point=True)
        reason = "two clear pressure derivative regions; bp taken where first region settles toward baseline"
    else:
        early_region_count = min(2, region_count)
        early_regions = peaks[:early_region_count]
        bp_region = max(early_regions, key=lambda region: region[3])
        best_index = bp_region[1]
        reason = "three or more clear pressure derivative regions; bp taken as strongest early peak"

    _region_start, _peak_index, region_end, best_score = bp_region

    window_n = ms_to_samples(120.0, samplerate, minimum=3)
    search_start = max(1, prp_index + ms_to_samples(config.bp_search_start_ms, samplerate, minimum=1))
    search_end = min(shared_features.abs_d1.size - 1, int(round(shared_features.abs_d1.size * 0.95)))
    confidence = float(np.clip(best_score, 0.0, 1.0))
    return BPResult(
        start_index=max(search_start, best_index - window_n),
        end_index=min(search_end, best_index + window_n),
        bp_index=best_index,
        confidence=confidence,
        reason=reason,
    )


def _pressure_sp_after_bp(
    pressure_signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    bp_index: int | None,
    *,
    config: MultivariatConfig,
    shared_pressure: SharedSignalFeatures | None = None,
) -> SPResult:
    x = validate_signal_1d(pressure_signal)
    if prp_index is None:
        return SPResult(None, None, None, 0.0, "missing prp")

    shared_features = shared_pressure or compute_shared_features(x, samplerate)
    pressure = shared_features.smooth_15
    peaks = _pressure_derivative_peaks(
        shared_features=shared_features,
        samplerate=samplerate,
        prp_index=prp_index,
        start_ms=config.bp_search_start_ms,
        min_gap_ms=config.pressure_peak_min_gap_ms,
        threshold_ratio=config.pressure_peak_threshold_ratio,
    )
    if not peaks:
        return SPResult(None, None, None, 0.0, "no pressure derivative sp peak found")

    region_count = len(peaks)
    min_after_bp_n = ms_to_samples(config.sp_min_after_bp_ms, samplerate, minimum=1)
    if region_count < 2:
        return SPResult(None, None, None, 0.0, "fewer than two clear pressure derivative regions; no sp")

    baseline_window_n = ms_to_samples(500.0, samplerate, minimum=3)
    baseline_start = max(0, prp_index - baseline_window_n)
    baseline_slice = pressure[baseline_start:prp_index] if prp_index > baseline_start else pressure[:baseline_window_n]
    baseline = float(np.median(baseline_slice)) if baseline_slice.size else float(pressure[0])
    tail_n = min(pressure.size, ms_to_samples(1000.0, samplerate, minimum=3))
    final_level = float(np.median(pressure[-tail_n:]))
    total_motion = max(abs(final_level - baseline), 1e-12)
    progress = (
        (pressure - baseline) / total_motion
        if final_level >= baseline
        else (baseline - pressure) / total_motion
    )
    event_limit_idx = pressure.size - 1
    progress_threshold = min(0.92, max(0.75, config.sp_progress_ratio))
    if bp_index is not None and bp_index + 1 < pressure.size:
        for index in range(bp_index + 1, pressure.size):
            if float(progress[index]) >= progress_threshold:
                event_limit_idx = index
                break

    search_limit_n = ms_to_samples(400.0, samplerate, minimum=1)
    valid_peaks = [
        (start, peak, end, score)
        for start, peak, end, score in peaks
        if (bp_index is None or start > bp_index + min_after_bp_n)
        and start <= event_limit_idx + search_limit_n
    ]
    if not valid_peaks:
        return SPResult(None, None, None, 0.0, "no later pressure derivative region inside main event")

    sp_region = valid_peaks[-1]
    if region_count == 2:
        reason = "two clear pressure derivative regions; sp taken as start of later separate region"
    else:
        reason = "last later separate pressure derivative region inside main event used for sp"

    best_start, _best_peak, best_end, best_score = sp_region
    if bp_index is not None and best_start <= bp_index + min_after_bp_n:
        return SPResult(None, None, None, 0.0, "pressure sp region did not occur sufficiently after bp")

    window_n = ms_to_samples(120.0, samplerate, minimum=3)
    return SPResult(
        start_index=max(0, best_start - window_n),
        end_index=min(shared_features.abs_d1.size - 1, best_end + window_n),
        sp_index=best_start,
        confidence=float(np.clip(best_score, 0.0, 1.0)),
        reason=reason,
    )


def detect_BP_multivariat(
    pressure_signal: np.ndarray,
    travel_signal: np.ndarray | None,
    samplerate: int,
    prp_index: int | None,
    strain_signal: np.ndarray | None = None,
    config: MultivariatConfig | None = None,
    shared_pressure: SharedSignalFeatures | None = None,
) -> BPResult:
    cfg = config or MultivariatConfig()
    pressure_bp = _pressure_bp_local_minimum(
        pressure_signal,
        samplerate,
        prp_index,
        config=cfg,
        shared_pressure=shared_pressure,
    )

    if travel_signal is None:
        return pressure_bp

    travel = gaussian_smooth_signal(
        travel_signal,
        samplerate,
        cfg.bp_smooth_ms,
        polyorder=2,
    )
    baseline, baseline_std, baseline_n = _baseline_stats(
        travel,
        samplerate,
        cfg.prp_baseline_ms,
    )
    travel_range = max(float(np.max(travel) - np.min(travel)), baseline_std, 1e-12)
    tail_n = min(travel.size, ms_to_samples(1000.0, samplerate, minimum=3))
    final_level = float(np.median(travel[-tail_n:]))
    direction = 1.0 if final_level >= baseline else -1.0
    d1 = np.gradient(travel).astype(np.float32)
    directional_slope = (
        np.maximum(d1, 0.0).astype(np.float32)
        if direction >= 0.0
        else np.maximum(-d1, 0.0).astype(np.float32)
    )
    slope_ref = max(float(np.percentile(directional_slope, 95)), 1e-12)
    slope_threshold = cfg.bp_slope_threshold_ratio * slope_ref
    sustain_n = ms_to_samples(60.0, samplerate, minimum=3)
    slope_window_n = ms_to_samples(30.0, samplerate, minimum=3)
    future_n = ms_to_samples(150.0, samplerate, minimum=sustain_n)
    start_hint = baseline_n + ms_to_samples(cfg.bp_search_start_ms, samplerate, minimum=1)
    if prp_index is not None:
        pre_prp_tol = ms_to_samples(cfg.bp_pre_prp_tolerance_ms, samplerate, minimum=1)
        start_hint = max(1, min(start_hint, max(1, prp_index - pre_prp_tol)))
    upper = travel.size - max(sustain_n, future_n)
    if upper <= start_hint:
        return BPResult(None, None, None, 0.0, "not enough samples for multivariat bp")

    deviation = travel - baseline if direction >= 0.0 else baseline - travel
    deviation_threshold = max(0.12 * cfg.bp_min_progress_ratio * travel_range, 2.5 * baseline_std, 1e-8)
    sustain_deviation_threshold = max(0.28 * cfg.bp_min_progress_ratio * travel_range, 3.0 * baseline_std, 1e-8)
    slope_mean_threshold = 0.16 * slope_threshold
    slope_peak_threshold = 0.35 * slope_threshold
    min_progress = max(0.30 * cfg.bp_min_progress_ratio * travel_range, 2.5 * baseline_std)
    travel_bp: BPResult | None = None
    for index in range(start_hint, upper + 1):
        sustain_window = deviation[index:index + sustain_n]
        if sustain_window.size < sustain_n:
            continue
        mean_deviation = float(np.mean(sustain_window))
        current_deviation = float(deviation[index])
        if current_deviation < deviation_threshold and mean_deviation < sustain_deviation_threshold:
            continue

        slope_window = directional_slope[index:index + slope_window_n]
        if slope_window.size < slope_window_n:
            continue
        mean_slope = float(np.mean(slope_window))
        peak_slope = float(np.max(slope_window))
        if mean_slope < slope_mean_threshold and peak_slope < slope_peak_threshold:
            continue

        future_window = travel[index:index + future_n]
        if future_window.size < future_n:
            continue
        future_level = float(np.median(future_window))
        future_progress = (
            future_level - baseline
            if direction >= 0.0
            else baseline - future_level
        )
        if future_progress < min_progress:
            continue
        deviation_score = min(max(mean_deviation, current_deviation) / (sustain_deviation_threshold + 1e-12), 1.0)
        slope_score = min(max(mean_slope / (slope_mean_threshold + 1e-12), peak_slope / (slope_peak_threshold + 1e-12)), 1.0)
        progress_score = min(future_progress / (min_progress + 1e-12), 1.0)
        confidence = float(np.clip(0.35 * deviation_score + 0.35 * slope_score + 0.30 * progress_score, 0.0, 1.0))
        travel_bp = BPResult(
            start_index=index,
            end_index=min(travel.size - 1, index + sustain_n - 1),
            bp_index=index,
            confidence=confidence,
            reason="multivariat first travel onset above deviation and slope noise floors used for bp",
        )
        break

    if travel_bp is None:
        return BPResult(None, None, None, 0.0, "no multivariat bp found")

    tolerance_n = ms_to_samples(cfg.bp_consensus_tolerance_ms, samplerate, minimum=1)
    if _indices_within_tolerance(travel_bp.bp_index, pressure_bp.bp_index, tolerance_n):
        return BPResult(
            start_index=travel_bp.start_index,
            end_index=travel_bp.end_index,
            bp_index=travel_bp.bp_index,
            confidence=min(travel_bp.confidence, pressure_bp.confidence),
            reason="multivariat travel onset confirmed by pressure minimum within tolerance used for bp",
        )

    return BPResult(None, None, None, 0.0, "travel bp and pressure bp disagree beyond tolerance")


def detect_SP_multivariat(
    pressure_signal: np.ndarray,
    travel_signal: np.ndarray | None,
    samplerate: int,
    prp_index: int | None,
    bp_index: int | None,
    strain_signal: np.ndarray | None = None,
    config: MultivariatConfig | None = None,
    shared_pressure: SharedSignalFeatures | None = None,
) -> SPResult:
    cfg = config or MultivariatConfig()
    pressure_sp = _pressure_sp_after_bp(
        pressure_signal=pressure_signal,
        samplerate=samplerate,
        prp_index=prp_index,
        bp_index=bp_index,
        config=cfg,
        shared_pressure=shared_pressure,
    )

    if travel_signal is None:
        return pressure_sp

    travel = gaussian_smooth_signal(
        travel_signal,
        samplerate,
        cfg.sp_smooth_ms,
        polyorder=2,
    )
    baseline, baseline_std, baseline_n = _baseline_stats(
        travel,
        samplerate,
        cfg.prp_baseline_ms,
    )
    travel_range = max(float(np.max(travel) - np.min(travel)), baseline_std, 1e-12)
    tail_n = min(travel.size, ms_to_samples(1000.0, samplerate, minimum=3))
    final_level = float(np.median(travel[-tail_n:]))
    total_motion = max(abs(final_level - baseline), 1e-12)
    plateau_ms = adaptive_ms_for_signal(
        base_ms=cfg.sp_plateau_ms,
        n_samples=travel.size,
        samplerate=samplerate,
        ratio=cfg.sp_plateau_ratio,
        min_ms=cfg.sp_plateau_min_ms,
        max_ms=cfg.sp_plateau_max_ms,
    )
    plateau_n = ms_to_samples(plateau_ms, samplerate, minimum=3)
    future_n = ms_to_samples(cfg.sp_future_verify_ms, samplerate, minimum=plateau_n)
    min_after_bp_n = ms_to_samples(cfg.sp_min_after_bp_ms, samplerate, minimum=1)
    search_start = max(baseline_n, 1)
    if bp_index is not None:
        search_start = max(search_start, bp_index + min_after_bp_n)
    upper = travel.size - plateau_n
    if upper <= search_start:
        return SPResult(None, None, None, 0.0, "not enough samples for multivariat sp")

    d1 = np.gradient(travel).astype(np.float32)
    slope_ref = max(float(np.percentile(np.abs(d1), 95)), 1e-12)
    local_slope_threshold = 0.10 * slope_ref
    final_tolerance = max(cfg.sp_final_tolerance_ratio * travel_range, 6.0 * baseline_std)
    future_motion_limit = max(
        cfg.sp_future_motion_ratio * travel_range,
        cfg.sp_min_motion_floor,
        6.0 * baseline_std,
    )

    travel_sp: SPResult | None = None
    for index in range(search_start, upper + 1):
        plateau_window = travel[index:index + plateau_n]
        if plateau_window.size < plateau_n:
            continue

        candidate_level = float(np.median(plateau_window))
        local_range = float(np.max(plateau_window) - np.min(plateau_window))
        local_slope = float(np.mean(np.abs(d1[index:index + plateau_n])))
        remaining = abs(final_level - candidate_level)
        progress = abs(candidate_level - baseline) / total_motion

        if progress < cfg.sp_progress_ratio:
            continue
        if remaining > final_tolerance:
            continue
        if local_range > future_motion_limit:
            continue
        if local_slope > local_slope_threshold:
            continue

        future_end = min(travel.size, index + future_n)
        future_window = travel[index:future_end]
        if future_window.size >= plateau_n:
            future_motion = float(np.max(future_window) - np.min(future_window))
            if future_motion > future_motion_limit:
                continue

        quiet_score = 1.0 - min(local_range / (future_motion_limit + 1e-12), 1.0)
        final_score = 1.0 - min(remaining / (final_tolerance + 1e-12), 1.0)
        progress_score = min(progress, 1.0)
        confidence = float(
            np.clip(
                0.35 * quiet_score + 0.35 * final_score + 0.30 * progress_score,
                0.0,
                1.0,
            )
        )
        travel_sp = SPResult(
            start_index=index,
            end_index=min(travel.size - 1, index + plateau_n - 1),
            sp_index=index,
            confidence=confidence,
            reason="multivariat first true near-final travel flattening used for sp",
        )
        break

    if travel_sp is None:
        return SPResult(None, None, None, 0.0, "no multivariat sp found")

    tolerance_n = ms_to_samples(cfg.sp_consensus_tolerance_ms, samplerate, minimum=1)
    if _indices_within_tolerance(travel_sp.sp_index, pressure_sp.sp_index, tolerance_n):
        return SPResult(
            start_index=travel_sp.start_index,
            end_index=travel_sp.end_index,
            sp_index=travel_sp.sp_index,
            confidence=min(travel_sp.confidence, pressure_sp.confidence),
            reason="multivariat travel plateau confirmed by pressure sp within tolerance used for sp",
        )

    return SPResult(None, None, None, 0.0, "travel sp and pressure sp disagree beyond tolerance")


# ------------ Target builders --------- #

# if PRP is found, PRP will be the center of the gaussian bump
def build_prp_target(
    n_samples: int,
    prp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0,
) -> np.ndarray:
    if prp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = prp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms,
    )

def build_bp_target(
    n_samples: int,
    bp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0,
) -> np.ndarray:
    if bp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = bp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms,
    )

def build_sp_target(
    n_samples: int,
    sp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0,
) -> np.ndarray:
    if sp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = sp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms,
    )
