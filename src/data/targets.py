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
    settle_threshold_ratio: float = 0.12
    min_peak_motion: float = 1e-6
    final_plateau_window_ms: float = 800.0
    acceleration_sigma_mult: float = 2.0
    late_search_ratio: float = 0.35
    pre_window_ms: float = 40.0
    post_window_ms: float = 40.0
    noise_window_ms: float = 40.0
    future_confirm_ms: float = 150.0
    min_change_multiplier: float = 3.3
    min_absolute_change: float = 8e-6
    min_future_net_change: float = 7e-6
    inner_region_ratio: float = 0.7
    min_region_width: int = 3


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

def ms_to_samples(ms: float, samplerate: int, minimum: int = 1) -> int: 
    n = int(round((ms / 1000.0) * samplerate))
    return max(minimum, n)

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

def safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    std = float(np.std(x))
    return max(std, eps)

    # Gaussian bump to get a soft location target (trying to start with a "general" location for PRP)
def gaussian(n_samples: int, center: int, samplerate: int, sigma_ms: float) -> np.ndarray:
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

    smooth_10 = moving_avg(x, smooth_10_n)
    d1_10 = np.gradient(smooth_10).astype(np.float32)
    d2_10 = np.gradient(d1_10).astype(np.float32)

    abs_d1 = np.abs(d1_10).astype(np.float32)
    abs_d2 = np.abs(d2_10).astype(np.float32)
    
    smooth_15 = moving_avg(x, smooth_15_n)

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
        window_start = 1 - window_size + 1

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
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="missing prp")

    shared_features = shared or compute_shared_features(x, samplerate)

    x_smooth = shared_features.smooth_10
    d1 = shared_features.d1_10
    d2 = shared_features.d2_10
    motion = shared_features.abs_d1


    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    plateau_window_n = min(
        ms_to_samples(cfg.final_plateau_window_ms, samplerate),
        max(20, x.size // 4),
    )

    edge_guard = max(5, smooth_n * 2)

    if motion.size > 2 * edge_guard:
        trimmed_motion = motion[edge_guard: x.size - edge_guard]
    else:
        trimmed_motion = motion

    peak_motion = float(np.max(trimmed_motion))
    if peak_motion < cfg.min_peak_motion:
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="no significant motion")

    settle_threshold = cfg.settle_threshold_ratio * peak_motion

    plateau_start: int | None = None
    last_search_start = max(edge_guard, signal.size - plateau_window_n - edge_guard)

    for i in range(last_search_start, edge_guard - 1, -1):
        window_motion = motion[i:i + plateau_window_n]
        if window_motion.size < plateau_window_n:
            continue
        if float(np.max(window_motion)) <= settle_threshold:
            plateau_start = i
        elif plateau_start is not None:
            break

    if plateau_start is None:
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="no final plateau found")

    base_search_start = max(edge_guard, prp_index + 1)
    search_end = plateau_start
    total_search_width = search_end - base_search_start
    late_search_start = search_end - int(round(total_search_width * cfg.late_search_ratio))
    search_start = max(base_search_start, late_search_start)

    if search_end <= search_start + 3:
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="no pre-plateau region")

    signed_accel = d2[search_start:search_end]
    if signed_accel.size > 0:
        overall_direction = float(x_smooth[plateau_start] - x_smooth[prp_index])
        direction_sign = 1.0 if overall_direction >= 0.0 else -1.0
        directional_accel = signed_accel * direction_sign

        accel_baseline = directional_accel[:max(5, min(len(directional_accel), ms_to_samples(200.0, samplerate)))]
        accel_threshold = float(np.mean(accel_baseline) + cfg.acceleration_sigma_mult * safe_std(accel_baseline))
        accel_threshold = max(accel_threshold, 0.1 * float(np.max(directional_accel)))

        accel_mask = directional_accel >= accel_threshold
        accel_strength = np.clip(directional_accel / (np.max(directional_accel) + 1e-8), 0.0, None)
        accel_regions = build_regions_with_stats(accel_mask, accel_strength)

        selected_accel_region: PRPRegion | None = None
        fallback_accel_region: PRPRegion | None = None

        for region in accel_regions:
            if fallback_accel_region is None:
                fallback_accel_region = region
            if region.width >= cfg.min_region_width:
                selected_accel_region = region
                break

        if selected_accel_region is None:
            selected_accel_region = fallback_accel_region

        if selected_accel_region is not None:
            selected_accel_region = tighten_region(
                region=selected_accel_region,
                strength_signal=accel_strength,
                inner_region_ratio=cfg.inner_region_ratio,
            )

            region_start = search_start + selected_accel_region.start_index
            region_end = search_start + selected_accel_region.end_index
            sp_index = region_start

            return SPResult(
                start_index=region_start,
                end_index=region_end,
                sp_index=sp_index,
                confidence=float(selected_accel_region.peak_strength),
                reason="Select start of earliest strong acceleration region in late pre-plateau window",
            )

    search_signal = x_smooth[search_start:search_end]
    if search_signal.size < 5:
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="empty sp search window")

    # Reverse the pre-plateau segment so the latest strong region before plateau
    # becomes the earliest strong region in reversed time.
    reversed_signal = search_signal[::-1].copy()

    pre_window_n = ms_to_samples(cfg.pre_window_ms, samplerate)
    post_window_n = ms_to_samples(cfg.post_window_ms, samplerate)
    noise_window_n = ms_to_samples(cfg.noise_window_ms, samplerate)
    future_confirm_n = ms_to_samples(cfg.future_confirm_ms, samplerate)

    local_std = compute_local_std(reversed_signal, noise_window_n)
    local_change = compute_local_change(reversed_signal, pre_window_n, post_window_n)
    change_strength = compute_change_strength(local_change=local_change, local_std=local_std)
    future_net_rise = compute_future_net_rise(reversed_signal, future_confirm_n)
    future_net_drop = compute_future_net_drop(reversed_signal, future_confirm_n)

    candidate_mask = compute_change_candidates(
        local_change=local_change,
        change_strength=change_strength,
        future_net_rise=future_net_rise,
        future_net_drop=future_net_drop,
        min_change_multiplier=cfg.min_change_multiplier,
        min_absolute_change=cfg.min_absolute_change,
        min_future_net_change=cfg.min_future_net_change,
    )

    reversed_regions = build_regions_with_stats(candidate_mask, change_strength)

    selected_region: PRPRegion | None = None
    fallback_region: PRPRegion | None = None

    for region in reversed_regions:
        if region.peak_strength >= cfg.min_change_multiplier:
            if fallback_region is None:
                fallback_region = region
            if region.width >= cfg.min_region_width:
                selected_region = region
                break

    if selected_region is None:
        selected_region = fallback_region

    if selected_region is None:
        return SPResult(start_index=None, end_index=None, sp_index=None, confidence=0.0, reason="no strong sp region before plateau")

    selected_region = tighten_region(
        region=selected_region,
        strength_signal=change_strength,
        inner_region_ratio=cfg.inner_region_ratio,
    )

    region_start = search_end - 1 - selected_region.end_index
    region_end = search_end - 1 - selected_region.start_index
    sp_index = (region_start + region_end) // 2

    return SPResult(
        start_index=region_start,
        end_index=region_end,
        sp_index=sp_index,
        confidence=float(selected_region.peak_strength),
        reason="Select midpoint of latest strong candidate region before plateau",
    )
    

# ------------ Target builders --------- #

# if PRP is found, PRP will be the center of the gaussian bump
def build_prp_target(
    n_samples: int,
    prp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0 
) -> np.ndarray:
    if prp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = prp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms
    )

def build_bp_target(
    n_samples: int,
    bp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0 
) -> np.ndarray:
    if bp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = bp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms
    )

def build_sp_target(
    n_samples: int,
    sp_index: int | None,
    samplerate: int,
    sigma_ms: float = 10.0 
) -> np.ndarray:
    if sp_index is None:
        return np.zeros(n_samples, dtype = np.float32)

    return gaussian(
        n_samples = n_samples,
        center = sp_index, 
        samplerate = samplerate,
        sigma_ms = sigma_ms
    )
