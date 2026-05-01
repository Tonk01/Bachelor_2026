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


@dataclass(frozen=True)
class MultiSensorBPConfig:
    search_start_offset_ms: float = 60.0
    search_end_ratio: float = 0.75
    travel_smooth_ms: float = 25.0
    sustain_window_ms: float = 80.0
    net_drop_window_ms: float = 200.0
    onset_threshold_ratio: float = 0.20
    sustain_threshold_ratio: float = 0.15
    min_net_drop_ratio: float = 0.03
    max_pressure_bp_shift_ms: float = 800.0
    consensus_window_ms: float = 600.0
    fallback_to_pressure_bp: bool = True

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


@dataclass(frozen=True)
class MultiSensorSPConfig:
    search_start_offset_ms: float = 80.0
    search_end_ratio: float = 0.95
    travel_smooth_ms: float = 25.0
    motion_window_ms: float = 200.0
    sustain_window_ms: float = 80.0
    plateau_window_ms: float = 150.0
    motion_threshold_ratio: float = 0.25
    plateau_threshold_ratio: float = 0.12
    max_settle_motion_ratio: float = 0.02
    max_pressure_sp_shift_ms: float = 1200.0
    consensus_window_ms: float = 700.0
    fallback_to_pressure_sp: bool = True


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


def detect_bp_multisensor(
    pressure_signal: np.ndarray,
    travel_signal: np.ndarray | None,
    samplerate: int,
    prp_index: int | None,
    strain_signal: np.ndarray | None = None,
    pressure_config: BPConfig | None = None,
    multisensor_config: MultiSensorBPConfig | None = None,
    shared_pressure: SharedSignalFeatures | None = None,
) -> BPResult:
    pressure_bp = detect_bp(
        pressure_signal,
        samplerate=samplerate,
        prp_index=prp_index,
        config=pressure_config,
        shared=shared_pressure,
    )

    cfg = multisensor_config or MultiSensorBPConfig()

    if prp_index is None:
        return pressure_bp

    if samplerate <= 0:
        raise ValueError("samplerate must be more than 0")

    smooth_n = ms_to_samples(cfg.travel_smooth_ms, samplerate, minimum=3)
    sustain_n = ms_to_samples(cfg.sustain_window_ms, samplerate, minimum=3)
    net_motion_n = ms_to_samples(cfg.net_drop_window_ms, samplerate, minimum=3)
    offset_n = ms_to_samples(cfg.search_start_offset_ms, samplerate)
    max_shift_n = ms_to_samples(cfg.max_pressure_bp_shift_ms, samplerate)
    def _main_motion_region(motion_signal: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, bool, float, float] | None:
        motion = validate_signal_1d(motion_signal)
        search_start = max(prp_index + offset_n, 1)
        search_end = min(int(round(motion.size * cfg.search_end_ratio)), motion.size - 1)

        if pressure_bp.bp_index is not None:
            search_end = min(search_end, pressure_bp.bp_index + max_shift_n)

        if search_end <= search_start + sustain_n:
            return None

        smooth_motion = moving_avg(motion.astype(np.float32), smooth_n)
        d1_motion = np.gradient(smooth_motion).astype(np.float32)
        motion_window = smooth_motion[search_start:search_end]
        window_net_motion = float(motion_window[-1] - motion_window[0]) if motion_window.size >= 2 else 0.0
        positive_direction = window_net_motion >= 0.0

        direction_strength = (
            np.maximum(d1_motion, 0.0).astype(np.float32)
            if positive_direction
            else np.maximum(-d1_motion, 0.0).astype(np.float32)
        )

        strength_window = direction_strength[search_start:search_end]
        if strength_window.size == 0:
            return None

        reference_peak = float(np.max(strength_window))
        if reference_peak <= 0.0:
            return None

        onset_threshold = cfg.onset_threshold_ratio * reference_peak
        motion_range = float(np.max(smooth_motion[search_start:search_end]) - np.min(smooth_motion[search_start:search_end]))
        min_net_drop = cfg.min_net_drop_ratio * max(motion_range, 1e-8)

        motion_mask = strength_window >= onset_threshold
        regions = build_regions_with_stats(motion_mask, strength_window)
        valid_regions: list[PRPRegion] = []

        for region in regions:
            region_start = search_start + region.start_index
            region_end = search_start + region.end_index

            if region_end - region_start + 1 < sustain_n:
                continue
            if region_start + net_motion_n >= motion.size:
                continue

            net_motion = float(smooth_motion[region_start + net_motion_n] - smooth_motion[region_start])
            if positive_direction:
                if net_motion < min_net_drop:
                    continue
            else:
                if net_motion > -min_net_drop:
                    continue

            valid_regions.append(region)

        if not valid_regions:
            return None

        main_region = max(valid_regions, key=lambda r: (r.area_strength, r.peak_strength, -r.start_index))
        main_start = search_start + main_region.start_index
        main_end = search_start + main_region.end_index
        confidence = float(
            np.clip(main_region.area_strength / (sum(r.area_strength for r in valid_regions) + 1e-8), 0.0, 1.0)
        )
        return main_start, main_end, smooth_motion, direction_strength, positive_direction, onset_threshold, confidence

    def _strain_shift_inside_region(
        smooth_strain: np.ndarray,
        region_start: int,
        region_end: int,
    ) -> int | None:
        if region_end <= region_start + sustain_n:
            return None

        d1_strain = np.gradient(smooth_strain).astype(np.float32)
        local_window = d1_strain[region_start:region_end + 1]
        if local_window.size < sustain_n:
            return None

        positive_direction = float(smooth_strain[region_end] - smooth_strain[region_start]) >= 0.0
        direction_strength = (
            np.maximum(d1_strain, 0.0).astype(np.float32)
            if positive_direction
            else np.maximum(-d1_strain, 0.0).astype(np.float32)
        )
        local_strength = direction_strength[region_start:region_end + 1]
        peak = float(np.max(local_strength))
        if peak <= 0.0:
            return None

        threshold = max(cfg.onset_threshold_ratio * peak, 0.35 * peak)
        upper_bound = region_end - sustain_n + 1
        for i in range(region_start, upper_bound + 1):
            if float(direction_strength[i]) < threshold:
                continue
            sustain_slice = direction_strength[i:i + sustain_n]
            if sustain_slice.size < sustain_n:
                continue
            if float(np.mean(sustain_slice)) < 0.75 * threshold:
                continue
            return i
        return None

    travel_region = _main_motion_region(travel_signal) if travel_signal is not None else None
    if travel_region is not None:
        region_start, region_end, smooth_travel, direction_strength, positive_direction, onset_threshold, confidence = travel_region
        if strain_signal is not None:
            smooth_strain = moving_avg(validate_signal_1d(strain_signal).astype(np.float32), smooth_n)
            strain_candidate = _strain_shift_inside_region(smooth_strain, region_start, region_end)
            if strain_candidate is not None:
                direction_label = "positive" if positive_direction else "negative"
                return BPResult(
                    start_index=region_start,
                    end_index=region_end,
                    bp_index=strain_candidate,
                    confidence=confidence,
                    reason=f"strain shift selected inside travel main {direction_label} motion region",
                )

        upper_bound = min(region_end, len(direction_strength) - sustain_n)
        for i in range(region_start, upper_bound + 1):
            if float(direction_strength[i]) < onset_threshold:
                continue
            sustain_slice = direction_strength[i:i + sustain_n]
            if sustain_slice.size < sustain_n:
                continue
            if float(np.mean(sustain_slice)) < cfg.sustain_threshold_ratio * float(np.max(direction_strength[region_start:region_end + 1])):
                continue
            direction_label = "positive" if positive_direction else "negative"
            return BPResult(
                start_index=region_start,
                end_index=region_end,
                bp_index=i,
                confidence=confidence,
                reason=f"travel main {direction_label} motion region onset used for bp",
            )

    strain_region = _main_motion_region(strain_signal) if strain_signal is not None else None
    if strain_region is not None:
        region_start, region_end, _smooth, _strength, positive_direction, _threshold, confidence = strain_region
        direction_label = "positive" if positive_direction else "negative"
        return BPResult(
            start_index=region_start,
            end_index=region_end,
            bp_index=region_start,
            confidence=confidence,
            reason=f"strain main {direction_label} motion region onset used as bp fallback",
        )

    return pressure_bp


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


def detect_sp_multisensor(
    pressure_signal: np.ndarray,
    travel_signal: np.ndarray | None,
    samplerate: int,
    prp_index: int | None,
    bp_index: int | None,
    strain_signal: np.ndarray | None = None,
    pressure_config: SPConfig | None = None,
    multisensor_config: MultiSensorSPConfig | None = None,
    shared_pressure: SharedSignalFeatures | None = None,
) -> SPResult:
    pressure_sp = detect_sp(
        pressure_signal,
        samplerate=samplerate,
        prp_index=prp_index,
        config=pressure_config,
        shared=shared_pressure,
    )

    cfg = multisensor_config or MultiSensorSPConfig()

    if prp_index is None:
        return pressure_sp

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    smooth_n = ms_to_samples(cfg.travel_smooth_ms, samplerate, minimum=3)
    motion_n = ms_to_samples(cfg.motion_window_ms, samplerate, minimum=3)
    sustain_n = ms_to_samples(cfg.sustain_window_ms, samplerate, minimum=3)
    plateau_n = ms_to_samples(cfg.plateau_window_ms, samplerate, minimum=3)
    offset_n = ms_to_samples(cfg.search_start_offset_ms, samplerate)
    max_shift_n = ms_to_samples(cfg.max_pressure_sp_shift_ms, samplerate)
    def _settle_region(motion_signal: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, bool, float, float] | None:
        motion = validate_signal_1d(motion_signal)
        search_anchor = bp_index if bp_index is not None else prp_index
        search_start = max(search_anchor + offset_n, 1)
        search_end = min(int(round(motion.size * cfg.search_end_ratio)), motion.size - 1)

        if pressure_sp.sp_index is not None:
            search_end = min(search_end, pressure_sp.sp_index + max_shift_n)

        if search_end <= search_start + max(motion_n, plateau_n):
            return None

        smooth_motion = moving_avg(motion.astype(np.float32), smooth_n)
        d1_motion = np.gradient(smooth_motion).astype(np.float32)

        direction_window = smooth_motion[search_start:search_end]
        if direction_window.size < 2:
            return None

        window_net_motion = float(direction_window[-1] - direction_window[0])
        positive_direction = window_net_motion >= 0.0

        direction_strength = (
            np.maximum(d1_motion, 0.0).astype(np.float32)
            if positive_direction
            else np.maximum(-d1_motion, 0.0).astype(np.float32)
        )
        strength_window = direction_strength[search_start:search_end]
        if strength_window.size == 0:
            return None

        reference_peak = float(np.max(strength_window))
        if reference_peak <= 0.0:
            return None

        motion_threshold = cfg.motion_threshold_ratio * reference_peak
        motion_range = float(np.max(direction_window) - np.min(direction_window))
        max_settle_motion = cfg.max_settle_motion_ratio * max(motion_range, 1e-8)

        motion_start: int | None = None
        motion_peak_index: int | None = None
        upper_motion_bound = min(search_end - sustain_n - 1, motion.size - 2)

        for i in range(search_start, upper_motion_bound + 1):
            if float(direction_strength[i]) < motion_threshold:
                continue
            sustain_slice = direction_strength[i:i + sustain_n]
            if sustain_slice.size < sustain_n:
                continue
            if float(np.mean(sustain_slice)) < motion_threshold:
                continue
            motion_start = i
            local_peak_offset = int(np.argmax(direction_strength[i:search_end]))
            motion_peak_index = i + local_peak_offset
            break

        if motion_start is None or motion_peak_index is None:
            return None

        settle_start = max(motion_peak_index + 1, motion_start + sustain_n // 2)
        upper_plateau_bound = min(search_end - plateau_n - 1, motion.size - plateau_n - 1)
        for i in range(settle_start, upper_plateau_bound + 1):
            plateau_slice = direction_strength[i:i + plateau_n]
            if plateau_slice.size < plateau_n:
                continue
            plateau_threshold = cfg.plateau_threshold_ratio * reference_peak
            if float(np.mean(plateau_slice)) > plateau_threshold:
                continue
            if float(np.max(plateau_slice)) > motion_threshold:
                continue

            net_motion = float(smooth_motion[i + plateau_n] - smooth_motion[i])
            if positive_direction:
                if net_motion > max_settle_motion:
                    continue
            else:
                if net_motion < -max_settle_motion:
                    continue

            confidence = float(
                np.clip(
                    1.0 - (float(np.mean(plateau_slice)) / (reference_peak + 1e-8)),
                    0.0,
                    1.0,
                )
            )
            region_end = i + plateau_n - 1
            return i, region_end, smooth_motion, direction_strength, positive_direction, reference_peak, confidence

        return None

    def _strain_settle_inside_region(
        smooth_strain: np.ndarray,
        region_start: int,
        region_end: int,
    ) -> int | None:
        if region_end <= region_start + plateau_n:
            return None

        d1_strain = np.gradient(smooth_strain).astype(np.float32)
        local_window = d1_strain[region_start:region_end + 1]
        if local_window.size < plateau_n:
            return None

        positive_direction = float(smooth_strain[region_end] - smooth_strain[region_start]) >= 0.0
        direction_strength = (
            np.maximum(d1_strain, 0.0).astype(np.float32)
            if positive_direction
            else np.maximum(-d1_strain, 0.0).astype(np.float32)
        )
        local_strength = direction_strength[region_start:region_end + 1]
        peak = float(np.max(local_strength))
        if peak <= 0.0:
            return None

        plateau_threshold = cfg.plateau_threshold_ratio * peak
        upper_bound = region_end - plateau_n + 1
        for i in range(region_start, upper_bound + 1):
            plateau_slice = direction_strength[i:i + plateau_n]
            if plateau_slice.size < plateau_n:
                continue
            if float(np.mean(plateau_slice)) > plateau_threshold:
                continue
            return i
        return None

    travel_region = _settle_region(travel_signal) if travel_signal is not None else None
    if travel_region is not None:
        region_start, region_end, _smooth, _strength, positive_direction, _reference_peak, confidence = travel_region
        if strain_signal is not None:
            smooth_strain = moving_avg(validate_signal_1d(strain_signal).astype(np.float32), smooth_n)
            strain_candidate = _strain_settle_inside_region(smooth_strain, region_start, region_end)
            if strain_candidate is not None:
                direction_label = "positive" if positive_direction else "negative"
                return SPResult(
                    start_index=region_start,
                    end_index=region_end,
                    sp_index=strain_candidate,
                    confidence=confidence,
                    reason=f"strain settling selected inside travel {direction_label} plateau region",
                )

        direction_label = "positive" if positive_direction else "negative"
        return SPResult(
            start_index=region_start,
            end_index=region_end,
            sp_index=region_start,
            confidence=confidence,
            reason=f"travel {direction_label} motion settled into plateau region used for sp",
        )

    strain_region = _settle_region(strain_signal) if strain_signal is not None else None
    if strain_region is not None:
        region_start, region_end, _smooth, _strength, positive_direction, _reference_peak, confidence = strain_region
        direction_label = "positive" if positive_direction else "negative"
        return SPResult(
            start_index=region_start,
            end_index=region_end,
            sp_index=region_start,
            confidence=confidence,
            reason=f"strain {direction_label} plateau region used as sp fallback",
        )

    return pressure_sp
        
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
