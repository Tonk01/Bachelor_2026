from __future__ import annotations

from dataclasses import dataclass
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
    confidence: float
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
    search_start_offset_ms: float = 400.0
    search_end_ratio: float = 0.75
    peak_sigma_mult: float = 3.0
    local_window_ms: float = 25.0
    min_index: int = 0

@dataclass(frozen = True)
class BPResult:
    bp_index: int | None
    confidence: float
    reason: str


@dataclass(frozen = True)
class SPConfig:
    smooth_ms: float = 10.0
    settle_threshold_ratio: float = 0.2
    significant_motion_ratio: float = 0.35
    settle_window_ms: float = 200.0
    min_peak_motion: float = 1e-6
    peak_motion_percentile: float = 99.9
    final_plateau_ms: float = 1000.0
    final_level_tolerance_ratio: float = 0.14


@dataclass(frozen = True)
class SPResult:
    sp_index: int | None
    confidence: float
    reason: str

def ms_to_samples(ms: float, samplerate: int, minimum: int = 1) -> int: 
    n = int(round((ms / 1000.0) * samplerate))
    return max(minimum, n)

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

def smooth_signal(signal: np.ndarray, config: PRPConfig) -> np.ndarray:
    return moving_avg(signal.astype(np.float32), config.smooth_samples)

def compute_local_std(signal: np.ndarray, window_samples: int, min_std: float = 1e-12) -> np.ndarray:
    n = len(signal)
    local_std = np.zeros(n, dtype=np.float32)
    half = window_samples // 2

    for i in range(n):
        start = max(0, i - half)
        stop = min(n, i + half + 1)
        window = signal[start:stop]

        if len(window) < 2:
            local_std[i] = min_std
        else: 
            local_std[i] = max(float(np.std(window)), min_std)
    
    return local_std

def compute_local_change(signal: np.ndarray, pre_window_samples: int, post_window_samples: int) -> np.ndarray:
    n = len(signal)
    local_change = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pre_start = max(0, i - pre_window_samples)
        pre_stop = i

        post_start = i
        post_stop = min(n, i + post_window_samples)

        pre_window = signal[pre_start:pre_stop]
        post_window = signal[post_start:post_stop]

        if len(pre_window) == 0 or len(post_window) == 0:
            local_change[i] = 0.0
        else:
            local_change[i] = float(np.mean(post_window) - np.mean(pre_window))

    return local_change

def compute_future_net_rise(signal: np.ndarray, future_samples: int,) -> np.ndarray:
    n = len(signal)
    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        stop = min(n, i + future_samples + 1)
        future_window = signal[i:stop]

        if len(future_window) == 0:
            out[i] = 0.0
        else:
            out[i] = float(np.max(future_window) - signal[i])

    return out

def compute_future_net_drop(signal: np.ndarray, future_samples: int,) -> np.ndarray:
    n = len(signal)
    out = np.zeros(n, dtype=np.float32)

    for i in range(n):
        stop = min(n, i + future_samples + 1)
        future_window = signal[i:stop]

        if len(future_window) == 0:
            out[i] = 0.0
        else:
            out[i] = float(signal[i] - np.min(future_window))

    return out
    

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
    is_drop = local_change < 0

    future_confirm = np.where(is_rise, future_net_rise, future_net_drop)

    return(change_strength >= min_change_multiplier) & (np.abs(local_change) >= min_absolute_change) & (future_confirm >= min_future_net_change)

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

def select_earliest_strong_region(regions: list[PRPRegion], config: PRPConfig,) -> PRPRegion | None: 
    
    for region in regions:
        if region.peak_strength >= config.min_change_multiplier:
            return region
    return None

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
            confidence=None,
            reason="No candidate region passed the minimum peak strength"
        )
    
    return PRPResult(
        start_index=selected_regions.start_index,
        end_index=selected_regions.end_index,
        prp_index=selected_regions.midpoint_index,
        confidence=selected_regions.peak_strength,
        reason=" Select midpoint of earliest strong candidate region"
    )

def detect_prp(signal: np.ndarray, config: PRPConfig) -> PRPResult:
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("detect prp excepts 1D signal")
    
    if len(signal) == 0:
        raise ValueError("detect prp expects a non empty signal")


    smoothed = smooth_signal(signal, config)

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

    selected_region = select_earliest_strong_region(
        regions=regions,
        config=config,
    )

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
    
# boolean array catching real drops, possible PRP's
def compute_drop_candidates(
        strength_signal: np.ndarray, 
        future_drop: np.ndarray,  
        min_drop_multiplier: float, 
        min_absolute_drop: float,
        ) -> np.ndarray:
    
    return (
        (strength_signal >= min_drop_multiplier) & (future_drop >= min_absolute_drop))


def find_active_window(
    signal: np.ndarray,
    samplerate: int,
    smooth_n: int,
    active_window_ms: float,
) -> tuple[int, int]:
    x_smooth = moving_avg(signal, smooth_n)
    d1 = np.gradient(x_smooth)
    activity = np.abs(d1)

    edge_guard = max(5, smooth_n * 2)
    if activity.size > 2 * edge_guard:
        trimmed_activity = activity[edge_guard:-edge_guard]
        peak = int(np.argmax(trimmed_activity)) + edge_guard
    else:
        peak = int(np.argmax(activity))

    pad = ms_to_samples(active_window_ms, samplerate)
    start = max(0, peak - pad)
    end = min(signal.size, peak + pad)
    return start, end

def detect_bp(
    signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    config: BPConfig | None = None,
) -> BPResult:
    cfg = config or BPConfig()

    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")

    if signal.size == 0:
        return BPResult(bp_index=None, confidence=0.0, reason="empty signal")

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    if prp_index is None:
        return BPResult(bp_index=None, confidence=0.0, reason="missing prp")

    x = np.asarray(signal, dtype=np.float32)

    if not np.isfinite(x).all():
        return BPResult(bp_index=None, confidence=0.0, reason="non-finite signal")

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    offset_n = ms_to_samples(cfg.search_start_offset_ms, samplerate)
    local_window_n = ms_to_samples(cfg.local_window_ms, samplerate)

    x_smooth = moving_avg(x, smooth_n)
    d1 = np.gradient(x_smooth)
    d2 = np.gradient(d1)
    abs_d2 = np.abs(d2)

    search_end = min(int(round(x.size * cfg.search_end_ratio)), x.size)
    primary_start = max(cfg.min_index, prp_index + offset_n)
    fallback_start = max(cfg.min_index, prp_index + 1)

    if search_end <= fallback_start + local_window_n:
        return BPResult(bp_index=None, confidence=0.0, reason="search window too short")

    baseline_start = max(0, prp_index - local_window_n)
    baseline = abs_d2[baseline_start:prp_index]
    if baseline.size == 0:
        baseline = abs_d2[:local_window_n]

    threshold = float(np.mean(baseline) + cfg.peak_sigma_mult * safe_std(baseline))

    def _scan_for_peak(window_start: int) -> BPResult | None:
        for i in range(window_start + 1, search_end - 1):
            value = float(abs_d2[i])

            left = max(window_start, i - local_window_n)
            right = min(search_end, i + local_window_n + 1)
            local_slice = abs_d2[left:right]

            is_local_peak = value >= float(np.max(local_slice))
            strong_enough = value > threshold

            if is_local_peak and strong_enough:
                confidence = float(np.clip(value / (threshold + 1e-8), 0.0, 1.0))
                return BPResult(bp_index=i, confidence=confidence, reason="detected")
        return None

    if primary_start + local_window_n < search_end:
        primary_result = _scan_for_peak(primary_start)
        if primary_result is not None:
            return primary_result

    fallback_result = _scan_for_peak(fallback_start)
    if fallback_result is not None:
        return fallback_result

    return BPResult(bp_index=None, confidence=0.0, reason="no bp found")


def detect_sp(
    signal: np.ndarray,
    samplerate: int,
    config: SPConfig | None = None,
) -> SPResult:
    cfg = config or SPConfig()

    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")

    if signal.size == 0:
        return SPResult(sp_index=None, confidence=0.0, reason="empty signal")

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    x = np.asarray(signal, dtype=np.float32)

    if not np.isfinite(x).all():
        return SPResult(sp_index=None, confidence=0.0, reason="non-finite signal")

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    settle_window_n = ms_to_samples(cfg.settle_window_ms, samplerate)
    final_plateau_n = ms_to_samples(cfg.final_plateau_ms, samplerate)

    x_smooth = moving_avg(x, smooth_n)
    motion = np.abs(np.gradient(x_smooth))
    edge_guard = max(5, smooth_n * 2)

    if motion.size > 2 * edge_guard:
        trimmed_motion = motion[edge_guard: signal.size - edge_guard]
    else:
        trimmed_motion = motion

    peak_motion = float(np.percentile(trimmed_motion, cfg.peak_motion_percentile))
    if peak_motion < cfg.min_peak_motion:
        return SPResult(sp_index=None, confidence=0.0, reason="no significant motion")

    plateau_n = min(final_plateau_n, max(settle_window_n, signal.size // 5))
    plateau_start = max(edge_guard, signal.size - plateau_n)
    plateau = x_smooth[plateau_start: signal.size - edge_guard] if signal.size - edge_guard > plateau_start else x_smooth[plateau_start:]
    if plateau.size == 0:
        plateau = x_smooth[max(0, signal.size - final_plateau_n):]
    if plateau.size == 0:
        return SPResult(sp_index=None, confidence=0.0, reason="no final plateau window")

    final_level = float(np.mean(plateau))
    signal_range = float(np.max(x_smooth) - np.min(x_smooth))
    plateau_std = safe_std(plateau)
    level_tolerance = max(
        3.0 * plateau_std,
        cfg.final_level_tolerance_ratio * signal_range,
    )
    settle_threshold = cfg.settle_threshold_ratio * peak_motion

    search_end = max(edge_guard + 1, plateau_start)
    for i in range(search_end - 1, edge_guard, -1):
        local_motion = motion[i:min(signal.size, i + settle_window_n)]
        if local_motion.size == 0:
            continue

        distance_from_final = abs(float(x_smooth[i]) - final_level)
        max_motion = float(np.max(local_motion))

        if distance_from_final > level_tolerance or max_motion > settle_threshold:
            sp_index = min(i + 1, signal.size - 1)
            confidence = float(
                np.clip(
                    max(
                        distance_from_final / (level_tolerance + 1e-8),
                        max_motion / (settle_threshold + 1e-8),
                    ) - 1.0,
                    0.0,
                    1.0,
                )
            )
            return SPResult(sp_index=sp_index, confidence=confidence, reason="detected")

    return SPResult(sp_index=None, confidence=0.0, reason="no sp found")
    
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
