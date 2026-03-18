from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass (frozen = True)
class PRPconfig:
    smooth_ms: float = 10.0
    baseline_ms: float = 80.0
    search_end_ratio: float = 0.35
    min_event_ms: float = 12.0
    slope_sigma_mult: float = 3.0 
    min_amplitude_sigma: float = 2.0
    min_index: int = 0

@dataclass (frozen = True)
class PRPResult:
    prp_index: int | None
    confidence: float
    reason: str
    direction: str | None = None        # rise or falling. (open or close)


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

def ms_to_samples(ms: float, samplerate: int, minimum: int = 1) -> int: 
    n = int(round((ms / 1000.0) * samplerate))
    return max(minimum, n)

def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    
    kernel = np.ones(window, dtype = np.float32) / float(window)
    return np.convolve(x, kernel, mode = "same").astype(np.float32)

def safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    std = float(np.std(x))
    return max(std, eps)

def detect_prp(signal: np.ndarray, samplerate: int, config: PRPconfig | None = None) -> PRPResult:
    cfg = config or PRPconfig()

    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")

    if signal.size == 0:
        return PRPResult(prp_index=None, confidence=0.0, reason="empty signal", direction = None)

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    x = np.asarray(signal, dtype=np.float32)

    if not np.isfinite(x).all():
        return PRPResult(prp_index=None, confidence=0.0, reason="non-finite signal", direction = None)

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    baseline_n = ms_to_samples(cfg.baseline_ms, samplerate)
    min_event_n = ms_to_samples(cfg.min_event_ms, samplerate) 

    x_smooth = moving_avg(x, smooth_n)
    dx = np.diff(x_smooth, prepend=x_smooth[0])

    search_end = max(cfg.min_index + 1, int(round(x.size * cfg.search_end_ratio)))
    search_end = min(search_end, x.size)

    if search_end <= baseline_n + min_event_n:
        return PRPResult(prp_index=None, confidence=0.0, reason="signal too short for PRP", direction = None)

    baseline_x = x_smooth[:baseline_n]
    baseline_dx = dx[:baseline_n]

    baseline_mean = float(np.mean(baseline_x))
    baseline_std = safe_std(baseline_x)

    slope_mean = float(np.mean(baseline_dx))
    slope_std = safe_std(baseline_dx)

    pos_slope_threshold = slope_mean + cfg.slope_sigma_mult * slope_std
    neg_slope_threshold = slope_mean - cfg.slope_sigma_mult * slope_std

    upper_amplitude_threshold = baseline_mean + cfg.min_amplitude_sigma * baseline_std
    lower_amplitude_threshold = baseline_mean - cfg.min_amplitude_sigma * baseline_std

    # debugg
    print("baseline_mean           :", baseline_mean)
    print("baseline_std            :", baseline_std)
    print("slope_mean              :", slope_mean)
    print("slope_std               :", slope_std)
    print("pos_slope_threshold     :", pos_slope_threshold)
    print("neg_slope_threshold     :", neg_slope_threshold)
    print("upper_amp_threshold     :", upper_amplitude_threshold)
    print("lower_amp_threshold     :", lower_amplitude_threshold)
    print("baseline_n              :", baseline_n)
    print("min_event_n              :", min_event_n)
    print("search_end              :", search_end)

    start_idx = max(cfg.min_index, baseline_n)

    for i in range(start_idx, search_end - min_event_n + 1):
        local_dx = dx[i:i + min_event_n]
        local_x = x_smooth[i:i + min_event_n]

        mean_slope = float(np.mean(local_dx))
        max_amp = float(np.max(local_x))
        min_amp = float(np.min(local_x))

        pos_count = int(np.sum(local_dx > 0.0))
        neg_count = int(np.sum(local_dx < 0.0))

        # opening sequence check
        rise_slope_ok = mean_slope > pos_slope_threshold
        rise_amplitude_ok = max_amp > upper_amplitude_threshold
        rise_persistence_ok = pos_count >= max(1, int(0.8 * min_event_n))

        # closing sequence check
        fall_slope_ok = mean_slope < neg_slope_threshold
        fall_amplitude_ok = min_amp < lower_amplitude_threshold
        fall_persistence_ok = neg_count >= max(1, int(0.5 * min_event_n))

        rise_valid = rise_slope_ok and rise_amplitude_ok and rise_persistence_ok
        fall_valid = fall_slope_ok and fall_amplitude_ok and fall_persistence_ok

        rise_score = 0.0
        if rise_valid:
            slope_score = mean_slope / (pos_slope_threshold + 1e-8)
            amp_score = (max_amp - baseline_mean) / baseline_std
            rise_score = 0.25 * slope_score + 0.1 * amp_score

        fall_score = 0.0
        if fall_valid:
            slope_score = abs(mean_slope) / (abs(neg_slope_threshold) + 1e-8)
            amp_score = (baseline_mean - min_amp) / baseline_std
            fall_score = 0.25 * slope_score + 0.1 * amp_score
    
        if rise_valid:
            confidence = float(np.clip(rise_score, 0.0, 1.0))

            return PRPResult(prp_index = i, confidence = confidence, reason = "detected", direction = "rise")

        if  fall_valid:
            confidence = float(np.clip(fall_score, 0.0, 1.0))

            return PRPResult(prp_index = i, confidence = confidence, reason = "detected", direction = "fall")

    return PRPResult(prp_index = None, confidence = 0.0, reason = "no prp found", direction = None)

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
