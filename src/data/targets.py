from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass (frozen = True)
class PRPconfig:
    smooth_ms: float = 10.0
    baseline_ms: float = 80.0
    search_end_ratio: float = 0.35
    min_rise_ms: float = 12.0
    slope_sigma_mult: float = 3.0 
    min_amplitude_sigma: float = 2.0
    min_index: int = 0

@dataclass (frozen = True)
class PRPResult:
    prp_index: int | None
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
        return PRPResult(prp_index=None, confidence=0.0, reason="empty signal")

    if samplerate <= 0:
        raise ValueError("samplerate must be above 0")

    x = np.asarray(signal, dtype=np.float32)

    if not np.isfinite(x).all():
        return PRPResult(prp_index=None, confidence=0.0, reason="non-finite signal")

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    baseline_n = ms_to_samples(cfg.baseline_ms, samplerate)
    min_rise_n = ms_to_samples(cfg.min_rise_ms, samplerate)

    x_smooth = moving_avg(x, smooth_n)
    dx = np.diff(x_smooth, prepend=x_smooth[0])

    search_end = max(cfg.min_index + 1, int(round(x.size * cfg.search_end_ratio)))
    search_end = min(search_end, x.size)

    if search_end <= baseline_n + min_rise_n:
        return PRPResult(prp_index=None, confidence=0.0, reason="signal too short for PRP")

    baseline_x = x_smooth[:baseline_n]
    baseline_dx = dx[:baseline_n]

    baseline_mean = float(np.mean(baseline_x))
    baseline_std = safe_std(baseline_x)

    slope_mean = float(np.mean(baseline_dx))
    slope_std = safe_std(baseline_dx)

    slope_threshold = slope_mean + cfg.slope_sigma_mult * slope_std
    amplitude_threshold = baseline_mean + cfg.min_amplitude_sigma * baseline_std

    # debugg
    print("baseline_mean       :", baseline_mean)
    print("baseline_std        :", baseline_std)
    print("slope_mean          :", slope_mean)
    print("slope_std           :", slope_std)
    print("slope_threshold     :", slope_threshold)
    print("amplitude_threshold :", amplitude_threshold)
    print("baseline_n          :", baseline_n)
    print("min_rise_n          :", min_rise_n)
    print("search_end          :", search_end)

    start_idx = max(cfg.min_index, baseline_n)

    best_i = None
    best_mean_slope = -1e18
    best_max_amp = -1e18
    best_pos_count = -1

    for i in range(start_idx, search_end - min_rise_n):
        local_dx = dx[i:i + min_rise_n]
        local_x = x_smooth[i:i + min_rise_n]

        mean_slope = float(np.mean(local_dx))
        max_amp = float(np.max(local_x))
        pos_count = int(np.sum(local_dx > 0.0))

        if mean_slope > best_mean_slope:
            best_i = i
            best_mean_slope = mean_slope
            best_max_amp = max_amp
            best_pos_count = pos_count

        slope_ok = mean_slope > slope_threshold
        amplitude_ok = max_amp > amplitude_threshold
        persistence_ok = pos_count >= max(1, int(0.8 * min_rise_n))

        if slope_ok and amplitude_ok and persistence_ok:
            slope_score = mean_slope / (slope_threshold + 1e-8)
            amp_score = (max_amp - baseline_mean) / baseline_std
            confidence = float(np.clip(0.25 * slope_score + 0.1 * amp_score, 0.0, 1.0))

            #debugg
            print("detected_i          :", i)
            print("detected_mean_slope :", mean_slope)
            print("detected_max_amp    :", max_amp)
            print("detected_pos_count  :", pos_count)

            return PRPResult(prp_index=i, confidence=confidence, reason="detected")

    # debugg
    print("best_i              :", best_i)
    print("best_mean_slope     :", best_mean_slope)
    print("best_max_amp        :", best_max_amp)
    print("best_pos_count      :", best_pos_count)

    return PRPResult(prp_index=None, confidence=0.0, reason="no prp found")
    
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