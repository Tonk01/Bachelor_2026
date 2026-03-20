import numpy as np
import pytest

from src.data.targets import (
    detect_bp,
    detect_prp,
    moving_avg,
    ms_to_samples,
    safe_std,
    compute_future_drop,
    compute_drop_candidates,
    build_regions_with_stats,
    select_earliest_strong_region,
    tighten_region, 
    PRPConfig,
    PRPRegion,
)

def test_ms_to_sample():
    assert ms_to_samples(10, 400) == 4

def test_moving_avg_identify():
    x = np.array([1, 2, 3], dtype=np.float32)
    y = moving_avg(x, 1)
    assert np.allclose(x, y)

def test_safe_std():
    x = np.array([5, 5, 5], dtype=np.float32)
    assert safe_std(x) > 0

def test_compute_future_drop():
    signal = np.array([4, 5, 3, 2, 1], dtype=np.float32)
    future_drop = compute_future_drop(signal, future_samples = 2)

    assert future_drop.shape == signal.shape
    assert future_drop[0] > 0

def test_compute_drop_candidates():
    drop_strength = np.array([1.0, 3.0, 5.0], dtype = np.float32)
    future_drop = np.array([0.0, 1e-7, 1e-4], dtype = np.float32)

    mask = compute_drop_candidates(drop_strength, future_drop=future_drop, min_drop_multiplier=3.0, min_absolute_drop=1e-6)

    assert np.array_equal(mask, np.array([False, False, True]))

def test_build_regions_with_stats():
    mask = np.array([False, True, True, False])
    drop_strength = np.array([0, 3, 5, 0], dtype=np.float32)

    regions = build_regions_with_stats(mask, drop_strength)

    assert len(regions) == 1
    assert regions[0].start_index == 1
    assert regions[0].end_index == 2

def test_select_earliest_region():
    config = PRPConfig(min_drop_multiplier = 3.0)

    regions = [
        PRPRegion(10, 12, peak_strength=2.0, mean_strength=1.5),
        PRPRegion(20, 25, peak_strength=3.5, mean_strength=3.0),
        PRPRegion(30, 35, peak_strength=6.0, mean_strength=5.0)
    ]

    selected = select_earliest_strong_region(regions, config)

    assert selected.start_index == 20

def test_tighten_region_shrink():
    region = PRPRegion(10, 16, peak_strength=10.0, mean_strength=6.0)

    drop_strength = np.array(
        [0]*10 + [6, 7, 8, 10, 8, 7, 5] + [0]*5,
        dtype=np.float32
    )

    tightend = tighten_region(region, drop_strength, 0.8)
    assert tightend.start_index >= region.start_index
    assert tightend.end_index <= region.end_index

def test_detect_prp_no_event():
    config = PRPConfig()
    signal = np.ones(400, dtype=np.float32)

    result = detect_prp(signal, config)
    assert result.prp_index is None

def test_detect_prp_simple_drop():
    config = PRPConfig()
    signal = np.ones(400, dtype=np.float32)

    signal[100:140] = np.linspace(1.0, 0.2, 40)
    signal[140:] = 0.2

    result = detect_prp(signal, config)
    
    assert result.prp_index is not None
    assert result.start_index <= result.prp_index <= result.end_index

def test_detect_prp_prefer_early():
    config = PRPConfig(min_drop_multiplier=2.0)
    signal = np.ones(500, dtype=np.float32)

    signal[80:120] = np.linspace(1.0, 0.6, 40)
    signal[120:] = 0.6
    
    signal[300:340] = np.linspace(0.6, 0.1, 40)
    signal[340:] = 0.1
    
    result = detect_prp(signal, config)
    assert result.prp_index < 200

def test_moving_avg_constant_signal():
    x = np.ones(20, dtype=np.float32)
    y = moving_avg(x, 5)
    assert np.allclose(y, 1.0)

def test_detect_bp_simple_rise():
    samplerate = 400
    signal = np.zeros(400, dtype=np.float32)

    signal[120:200] = np.linspace(0, 10, 80, dtype=np.float32)
    result = detect_bp(signal, samplerate, prp_index=120)

    assert result.bp_index is not None
    assert result.bp_index >= 120


def test_detect_bp_simple_fall():
    samplerate = 400
    signal = np.ones(400, dtype=np.float32) * 10

    signal[120:200] = np.linspace(10, 0, 80, dtype=np.float32)
    signal[200:] = 0
    result = detect_bp(signal, samplerate, prp_index=120)

    assert result.bp_index is not None
    assert result.bp_index >= 120
