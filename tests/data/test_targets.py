import numpy as np
import pytest

from src.data.targets import (
    BPConfig,
    PRPConfig,
    PRPRegion,
    SPConfig,
    build_regions_with_stats,
    compute_change_candidates,
    compute_future_net_drop,
    compute_future_net_rise,
    detect_bp,
    detect_prp,
    detect_sp,
    moving_avg,
    ms_to_samples,
    safe_std,
    select_earliest_strong_region,
    tighten_region,
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
    future_drop = compute_future_net_drop(signal, future_samples = 2)

    assert future_drop.shape == signal.shape
    assert future_drop[0] > 0

def test_compute_future_rise():
    signal = np.array([1, 2, 5, 4, 3], dtype=np.float32)
    future_rise = compute_future_net_rise(signal, future_samples=2)

    assert future_rise.shape == signal.shape
    assert future_rise[0] > 0

def test_compute_change_candidates():
    change_strength = np.array([1.0, 3.0, 5.0], dtype = np.float32)
    local_change = np.array([1e-7, 1e-7, 1e-4], dtype=np.float32)
    future_rise = np.array([0.0, 1e-7, 1e-4], dtype = np.float32)
    future_drop = np.array([0.0, 1e-7, 1e-4], dtype = np.float32)

    mask = compute_change_candidates(
        local_change=local_change,
        change_strength=change_strength,
        future_net_rise=future_rise,
        future_net_drop=future_drop,
        min_change_multiplier=3.0,
        min_absolute_change=1e-6,
        min_future_net_change=1e-6,
    )

    assert np.array_equal(mask, np.array([False, False, True]))

def test_build_regions_with_stats():
    mask = np.array([False, True, True, False])
    drop_strength = np.array([0, 3, 5, 0], dtype=np.float32)

    regions = build_regions_with_stats(mask, drop_strength)

    assert len(regions) == 1
    assert regions[0].start_index == 1
    assert regions[0].end_index == 2

def test_select_earliest_region():
    regions = [
        PRPRegion(10, 12, peak_strength=2.0, mean_strength=1.5, area_strength=4.0),
        PRPRegion(20, 25, peak_strength=3.5, mean_strength=3.0, area_strength=9.0),
        PRPRegion(30, 35, peak_strength=6.0, mean_strength=5.0, area_strength=15.0)
    ]

    selected = select_earliest_strong_region(regions)

    assert selected.start_index == 10

def test_tighten_region_shrink():
    region = PRPRegion(10, 16, peak_strength=10.0, mean_strength=6.0, area_strength=51.0)

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
    config = PRPConfig(min_change_multiplier=2.0)
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
    result = detect_bp(signal, samplerate, prp_index=120, config=BPConfig())

    assert result.bp_index is not None
    assert result.bp_index >= 120


def test_detect_bp_simple_fall():
    samplerate = 400
    signal = np.ones(400, dtype=np.float32) * 10

    signal[120:200] = np.linspace(10, 0, 80, dtype=np.float32)
    signal[200:] = 0
    result = detect_bp(signal, samplerate, prp_index=120, config=BPConfig())

    assert result.bp_index is not None
    assert result.bp_index >= 120


def test_detect_sp_simple_rise():
    samplerate = 400
    signal = np.zeros(400, dtype=np.float32)

    signal[120:200] = np.linspace(0, 10, 80, dtype=np.float32)
    signal[200:] = 10
    result = detect_sp(signal, samplerate, prp_index=120, config=SPConfig())

    assert result.sp_index is not None
    assert result.sp_index >= 160


def test_detect_sp_simple_fall():
    samplerate = 400
    signal = np.ones(400, dtype=np.float32) * 10

    signal[120:200] = np.linspace(10, 0, 80, dtype=np.float32)
    signal[200:] = 0
    result = detect_sp(signal, samplerate, prp_index=120, config=SPConfig())

    assert result.sp_index is not None
    assert result.sp_index >= 170
