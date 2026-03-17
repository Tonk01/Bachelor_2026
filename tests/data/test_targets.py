import numpy as np
import pytest

from src.data.targets import ms_to_samples, moving_avg, safe_std, gaussian, build_prp_target, detect_prp


def test_ms_to_sample():
    assert ms_to_samples(10, 400) == 4
def test_ms_to_samples_round():
    assert ms_to_samples(12, 400) == 5 # -> 4.8, but rounds to 5
def test_ms_to_samples_minimum():
    assert ms_to_samples(1, 400) == 1


def test_moving_avg_window():
    x = np.array([1, 2, 3, 4], dtype = np.float32)
    y = moving_avg(x, 1)
    assert np.allclose(x, y)

def test_moving_avg_smoothing():
    x = np.array([0, 0, 10, 0, 0], dtype = np.float32)
    y = moving_avg(x, 3)

    assert y[2] < 10
    assert y[1] > 0

def test_safe_std():
    x = np.array([1,2,3], dtype = np.float32)
    assert safe_std(x) > 0

def test_safe_std_zero_signal():
    x = np.array([5, 5, 5], dtype = np.float32)
    assert safe_std(x) > 0

def test_gaussian_peak_location():
    g = gaussian(n_samples=100, center=50, samplerate=400, sigma_ms=10)

    assert np.argmax(g) == 50
    assert g[50] == g.max()

def test_gaussian_shape():
    g = gaussian(100, 50, 400, 10)
    assert g.shape == (100,)


def test_build_prp_target_none():
    t = build_prp_target(100, None, 400)
    assert t.sum() == 0

def test_build_prp_target_center():
    t = build_prp_target(100, 30, 400)
    
    assert t.argmax() == 30
    assert t[30] > 0

def test_detect_prp_simple_rise():
    
    samplerate = 400
    signal = np.zeros(400)

    signal[120:200] = np.linspace(0, 10, 80)
    result = detect_prp(signal, samplerate)

    assert result.prp_index is not None
    assert result.prp_index >= 100

def test_detect_prp_no_event():
    
    samplerate = 400
    signal = np.ones(400)

    result = detect_prp(signal, samplerate)
    assert result.prp_index is None

def test_detect_prp_invalid_shape():

    signal = np.zeros((10, 10))

    with pytest.raises(ValueError):
        detect_prp(signal, 400)