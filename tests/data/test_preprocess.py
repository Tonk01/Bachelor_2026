import math

import numpy as np
import pytest

from src.data.data_loader import Event
from src.data.preprocess import EventProcessor


def make_event(
    *,
    samplerate: int,
    signal: list[float],
    eventid: int = 1,
    valvetag: str = "valve-1",
    sitename: str = "site-1",
    path: str = "event.json",
) -> Event:
    return Event(
        eventid=eventid,
        valvetag=valvetag,
        sitename=sitename,
        samplerate=samplerate,
        signal=signal,
        n_samples=len(signal),
        duration_sec=len(signal) / samplerate if samplerate > 0 else 0.0,
        path=path,
    )


def test_preprocess_event_resamples_1000hz_to_400hz() -> None:
    processor = EventProcessor()
    event = make_event(samplerate=1000, signal=list(np.linspace(0.0, 1.0, 1000)))

    processed = processor.preprocess_event(event)

    assert processed.samplerate == 400
    assert processed.n_samples == 400
    assert math.isclose(processed.duration_sec, 1.0)
    assert processed.resampled_signal.shape == (400,)
    assert processed.normalized_signal.shape == (400,)


def test_preprocess_event_keeps_400hz_length() -> None:
    processor = EventProcessor()
    event = make_event(samplerate=400, signal=list(np.linspace(0.0, 1.0, 400)))

    processed = processor.preprocess_event(event)

    assert processed.samplerate == 400
    assert processed.n_samples == 400
    assert math.isclose(processed.duration_sec, 1.0)
    assert processed.resampled_signal.shape == (400,)
    assert processed.normalized_signal.shape == (400,)


def test_preprocess_event_rejects_unsupported_samplerate() -> None:
    processor = EventProcessor()
    event = make_event(samplerate=123, signal=[1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Samplerate 123 not allowed"):
        processor.preprocess_event(event)


def test_preprocess_event_rejects_empty_signal() -> None:
    processor = EventProcessor()
    event = make_event(samplerate=400, signal=[])

    with pytest.raises(ValueError, match="signal is empty"):
        processor.preprocess_event(event)


def test_preprocess_all_collects_errors_and_continues() -> None:
    processor = EventProcessor()
    valid_event = make_event(samplerate=400, signal=list(np.linspace(0.0, 1.0, 400)))
    invalid_event = make_event(
        samplerate=123,
        signal=[1.0, 2.0, 3.0, 4.0],
        path="bad-event.json",
    )

    processed, errors = processor.preprocess_all([valid_event, invalid_event])

    assert len(processed) == 1
    assert len(errors) == 1
    assert errors[0][0] == "bad-event.json"
    assert "Samplerate 123 not allowed" in errors[0][1]
