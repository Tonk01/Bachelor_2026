from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .data_loader import Event

@dataclass(frozen = True)
class PreprocessingConfig:
    allowed_samplerates: tuple[int, ...] = (400, 1000)
    normalize: bool = True
    norm_eps: float = 1e-8

@dataclass
class ProcessedEvent:
    eventid: int
    valvetag: str
    sitename: str
    samplerate: int
    signal: np.ndarray
    n_samples: int
    duration_sec: float
    path: str

class EventProcessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.allowed_samplerates = set(self.config.allowed_samplerates)

    def valid_samplerate(self, samplerate: int) -> None:
        if samplerate not in self.allowed_samplerates:

            allowed = sorted(self.allowed_samplerates)
            raise ValueError("Samplerate {samplerate} not allowed")

    def preprocess_signal(self, signal: list[float]) -> np.ndarray:
        x = np.asarray(signal, dtype = np.float32)

        if x.ndim != 1:
            raise ValueError("signal size must be 1D")
        
        if x.size == 0:
            raise ValueError("Signal is empty")
        
        if not np.isfinite(x).all():
            raise ValueError("Signal has NaN or inf")
        
        if self.config.normalize:
            mean = float(x.mean())
            std = float(x.std())        # standard divation

            if std < self.config.norm_eps:
                x = x - mean
            else: 
                x = (x - mean) / std
            
        return x
    
    def preprocess_event(self, event: Event) -> ProcessedEvent:
        self.valid_samplerate(event.samplerate)
        
        x = self.preprocess_signal(event.signal)

        return ProcessedEvent(
            eventid = event.eventid,
            valvetag = event.valvetag,
            sitename = event.sitename,
            samplerate = event.samplerate,
            signal = x,
            n_samples = int(x.size),
            duration_sec = float(x.size / event.samplerate),
            path = event.path,
        )

    # Loops over the events and processes them building rows for each event
    def preprocess_all(
        self, 
        events: Iterable[Event],
    ) -> tuple[list[ProcessedEvent], list[tuple[str, str]]]:
        processed: list[ProcessedEvent] = []
        errors: list[tuple[str, str]] = []

        for event in events:
            try:
                processed.append(self.preprocess_event(event))
            except Exception as e:
                errors.append((event.path, str(e)))

        return processed, errors
