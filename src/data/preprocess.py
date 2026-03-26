from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.signal import resample_poly

from .data_loader import Event

@dataclass (frozen = True)
class PreprocessingConfig:
    allowed_samplerate: tuple[int, ...] = (400, 1000)
    target_samplerate: int = 400
    normalize_for_model: bool = True
    normal_eps: float = 1e-8

@dataclass
class ProcessedEvent:
    eventid: int
    valvetag: str
    sitename: str

    original_samplerate: int
    samplerate: int

    resampled_signal: np.ndarray
    normalized_signal: np.ndarray

    n_samples: int
    duration_sec: float
    path: str

class EventProcessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.allowed_samplerates = set(self.config.allowed_samplerate)

        if self.config.target_samplerate <= 0:
            raise ValueError("target_samplerate must be above 0")
        
    def validate_samplerate(self, samplerate: int) -> None:
        if samplerate not in self.allowed_samplerates:
            allowed = sorted(self.allowed_samplerates)
            
            raise ValueError(f"Samplerate {samplerate} not allowed.")
        
    def validate_signal(self, signal: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float32)

        if x.ndim != 1:
            raise ValueError("signal must be 1D")
        
        if x.size == 0:
            raise ValueError("signal is emoty")

        if not np.isfinite(x).all():
            raise ValueError("signal contains NaN or INF")
        
        return x
        
    def resample_signal(self, signal: list[float] | np.ndarray, from_samplerate: int) -> np.ndarray:
        x = self.validate_signal(signal)

        if from_samplerate <= 0:
            raise ValueError("from_samplerate must be above 0")
        
        if from_samplerate == self.config.target_samplerate:
            return x.copy()
        
        x_resampled = resample_poly(x, up=self.config.target_samplerate, down = from_samplerate).astype(np.float32)

        if x_resampled.size == 0:
            raise ValueError("resampled signal is empty")
        
        return x_resampled
    
    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        x = self.validate_signal(signal)

        if not self.config.normalize_for_model:
            return x.copy()

        mean = float(np.mean(x))
        std = float(np.std(x))

        if std < self.config.normal_eps:
            return (x - mean).astype(np.float32)
        
        return ((x - mean) / std).astype(np.float32)
    
    def preprocess_event(self, event: Event) -> ProcessedEvent:
        self.validate_samplerate(event.samplerate)

        resampled_signal = self.resample_signal(
            signal=event.signal,
            from_samplerate=event.samplerate,
        )

        normalized_signal = self.normalize_signal(resampled_signal)

        return ProcessedEvent(
            eventid=event.eventid,
            valvetag=event.valvetag,
            sitename=event.sitename,
            original_samplerate=event.samplerate,
            samplerate=self.config.target_samplerate,
            resampled_signal=resampled_signal,
            normalized_signal=normalized_signal,
            n_samples=int(resampled_signal.size),
            duration_sec=float(resampled_signal.size / self.config.target_samplerate),
            path=event.path,
        )
    
    def preprocess_all(self, events: Iterable[Event]) -> tuple[list[ProcessedEvent], list[tuple[str, str]]]:
        processed: list[ProcessedEvent] = []
        errors: list[tuple[str, str]] = []

        for event in events:
            try:
                processed.append(self.preprocess_event(event))
            except Exception as e:
                errors.append((event.path, str(e)))

        return processed, errors