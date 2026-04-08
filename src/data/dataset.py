from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from collections import Counter

import numpy as np

import torch
from torch.utils.data import Dataset

import time

from .data_loader import load_event, find_event_files
from .preprocess import EventProcessor, PreprocessingConfig

class ValveDataset(Dataset):
    def __init__(self, *event_roots: str | Path, preprocessor: EventProcessor | None = None, target_builder: Callable) -> None:
        self.preprocessor = preprocessor or EventProcessor(PreprocessingConfig())
        self.target_builder = target_builder
        self.file_paths = find_event_files(*event_roots)
        self.skipped_samplerates: Counter[int] = Counter()
        self.other_load_errors = 0

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index: int) -> dict[str, Any] | None:
        path = self.file_paths[index]

        try:
            t0 = time.perf_counter()
            event = load_event(path)
            t1 = time.perf_counter()

            processed = self.preprocessor.preprocess_event(event)
            t2 = time.perf_counter()

            length = int(processed.n_samples)

            x = torch.from_numpy(processed.normalized_signal).to(torch.float32).unsqueeze(0)
            y = self.target_builder(processed).to(torch.float32)
            t3 = time.perf_counter()

            print(f"load={t1-t0:.4f}s preprocess={t2-t1:.4f}s target={t3-t2:.4f}s total={t3-t0:.4f}s")
        except ValueError as exc:
            if "Samplerate" in str(exc):
                samplerate = getattr(locals().get("event"), "samplerate", None)
                if samplerate is not None:
                    self.skipped_samplerates[samplerate] += 1
                    if self.skipped_samplerates[samplerate] == 1:
                        print(f"Skipping unsupported samplerate {samplerate} for {path}")
                return None
            self.other_load_errors += 1
            if self.other_load_errors <= 5:
                print(f"Skipping invalid event file {path}: {exc}")
            return None
        except Exception as exc:
            self.other_load_errors += 1
            if self.other_load_errors <= 5:
                print(f"Skipping invalid event file {path}: {exc}")
            return None

        return {
            "x": x,
            "y": y,
            "length": length,
            "meta": {
                "eventid": processed.eventid,
                "valvetag": processed.valvetag,
                "sitename": processed.sitename,
                "path": processed.path,
                "original_samplerate": processed.original_samplerate,
                "samplerate": processed.samplerate,
                "duration_sec": processed.duration_sec,
            },
        }
