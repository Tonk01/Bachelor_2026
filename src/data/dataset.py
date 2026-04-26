from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from collections import Counter

import numpy as np

import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

import time

from .data_loader import load_event, find_event_files, peak_event_metadata
from .preprocess import EventProcessor, PreprocessingConfig

class ValveDataset(Dataset):
    def __init__(self, *event_roots: str | Path, preprocessor: EventProcessor | None = None, target_builder: Callable[[any, torch.tensor]]) -> None:
        self.preprocessor = preprocessor or EventProcessor(PreprocessingConfig())
        self.target_builder = target_builder
        
        all_file_paths = find_event_files(*event_roots)

        self.file_paths: list[Path] = []
        self.estimated_durations: list[float] = []
        self.estimated_lengths: list[int] = []

        self.skipped_samplerates: Counter[int] = Counter()
        self.other_load_errors = 0

        def scan_one(path: Path):
            metadata = peak_event_metadata(path)
            if metadata is None:
                return None
            
            estimated_length = max(1, int(round(metadata.duration_sec * 400)))
            return path, metadata.duration_sec, estimated_length

        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, result in enumerate(executor.map(scan_one, all_file_paths), start=1):

                if i % 1000 == 0:
                    print(f"Scanned {i} / {len(all_file_paths)} files")

                if result is None:
                    continue

                path, duration_sec, estimated_length = result

                self.file_paths.append(path)
                self.estimated_durations.append(duration_sec)
                self.estimated_lengths.append(estimated_length)


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

            target_time = t3 - t2

            if target_time > 1.0:
                print(f"preprocess={t2-t1:.4f}s target={t3-t2:.4f}s total={t3-t0:.4f}s, {path}")

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
