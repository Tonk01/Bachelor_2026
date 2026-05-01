from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_loader import load_event, peak_event_metadata
from .preprocess import EventProcessor, PreprocessingConfig, ProcessedEvent


SENSOR_NAMES = ("pressure", "strain", "travel")
MULTISENSOR_SAMPLERATES = (10, 50, 200, 250, 400, 800, 1000, 2000)


class MultiSensorValveDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        preprocessor: EventProcessor | None = None,
        target_builder: Callable[..., torch.Tensor],
        skip_flat_signals: bool = True,
        min_std: float = 1e-12,
        max_duration_mismatch_sec: float = 0.025,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.preprocessor = preprocessor or EventProcessor(
            PreprocessingConfig(allowed_samplerate=MULTISENSOR_SAMPLERATES)
        )
        self.target_builder = target_builder
        self.skip_flat_signals = skip_flat_signals
        self.min_std = min_std
        self.max_duration_mismatch_sec = max_duration_mismatch_sec

        self.event_groups: list[dict[str, Path]] = []
        self.estimated_lengths: list[int] = []
        self.skipped_reasons: Counter[str] = Counter()

        for group in self._iter_complete_event_groups():
            metadata_by_sensor = {
                sensor_name: peak_event_metadata(path)
                for sensor_name, path in group.items()
            }
            if any(metadata is None for metadata in metadata_by_sensor.values()):
                self.skipped_reasons["missing_metadata"] += 1
                continue

            durations = [
                metadata.duration_sec
                for metadata in metadata_by_sensor.values()
                if metadata is not None
            ]
            if max(durations) - min(durations) > self.max_duration_mismatch_sec:
                self.skipped_reasons["duration_mismatch"] += 1
                continue

            estimated_length = max(
                1,
                int(round(min(durations) * self.preprocessor.config.target_samplerate)),
            )
            self.event_groups.append(group)
            self.estimated_lengths.append(estimated_length)

    def _iter_complete_event_groups(self) -> list[dict[str, Path]]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        groups: list[dict[str, Path]] = []
        for site_dir in sorted(path for path in self.dataset_root.iterdir() if path.is_dir()):
            sensor_files = {
                sensor_name: {
                    path.name: path
                    for path in (site_dir / "events" / sensor_name).glob("*.json")
                }
                for sensor_name in SENSOR_NAMES
            }
            complete_names = sorted(set.intersection(*(set(paths) for paths in sensor_files.values())))
            for filename in complete_names:
                groups.append(
                    {
                        sensor_name: sensor_files[sensor_name][filename]
                        for sensor_name in SENSOR_NAMES
                    }
                )
        return groups

    def __len__(self) -> int:
        return len(self.event_groups)

    def _load_processed_sensors(self, group: dict[str, Path]) -> dict[str, ProcessedEvent]:
        processed_by_sensor: dict[str, ProcessedEvent] = {}
        for sensor_name, path in group.items():
            event = load_event(path)
            processed_by_sensor[sensor_name] = self.preprocessor.preprocess_event(event)
        return processed_by_sensor

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        group = self.event_groups[index]

        try:
            processed_by_sensor = self._load_processed_sensors(group)
            min_length = min(processed.n_samples for processed in processed_by_sensor.values())

            channel_arrays = []
            for sensor_name in SENSOR_NAMES:
                processed = processed_by_sensor[sensor_name]
                signal = processed.normalized_signal[:min_length]
                if self.skip_flat_signals and float(np.std(signal)) < self.min_std:
                    self.skipped_reasons[f"flat_{sensor_name}"] += 1
                    return None
                channel_arrays.append(signal)

            pressure_event = processed_by_sensor["pressure"]
            pressure_for_targets = replace(
                pressure_event,
                resampled_signal=pressure_event.resampled_signal[:min_length],
                normalized_signal=pressure_event.normalized_signal[:min_length],
                n_samples=min_length,
                duration_sec=min_length / float(pressure_event.samplerate),
            )

            x_np = np.stack(channel_arrays, axis=0).astype(np.float32)
            x = torch.from_numpy(x_np).to(torch.float32)
            y = self.target_builder(pressure_for_targets, processed_by_sensor).to(torch.float32)

        except Exception as exc:
            self.skipped_reasons[type(exc).__name__] += 1
            if self.skipped_reasons[type(exc).__name__] <= 5:
                print(f"Skipping invalid multisensor event {group['pressure']}: {exc}")
            return None

        return {
            "x": x,
            "y": y,
            "length": min_length,
            "meta": {
                "eventid": pressure_for_targets.eventid,
                "valvetag": pressure_for_targets.valvetag,
                "sitename": pressure_for_targets.sitename,
                "samplerate": pressure_for_targets.samplerate,
                "duration_sec": pressure_for_targets.duration_sec,
                "paths": {
                    sensor_name: str(path)
                    for sensor_name, path in group.items()
                },
                "original_samplerates": {
                    sensor_name: processed_by_sensor[sensor_name].original_samplerate
                    for sensor_name in SENSOR_NAMES
                },
            },
        }
