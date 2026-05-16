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


def _normalize_target_payload(
    target_payload: torch.Tensor | dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(target_payload, dict):
        y = target_payload["y"].to(torch.float32)
        target_valid_mask = target_payload.get("target_valid_mask")
        if target_valid_mask is not None:
            target_valid_mask = torch.as_tensor(target_valid_mask)
        return y, target_valid_mask

    return target_payload.to(torch.float32), None


class AllSensorAvailabilityValveDataset(Dataset):
    def __init__(
        self,
        pressure_dataset_root: str | Path,
        two_sensor_dataset_root: str | Path,
        *,
        preprocessor: EventProcessor | None = None,
        target_builder: Callable[..., torch.Tensor | dict[str, Any]],
        skip_flat_signals: bool = True,
        min_std: float = 1e-12,
        max_duration_mismatch_sec: float = 0.025,
    ) -> None:
        self.pressure_dataset_root = Path(pressure_dataset_root)
        self.two_sensor_dataset_root = Path(two_sensor_dataset_root)
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

        for group in self._iter_available_event_groups():
            if "pressure" not in group:
                self.skipped_reasons["missing_pressure"] += 1
                continue

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

    def _iter_available_event_groups(self) -> list[dict[str, Path]]:
        groups: dict[tuple[str, str], dict[str, Path]] = {}

        if self.pressure_dataset_root.exists():
            for site_dir in sorted(path for path in self.pressure_dataset_root.iterdir() if path.is_dir()):
                events_dir = site_dir / "events"
                if not events_dir.exists():
                    continue

                for path in events_dir.glob("*.json"):
                    key = (site_dir.name, path.name)
                    group = groups.setdefault(key, {})
                    group.setdefault("pressure", path)

        if self.two_sensor_dataset_root.exists():
            for site_dir in sorted(path for path in self.two_sensor_dataset_root.iterdir() if path.is_dir()):
                events_dir = site_dir / "events"
                if not events_dir.exists():
                    continue

                for sensor_name in ("strain", "travel"):
                    sensor_dir = events_dir / sensor_name
                    if not sensor_dir.exists():
                        continue

                    for path in sensor_dir.glob("*.json"):
                        key = (site_dir.name, path.name)
                        group = groups.setdefault(key, {})
                        group.setdefault(sensor_name, path)

        return [
            groups[key]
            for key in sorted(groups.keys())
        ]

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

            channel_arrays: list[np.ndarray] = []
            sensor_presence: list[float] = []
            for sensor_name in SENSOR_NAMES:
                processed = processed_by_sensor.get(sensor_name)
                if processed is None:
                    channel_arrays.append(np.zeros(min_length, dtype=np.float32))
                    sensor_presence.append(0.0)
                    continue

                signal = processed.normalized_signal[:min_length]
                if self.skip_flat_signals and float(np.std(signal)) < self.min_std:
                    self.skipped_reasons[f"flat_{sensor_name}"] += 1
                    return None
                channel_arrays.append(signal)
                sensor_presence.append(1.0)

            pressure_event = processed_by_sensor["pressure"]
            pressure_for_targets = replace(
                pressure_event,
                resampled_signal=pressure_event.resampled_signal[:min_length],
                normalized_signal=pressure_event.normalized_signal[:min_length],
                n_samples=min_length,
                duration_sec=min_length / float(pressure_event.samplerate),
            )

            truncated_processed_by_sensor = {
                sensor_name: replace(
                    processed,
                    resampled_signal=processed.resampled_signal[:min_length],
                    normalized_signal=processed.normalized_signal[:min_length],
                    n_samples=min_length,
                    duration_sec=min_length / float(processed.samplerate),
                )
                for sensor_name, processed in processed_by_sensor.items()
            }

            x_np = np.stack(channel_arrays, axis=0).astype(np.float32)
            x = torch.from_numpy(x_np).to(torch.float32)
            target_payload = self.target_builder(
                pressure_for_targets,
                truncated_processed_by_sensor,
            )
            y, target_valid_mask = _normalize_target_payload(target_payload)

        except Exception as exc:
            self.skipped_reasons[type(exc).__name__] += 1
            if self.skipped_reasons[type(exc).__name__] <= 5:
                pressure_path = group.get("pressure") or next(iter(group.values()))
                print(f"Skipping invalid mixed-sensor event {pressure_path}: {exc}")
            return None

        return {
            "x": x,
            "y": y,
            "sensor_presence": torch.tensor(sensor_presence, dtype=torch.float32),
            "target_valid_mask": target_valid_mask,
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
                    for sensor_name in processed_by_sensor.keys()
                },
            },
        }
