from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import build_bp_target, build_prp_target, build_sp_target


@dataclass(frozen=True)
class ManualLabelEntry:
    event: str
    site: str
    valvetag: str
    status: str
    prp_time_sec: float | None
    bp_time_sec: float | None
    sp_time_sec: float | None
    prp_index_400hz: int | None
    bp_index_400hz: int | None
    sp_index_400hz: int | None
    note: str
    pressure_path: str

    @property
    def is_rejected(self) -> bool:
        return self.status.strip().lower() == "rejected"

    @property
    def is_labeled(self) -> bool:
        return self.status.strip().lower() == "labeled"


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_manual_labels(csv_path: str | Path) -> dict[str, ManualLabelEntry]:
    path = Path(csv_path)
    if not path.exists():
        return {}

    labels: dict[str, ManualLabelEntry] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = str(row.get("event", "")).strip()
            if not event:
                continue
            labels[event] = ManualLabelEntry(
                event=event,
                site=str(row.get("site", "")).strip(),
                valvetag=str(row.get("valvetag", "")).strip(),
                status=str(row.get("status", "")).strip(),
                prp_time_sec=_to_float(row.get("prp_time_sec")),
                bp_time_sec=_to_float(row.get("bp_time_sec")),
                sp_time_sec=_to_float(row.get("sp_time_sec")),
                prp_index_400hz=_to_int(row.get("prp_index_400hz")),
                bp_index_400hz=_to_int(row.get("bp_index_400hz")),
                sp_index_400hz=_to_int(row.get("sp_index_400hz")),
                note=str(row.get("note", "")).strip(),
                pressure_path=str(row.get("pressure_path", "")).strip(),
            )
    return labels


def manual_label_targets(
    *,
    entry: ManualLabelEntry,
    n_samples: int,
    samplerate: int,
    sigma_ms: float,
) -> torch.Tensor:
    def choose_index(time_sec: float | None, index_400hz: int | None) -> int | None:
        if time_sec is not None:
            index = int(round(time_sec * float(samplerate)))
        elif index_400hz is not None:
            index = int(round(index_400hz * float(samplerate) / 400.0))
        else:
            return None
        return max(0, min(index, n_samples - 1))

    prp_index = choose_index(entry.prp_time_sec, entry.prp_index_400hz)
    bp_index = choose_index(entry.bp_time_sec, entry.bp_index_400hz)
    sp_index = choose_index(entry.sp_time_sec, entry.sp_index_400hz)

    y = torch.from_numpy(
        np.stack(
            [
                build_prp_target(n_samples, prp_index, samplerate, sigma_ms=sigma_ms),
                build_bp_target(n_samples, bp_index, samplerate, sigma_ms=sigma_ms),
                build_sp_target(n_samples, sp_index, samplerate, sigma_ms=sigma_ms),
            ],
            axis=0,
        )
    ).to(torch.float32)
    return y


def manual_label_valid_mask(
    *,
    entry: ManualLabelEntry,
    n_samples: int,
    samplerate: int,
) -> torch.Tensor:
    indices = manual_label_indices(
        entry=entry,
        n_samples=n_samples,
        samplerate=samplerate,
    )
    return torch.tensor(
        [
            float(indices["PRP"] is not None),
            float(indices["BP"] is not None),
            float(indices["SP"] is not None),
        ],
        dtype=torch.float32,
    )


def manual_label_indices(
    *,
    entry: ManualLabelEntry,
    n_samples: int,
    samplerate: int,
) -> dict[str, int | None]:
    def choose_index(time_sec: float | None, index_400hz: int | None) -> int | None:
        if time_sec is not None:
            index = int(round(time_sec * float(samplerate)))
        elif index_400hz is not None:
            index = int(round(index_400hz * float(samplerate) / 400.0))
        else:
            return None
        return max(0, min(index, n_samples - 1))

    return {
        "PRP": choose_index(entry.prp_time_sec, entry.prp_index_400hz),
        "BP": choose_index(entry.bp_time_sec, entry.bp_index_400hz),
        "SP": choose_index(entry.sp_time_sec, entry.sp_index_400hz),
    }


def sample_event_name(sample: dict[str, Any]) -> str:
    meta = sample.get("meta", {})
    paths = meta.get("paths", {})
    pressure_path = paths.get("pressure")
    if pressure_path:
        return Path(str(pressure_path)).stem

    site = str(meta.get("site") or meta.get("sitename") or "").strip()
    eventid = meta.get("eventid")
    if site and eventid is not None:
        return f"{site}_{eventid}"
    raise KeyError("Could not infer event name from sample metadata")


class ManualLabelOverrideDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        manual_labels: dict[str, ManualLabelEntry],
        sigma_ms: float,
        manual_weight: float = 5.0,
    ) -> None:
        self.dataset = dataset
        self.manual_labels = manual_labels
        self.sigma_ms = sigma_ms
        self.manual_weight = manual_weight
        self.estimated_lengths = getattr(dataset, "estimated_lengths", None)
        self.skipped_reasons = getattr(dataset, "skipped_reasons", {})
        self.event_groups = getattr(dataset, "event_groups", None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        sample = self.dataset[index]
        if sample is None:
            return None

        sample = dict(sample)
        meta = dict(sample.get("meta", {}))
        event_name = sample_event_name(sample)
        entry = self.manual_labels.get(event_name)

        if entry is None:
            meta["event"] = event_name
            meta["label_source"] = meta.get("label_source", "heuristic")
            meta["sample_weight"] = float(meta.get("sample_weight", 1.0))
            sample["meta"] = meta
            sample["sample_weight"] = float(meta["sample_weight"])
            return sample

        if entry.is_rejected:
            return None

        meta["event"] = event_name
        meta["manual_label_status"] = entry.status
        meta["manual_label_note"] = entry.note

        if entry.is_labeled:
            length = int(sample["length"])
            samplerate = int(meta.get("samplerate", 400))
            sample["y"] = manual_label_targets(
                entry=entry,
                n_samples=length,
                samplerate=samplerate,
                sigma_ms=self.sigma_ms,
            )
            sample["target_valid_mask"] = manual_label_valid_mask(
                entry=entry,
                n_samples=length,
                samplerate=samplerate,
            )
            meta["label_source"] = "manual"
            meta["sample_weight"] = float(self.manual_weight)
            sample["sample_weight"] = float(self.manual_weight)
        else:
            meta["label_source"] = "heuristic"
            meta["sample_weight"] = 1.0
            sample["sample_weight"] = 1.0
            if "target_valid_mask" not in sample:
                sample["target_valid_mask"] = (sample["y"].amax(dim=1) > 0).to(torch.float32)

        sample["meta"] = meta
        return sample

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)
