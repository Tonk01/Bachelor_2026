from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import (
    cache_lengths_path,
    cache_meta_path,
    cache_sample_dir,
    cache_sample_path,
    cache_split_path,
    load_cache_meta,
)


class CachedValveDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str = "all") -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split

        if not cache_meta_path(self.cache_dir).exists():
            raise FileNotFoundError(f"Cache metadata not found: {cache_meta_path(self.cache_dir)}")

        self.meta = load_cache_meta(self.cache_dir)
        self._all_lengths = np.load(cache_lengths_path(self.cache_dir)).astype(np.int64)

        if split == "all":
            self.indices = np.arange(len(self._all_lengths), dtype=np.int64)
        else:
            split_path = cache_split_path(self.cache_dir, split)
            if not split_path.exists():
                raise FileNotFoundError(f"Cache split not found: {split_path}")
            self.indices = np.load(split_path).astype(np.int64)

        self.estimated_lengths = self._all_lengths[self.indices].astype(np.int64).tolist()

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache_index = int(self.indices[index])
        sample_path = self._sample_path(cache_index)
        payload = torch.load(sample_path, map_location="cpu")

        sample = {
            "x": payload["x"].to(torch.float32),
            "y": payload["y"].to(torch.float32),
            "length": int(payload["length"]),
            "meta": dict(payload.get("meta", {})),
        }
        if "sensor_presence" in payload:
            sample["sensor_presence"] = payload["sensor_presence"].to(torch.float32)
        if "target_valid_mask" in payload:
            sample["target_valid_mask"] = payload["target_valid_mask"].to(torch.float32)
        if "sample_weight" in payload:
            sample["sample_weight"] = float(payload["sample_weight"])
        return sample

    def _sample_path(self, cache_index: int) -> Path:
        numeric_path = cache_sample_path(self.cache_dir, cache_index)
        if numeric_path.exists():
            return numeric_path

        matches = sorted(cache_sample_dir(self.cache_dir).glob(f"{cache_index:06d}_*.pt"))
        if matches:
            return matches[0]

        return numeric_path
