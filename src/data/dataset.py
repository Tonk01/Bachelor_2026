from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset

class ValveDataset(Dataset):
    def __init__(self, *cache_roots: str | Path) -> None:
        self.file_paths: list[Path] = []

        for root in cache_roots:
            root_path = Path(root)
            self.file_paths.extend(sorted(root_path.rglob("*.pt")))

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem(self, index: int) -> dict[str, Any] | None:
        path = self.file_paths[index]

        try:
            sample = torch.load(path)

            x = sample["x"].to(torch.float32).unsqueeze(0)
            y = sample["y"].to(torch.float32)

            return {
                "x": x,
                "y": y,
                "length": int(sample["length"]),
                "meta": sample["meta"],
            }
        
        except Exception as e:
            print(f"Skipping cached sample at {path}: {e}")
            return None