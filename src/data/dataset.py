from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset

from .data_loader import find_event_files, load_event
from .preprocess import EventProcessor, PreprocessingConfig
from .targets import (PRPConfig, BPConfig, SPConfig, detect_bp, detect_prp, detect_sp, build_prp_target, build_bp_target, build_sp_target)

class ValveDataset(Dataset):
    def __init__(   
        self,
        *roots: str | Path,
        preprocessing_config: PreprocessingConfig | None = None,
        prp_config: PRPConfig | None = None,
        bp_config: BPConfig | None = None,
        sp_config: SPConfig | None = None,
    ) -> None:
        self.file_paths = find_event_files(*roots)
        self.processor = EventProcessor(preprocessing_config)

        self.prp_config = prp_config or PRPConfig()
        self.bp_config = bp_config or BPConfig()
        self.sp_config = sp_config or SPConfig()

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.file_paths[index]

        event = load_event(path)
        processed = self.processor.preprocess_event(event)

        signal = processed.signal
        samplerate = processed.samplerate
        n_samples = processed.n_samples

        prp_result = detect_prp(signal, self.prp_config)
        bp_result = detect_bp(signal, samplerate, prp_result.prp_index, self.bp_config)
        sp_result = detect_sp(signal, samplerate, bp_result.bp_index, self.sp_config)

        prp_target = build_prp_target(
            n_samples=n_samples,
            prp_index=prp_result.prp_index,
            samplerate=samplerate,
        )

        bp_target = build_bp_target(
            n_samples=n_samples,
            bp_index=bp_result.bp_index,
            samplerate=samplerate,
        )

        sp_target = build_sp_target(
            n_samples=n_samples,
            sp_index=sp_result.sp_index,
            samplerate=samplerate,
        )

        x = torch.from_numpy(processed.signal).to(torch.float32).unsqueeze(0)
        y = torch.from_numpy(np.stack([prp_target, bp_target, sp_target], axis=0)).to(torch.float32)


        return{
            "x": x,
            "y": y,
            "meta": {
                "eventid": processed.eventid,
                "valvetag": processed.valvetag,
                "sitename": processed.sitename,
                "samplerate": processed.samplerate,
                "n_samples": processed.n_samples,
                "duration_sec": processed.duration_sec,
                "path": processed.path,
                "prp_index": prp_result.prp_index,
                "bp_index": bp_result.bp_index,
                "sp_index": sp_result.sp_index,
            },
        }