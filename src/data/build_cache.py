from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data_loader import load_event, find_event_files
from .preprocess import EventProcessor, PreprocessingConfig
from .targets import ( PRPConfig, BPConfig, SPConfig, detect_prp, detect_bp, detect_sp, build_prp_target, build_bp_target, build_sp_target)

def build_cached_sample(
        raw_path: str | Path,
        preprocessing_config: PreprocessingConfig | None = None,
        prp_config: PRPConfig | None = None,
        bp_config: BPConfig | None = None,
        sp_config: SPConfig | None = None,
) -> dict[str, Any]:
    processor = EventProcessor(preprocessing_config)

    prp_cfg = prp_config or PRPConfig()
    bp_cfg = bp_config or BPConfig()
    sp_cfg = sp_config or SPConfig()

    event = load_event(raw_path)
    processed = processor.preprocess_event(event)

    model_signal = processed.normalized_signal
    label_signal = processed.resampled_signal

    samplerate = processed.samplerate
    n_samples = processed.n_samples

    prp_result = detect_prp(label_signal, prp_cfg)
    bp_result = detect_bp(label_signal, samplerate, prp_result.prp_index, bp_cfg)
    sp_result = detect_sp(label_signal, samplerate, prp_result.prp_index, sp_cfg)

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

    y = np.stack([prp_target, bp_target, sp_target], axis=0).astype(np.float32)

    return {
        "x": model_signal.astype(np.float32),
        "y": y,
        "length": n_samples,
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

def save_cached_sample(sample: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = {
        "x": torch.from_numpy(sample["x"]).to(torch.float32),
        "y": torch.from_numpy(sample["y"]).to(torch.float32),
        "length": int(sample["length"]),
        "meta": sample["meta"],
    }

    torch.save(to_save, output_path)

def make_cache_path(raw_path: str | Path, cache_dir: str | Path) -> Path:
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir)

    return cache_dir / f"{raw_path.stem}.pt"

def build_and_save_cached_sample(
    raw_path: str | Path,
    cache_dir: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    prp_config: PRPConfig | None = None,
    bp_config: BPConfig | None = None,
    sp_config: SPConfig | None = None,
) -> Path:
    sample = build_cached_sample(
        raw_path=raw_path,
        preprocessing_config=preprocessing_config,
        prp_config=prp_config,
        bp_config=bp_config,
        sp_config=sp_config,
    )
    
    output_path = make_cache_path(raw_path=raw_path, cache_dir=cache_dir)
    save_cached_sample(sample=sample, output_path=output_path)

    return output_path

def build_cache_paths(
    raw_paths: list[str | Path],
    cache_dir: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    prp_config: PRPConfig | None = None,
    bp_config: BPConfig | None = None,
    sp_config: SPConfig | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    saved_paths: list[Path] = []
    errors: list[tuple[str, str]] = []


    for raw_path in raw_paths:
        try:
            output_path = make_cache_path(raw_paths, cache_dir)

            if output_path.exists():
                saved_paths.append(output_path)
                continue

            output_path = build_and_save_cached_sample(
                raw_path=raw_path,
                cache_dir=cache_dir,
                preprocessing_config=preprocessing_config,
                prp_config=prp_config,
                bp_config=bp_config,
                sp_config=sp_config,
            )
            saved_paths.append(output_path)
        except Exception as e:
            errors.append((str(raw_path), str(e)))

    return saved_paths, errors
    
def build_cache_roots(
    *roots: str | Path,
    cache_dir: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    prp_config: PRPConfig | None = None,
    bp_config: BPConfig | None = None,
    sp_config: SPConfig | None = None,
) -> tuple[list[Path], list[tuple[str, str]]]:
    raw_paths = find_event_files(*roots)

    return build_cache_paths(
        raw_paths=raw_paths,
        cache_dir=cache_dir,
        preprocessing_config=preprocessing_config,
        prp_config=prp_config,
        bp_config=bp_config,
        sp_config=sp_config,
    )