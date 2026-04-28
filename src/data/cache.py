from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT_VERSION = 1


def cache_meta_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "meta.json"


def cache_sample_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "samples"


def cache_sample_path(cache_dir: str | Path, sample_index: int) -> Path:
    return cache_sample_dir(cache_dir) / f"{sample_index:06d}.pt"


def cache_split_path(cache_dir: str | Path, split_name: str) -> Path:
    return Path(cache_dir) / f"{split_name}_idx.npy"


def cache_raw_split_path(cache_dir: str | Path, split_name: str) -> Path:
    return Path(cache_dir) / f"raw_{split_name}_idx.npy"


def cache_lengths_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "sample_lengths.npy"


def cache_source_indices_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "source_indices.npy"


def _normalize_meta_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_meta_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_meta_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_meta_value(item)
            for key, item in value.items()
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _safe_filename_part(value: Any) -> str:
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text or "unknown"


def _sample_filename(cache_index: int, sample_meta: dict[str, Any]) -> str:
    source_path = sample_meta.get("path")
    if source_path is None:
        paths = sample_meta.get("paths", {})
        source_path = paths.get("pressure") if isinstance(paths, dict) else None

    if source_path is not None:
        event_name = Path(source_path).stem
    else:
        event_name = f"{sample_meta.get('valvetag')}_{sample_meta.get('eventid')}"

    return f"{cache_index:06d}_{_safe_filename_part(event_name)}.pt"


def load_cache_meta(cache_dir: str | Path) -> dict[str, Any]:
    with cache_meta_path(cache_dir).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def cache_is_compatible(cache_dir: str | Path, expected_meta: dict[str, Any]) -> bool:
    meta_path = cache_meta_path(cache_dir)
    if not meta_path.exists():
        return False

    try:
        actual = load_cache_meta(cache_dir)
    except Exception:
        return False

    expected = {
        key: _normalize_meta_value(value)
        for key, value in expected_meta.items()
    }
    return all(actual.get(key) == value for key, value in expected.items())


def build_cache_from_dataset(
    *,
    dataset: Dataset,
    cache_dir: str | Path,
    split_indices: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    sample_dir = cache_sample_dir(cache_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    lengths: list[int] = []
    source_indices: list[int] = []
    source_to_cache: dict[int, int] = {}
    skipped_raw_samples = 0

    n_items = len(dataset)
    for source_index in range(n_items):
        sample = dataset[source_index]
        if sample is None:
            skipped_raw_samples += 1
            continue

        length = int(sample["length"])
        cache_index = len(lengths)

        x = sample["x"].detach().cpu().to(torch.float32).contiguous()
        y = sample["y"].detach().cpu().to(torch.float32).contiguous()
        sample_meta = dict(sample.get("meta", {}))
        sample_meta["cache_index"] = cache_index
        sample_meta["source_index"] = source_index
        sample_filename = _sample_filename(cache_index, sample_meta)

        torch.save(
            {
                "x": x,
                "y": y,
                "length": length,
                "meta": sample_meta,
            },
            sample_dir / sample_filename,
        )

        lengths.append(length)
        source_indices.append(source_index)
        source_to_cache[source_index] = cache_index

        if (source_index + 1) % 100 == 0 or source_index + 1 == n_items:
            print(
                f"Cached {source_index + 1} / {n_items} source samples"
                f" ({len(lengths)} usable)"
            )

    split_sizes: dict[str, int] = {}
    for split_name, raw_indices in split_indices.items():
        raw_indices = np.asarray(raw_indices, dtype=np.int64)
        np.save(cache_raw_split_path(cache_dir, split_name), raw_indices)
        cached_indices = np.asarray(
            [
                source_to_cache[int(source_index)]
                for source_index in raw_indices
                if int(source_index) in source_to_cache
            ],
            dtype=np.int64,
        )
        np.save(cache_split_path(cache_dir, split_name), cached_indices)
        split_sizes[split_name] = int(len(cached_indices))

    np.save(cache_lengths_path(cache_dir), np.asarray(lengths, dtype=np.int64))
    np.save(cache_source_indices_path(cache_dir), np.asarray(source_indices, dtype=np.int64))

    payload = {
        "format_version": FORMAT_VERSION,
        **{
            key: _normalize_meta_value(value)
            for key, value in meta.items()
        },
        "num_samples": int(len(lengths)),
        "skipped_raw_samples": int(skipped_raw_samples),
        "split_sizes": split_sizes,
    }
    _write_json(cache_meta_path(cache_dir), payload)
    return cache_dir


def ensure_dataset_cache(
    *,
    dataset: Dataset,
    cache_dir: str | Path,
    split_indices: dict[str, np.ndarray],
    expected_meta: dict[str, Any],
    rebuild: bool = False,
) -> Path:
    cache_dir = Path(cache_dir)
    if not rebuild and cache_is_compatible(cache_dir, expected_meta):
        print("Using compatible cache:", cache_dir)
        return cache_dir

    print("Building cache:", cache_dir)
    return build_cache_from_dataset(
        dataset=dataset,
        cache_dir=cache_dir,
        split_indices=split_indices,
        meta=expected_meta,
    )
