from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import os
import random
import sys

import numpy as np
from dotenv import load_dotenv
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=str(dotenv_path), override=False)

from src.data.cache import cache_is_compatible, cache_raw_split_path, ensure_dataset_cache
from src.data.cached_dataset import CachedValveDataset
from src.data.dataset import ValveDataset
from src.train.train_core import TrainConfig, fit
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (PRPConfig, BPConfig, SPConfig, detect_prp, detect_bp, detect_sp, build_prp_target, build_bp_target, build_sp_target)


SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 400
MAX_TEST_SAMPLES = 400
WORKERS = 0
PREFETCH_FACTOR = 2
CACHE_NAME = "valve-1sensor-v1"
CACHE_TARGET_VERSION = 1

class ValveSubset(Subset):
    @property
    def estimated_lengths(self) -> list[int]:
        return [self.dataset.estimated_lengths[i] for i in self.indices]



def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_dataset_path() -> Path:
    dataset_path = os.getenv("DATASET_PATH")

    if not dataset_path:
        raise ValueError("DATASET_PATH is missing. Add it to .env")
    
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise ValueError(f"DATASET_PATH does not exist: {dataset_root}")
    
    return dataset_root


def _default_cache_dir() -> Path:
    return PROJECT_ROOT / "data" / "cache" / CACHE_NAME

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _subset_indices(indices: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(indices) <= max_items:
        return np.sort(indices).astype(np.int64)
    
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size = max_items, replace = False)
    return np.sort(chosen).astype(np.int64)


def _validate_split_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac

    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"split fractions not working")

    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError("All split fractions must be greater than 0")

def _split_indices(
        n_items: int,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if n_items <= 0:
        raise ValueError("Cannot split empty dataset")
    
    indices = np.arange(n_items, dtype = np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(round(n_items * train_frac))
    n_val = int(round(n_items * val_frac))

    n_train = min(n_train, n_items - 2)
    n_val = min(n_val, n_items - n_train - 1)
    n_test = n_items - n_train - n_val

    if n_train <= 0:
        raise ValueError("Train split is empty")
    
    if n_val <= 0:
        raise ValueError("Validation split is empty")
    
    if n_test <= 0:
        raise ValueError("Test split is empty")
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return(
        np.sort(train_idx).astype(np.int64),
        np.sort(val_idx).astype(np.int64),
        np.sort(test_idx).astype(np.int64),
    )

def _summarize_subset(name: str, subset: Subset) -> None:
    print(f"{name} samples: {len(subset)}")


def _build_train_config(
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        workers: int,
        prefetch_factor: int,
) -> TrainConfig:
    
    return TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        prefetch_factor=prefetch_factor,
    )


def build_targets(processed_event) -> torch.Tensor:
    signal = processed_event.resampled_signal
    samplerate = processed_event.samplerate
    n_samples = processed_event.n_samples

    prp_result = detect_prp(
        signal=signal,
        config=PRPConfig(samplerate=samplerate),
    )

    bp_result = detect_bp(
        signal=signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=BPConfig(),
    )

    sp_result = detect_sp(
        signal=signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=SPConfig()
    )

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

    y = np.stack([prp_target, bp_target, sp_target], axis=0)
    return torch.from_numpy(y).to(torch.float32)

def main() -> None:
    seed = _env_int("SEED", SEED)
    batch_size = _env_int("BATCH_SIZE", BATCH_SIZE)
    epochs = _env_int("EPOCHS", EPOCHS)
    learning_rate = _env_float("LEARNING_RATE", LEARNING_RATE)
    weight_decay = _env_float("WEIGHT_DECAY", WEIGHT_DECAY)

    train_frac = _env_float("TRAIN_FRAC", TRAIN_FRAC)
    val_frac = _env_float("VAL_FRAC", VAL_FRAC)
    test_frac = _env_float("TEST_FRAC", TEST_FRAC)

    workers = _env_int("WORKERS", WORKERS)
    prefetch_factor = _env_int("PREFETCH_FACTOR", PREFETCH_FACTOR)
    max_train_samples = _env_int("MAX_TRAIN_SAMPLES", MAX_TRAIN_SAMPLES)
    max_val_samples = _env_int("MAX_VAL_SAMPLES", MAX_VAL_SAMPLES)
    max_test_samples = _env_int("MAX_TEST_SAMPLES", MAX_TEST_SAMPLES)
    use_cache = _env_bool("USE_CACHE", True)
    rebuild_cache = _env_bool("REBUILD_CACHE", False)
    cache_only = _env_bool("CACHE_ONLY", False)
    cache_dir = Path(os.getenv("CACHE_DIR", str(_default_cache_dir())))
    if cache_only and not use_cache:
        raise ValueError("CACHE_ONLY=1 requires USE_CACHE=1")

    _set_seed(seed)

    print("Valve Event Detection")
    print("Project root:", PROJECT_ROOT)

    dataset_root = _require_dataset_path()
    print("Dataset path:", dataset_root)

    preprocessor = EventProcessor(PreprocessingConfig())
    expected_cache_meta = {
        "cache_name": CACHE_NAME,
        "mode": "1sensor",
        "dataset_root": str(dataset_root),
        "in_channels": 1,
        "target_samplerate": preprocessor.config.target_samplerate,
        "allowed_samplerate": list(preprocessor.config.allowed_samplerate),
        "normalize_for_model": preprocessor.config.normalize_for_model,
        "normal_eps": preprocessor.config.normal_eps,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_seed": seed,
        "target_version": CACHE_TARGET_VERSION,
        "sensors": ["pressure"],
    }

    full_train_dataset: Subset | CachedValveDataset
    full_val_dataset: Subset | CachedValveDataset
    full_test_dataset: Subset | CachedValveDataset
    raw_split_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    n_total: int | None = None

    if use_cache:
        if rebuild_cache or not cache_is_compatible(cache_dir, expected_cache_meta):
            full_dataset = ValveDataset(
                dataset_root,
                preprocessor=preprocessor,
                target_builder=build_targets,
            )
            n_total = len(full_dataset)
            print(f"Discovered event file: {n_total}")
            if n_total == 0:
                raise ValueError("No event file")

            raw_split_indices = _split_indices(
                n_items=n_total,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
            )
            ensure_dataset_cache(
                dataset=full_dataset,
                cache_dir=cache_dir,
                split_indices={
                    "train": raw_split_indices[0],
                    "val": raw_split_indices[1],
                    "test": raw_split_indices[2],
                },
                expected_meta=expected_cache_meta,
                rebuild=True,
            )

        full_train_dataset = CachedValveDataset(cache_dir, split="train")
        full_val_dataset = CachedValveDataset(cache_dir, split="val")
        full_test_dataset = CachedValveDataset(cache_dir, split="test")
        n_total = int(full_train_dataset.meta["num_samples"])
        print("Train samples:", len(full_train_dataset))
        print("Val samples:", len(full_val_dataset))
        print("Test samples:", len(full_test_dataset))
        if cache_only:
            print("Cache ready:", cache_dir)
            return
    else:
        full_dataset = ValveDataset(dataset_root, preprocessor=preprocessor, target_builder=build_targets)

        n_total = len(full_dataset)
        print(f"Discovered event file: {n_total}")
        if n_total == 0:
            raise ValueError("No event file")
        
        raw_split_indices = _split_indices(
            n_items=n_total,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )

        full_train_dataset = ValveSubset(full_dataset, raw_split_indices[0].tolist())
        full_val_dataset = ValveSubset(full_dataset, raw_split_indices[1].tolist())
        full_test_dataset = ValveSubset(full_dataset, raw_split_indices[2].tolist())

    train_local_idx = _subset_indices(
        np.arange(len(full_train_dataset), dtype=np.int64),
        max_train_samples,
        seed,
    )
    val_local_idx = _subset_indices(
        np.arange(len(full_val_dataset), dtype=np.int64),
        max_val_samples,
        seed + 1,
    )
    test_local_idx = _subset_indices(
        np.arange(len(full_test_dataset), dtype=np.int64),
        max_test_samples,
        seed + 2,
    )

    train_dataset = ValveSubset(full_train_dataset, train_local_idx.tolist())
    val_dataset = ValveSubset(full_val_dataset, val_local_idx.tolist())
    test_dataset = ValveSubset(full_test_dataset, test_local_idx.tolist())

    _summarize_subset("Train", train_dataset)
    _summarize_subset("Val", val_dataset)
    _summarize_subset("Test", test_dataset)

    train_config = _build_train_config(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        prefetch_factor=prefetch_factor,
    )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / f"valve-cnn-{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    if use_cache:
        np.save(run_dir / "train_idx.npy", np.load(cache_raw_split_path(cache_dir, "train")))
        np.save(run_dir / "val_idx.npy", np.load(cache_raw_split_path(cache_dir, "val")))
        np.save(run_dir / "test_idx.npy", np.load(cache_raw_split_path(cache_dir, "test")))
    elif raw_split_indices is not None:
        np.save(run_dir / "train_idx.npy", raw_split_indices[0])
        np.save(run_dir / "val_idx.npy", raw_split_indices[1])
        np.save(run_dir / "test_idx.npy", raw_split_indices[2])

    model, history = fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
    )

    history_json = {
        key: [float(x) for x in values]
        for key, values in history.items()
    }

    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history_json, fp, indent=2)

    torch.save(model.state_dict(), run_dir / "best_model.pt")

    best_val_loss = min(history["val_loss"]) if history ["val_loss"] else None
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else None
    total_training_seconds = history.get("total_seconds", [None])[-1]
    average_epoch_seconds = (
        float(np.mean(history["epoch_seconds"]))
        if history.get("epoch_seconds")
        else None
    )

    run_summary = {
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "workers": workers,
        "prefetch_factor": prefetch_factor,
        "use_cache": use_cache,
        "cache_dir": str(cache_dir) if use_cache else None,
        "dataset_path": str(dataset_root),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "total_samples": int(n_total) if n_total is not None else None,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "test_samples": int(len(test_dataset)),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "total_training_seconds": (
            float(total_training_seconds)
            if total_training_seconds is not None
            else None
        ),
        "average_epoch_seconds": average_epoch_seconds,
        "best_model_file": "best_model.pt",
        "history_file": "history.json",
        "train_indices_file": "train_idx.npy",
        "val_indices_file": "val_idx.npy",
        "test_indices_file": "test_idx.npy"
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, indent=2)

    print("Training complete")
    print("Best val_loss:", best_val_loss)
    
if __name__ == "__main__":
    main()
