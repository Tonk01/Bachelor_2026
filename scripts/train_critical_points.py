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
load_dotenv(dotenv_path=str(dotenv_path), override=True)

from src.data.dataset import ValveDataset
from src.train.train_core import TrainConfig, fit
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (PRPConfig, BPConfig, SPConfig, detect_prp, detect_bp, detect_sp, build_prp_target, build_bp_target, build_sp_target)


SEED = 42
BATCH_SIZE = 24
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 20
MAX_TEST_SAMPLES = 20
WORKERS = 0



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


def _require_dataset_path() -> Path:
    dataset_path = os.getenv("DATASET_PATH")

    if not dataset_path:
        raise ValueError("DATASET_PATH is missing. Add it to .env")
    
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise ValueError(f"DATASET_PATH does not exist: {dataset_root}")
    
    return dataset_root

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
) -> TrainConfig:
    
    return TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
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
    max_train_samples = _env_int("MAX_TRAIN_SAMPLES", MAX_TRAIN_SAMPLES)
    max_val_samples = _env_int("MAX_VAL_SAMPLES", MAX_VAL_SAMPLES)
    max_test_samples = _env_int("MAX_TEST_SAMPLES", MAX_TEST_SAMPLES)

    _set_seed(seed)

    print("Valve Event Detection")
    print("Project root:", PROJECT_ROOT)

    dataset_root = _require_dataset_path()
    print("Dataset path:", dataset_root)

    preprocessor = EventProcessor(PreprocessingConfig())

    full_dataset = ValveDataset(dataset_root, preprocessor=preprocessor, target_builder=build_targets)

    n_total = len(full_dataset)

    print(f"Discovered event file: {n_total}")
    if n_total == 0:
        raise ValueError("No event file")
    
    train_idx, val_idx, test_idx = _split_indices(
        n_items=n_total,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    train_idx = _subset_indices(train_idx, max_train_samples, seed)
    val_idx = _subset_indices(val_idx, max_val_samples, seed + 1)
    test_idx = _subset_indices(test_idx, max_test_samples, seed + 2)

    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset = Subset(full_dataset, val_idx.tolist())
    test_dataset = Subset(full_dataset, test_idx.tolist())

    _summarize_subset("Train", train_dataset)
    _summarize_subset("Val", val_dataset)
    _summarize_subset("Test", test_dataset)

    train_config = _build_train_config(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
    )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / f"valve-cnn-{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    np.save(run_dir / "train_idx.npy", train_idx)
    np.save(run_dir / "val_idx.npy", val_idx)
    np.save(run_dir / "test_idx.npy", test_idx)

    model, history = fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
    )

    history_json = {
        "train_loss": [float(x) for x in history ["train_loss"]],
        "val_loss": [float(x) for x in history ["val_loss"]],
    }

    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history_json, fp, indent=2)

    torch.save(model.state_dict(), run_dir / "best_model.pt")

    best_val_loss = min(history["val_loss"]) if history ["val_loss"] else None
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else None

    run_summary = {
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "workers": workers,
        "dataset_path": str(dataset_root),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "total_samples": int(n_total),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
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