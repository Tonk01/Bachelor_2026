from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from dotenv import load_dotenv
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=False)

from src.data.multisensor_dataset import MultiSensorValveDataset
from src.train.train_core import TrainConfig, fit
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (
    BPConfig,
    PRPConfig,
    SPConfig,
    build_bp_target,
    build_prp_target,
    build_sp_target,
    detect_bp,
    detect_prp,
    detect_sp,
)


DEFAULT_DATASET_PATH = Path("/mnt/d/Bachelor_data/data/raw_complete_3sensor")
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MAX_TRAIN_SAMPLES = 1500
MAX_VAL_SAMPLES = 300
MAX_TEST_SAMPLES = 300
WORKERS = 0
SENSOR_SAMPLERATES = (10, 50, 200, 250, 400, 800, 1000, 2000)


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


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None:
        return default
    return Path(value)


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
    chosen = rng.choice(indices, size=max_items, replace=False)
    return np.sort(chosen).astype(np.int64)


def _split_indices(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("Split fractions must sum to 1.0")
    if n_items < 3:
        raise ValueError("Need at least three samples to create train/val/test splits")

    indices = np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = min(int(round(n_items * train_frac)), n_items - 2)
    n_val = min(int(round(n_items * val_frac)), n_items - n_train - 1)

    return (
        np.sort(indices[:n_train]).astype(np.int64),
        np.sort(indices[n_train:n_train + n_val]).astype(np.int64),
        np.sort(indices[n_train + n_val:]).astype(np.int64),
    )


def build_targets(processed_event) -> torch.Tensor:
    signal = processed_event.resampled_signal
    samplerate = processed_event.samplerate
    n_samples = processed_event.n_samples

    prp_result = detect_prp(signal=signal, config=PRPConfig(samplerate=samplerate))
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
        config=SPConfig(),
    )

    y = np.stack(
        [
            build_prp_target(n_samples, prp_result.prp_index, samplerate),
            build_bp_target(n_samples, bp_result.bp_index, samplerate),
            build_sp_target(n_samples, sp_result.sp_index, samplerate),
        ],
        axis=0,
    )
    return torch.from_numpy(y).to(torch.float32)


def main() -> None:
    seed = _env_int("SEED", SEED)
    batch_size = _env_int("BATCH_SIZE", BATCH_SIZE)
    epochs = _env_int("EPOCHS", EPOCHS)
    learning_rate = _env_float("LEARNING_RATE", LEARNING_RATE)
    weight_decay = _env_float("WEIGHT_DECAY", WEIGHT_DECAY)
    workers = _env_int("WORKERS", WORKERS)

    train_frac = _env_float("TRAIN_FRAC", TRAIN_FRAC)
    val_frac = _env_float("VAL_FRAC", VAL_FRAC)
    test_frac = _env_float("TEST_FRAC", TEST_FRAC)
    max_train_samples = _env_int("MAX_TRAIN_SAMPLES", MAX_TRAIN_SAMPLES)
    max_val_samples = _env_int("MAX_VAL_SAMPLES", MAX_VAL_SAMPLES)
    max_test_samples = _env_int("MAX_TEST_SAMPLES", MAX_TEST_SAMPLES)
    dataset_root = _env_path("MULTISENSOR_DATASET_PATH", DEFAULT_DATASET_PATH)

    _set_seed(seed)

    print("Valve Event Detection - 3 Sensor")
    print("Project root:", PROJECT_ROOT)
    print("Dataset path:", dataset_root)

    preprocessor = EventProcessor(
        PreprocessingConfig(allowed_samplerate=SENSOR_SAMPLERATES)
    )
    full_dataset = MultiSensorValveDataset(
        dataset_root,
        preprocessor=preprocessor,
        target_builder=build_targets,
    )

    n_total = len(full_dataset)
    print("Discovered complete 3-sensor events:", n_total)
    print("Skipped during discovery:", dict(full_dataset.skipped_reasons))
    if n_total == 0:
        raise ValueError("No complete 3-sensor events found")

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

    train_dataset = ValveSubset(full_dataset, train_idx.tolist())
    val_dataset = ValveSubset(full_dataset, val_idx.tolist())
    test_dataset = ValveSubset(full_dataset, test_idx.tolist())

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        in_channels=3,
    )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / f"valve-cnn-3sensor-{run_timestamp}"
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
        "train_loss": [float(x) for x in history["train_loss"]],
        "val_loss": [float(x) for x in history["val_loss"]],
    }
    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history_json, fp, indent=2)

    torch.save(model.state_dict(), run_dir / "best_model.pt")

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else None
    run_summary = {
        "mode": "3sensor",
        "input_channels": ["pressure", "strain", "travel"],
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "workers": workers,
        "dataset_path": str(dataset_root),
        "total_samples": int(n_total),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "best_model_file": "best_model.pt",
        "history_file": "history.json",
        "train_indices_file": "train_idx.npy",
        "val_indices_file": "val_idx.npy",
        "test_indices_file": "test_idx.npy",
        "skipped_reasons": dict(full_dataset.skipped_reasons),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, indent=2)

    print("3-sensor training complete")
    print("Best val_loss:", best_val_loss)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
