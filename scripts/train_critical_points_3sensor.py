from __future__ import annotations

from datetime import datetime
import csv
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

from src.data.cache import cache_is_compatible, cache_raw_split_path, ensure_dataset_cache
from src.data.cached_dataset import CachedValveDataset
from src.data.manual_labels import ManualLabelOverrideDataset, load_manual_labels
from src.data.all_sensor_dataset import AllSensorAvailabilityValveDataset
from src.data.multisensor_dataset import MultiSensorValveDataset
from src.train.train_core import TrainConfig, fit
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (
    BPConfig,
    MultivariatConfig,
    PRPConfig,
    SPConfig,
    build_bp_target,
    build_prp_target,
    build_sp_target,
    detect_BP_multivariat,
    detect_PRP_multivariat,
    detect_SP_multivariat,
    detect_bp,
    detect_prp,
    detect_sp,
)


DEFAULT_DATASET_PATH = Path("/mnt/d/Bachelor_data/data/raw_complete_3sensor")
DEFAULT_PRESSURE_DATASET_PATH = Path("/mnt/d/Bachelor_data/data/raw_1sensor")
DEFAULT_TWO_SENSOR_DATASET_PATH = Path("/mnt/d/Bachelor_data/data/raw_2SENSOR")
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
EARLY_STOPPING_PATIENCE = 0
EARLY_STOPPING_MIN_DELTA = 0.0
MODEL_TYPE = "cnn"
TARGET_SIGMA_MS = 50.0
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MAX_TRAIN_SAMPLES = 1500
MAX_VAL_SAMPLES = 300
MAX_TEST_SAMPLES = 300
WORKERS = max(1, min(4, (os.cpu_count() or 1) // 2 or 1))
PREFETCH_FACTOR = 2
SENSOR_SAMPLERATES = (200, 400, 800, 1000)
CACHE_NAME = "valve-3sensor-v1"
CACHE_TARGET_VERSION = 13
TARGET_STRATEGY = "multivariat"
DATASET_MODE = "complete_3sensor"
POINT_NAMES = ("PRP", "BP", "SP")
TIME_TOLERANCES_SEC = (0.05, 0.1, 0.2)
SITE_FILTER = ""
TRAIN_SITE_OVERSAMPLE = ""
TRAIN_SITE_OVERSAMPLE_FACTOR = 1
USE_MANUAL_LABELS = True
MANUAL_LABELS_CSV = PROJECT_ROOT / "data" / "manual_labels" / "manual_labels.csv"
MANUAL_SPLIT_JSON = PROJECT_ROOT / "data" / "manual_labels" / "manual_split.json"
MANUAL_SAMPLE_WEIGHT = 5.0
MANUAL_TRAIN_OVERSAMPLE_FACTOR = 1
EXCLUDE_EVENTS_CSV = PROJECT_ROOT / "events_to_be_deleted - Ark 1.csv"


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_site_set(name: str, default: str = "") -> set[str]:
    value = os.getenv(name, default)
    return {part.strip() for part in value.split(",") if part.strip()}


def _default_cache_dir(cache_name: str = CACHE_NAME) -> Path:
    return PROJECT_ROOT / "data" / "cache" / cache_name


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _sample_site(dataset: Subset | CachedValveDataset, local_index: int) -> str | None:
    sample = dataset[int(local_index)]
    if sample is None:
        return None
    meta = sample.get("meta", {})
    paths = meta.get("paths", {})
    pressure_path = paths.get("pressure")
    if pressure_path:
        parts = Path(str(pressure_path)).parts
        if "raw_complete_3sensor" in parts:
            root_index = parts.index("raw_complete_3sensor")
            if root_index + 1 < len(parts):
                return parts[root_index + 1]

    return str(meta.get("sitename", ""))


def _sample_label_source(dataset: Subset | CachedValveDataset, local_index: int) -> str | None:
    sample = dataset[int(local_index)]
    if sample is None:
        return None
    return str(sample.get("meta", {}).get("label_source", "heuristic"))


def _filter_local_indices_by_site(
    dataset: Subset | CachedValveDataset,
    local_indices: np.ndarray,
    allowed_sites: set[str],
) -> np.ndarray:
    if not allowed_sites:
        return local_indices

    kept = [
        int(index)
        for index in local_indices
        if _sample_site(dataset, int(index)) in allowed_sites
    ]
    return np.asarray(kept, dtype=np.int64)


def _oversample_local_indices_by_site(
    dataset: Subset | CachedValveDataset,
    local_indices: np.ndarray,
    sites: set[str],
    factor: int,
) -> np.ndarray:
    if not sites or factor <= 1 or len(local_indices) == 0:
        return local_indices

    matching = _filter_local_indices_by_site(dataset, local_indices, sites)
    if len(matching) == 0:
        return local_indices

    repeated = [local_indices]
    repeated.extend(matching for _ in range(factor - 1))
    return np.concatenate(repeated).astype(np.int64)


def _oversample_local_indices_by_label_source(
    dataset: Subset | CachedValveDataset,
    local_indices: np.ndarray,
    label_source: str,
    factor: int,
) -> np.ndarray:
    if factor <= 1 or len(local_indices) == 0:
        return local_indices

    matching = np.asarray(
        [
            int(index)
            for index in local_indices
            if _sample_label_source(dataset, int(index)) == label_source
        ],
        dtype=np.int64,
    )
    if len(matching) == 0:
        return local_indices

    repeated = [local_indices]
    repeated.extend(matching for _ in range(factor - 1))
    return np.concatenate(repeated).astype(np.int64)


def _subset_indices(indices: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(indices) <= max_items:
        return np.sort(indices).astype(np.int64)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_items, replace=False)
    return np.sort(chosen).astype(np.int64)


def _split_counts(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[int, int, int]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("Split fractions must sum to 1.0")
    if n_items < 3:
        raise ValueError("Need at least three samples to create train/val/test splits")

    n_train = min(int(round(n_items * train_frac)), n_items - 2)
    n_val = min(int(round(n_items * val_frac)), n_items - n_train - 1)
    n_test = n_items - n_train - n_val
    return int(n_train), int(n_val), int(n_test)


def _split_indices(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train, n_val, _n_test = _split_counts(n_items, train_frac, val_frac, test_frac)

    return (
        np.sort(indices[:n_train]).astype(np.int64),
        np.sort(indices[n_train:n_train + n_val]).astype(np.int64),
        np.sort(indices[n_train + n_val:]).astype(np.int64),
    )


def _event_name_from_group(group: dict[str, Path]) -> str:
    return group["pressure"].stem


def _load_excluded_events(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.reader(fp))

    if not rows:
        return set()

    headers = [str(cell).strip() for cell in rows[0]]
    excluded: set[str] = set()
    for row in rows[1:]:
        for col_index, raw_value in enumerate(row):
            if col_index >= len(headers):
                continue
            site = headers[col_index].strip()
            value = str(raw_value).strip()
            if not site or not value:
                continue
            if not value.isdigit():
                continue
            excluded.add(f"{site}_{value}")
    return excluded


def _filter_dataset_excluded_events(
    dataset: MultiSensorValveDataset | AllSensorAvailabilityValveDataset,
    excluded_events: set[str],
) -> int:
    if not excluded_events:
        return 0

    kept_groups: list[dict[str, Path]] = []
    kept_lengths: list[int] = []
    removed = 0
    for group, estimated_length in zip(dataset.event_groups, dataset.estimated_lengths):
        event_name = _event_name_from_group(group)
        if event_name in excluded_events:
            removed += 1
            continue
        kept_groups.append(group)
        kept_lengths.append(estimated_length)

    dataset.event_groups = kept_groups
    dataset.estimated_lengths = kept_lengths
    return removed


def _load_manual_split(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    splits = payload.get("splits", {})
    return {
        split_name: {str(event) for event in splits.get(split_name, [])}
        for split_name in ("train", "val", "test")
    }


def _split_indices_with_manual_split(
    dataset: MultiSensorValveDataset | AllSensorAvailabilityValveDataset,
    *,
    manual_split: dict[str, set[str]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_items = len(dataset)
    n_train_target, n_val_target, n_test_target = _split_counts(
        n_items,
        train_frac,
        val_frac,
        test_frac,
    )

    event_to_index = {
        _event_name_from_group(group): raw_index
        for raw_index, group in enumerate(dataset.event_groups)
    }
    assigned: dict[str, set[int]] = {}
    seen_assigned: set[int] = set()
    for split_name in ("train", "val", "test"):
        split_indices: set[int] = set()
        for event in manual_split.get(split_name, set()):
            raw_index = event_to_index.get(event)
            if raw_index is None or raw_index in seen_assigned:
                continue
            split_indices.add(int(raw_index))
            seen_assigned.add(int(raw_index))
        assigned[split_name] = split_indices

    remaining = np.asarray(
        [index for index in range(n_items) if index not in seen_assigned],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    remaining_targets = {
        "train": max(0, n_train_target - len(assigned["train"])),
        "val": max(0, n_val_target - len(assigned["val"])),
        "test": max(0, n_test_target - len(assigned["test"])),
    }

    start = 0
    extra: dict[str, np.ndarray] = {}
    for split_name in ("train", "val"):
        count = remaining_targets[split_name]
        extra[split_name] = np.sort(remaining[start:start + count]).astype(np.int64)
        start += count
    extra["test"] = np.sort(remaining[start:]).astype(np.int64)

    train_indices = np.sort(
        np.concatenate(
            [
                np.asarray(sorted(assigned["train"]), dtype=np.int64),
                extra["train"],
            ]
        )
    ).astype(np.int64)
    val_indices = np.sort(
        np.concatenate(
            [
                np.asarray(sorted(assigned["val"]), dtype=np.int64),
                extra["val"],
            ]
        )
    ).astype(np.int64)
    test_indices = np.sort(
        np.concatenate(
            [
                np.asarray(sorted(assigned["test"]), dtype=np.int64),
                extra["test"],
            ]
        )
    ).astype(np.int64)

    return train_indices, val_indices, test_indices


def _build_base_dataset(
    *,
    dataset_mode: str,
    dataset_root: Path,
    pressure_dataset_root: Path,
    two_sensor_dataset_root: Path,
    preprocessor: EventProcessor,
):
    if dataset_mode == "all_sensor_availability":
        return AllSensorAvailabilityValveDataset(
            pressure_dataset_root,
            two_sensor_dataset_root,
            preprocessor=preprocessor,
            target_builder=build_targets,
        )
    if dataset_mode == "complete_3sensor":
        return MultiSensorValveDataset(
            dataset_root,
            preprocessor=preprocessor,
            target_builder=build_targets,
        )
    raise ValueError(
        "DATASET_MODE must be 'complete_3sensor' or 'all_sensor_availability'"
    )


def build_targets(processed_event, processed_by_sensor) -> torch.Tensor:
    signal = processed_event.resampled_signal
    samplerate = processed_event.samplerate
    n_samples = processed_event.n_samples
    travel_event = processed_by_sensor.get("travel")
    strain_event = processed_by_sensor.get("strain")
    travel_signal = (
        travel_event.resampled_signal[:n_samples]
        if travel_event is not None
        else None
    )
    strain_signal = (
        strain_event.resampled_signal[:n_samples]
        if strain_event is not None
        else None
    )
    target_strategy = os.getenv("TARGET_STRATEGY", TARGET_STRATEGY).strip().lower()

    if target_strategy == "multivariat":
        multivariat_config = MultivariatConfig()
        prp_result = detect_PRP_multivariat(
            pressure_signal=signal,
            samplerate=samplerate,
            config=multivariat_config,
        )
    else:
        prp_result = detect_prp(signal=signal, config=PRPConfig(samplerate=samplerate))

    if target_strategy == "multivariat":
        bp_result = detect_BP_multivariat(
            pressure_signal=signal,
            travel_signal=travel_signal,
            samplerate=samplerate,
            prp_index=prp_result.prp_index,
            strain_signal=strain_signal,
            config=multivariat_config,
        )
        sp_result = detect_SP_multivariat(
            pressure_signal=signal,
            travel_signal=travel_signal,
            samplerate=samplerate,
            prp_index=prp_result.prp_index,
            bp_index=bp_result.bp_index,
            strain_signal=strain_signal,
            config=multivariat_config,
        )
    else:
        if travel_signal is None:
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
        else:
            bp_result = detect_BP_multivariat(
                pressure_signal=signal,
                travel_signal=travel_signal,
                samplerate=samplerate,
                prp_index=prp_result.prp_index,
                strain_signal=strain_signal,
                config=MultivariatConfig(),
            )
            sp_result = detect_SP_multivariat(
                pressure_signal=signal,
                travel_signal=travel_signal,
                samplerate=samplerate,
                prp_index=prp_result.prp_index,
                bp_index=bp_result.bp_index,
                strain_signal=strain_signal,
                config=MultivariatConfig(),
            )

    y = np.stack(
        [
            build_prp_target(
                n_samples,
                prp_result.prp_index,
                samplerate,
                sigma_ms=TARGET_SIGMA_MS,
            ),
            build_bp_target(
                n_samples,
                bp_result.bp_index,
                samplerate,
                sigma_ms=TARGET_SIGMA_MS,
            ),
            build_sp_target(
                n_samples,
                sp_result.sp_index,
                samplerate,
                sigma_ms=TARGET_SIGMA_MS,
            ),
        ],
        axis=0,
    )
    y_tensor = torch.from_numpy(y).to(torch.float32)
    target_valid_mask = (y_tensor.amax(dim=1) > 0).to(torch.float32)
    return {
        "y": y_tensor,
        "target_valid_mask": target_valid_mask,
    }


@torch.no_grad()
def evaluate_point_metrics(
    model: torch.nn.Module,
    dataset: Subset | CachedValveDataset,
    device: torch.device,
) -> dict[str, object]:
    model.eval()

    per_point_abs_errors: dict[str, list[float]] = {name: [] for name in POINT_NAMES}
    per_point_signed_errors: dict[str, list[float]] = {name: [] for name in POINT_NAMES}
    per_point_confidences: dict[str, list[float]] = {name: [] for name in POINT_NAMES}
    valid_order_count = 0
    num_events = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        if sample is None:
            continue

        x = sample["x"].unsqueeze(0).to(device)
        y = sample["y"]
        sensor_presence = sample.get("sensor_presence")
        if sensor_presence is None:
            sensor_presence = torch.ones(x.shape[1], dtype=torch.float32)
        sensor_presence = sensor_presence.unsqueeze(0).to(device)
        target_valid_mask = sample.get("target_valid_mask")
        if target_valid_mask is None:
            target_valid_mask = (y.amax(dim=1) > 0).to(torch.bool)
        else:
            target_valid_mask = target_valid_mask.to(torch.bool)
        length = int(sample["length"])
        samplerate = int(sample["meta"]["samplerate"])

        logits = model(x, sensor_presence=sensor_presence)[0, :, :length]
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y[:, :length].cpu().numpy()

        pred_indices: list[int] = []
        valid_target_points = 0
        for channel_index, point_name in enumerate(POINT_NAMES):
            if not bool(target_valid_mask[channel_index].item()):
                continue
            pred_index = int(np.argmax(probs[channel_index]))
            target_index = int(np.argmax(targets[channel_index]))
            pred_time = pred_index / float(samplerate)
            target_time = target_index / float(samplerate)
            signed_error = pred_time - target_time
            abs_error = abs(signed_error)
            confidence = float(probs[channel_index][pred_index])

            per_point_abs_errors[point_name].append(abs_error)
            per_point_signed_errors[point_name].append(signed_error)
            per_point_confidences[point_name].append(confidence)
            pred_indices.append(pred_index)
            valid_target_points += 1

        if valid_target_points == len(POINT_NAMES) and pred_indices[0] <= pred_indices[1] <= pred_indices[2]:
            valid_order_count += 1
            num_events += 1
        elif valid_target_points == len(POINT_NAMES):
            num_events += 1

    if num_events == 0:
        return {
            "num_events": 0,
            "valid_order_fraction": None,
            "points": {},
        }

    point_metrics: dict[str, object] = {}
    for point_name in POINT_NAMES:
        abs_errors = np.asarray(per_point_abs_errors[point_name], dtype=np.float64)
        signed_errors = np.asarray(per_point_signed_errors[point_name], dtype=np.float64)
        confidences = np.asarray(per_point_confidences[point_name], dtype=np.float64)

        point_metrics[point_name] = {
            "mean_abs_error_sec": float(np.mean(abs_errors)),
            "median_abs_error_sec": float(np.median(abs_errors)),
            "p90_abs_error_sec": float(np.percentile(abs_errors, 90)),
            "mean_signed_error_sec": float(np.mean(signed_errors)),
            "median_signed_error_sec": float(np.median(signed_errors)),
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "within_tolerance_fraction": {
                f"{tolerance:.2f}s": float(np.mean(abs_errors <= tolerance))
                for tolerance in TIME_TOLERANCES_SEC
            },
        }

    return {
        "num_events": int(num_events),
        "valid_order_fraction": float(valid_order_count / num_events),
        "points": point_metrics,
    }


def main() -> None:
    global TARGET_SIGMA_MS

    if "--help" in sys.argv or "-h" in sys.argv:
        print("Train 3-sensor model via environment variables.")
        print("Common variables:")
        print("  CACHE_NAME")
        print("  REBUILD_CACHE")
        print("  CACHE_ONLY")
        print("  USE_CACHE")
        print("  USE_MANUAL_LABELS")
        print("  MANUAL_LABELS_CSV")
        print("  MANUAL_SPLIT_JSON")
        print("  EXCLUDE_EVENTS_CSV")
        print("  MANUAL_SAMPLE_WEIGHT")
        print("  MANUAL_TRAIN_OVERSAMPLE_FACTOR")
        print("  MODEL_TYPE")
        print("  TARGET_SIGMA_MS")
        print("  TARGET_STRATEGY")
        print("  MAX_TRAIN_SAMPLES")
        print("  MAX_VAL_SAMPLES")
        print("  MAX_TEST_SAMPLES")
        print("  EPOCHS")
        print("  BATCH_SIZE")
        print("  SITE_FILTER")
        print("  TRAIN_SITE_OVERSAMPLE")
        print("  TRAIN_SITE_OVERSAMPLE_FACTOR")
        print("  DATASET_MODE")
        print("  MULTISENSOR_DATASET_PATH")
        print("  PRESSURE_DATASET_PATH")
        print("  TWO_SENSOR_DATASET_PATH")
        raise SystemExit(0)

    seed = _env_int("SEED", SEED)
    batch_size = _env_int("BATCH_SIZE", BATCH_SIZE)
    epochs = _env_int("EPOCHS", EPOCHS)
    learning_rate = _env_float("LEARNING_RATE", LEARNING_RATE)
    weight_decay = _env_float("WEIGHT_DECAY", WEIGHT_DECAY)
    dropout = _env_float("DROPOUT", DROPOUT)
    early_stopping_patience = _env_int(
        "EARLY_STOPPING_PATIENCE",
        EARLY_STOPPING_PATIENCE,
    )
    early_stopping_min_delta = _env_float(
        "EARLY_STOPPING_MIN_DELTA",
        EARLY_STOPPING_MIN_DELTA,
    )
    model_type = os.getenv("MODEL_TYPE", MODEL_TYPE).strip().lower()
    TARGET_SIGMA_MS = _env_float("TARGET_SIGMA_MS", TARGET_SIGMA_MS)
    target_strategy = os.getenv("TARGET_STRATEGY", TARGET_STRATEGY).strip().lower()
    workers = _env_int("WORKERS", WORKERS)
    prefetch_factor = _env_int("PREFETCH_FACTOR", PREFETCH_FACTOR)

    train_frac = _env_float("TRAIN_FRAC", TRAIN_FRAC)
    val_frac = _env_float("VAL_FRAC", VAL_FRAC)
    test_frac = _env_float("TEST_FRAC", TEST_FRAC)
    max_train_samples = _env_int("MAX_TRAIN_SAMPLES", MAX_TRAIN_SAMPLES)
    max_val_samples = _env_int("MAX_VAL_SAMPLES", MAX_VAL_SAMPLES)
    max_test_samples = _env_int("MAX_TEST_SAMPLES", MAX_TEST_SAMPLES)
    site_filter = _env_site_set("SITE_FILTER", SITE_FILTER)
    train_site_oversample = _env_site_set("TRAIN_SITE_OVERSAMPLE", TRAIN_SITE_OVERSAMPLE)
    train_site_oversample_factor = _env_int(
        "TRAIN_SITE_OVERSAMPLE_FACTOR",
        TRAIN_SITE_OVERSAMPLE_FACTOR,
    )
    dataset_mode = os.getenv("DATASET_MODE", DATASET_MODE).strip().lower()
    dataset_root = _env_path("MULTISENSOR_DATASET_PATH", DEFAULT_DATASET_PATH)
    pressure_dataset_root = _env_path("PRESSURE_DATASET_PATH", DEFAULT_PRESSURE_DATASET_PATH)
    two_sensor_dataset_root = _env_path("TWO_SENSOR_DATASET_PATH", DEFAULT_TWO_SENSOR_DATASET_PATH)
    use_manual_labels = _env_bool("USE_MANUAL_LABELS", USE_MANUAL_LABELS)
    manual_labels_csv = _env_path("MANUAL_LABELS_CSV", Path(MANUAL_LABELS_CSV))
    manual_split_json = _env_path("MANUAL_SPLIT_JSON", Path(MANUAL_SPLIT_JSON))
    exclude_events_csv = _env_path("EXCLUDE_EVENTS_CSV", Path(EXCLUDE_EVENTS_CSV))
    manual_sample_weight = _env_float("MANUAL_SAMPLE_WEIGHT", MANUAL_SAMPLE_WEIGHT)
    manual_train_oversample_factor = _env_int(
        "MANUAL_TRAIN_OVERSAMPLE_FACTOR",
        MANUAL_TRAIN_OVERSAMPLE_FACTOR,
    )
    cache_name = os.getenv("CACHE_NAME", CACHE_NAME).strip()
    if not cache_name:
        raise ValueError("CACHE_NAME must be non-empty")
    use_cache = _env_bool("USE_CACHE", True)
    rebuild_cache = _env_bool("REBUILD_CACHE", False)
    cache_only = _env_bool("CACHE_ONLY", False)
    cache_dir = Path(os.getenv("CACHE_DIR", str(_default_cache_dir(cache_name))))
    if cache_only and not use_cache:
        raise ValueError("CACHE_ONLY=1 requires USE_CACHE=1")

    _set_seed(seed)

    print("Valve Event Detection - Mixed Sensor Availability")
    print("Project root:", PROJECT_ROOT)
    print("Dataset mode:", dataset_mode)
    if dataset_mode == "all_sensor_availability":
        print("Pressure dataset path:", pressure_dataset_root)
        print("Two-sensor dataset path:", two_sensor_dataset_root)
    else:
        print("Dataset path:", dataset_root)
    print("Cache name:", cache_name)
    print("Target sigma ms:", TARGET_SIGMA_MS)
    print("Target strategy:", target_strategy)
    if site_filter:
        print("Site filter:", ", ".join(sorted(site_filter)))
    if train_site_oversample:
        print(
            "Train site oversample:",
            ", ".join(sorted(train_site_oversample)),
            "x",
            train_site_oversample_factor,
        )
    if use_manual_labels:
        print("Manual labels csv:", manual_labels_csv)
        print("Manual split json:", manual_split_json)
        print("Manual sample weight:", manual_sample_weight)
        print("Manual train oversample factor:", manual_train_oversample_factor)
    if exclude_events_csv.exists():
        print("Exclude events csv:", exclude_events_csv)

    manual_labels = load_manual_labels(manual_labels_csv) if use_manual_labels else {}
    manual_split = (
        _load_manual_split(manual_split_json)
        if use_manual_labels and manual_split_json.exists()
        else None
    )
    excluded_events = _load_excluded_events(exclude_events_csv)
    manual_labeled_count = sum(1 for entry in manual_labels.values() if entry.is_labeled)
    manual_rejected_count = sum(1 for entry in manual_labels.values() if entry.is_rejected)

    preprocessor = EventProcessor(
        PreprocessingConfig(allowed_samplerate=SENSOR_SAMPLERATES)
    )
    expected_cache_meta = {
        "cache_name": cache_name,
        "mode": dataset_mode,
        "dataset_root": str(dataset_root) if dataset_mode != "all_sensor_availability" else None,
        "pressure_dataset_root": str(pressure_dataset_root) if dataset_mode == "all_sensor_availability" else None,
        "two_sensor_dataset_root": str(two_sensor_dataset_root) if dataset_mode == "all_sensor_availability" else None,
        "in_channels": 3,
        "sensor_mask_channels": 3,
        "target_samplerate": preprocessor.config.target_samplerate,
        "allowed_samplerate": list(preprocessor.config.allowed_samplerate),
        "normalize_for_model": preprocessor.config.normalize_for_model,
        "normal_eps": preprocessor.config.normal_eps,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_seed": seed,
        "target_version": CACHE_TARGET_VERSION,
        "target_sigma_ms": TARGET_SIGMA_MS,
        "target_strategy": target_strategy,
        "sensors": ["pressure", "strain", "travel"],
        "allow_missing_optional_sensors": dataset_mode == "all_sensor_availability",
        "skip_flat_signals": True,
        "min_std": 1e-12,
        "max_duration_mismatch_sec": 0.025,
        "use_manual_labels": use_manual_labels,
        "manual_labels_csv": str(manual_labels_csv) if use_manual_labels else None,
        "manual_labels_csv_exists": manual_labels_csv.exists() if use_manual_labels else False,
        "manual_labels_mtime_ns": manual_labels_csv.stat().st_mtime_ns if use_manual_labels and manual_labels_csv.exists() else None,
        "manual_split_json": str(manual_split_json) if use_manual_labels else None,
        "manual_split_json_exists": manual_split_json.exists() if use_manual_labels else False,
        "manual_split_json_mtime_ns": manual_split_json.stat().st_mtime_ns if use_manual_labels and manual_split_json.exists() else None,
        "manual_labeled_count": manual_labeled_count if use_manual_labels else 0,
        "manual_rejected_count": manual_rejected_count if use_manual_labels else 0,
        "manual_sample_weight": manual_sample_weight if use_manual_labels else 1.0,
        "exclude_events_csv": str(exclude_events_csv) if exclude_events_csv.exists() else None,
        "exclude_events_csv_exists": exclude_events_csv.exists(),
        "exclude_events_csv_mtime_ns": exclude_events_csv.stat().st_mtime_ns if exclude_events_csv.exists() else None,
        "excluded_events_count": len(excluded_events),
    }

    full_train_dataset: Subset | CachedValveDataset
    full_val_dataset: Subset | CachedValveDataset
    full_test_dataset: Subset | CachedValveDataset
    raw_split_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    n_total: int | None = None
    skipped_reasons: dict[str, int] = {}

    if use_cache:
        if rebuild_cache or not cache_is_compatible(cache_dir, expected_cache_meta):
            if rebuild_cache:
                print("Cache incompatible, rebuilding... (forced by REBUILD_CACHE=1)")
            else:
                print("Cache incompatible, rebuilding...")
            full_dataset = _build_base_dataset(
                dataset_mode=dataset_mode,
                dataset_root=dataset_root,
                pressure_dataset_root=pressure_dataset_root,
                two_sensor_dataset_root=two_sensor_dataset_root,
                preprocessor=preprocessor,
            )
            removed_before_override = _filter_dataset_excluded_events(full_dataset, excluded_events)
            if removed_before_override:
                print("Excluded events before cache build:", removed_before_override)
            if use_manual_labels:
                full_dataset = ManualLabelOverrideDataset(
                    full_dataset,
                    manual_labels=manual_labels,
                    sigma_ms=TARGET_SIGMA_MS,
                    manual_weight=manual_sample_weight,
                )
            n_total = len(full_dataset)
            print("Discovered source events:", n_total)
            print("Skipped during discovery:", dict(full_dataset.skipped_reasons))
            skipped_reasons = dict(full_dataset.skipped_reasons)
            if n_total == 0:
                raise ValueError("No source events found")

            if manual_split is not None:
                raw_split_indices = _split_indices_with_manual_split(
                    full_dataset.dataset if isinstance(full_dataset, ManualLabelOverrideDataset) else full_dataset,
                    manual_split=manual_split,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    seed=seed,
                )
            else:
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
        else:
            print("Using compatible cache:", cache_dir)

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
        full_dataset = _build_base_dataset(
            dataset_mode=dataset_mode,
            dataset_root=dataset_root,
            pressure_dataset_root=pressure_dataset_root,
            two_sensor_dataset_root=two_sensor_dataset_root,
            preprocessor=preprocessor,
        )
        removed_before_override = _filter_dataset_excluded_events(full_dataset, excluded_events)
        if removed_before_override:
            print("Excluded events before training:", removed_before_override)
        if use_manual_labels:
            full_dataset = ManualLabelOverrideDataset(
                full_dataset,
                manual_labels=manual_labels,
                sigma_ms=TARGET_SIGMA_MS,
                manual_weight=manual_sample_weight,
            )

        n_total = len(full_dataset)
        print("Discovered source events:", n_total)
        print("Skipped during discovery:", dict(full_dataset.skipped_reasons))
        skipped_reasons = dict(full_dataset.skipped_reasons)
        if n_total == 0:
            raise ValueError("No source events found")

        if manual_split is not None:
            raw_split_indices = _split_indices_with_manual_split(
                full_dataset.dataset if isinstance(full_dataset, ManualLabelOverrideDataset) else full_dataset,
                manual_split=manual_split,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
            )
        else:
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

    train_local_idx = _filter_local_indices_by_site(
        full_train_dataset,
        train_local_idx,
        site_filter,
    )
    val_local_idx = _filter_local_indices_by_site(
        full_val_dataset,
        val_local_idx,
        site_filter,
    )
    test_local_idx = _filter_local_indices_by_site(
        full_test_dataset,
        test_local_idx,
        site_filter,
    )
    train_local_idx = _oversample_local_indices_by_site(
        full_train_dataset,
        train_local_idx,
        train_site_oversample,
        train_site_oversample_factor,
    )
    train_local_idx = _oversample_local_indices_by_label_source(
        full_train_dataset,
        train_local_idx,
        "manual",
        manual_train_oversample_factor if use_manual_labels else 1,
    )

    if len(train_local_idx) == 0:
        raise ValueError("Training split is empty after SITE_FILTER/TRAIN_SITE_OVERSAMPLE.")
    if len(val_local_idx) == 0:
        raise ValueError("Validation split is empty after SITE_FILTER.")
    if len(test_local_idx) == 0:
        raise ValueError("Test split is empty after SITE_FILTER.")

    train_dataset = ValveSubset(full_train_dataset, train_local_idx.tolist())
    val_dataset = ValveSubset(full_val_dataset, val_local_idx.tolist())
    test_dataset = ValveSubset(full_test_dataset, test_local_idx.tolist())

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    manual_val_local_idx = np.asarray(
        [
            int(index)
            for index in val_local_idx
            if _sample_label_source(full_val_dataset, int(index)) == "manual"
        ],
        dtype=np.int64,
    )
    manual_test_local_idx = np.asarray(
        [
            int(index)
            for index in test_local_idx
            if _sample_label_source(full_test_dataset, int(index)) == "manual"
        ],
        dtype=np.int64,
    )
    manual_val_dataset = ValveSubset(full_val_dataset, manual_val_local_idx.tolist())
    manual_test_dataset = ValveSubset(full_test_dataset, manual_test_local_idx.tolist())
    if use_manual_labels:
        print("Manual val samples:", len(manual_val_dataset))
        print("Manual test samples:", len(manual_test_dataset))

    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        prefetch_factor=prefetch_factor,
        in_channels=3,
        dropout=dropout,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        model_type=model_type,
        sensor_mask_channels=3,
    )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_mode_name = "all-sensor" if dataset_mode == "all_sensor_availability" else "3sensor"
    lr_name = f"{learning_rate:.0e}".replace("+", "").replace("e-0", "e-").replace("e+0", "e")
    run_dir = artifacts_dir / f"valve-{model_type}-{run_mode_name}-lr{lr_name}-{run_timestamp}"
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

    device = torch.device(train_config.device)
    val_point_metrics = evaluate_point_metrics(model, val_dataset, device)
    test_point_metrics = evaluate_point_metrics(model, test_dataset, device)
    manual_val_point_metrics = (
        evaluate_point_metrics(model, manual_val_dataset, device)
        if len(manual_val_dataset) > 0
        else {"num_events": 0, "valid_order_fraction": None, "points": {}}
    )
    manual_test_point_metrics = (
        evaluate_point_metrics(model, manual_test_dataset, device)
        if len(manual_test_dataset) > 0
        else {"num_events": 0, "valid_order_fraction": None, "points": {}}
    )

    history_json = {
        key: [None if x is None else float(x) for x in values]
        for key, values in history.items()
    }
    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history_json, fp, indent=2)

    torch.save(model.state_dict(), run_dir / "best_model.pt")

    best_epoch = history.get("best_epoch", [None])[-1] if history.get("best_epoch") else None
    best_val_loss = (
        float(history["val_loss"][best_epoch - 1])
        if best_epoch is not None and history["val_loss"]
        else None
    )
    total_training_seconds = history.get("total_seconds", [None])[-1]
    average_epoch_seconds = (
        float(np.mean(history["epoch_seconds"]))
        if history.get("epoch_seconds")
        else None
    )
    run_summary = {
        "mode": "mixed_sensor_training",
        "dataset_mode": dataset_mode,
        "input_channels": ["pressure", "strain", "travel"],
        "sensor_mask_channels": 3,
        "sensor_presence_encoding": "3 binary channels concatenated across time",
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "model_type": model_type,
        "target_sigma_ms": TARGET_SIGMA_MS,
        "target_strategy": target_strategy,
        "workers": workers,
        "prefetch_factor": prefetch_factor,
        "use_cache": use_cache,
        "cache_name": cache_name,
        "cache_dir": str(cache_dir) if use_cache else None,
        "dataset_path": str(dataset_root) if dataset_mode != "all_sensor_availability" else None,
        "pressure_dataset_path": str(pressure_dataset_root) if dataset_mode == "all_sensor_availability" else None,
        "two_sensor_dataset_path": str(two_sensor_dataset_root) if dataset_mode == "all_sensor_availability" else None,
        "total_samples": int(n_total) if n_total is not None else None,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "test_samples": int(len(test_dataset)),
        "site_filter": sorted(site_filter),
        "train_site_oversample": sorted(train_site_oversample),
        "train_site_oversample_factor": train_site_oversample_factor,
        "use_manual_labels": use_manual_labels,
        "manual_labels_csv": str(manual_labels_csv) if use_manual_labels else None,
        "manual_split_json": str(manual_split_json) if use_manual_labels else None,
        "manual_labeled_count": manual_labeled_count if use_manual_labels else 0,
        "manual_rejected_count": manual_rejected_count if use_manual_labels else 0,
        "manual_sample_weight": manual_sample_weight if use_manual_labels else 1.0,
        "manual_train_oversample_factor": manual_train_oversample_factor if use_manual_labels else 1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stopped_early": bool(history.get("stopped_early", [False])[-1]),
        "stopped_epoch": history.get("stopped_epoch", [None])[-1] if history.get("stopped_epoch") else None,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
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
        "test_indices_file": "test_idx.npy",
        "skipped_reasons": skipped_reasons,
        "val_point_metrics": val_point_metrics,
        "test_point_metrics": test_point_metrics,
        "manual_val_point_metrics": manual_val_point_metrics,
        "manual_test_point_metrics": manual_test_point_metrics,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, indent=2)

    print("3-sensor training complete")
    print("Best val_loss:", best_val_loss)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
