from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.data.cached_dataset import CachedValveDataset
from src.data.data_loader import load_event
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.models.CNN_model import ValveEventCNN

RUN_NAME = "valve-cnn-20260508-190512"
NUM_RANDOM_EVENTS = 500
RANDOM_SEED = 42
POINT_NAMES = ["PRP", "BP", "SP"]

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "valve-1sensor-v1"
CACHE_SPLITS = ("train", "val", "test")


def load_model(run_dir: Path, device: torch.device) -> ValveEventCNN:
    model_path = run_dir / "best_model.pt"

    if not model_path.exists():
        raise ValueError(f"Missing model file: {model_path}")

    model = ValveEventCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def resolve_cache_dir(run_dir: Path) -> Path:
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)

        cache_dir_value = summary.get("cache_dir")
        if cache_dir_value:
            cache_dir = Path(cache_dir_value)
            if cache_dir.exists():
                return cache_dir

    if DEFAULT_CACHE_DIR.exists():
        return DEFAULT_CACHE_DIR

    raise FileNotFoundError(
        f"Could not resolve cache dir. Checked summary.json and {DEFAULT_CACHE_DIR}"
    )


@torch.no_grad()
def predict_points(
    model: ValveEventCNN,
    signal: np.ndarray,
    samplerate: int,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    x = torch.from_numpy(signal).to(torch.float32)
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    results: dict[str, dict[str, float | int]] = {}

    for c, name in enumerate(POINT_NAMES):
        values = probs[c]
        idx = int(np.argmax(values))
        conf = float(values[idx])

        results[name] = {
            "index": idx,
            "time_sec": idx / float(samplerate),
            "confidence": conf,
        }

    return results


def pseudo_label_points(
    y: torch.Tensor | np.ndarray,
    samplerate: int,
) -> dict[str, dict[str, float | int] | None]:
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    if y_np.ndim != 2:
        raise ValueError(f"Expected pseudo-label y shape (3, T), got {y_np.shape}")

    if y_np.shape[0] != len(POINT_NAMES):
        raise ValueError(
            f"Expected {len(POINT_NAMES)} pseudo-label channels, got {y_np.shape[0]}"
        )

    results: dict[str, dict[str, float | int] | None] = {}

    for c, name in enumerate(POINT_NAMES):
        values = y_np[c]

        if values.size == 0 or float(np.max(values)) <= 0.0:
            results[name] = None
            continue

        idx = int(np.argmax(values))
        conf = float(values[idx])

        results[name] = {
            "index": idx,
            "time_sec": idx / float(samplerate),
            "target_value": conf,
        }

    return results


def _meta_get(meta: Any, key: str) -> Any:
    if isinstance(meta, dict):
        return meta.get(key)

    if hasattr(meta, key):
        return getattr(meta, key)

    return None


def _add_path_keys(keys: set[tuple[str, str]], value: Any) -> None:
    if value is None:
        return

    path = Path(str(value))
    keys.add(("filename", path.name))
    keys.add(("stem", path.stem))


def _cache_keys_from_meta(meta: Any) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()

    for path_key in (
        "path",
        "file",
        "filename",
        "event_path",
        "source_path",
        "raw_path",
        "json_path",
    ):
        _add_path_keys(keys, _meta_get(meta, path_key))

    eventid = _meta_get(meta, "eventid")
    sitename = _meta_get(meta, "sitename")
    site = _meta_get(meta, "site")
    valvetag = _meta_get(meta, "valvetag")

    if eventid is not None:
        keys.add(("eventid", str(eventid)))

    site_value = sitename or site
    if site_value is not None and eventid is not None:
        keys.add(("site_eventid", f"{site_value}_{eventid}"))

    if valvetag is not None and eventid is not None:
        keys.add(("tag_eventid", f"{valvetag}_{eventid}"))

    return keys


def _event_keys_from_raw_event(event: Any, event_path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()

    _add_path_keys(keys, event_path)

    eventid = getattr(event, "eventid", None)
    sitename = getattr(event, "sitename", None)
    site = getattr(event, "site", None)
    valvetag = getattr(event, "valvetag", None)

    if eventid is not None:
        keys.add(("eventid", str(eventid)))

    site_value = sitename or site
    if site_value is not None and eventid is not None:
        keys.add(("site_eventid", f"{site_value}_{eventid}"))

    if valvetag is not None and eventid is not None:
        keys.add(("tag_eventid", f"{valvetag}_{eventid}"))

    return keys


def build_pseudo_label_lookup(cache_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for split in CACHE_SPLITS:
        dataset = CachedValveDataset(cache_dir, split=split)

        print(f"Loading pseudo-labels from cache split={split}, samples={len(dataset)}")

        for item_index in range(len(dataset)):
            item = dataset[item_index]

            if "y" not in item:
                raise KeyError(f"Cached item missing 'y' in split={split}, index={item_index}")

            y = item["y"]
            meta = item.get("meta", {})

            label_points = pseudo_label_points(y, samplerate=400)

            record = {
                "split": split,
                "item_index": item_index,
                "labels": label_points,
                "meta": meta,
            }

            keys = _cache_keys_from_meta(meta)

            for key in keys:
                lookup.setdefault(key, record)

    if not lookup:
        raise ValueError(f"No pseudo-label lookup entries were built from cache: {cache_dir}")

    return lookup


def find_pseudo_labels_for_event(
    *,
    event: Any,
    event_path: Path,
    pseudo_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    keys = _event_keys_from_raw_event(event, event_path)

    priority = ("filename", "stem", "site_eventid", "tag_eventid", "eventid")

    for key_type in priority:
        for key in keys:
            if key[0] == key_type and key in pseudo_lookup:
                return pseudo_lookup[key]

    return None


def compute_position_errors(
    predictions: dict[str, dict[str, float | int]],
    labels: dict[str, dict[str, float | int] | None],
    samplerate: int,
) -> dict[str, dict[str, float | int] | None]:
    errors: dict[str, dict[str, float | int] | None] = {}

    for name in POINT_NAMES:
        pred = predictions.get(name)
        label = labels.get(name)

        if pred is None or label is None:
            errors[name] = None
            continue

        pred_idx = int(pred["index"])
        label_idx = int(label["index"])

        error_samples = abs(pred_idx - label_idx)
        error_seconds = error_samples / float(samplerate)

        errors[name] = {
            "error_samples": int(error_samples),
            "error_seconds": float(error_seconds),
            "error_ms": float(error_seconds * 1000.0),
        }

    return errors


def plot_predictions(
    raw_signal: np.ndarray,
    raw_samplerate: int,
    normalized_signal: np.ndarray,
    samplerate: int,
    predictions: dict[str, dict[str, float | int]],
    labels: dict[str, dict[str, float | int] | None],
    errors: dict[str, dict[str, float | int] | None],
    event_path: Path,
    run_dir: Path,
    cache_record: dict[str, Any] | None,
) -> None:
    raw_time = np.arange(raw_signal.size, dtype=np.float32) / float(raw_samplerate)
    model_time = np.arange(normalized_signal.size, dtype=np.float32) / float(samplerate)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    colors = {"PRP": "blue", "BP": "orange", "SP": "green"}

    ax = axes[0]
    ax.plot(raw_time, raw_signal, label=f"raw signal {raw_samplerate} Hz", alpha=0.45)
    ax.set_title("Raw event signal")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(
        model_time,
        normalized_signal,
        label=f"normalized model input {samplerate} Hz",
        linewidth=1.0,
        alpha=0.9,
    )

    for name in POINT_NAMES:
        pred = predictions.get(name)
        label = labels.get(name)
        error = errors.get(name)

        if label is not None:
            label_t = float(label["time_sec"])
            label_idx = int(label["index"])

            ax.axvline(
                label_t,
                linestyle="-",
                color=colors[name],
                linewidth=1.5,
                alpha=0.75,
                label=f"{name} pseudo idx={label_idx}",
            )

        if pred is not None:
            pred_t = float(pred["time_sec"])
            pred_idx = int(pred["index"])
            conf = float(pred["confidence"])

            if error is not None:
                err_ms = float(error["error_ms"])
                label_text = f"{name} pred idx={pred_idx} conf={conf:.3f} err={err_ms:.1f}ms"
            else:
                label_text = f"{name} pred idx={pred_idx} conf={conf:.3f}"

            ax.axvline(
                pred_t,
                linestyle="--",
                color=colors[name],
                linewidth=2.0,
                alpha=0.95,
                label=label_text,
            )

    ax.set_title("Model input with predictions and cached pseudo-labels")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized pressure")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if cache_record is not None:
        cache_info = f"cache split={cache_record['split']} idx={cache_record['item_index']}"
    else:
        cache_info = "cache labels missing"

    fig.suptitle(
        f"{event_path.name} | Model: {run_dir.name} | {cache_info}",
        y=0.99,
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    run_dir = PROJECT_ROOT / "artifacts" / RUN_NAME
    raw_dir = PROJECT_ROOT / "src" / "data" / "raw"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("raw_dir:", raw_dir)
    print("raw_dir exists:", raw_dir.exists())
    print("run_dir:", run_dir)
    print("run_dir exists:", run_dir.exists())

    if not run_dir.exists():
        raise ValueError(f"Run not found: {run_dir}")

    cache_dir = resolve_cache_dir(run_dir)
    print("cache_dir:", cache_dir)
    print("cache_dir exists:", cache_dir.exists())

    pseudo_lookup = build_pseudo_label_lookup(cache_dir)

    all_event_paths = sorted(raw_dir.glob("*/events/*.json"))

    if not all_event_paths:
        raise ValueError(f"No raw event files found in: {raw_dir}")

    rng = np.random.default_rng(RANDOM_SEED)

    n = min(NUM_RANDOM_EVENTS, len(all_event_paths))
    selected_indices = rng.choice(len(all_event_paths), size=n, replace=False)
    selected_paths = [all_event_paths[i] for i in selected_indices]

    print("Using model:", run_dir)
    print(f"Found {len(all_event_paths)} raw events.")
    print(f"Randomly selected {len(selected_paths)} events.")

    processor = EventProcessor(
        PreprocessingConfig(
            allowed_samplerate=(200, 400, 800, 1000),
            target_samplerate=400,
            normalize_for_model=True,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, device)

    all_errors_ms: dict[str, list[float]] = {name: [] for name in POINT_NAMES}
    matched_events = 0
    missing_label_events = 0

    for i, event_path in enumerate(selected_paths, start=1):
        print(f"\n[{i}/{len(selected_paths)}] Using file: {event_path}")

        try:
            event = load_event(event_path)
            processed = processor.preprocess_event(event)

            predictions = predict_points(
                model=model,
                signal=processed.normalized_signal,
                samplerate=processed.samplerate,
                device=device,
            )

            cache_record = find_pseudo_labels_for_event(
                event=event,
                event_path=event_path,
                pseudo_lookup=pseudo_lookup,
            )

            if cache_record is None:
                missing_label_events += 1
                labels = {name: None for name in POINT_NAMES}
                print("Pseudo-labels: missing from cache lookup")
            else:
                matched_events += 1
                labels = cache_record["labels"]
                print(
                    f"Pseudo-labels: cache split={cache_record['split']} "
                    f"idx={cache_record['item_index']}"
                )

            errors = compute_position_errors(
                predictions=predictions,
                labels=labels,
                samplerate=processed.samplerate,
            )

            print("Predictions vs pseudo-labels:")
            for name in POINT_NAMES:
                pred = predictions.get(name)
                label = labels.get(name)
                error = errors.get(name)

                pred_text = (
                    f"pred_idx={int(pred['index'])} "
                    f"pred_time={float(pred['time_sec']):.3f}s "
                    f"conf={float(pred['confidence']):.4f}"
                    if pred is not None
                    else "pred=missing"
                )

                label_text = (
                    f"label_idx={int(label['index'])} "
                    f"label_time={float(label['time_sec']):.3f}s"
                    if label is not None
                    else "label=missing"
                )

                error_text = (
                    f"MAE={float(error['error_ms']):.1f}ms "
                    f"({int(error['error_samples'])} samples)"
                    if error is not None
                    else "MAE=missing"
                )

                print(f" {name}: {pred_text} | {label_text} | {error_text}")

                if error is not None:
                    all_errors_ms[name].append(float(error["error_ms"]))

            plot_predictions(
                raw_signal=np.asarray(event.signal, dtype=np.float32),
                raw_samplerate=event.samplerate,
                normalized_signal=processed.normalized_signal,
                samplerate=processed.samplerate,
                predictions=predictions,
                labels=labels,
                errors=errors,
                event_path=event_path,
                run_dir=run_dir,
                cache_record=cache_record,
            )

        except Exception as exc:
            print(f"Failed on {event_path}: {exc}")
            continue

    print("\nSummary")
    print(f"Matched cache labels for {matched_events}/{len(selected_paths)} events")
    print(f"Missing cache labels for {missing_label_events}/{len(selected_paths)} events")

    for name in POINT_NAMES:
        values = all_errors_ms[name]

        if not values:
            print(f"{name}: positional MAE unavailable")
            continue

        print(
            f"{name}: positional MAE = {float(np.mean(values)):.2f} ms "
            f"over {len(values)} events"
        )


if __name__ == "__main__":
    main()