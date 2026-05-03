from __future__ import annotations

import argparse
import json
from dataclasses import replace
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.data.data_loader import load_event
from src.utils.metrics import event_position_mae
from src.data.multisensor_dataset import MULTISENSOR_SAMPLERATES, SENSOR_NAMES
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (
    BPConfig,
    MultiSensorBPConfig,
    MultiSensorSPConfig,
    PRPConfig,
    SPConfig,
    detect_bp_multisensor,
    detect_prp,
    detect_sp_multisensor,
)
from src.models.CNN_model import ValveEventCNN
from src.models.TCN_model import ValveEventTCN


DEFAULT_DATASET_ROOT = Path("/mnt/d/Bachelor_data/data/raw_complete_3sensor")
POINT_NAMES = ("PRP", "BP", "SP")
SENSOR_COLORS = {
    "pressure": "tab:blue",
    "strain": "tab:orange",
    "travel": "tab:green",
}
POINT_COLORS = {
    "PRP": "tab:blue",
    "BP": "tab:red",
    "SP": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3-sensor model on one complete event."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Model run directory. Defaults to latest artifacts/valve-cnn-3sensor-*.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root with SITE/events/{pressure,strain,travel} JSON files.",
    )
    parser.add_argument(
        "--pressure-path",
        type=Path,
        default=None,
        help="Path to a pressure JSON file inside the complete 3-sensor dataset.",
    )
    parser.add_argument("--site", help="Site name, e.g. GRA or TROA.")
    parser.add_argument("--eventid", type=int, help="Event ID to find inside --site.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output PNG path. If omitted, shows an interactive plot.",
    )
    return parser.parse_args()


def _latest_run_dir() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "artifacts").glob("valve-cnn-3sensor-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No artifacts/valve-cnn-3sensor-* runs found.")
    return candidates[0]


def _resolve_pressure_path(args: argparse.Namespace) -> Path:
    if args.pressure_path is not None:
        return args.pressure_path

    if not args.site or args.eventid is None:
        raise ValueError("Provide either --pressure-path or both --site and --eventid.")

    pressure_path = (
        args.dataset_root
        / args.site
        / "events"
        / "pressure"
        / f"{args.site}_{args.eventid}.json"
    )
    if not pressure_path.exists():
        raise FileNotFoundError(f"Pressure event not found: {pressure_path}")
    return pressure_path


def _sensor_group_from_pressure_path(dataset_root: Path, pressure_path: Path) -> dict[str, Path]:
    site = pressure_path.stem.split("_", 1)[0]
    filename = pressure_path.name
    group = {
        "pressure": pressure_path,
        "strain": dataset_root / site / "events" / "strain" / filename,
        "travel": dataset_root / site / "events" / "travel" / filename,
    }

    missing = [sensor_name for sensor_name, path in group.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sensor files for {pressure_path.name}: {missing}")

    return group


def _load_processed_group(group: dict[str, Path]) -> tuple[dict, dict]:
    processor = EventProcessor(
        PreprocessingConfig(allowed_samplerate=MULTISENSOR_SAMPLERATES)
    )
    raw_by_sensor = {}
    processed_by_sensor = {}

    anchor = None
    for sensor_name in SENSOR_NAMES:
        event = load_event(group[sensor_name])
        if anchor is None:
            anchor = event
        elif (event.eventid, event.valvetag, event.sitename) != (
            anchor.eventid,
            anchor.valvetag,
            anchor.sitename,
        ):
            raise ValueError(f"Mismatched metadata for {sensor_name}: {group[sensor_name]}")
        raw_by_sensor[sensor_name] = event
        processed_by_sensor[sensor_name] = processor.preprocess_event(event)

    return raw_by_sensor, processed_by_sensor


def _prepare_model_input(processed_by_sensor: dict) -> tuple[torch.Tensor, dict, int]:
    min_length = min(processed.n_samples for processed in processed_by_sensor.values())
    channels = [
        processed_by_sensor[sensor_name].normalized_signal[:min_length]
        for sensor_name in SENSOR_NAMES
    ]
    x = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
    x = x.unsqueeze(0)

    pressure = processed_by_sensor["pressure"]
    pressure_for_targets = replace(
        pressure,
        resampled_signal=pressure.resampled_signal[:min_length],
        normalized_signal=pressure.normalized_signal[:min_length],
        n_samples=min_length,
        duration_sec=min_length / float(pressure.samplerate),
    )
    return x, pressure_for_targets, min_length


def _load_model(run_dir: Path, device: torch.device) -> torch.nn.Module:
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    summary_path = run_dir / "summary.json"
    model_type = "cnn"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
        model_type = str(summary.get("model_type", "cnn")).lower()

    if model_type == "tcn":
        model = ValveEventTCN(in_channels=3).to(device)
    elif model_type == "cnn":
        model = ValveEventCNN(in_channels=3).to(device)
    else:
        raise ValueError(f"Unknown model_type in {summary_path}: {model_type}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _predict_points(
    model: torch.nn.Module,
    x: torch.Tensor,
    samplerate: int,
    device: torch.device,
) -> tuple[dict[str, dict], np.ndarray]:
    logits = model(x.to(device))
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    predictions = {}
    for channel_index, name in enumerate(POINT_NAMES):
        values = probs[channel_index]
        index = int(np.argmax(values))
        predictions[name] = {
            "index": index,
            "time_sec": index / float(samplerate),
            "confidence": float(values[index]),
        }
    return predictions, probs


def _heuristic_points(processed_by_sensor: dict, pressure_event) -> dict[str, dict]:
    signal = pressure_event.resampled_signal
    samplerate = pressure_event.samplerate
    travel_signal = processed_by_sensor["travel"].resampled_signal[:pressure_event.n_samples]
    strain_signal = processed_by_sensor["strain"].resampled_signal[:pressure_event.n_samples]

    prp_result = detect_prp(signal, PRPConfig(samplerate=samplerate))
    bp_result = detect_bp_multisensor(
        pressure_signal=signal,
        travel_signal=travel_signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        strain_signal=strain_signal,
        pressure_config=BPConfig(),
        multisensor_config=MultiSensorBPConfig(),
    )
    sp_result = detect_sp_multisensor(
        pressure_signal=signal,
        travel_signal=travel_signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        bp_index=bp_result.bp_index,
        strain_signal=strain_signal,
        pressure_config=SPConfig(),
        multisensor_config=MultiSensorSPConfig(),
    )
    return {
        "PRP": {"index": prp_result.prp_index, "reason": prp_result.reason},
        "BP": {"index": bp_result.bp_index, "reason": bp_result.reason},
        "SP": {"index": sp_result.sp_index, "reason": sp_result.reason},
    }

def _position_mae_by_point(predictions: dict[str, dict], heuristic_points: dict[str, dict]) -> dict[str, float | None]:
    errors: dict[str, float | None] = {}

    for point_name in POINT_NAMES:
        pred_index = predictions[point_name]["index"]
        true_index = heuristic_points[point_name]["index"]

        if true_index is None or pred_index is None:
            errors[point_name] = None
            continue

        result = event_position_mae(true_indices=[true_index], pred_indices=[pred_index])
        errors[point_name] = result.mae_position

    return errors

def _normalize_for_plot(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    std = float(np.std(signal))
    if std < 1e-12:
        return np.zeros_like(signal)
    return ((signal - float(np.mean(signal))) / std).astype(np.float32)


def _plot(
    *,
    raw_by_sensor: dict,
    processed_by_sensor: dict,
    pressure_event,
    predictions: dict[str, dict],
    heuristic_points: dict[str, dict],
    probs: np.ndarray,
    run_dir: Path,
    save_path: Path | None,
) -> None:
    samplerate = pressure_event.samplerate
    time_axis = np.arange(pressure_event.n_samples, dtype=np.float32) / float(samplerate)
    raw_pressure = raw_by_sensor["pressure"]
    raw_time_axis = np.arange(raw_pressure.n_samples, dtype=np.float32) / float(raw_pressure.samplerate)

    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=False)

    axes[0].plot(
        raw_time_axis,
        np.asarray(raw_pressure.signal, dtype=np.float32),
        color=SENSOR_COLORS["pressure"],
        linewidth=1.0,
        alpha=0.9,
        label="raw pressure",
    )
    axes[0].set_ylabel("pressure")
    axes[0].set_title(f"{raw_pressure.sitename}_{raw_pressure.eventid} | raw pressure")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    for sensor_name in SENSOR_NAMES:
        signal = processed_by_sensor[sensor_name].resampled_signal[:pressure_event.n_samples]
        axes[1].plot(
            time_axis,
            _normalize_for_plot(signal),
            color=SENSOR_COLORS[sensor_name],
            linewidth=1.0,
            alpha=0.8,
            label=sensor_name,
        )
    axes[1].set_ylabel("z-score")
    axes[1].set_title(f"{run_dir.name} | normalized sensors")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    for point_name, color in POINT_COLORS.items():
        pred_index = predictions[point_name]["index"]
        axes[1].axvline(
            pred_index / float(samplerate),
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"pred {point_name}",
        )
        heuristic_index = heuristic_points[point_name]["index"]
        if heuristic_index is not None:
            axes[1].axvline(
                heuristic_index / float(samplerate),
                color=color,
                linestyle=":",
                linewidth=1.4,
                label=f"heuristic {point_name}",
            )

    for channel_index, point_name in enumerate(POINT_NAMES, start=2):
        axes[channel_index].plot(
            time_axis,
            probs[channel_index - 2],
            color=POINT_COLORS[point_name],
            linewidth=1.0,
            label=f"{point_name} probability",
        )
        pred_index = predictions[point_name]["index"]
        axes[channel_index].axvline(
            pred_index / float(samplerate),
            color=POINT_COLORS[point_name],
            linestyle="--",
            label=f"pred {pred_index}",
        )
        heuristic_index = heuristic_points[point_name]["index"]
        if heuristic_index is not None:
            axes[channel_index].axvline(
                heuristic_index / float(samplerate),
                color="black",
                linestyle=":",
                label=f"heuristic {heuristic_index}",
            )
        axes[channel_index].set_ylabel(point_name)
        axes[channel_index].grid(True, alpha=0.25)
        axes[channel_index].legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved plot:", save_path)
        return

    plt.show()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _latest_run_dir()
    pressure_path = _resolve_pressure_path(args).resolve()
    group = _sensor_group_from_pressure_path(dataset_root, pressure_path)

    print("Run dir:", run_dir)
    print("Pressure:", group["pressure"])
    print("Strain:  ", group["strain"])
    print("Travel:  ", group["travel"])

    raw_by_sensor, processed_by_sensor = _load_processed_group(group)
    x, pressure_event, _ = _prepare_model_input(processed_by_sensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run_dir, device)
    predictions, probs = _predict_points(model, x, pressure_event.samplerate, device)
    heuristic = _heuristic_points(processed_by_sensor, pressure_event)

    position_mae = _position_mae_by_point(predictions=predictions, heuristic_points=heuristic)

    print("\nEvent")
    pressure_raw = raw_by_sensor["pressure"]
    print(" site:", pressure_raw.sitename)
    print(" valve:", pressure_raw.valvetag)
    print(" eventid:", pressure_raw.eventid)
    print(" samplerate:", pressure_event.samplerate)
    print(" duration_sec:", f"{pressure_event.duration_sec:.3f}")

    print("\nPredictions vs heuristic")
    for point_name in POINT_NAMES:
        prediction = predictions[point_name]
        heuristic_index = heuristic[point_name]["index"]
        heuristic_time = (
            heuristic_index / float(pressure_event.samplerate)
            if heuristic_index is not None
            else None
        )
        mae_position = position_mae[point_name]

        if heuristic_time is not None:
            print(
                f" {point_name}: pred={prediction['index']} "
                f"t={prediction['time_sec']:.3f}s "
                f"conf={prediction['confidence']:.4f} | "
                f"heuristic={heuristic_index} "
                f"t={heuristic_time:.3f}s | "
                f"mae_pos={mae_position:.1f}"
            )
        else:
            print(
                f" {point_name}: pred={prediction['index']} "
                f"t={prediction['time_sec']:.3f}s "
                f"conf={prediction['confidence']:.4f} | "
                f"heuristic=None | "
                f"mae_pos=None"
            )

    _plot(
        raw_by_sensor=raw_by_sensor,
        processed_by_sensor=processed_by_sensor,
        pressure_event=pressure_event,
        predictions=predictions,
        heuristic_points=heuristic,
        probs=probs,
        run_dir=run_dir,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
