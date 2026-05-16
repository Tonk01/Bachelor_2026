from __future__ import annotations

import argparse
import json
from dataclasses import replace
import os
from pathlib import Path
import sys
from copy import copy
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.data.data_loader import load_event
from src.data.multisensor_dataset import MULTISENSOR_SAMPLERATES, SENSOR_NAMES
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (
    MultivariatConfig,
    detect_BP_multivariat,
    detect_PRP_multivariat,
    detect_SP_multivariat,
)
from src.models.CNN_model import ValveEventCNN
from src.models.TCN_model import ValveEventTCN, ValveEventTCNMedium, ValveEventTCNSmall
from src.utils.metrics import event_position_mae


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
        help="Model run directory. Defaults to latest artifacts/valve-{cnn,tcn}-3sensor-*.",
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
        help="Path to a pressure JSON file.",
    )
    parser.add_argument("--site", help="Site name, e.g. GRA or TROA.")
    parser.add_argument("--eventid", type=int, help="Event ID to find inside --site.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output PNG path. If omitted, shows an interactive plot.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.1,
        help="Minimum probability for accepting a detected PRP/BP/SP peak.",
    )
    parser.add_argument(
        "--ignore-start-sec",
        type=float,
        default=1.0,
        help="Ignore model peaks before this time to avoid start/edge spikes.",
    )
    parser.add_argument(
        "--ignore-end-sec",
        type=float,
        default=0.5,
        help="Ignore model peaks this many seconds before the signal end.",
    )
    parser.add_argument(
        "--event-window-before-sec",
        type=float,
        default=2.0,
        help="Search this many seconds before the largest pressure change.",
    )
    parser.add_argument(
        "--event-window-after-sec",
        type=float,
        default=20.0,
        help="Search this many seconds after the largest pressure change.",
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=0.05,
        help="Minimum time gap required between PRP, BP, and SP peaks.",
    )
    parser.add_argument(
        "--sp-min-after-bp-sec",
        type=float,
        default=0.05,
        help="Minimum time gap required from BP to SP.",
    )
    parser.add_argument(
        "--sp-pressure-prior-weight",
        type=float,
        default=0.0,
        help="Extra SP score weight for peaks close to a pressure change.",
    )
    parser.add_argument(
        "--sp-pressure-prior-window-sec",
        type=float,
        default=0.75,
        help="Window around an SP candidate used to score nearby pressure changes.",
    )
    parser.add_argument(
        "--max-peaks-per-channel",
        type=int,
        default=80,
        help="Keep only this many strongest candidate peaks per channel.",
    )
    parser.add_argument(
        "--pressure-only-fill-missing",
        action="store_true",
        help="Evaluate a pressure-only JSON by filling missing strain/travel channels with zeros.",
    )
    return parser.parse_args()


def _latest_run_dir() -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    candidates = sorted(
        [
            *artifacts_dir.glob("valve-cnn-3sensor-*"),
            *artifacts_dir.glob("valve-tcn-3sensor-*"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No artifacts/valve-{cnn,tcn}-3sensor-* runs found.")
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


def _load_pressure_only_group(pressure_path: Path) -> tuple[dict, dict]:
    processor = EventProcessor(
        PreprocessingConfig(allowed_samplerate=MULTISENSOR_SAMPLERATES)
    )
    pressure_event = load_event(pressure_path)
    pressure_processed = processor.preprocess_event(pressure_event)

    raw_by_sensor = {"pressure": pressure_event}
    processed_by_sensor = {"pressure": pressure_processed}

    zero_raw = [0.0] * pressure_event.n_samples
    for sensor_name in ("strain", "travel"):
        raw_event = copy(pressure_event)
        raw_event.signal = zero_raw
        raw_event.path = f"{pressure_event.path}::{sensor_name}_zeros"
        raw_by_sensor[sensor_name] = raw_event

        processed_event = copy(pressure_processed)
        processed_event.resampled_signal = np.zeros_like(pressure_processed.resampled_signal)
        processed_event.normalized_signal = np.zeros_like(pressure_processed.normalized_signal)
        processed_event.path = raw_event.path
        processed_by_sensor[sensor_name] = processed_event

    return raw_by_sensor, processed_by_sensor


def _prepare_model_input(
    processed_by_sensor: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict, int]:
    min_length = min(processed.n_samples for processed in processed_by_sensor.values())
    channels = [
        processed_by_sensor[sensor_name].normalized_signal[:min_length]
        for sensor_name in SENSOR_NAMES
    ]
    x = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
    x = x.unsqueeze(0)
    sensor_presence = torch.ones((1, len(SENSOR_NAMES)), dtype=torch.float32)

    pressure = processed_by_sensor["pressure"]
    pressure_for_targets = replace(
        pressure,
        resampled_signal=pressure.resampled_signal[:min_length],
        normalized_signal=pressure.normalized_signal[:min_length],
        n_samples=min_length,
        duration_sec=min_length / float(pressure.samplerate),
    )
    return x, sensor_presence, pressure_for_targets, min_length


def _load_model(run_dir: Path, device: torch.device) -> torch.nn.Module:
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    summary_path = run_dir / "summary.json"
    model_type = "cnn"
    in_channels = 3
    sensor_mask_channels = 0
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
        model_type = str(summary.get("model_type", "cnn")).lower()
        in_channels = int(summary.get("in_channels", in_channels))
        sensor_mask_channels = int(
            summary.get("sensor_mask_channels", sensor_mask_channels)
        )

    if model_type == "tcn":
        model = ValveEventTCN(
            in_channels=in_channels,
            sensor_mask_channels=sensor_mask_channels,
        ).to(device)
    elif model_type == "tcn_medium":
        model = ValveEventTCNMedium(
            in_channels=in_channels,
            sensor_mask_channels=sensor_mask_channels,
        ).to(device)
    elif model_type == "tcn_small":
        model = ValveEventTCNSmall(
            in_channels=in_channels,
            sensor_mask_channels=sensor_mask_channels,
        ).to(device)
    elif model_type == "cnn":
        model = ValveEventCNN(
            in_channels=in_channels,
            sensor_mask_channels=sensor_mask_channels,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type in {summary_path}: {model_type}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _postprocess_rule(
    *,
    min_prob: float,
    min_gap_sec: float,
    sp_min_after_bp_sec: float,
    max_peaks_per_channel: int,
) -> dict[str, float | int]:
    rule = {
        "min_prob": float(min_prob),
        "fallback_min_prob": float(min(0.02, min_prob)),
        "min_gap_sec": float(min_gap_sec),
        "sp_min_after_bp_sec": float(sp_min_after_bp_sec),
        "max_peaks_per_channel": int(max_peaks_per_channel),
        "sp_travel_prior_weight": 0.0,
    }
    return rule


def _travel_sp_prior(
    travel_signal: np.ndarray,
    samplerate: int,
) -> np.ndarray:
    signal = np.asarray(travel_signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    tail_n = max(3, min(signal.size, int(round(1.0 * samplerate))))
    final_level = float(np.median(signal[-tail_n:]))
    travel_range = max(float(np.max(signal) - np.min(signal)), 1e-8)
    closeness = 1.0 - np.clip(np.abs(signal - final_level) / travel_range, 0.0, 1.0)
    slope = np.abs(np.gradient(signal)).astype(np.float32)
    slope_scale = float(np.percentile(slope, 90)) if slope.size > 0 else 1.0
    slope_scale = max(slope_scale, 1e-8)
    quiet = 1.0 - np.clip(slope / slope_scale, 0.0, 1.0)
    return (closeness * quiet).astype(np.float32)


def _moving_average_for_window(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float32).copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(signal.astype(np.float32), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _pressure_event_window(
    pressure_signal: np.ndarray,
    samplerate: int,
    *,
    ignore_start_sec: float,
    ignore_end_sec: float,
    before_sec: float,
    after_sec: float,
) -> tuple[int, int, int | None]:
    n_samples = int(pressure_signal.size)
    if n_samples == 0:
        return 0, 0, None

    valid_start = max(0, int(round(ignore_start_sec * samplerate)))
    valid_end = min(n_samples, n_samples - int(round(ignore_end_sec * samplerate)))
    if valid_end <= valid_start:
        return 0, n_samples, None

    smooth_window = max(3, int(round(0.05 * samplerate)))
    smoothed = _moving_average_for_window(pressure_signal, smooth_window)
    derivative = np.abs(np.gradient(smoothed)).astype(np.float32)

    local = derivative[valid_start:valid_end]
    if local.size == 0 or float(np.max(local)) <= 0.0:
        return valid_start, valid_end, None

    event_center = valid_start + int(np.argmax(local))
    window_start = max(valid_start, event_center - int(round(before_sec * samplerate)))
    window_end = min(valid_end, event_center + int(round(after_sec * samplerate)))
    if window_end <= window_start:
        return valid_start, valid_end, event_center

    return window_start, window_end, event_center


def _pressure_change_prior(
    pressure_signal: np.ndarray,
    samplerate: int,
    window_sec: float,
) -> np.ndarray:
    n_samples = int(pressure_signal.size)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)

    smooth_window = max(3, int(round(0.05 * samplerate)))
    smoothed = _moving_average_for_window(pressure_signal, smooth_window)
    derivative = np.abs(np.gradient(smoothed)).astype(np.float32)
    max_value = float(np.max(derivative))
    if max_value <= 0.0:
        return np.zeros_like(derivative)

    derivative = derivative / max_value
    radius = max(1, int(round(window_sec * samplerate)))
    prior = np.zeros_like(derivative)
    for index in range(n_samples):
        start = max(0, index - radius)
        end = min(n_samples, index + radius + 1)
        prior[index] = float(np.max(derivative[start:end]))
    return prior


def _pressure_change_candidate_indices(
    pressure_signal: np.ndarray,
    samplerate: int,
    *,
    search_start: int,
    search_end: int,
    window_sec: float,
    max_candidates: int = 20,
) -> list[int]:
    if pressure_signal.size == 0 or search_end <= search_start:
        return []

    smooth_window = max(3, int(round(0.05 * samplerate)))
    smoothed = _moving_average_for_window(pressure_signal, smooth_window)
    derivative = np.abs(np.gradient(smoothed)).astype(np.float32)
    max_value = float(np.max(derivative[search_start:search_end]))
    if max_value <= 0.0:
        return []

    peaks, properties = find_peaks(derivative[search_start:search_end], height=max_value * 0.1)
    ranked = sorted(
        (
            (search_start + int(index), float(height))
            for index, height in zip(peaks, properties.get("peak_heights", []))
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    radius = max(1, int(round(window_sec * samplerate)))
    chosen: list[int] = []
    for index, _ in ranked:
        if any(abs(index - existing) <= radius for existing in chosen):
            continue
        chosen.append(index)
        if len(chosen) >= max_candidates:
            break
    return chosen


@torch.no_grad()
def _predict_points(
    model: torch.nn.Module,
    x: torch.Tensor,
    sensor_presence: torch.Tensor | None,
    samplerate: int,
    device: torch.device,
    min_prob: float = 0.1,
    pressure_signal: np.ndarray | None = None,
    travel_signal: np.ndarray | None = None,
    ignore_start_sec: float = 1.0,
    ignore_end_sec: float = 0.5,
    event_window_before_sec: float = 2.0,
    event_window_after_sec: float = 20.0,
    min_gap_sec: float = 0.05,
    sp_min_after_bp_sec: float = 0.05,
    sp_pressure_prior_weight: float = 0.0,
    sp_pressure_prior_window_sec: float = 0.75,
    max_peaks_per_channel: int = 80,
) -> tuple[dict[str, dict], np.ndarray]:
    if sensor_presence is None:
        logits = model(x.to(device))
    else:
        logits = model(
            x.to(device),
            sensor_presence=sensor_presence.to(device),
        )
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    n_samples = int(probs.shape[1])
    search_start = max(0, int(round(ignore_start_sec * samplerate)))
    search_end = min(n_samples, n_samples - int(round(ignore_end_sec * samplerate)))
    event_center = None

    if search_end <= search_start:
        search_start, search_end = 0, n_samples

    selected: list[tuple[int, float, float] | None] = [None, None, None]
    min_gap_samples = max(1, int(round(min_gap_sec * samplerate)))
    sp_min_after_bp_samples = max(min_gap_samples, int(round(sp_min_after_bp_sec * samplerate)))

    def _thresholded_argmax(values: np.ndarray, start: int, end: int) -> tuple[int, float, float] | None:
        window = values[start:end]
        if window.size == 0:
            return None

        local_index = int(np.argmax(window))
        confidence = float(window[local_index])
        if confidence < min_prob:
            return None

        index = start + local_index
        return index, confidence, confidence

    prp_values = probs[0]
    prp_candidate = _thresholded_argmax(prp_values, search_start, search_end)
    if prp_candidate is not None:
        prp_index, _prp_confidence, _prp_score = prp_candidate
        selected[0] = prp_candidate
        bp_start = min(search_end, prp_index + min_gap_samples)
        bp_values = probs[1]
        bp_candidate = _thresholded_argmax(bp_values, bp_start, search_end)
        if bp_candidate is not None:
            bp_index, _bp_confidence, _bp_score = bp_candidate
            selected[1] = bp_candidate
            sp_start = min(search_end, bp_index + sp_min_after_bp_samples)
            sp_values = probs[2]
            sp_candidate = _thresholded_argmax(sp_values, sp_start, search_end)
            if sp_candidate is not None:
                selected[2] = sp_candidate

    predictions = {}
    for channel_index, name in enumerate(POINT_NAMES):
        candidate = selected[channel_index]
        if candidate is None:
            search_values = probs[channel_index, search_start:search_end]
            max_confidence = (
                float(np.max(search_values))
                if search_values.size > 0
                else float(np.max(probs[channel_index]))
            )
            predictions[name] = {
                "index": None,
                "time_sec": None,
                "confidence": max_confidence,
                "detected": False,
                "search_start_sec": search_start / float(samplerate),
                "search_end_sec": search_end / float(samplerate),
                "event_center_sec": (
                    event_center / float(samplerate) if event_center is not None else None
                ),
                "selection_rule": "no_valid_candidate",
            }
            continue

        index, confidence, _score = candidate
        predictions[name] = {
            "index": index,
            "time_sec": index / float(samplerate),
            "confidence": confidence,
            "detected": True,
            "search_start_sec": search_start / float(samplerate),
            "search_end_sec": search_end / float(samplerate),
            "event_center_sec": (
                event_center / float(samplerate) if event_center is not None else None
            ),
            "selection_rule": "windowed_argmax",
        }
    return predictions, probs


def _heuristic_points(processed_by_sensor: dict, pressure_event) -> dict[str, dict]:
    signal = pressure_event.resampled_signal
    samplerate = pressure_event.samplerate
    travel_signal = processed_by_sensor["travel"].resampled_signal[:pressure_event.n_samples]
    strain_signal = processed_by_sensor["strain"].resampled_signal[:pressure_event.n_samples]

    multivariat_config = MultivariatConfig()
    prp_result = detect_PRP_multivariat(
        pressure_signal=signal,
        samplerate=samplerate,
        config=multivariat_config,
    )
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
    return {
        "PRP": {"index": prp_result.prp_index, "reason": prp_result.reason},
        "BP": {"index": bp_result.bp_index, "reason": bp_result.reason},
        "SP": {"index": sp_result.sp_index, "reason": sp_result.reason},
    }


def _position_mae_by_point(
    predictions: dict[str, dict],
    heuristic_points: dict[str, dict],
) -> dict[str, float | None]:
    errors: dict[str, float | None] = {}

    for point_name in POINT_NAMES:
        pred_index = predictions[point_name]["index"]
        true_index = heuristic_points[point_name]["index"]

        if true_index is None or pred_index is None:
            errors[point_name] = None
            continue

        result = event_position_mae(
            true_indices=[true_index],
            pred_indices=[pred_index],
        )
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

    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=False)

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
        raw_sensor = raw_by_sensor[sensor_name]
        raw_sensor_signal = np.asarray(raw_sensor.signal, dtype=np.float32)
        raw_sensor_time = np.arange(raw_sensor.n_samples, dtype=np.float32) / float(raw_sensor.samplerate)
        axes[1].plot(
            raw_sensor_time,
            _normalize_for_plot(raw_sensor_signal),
            color=SENSOR_COLORS[sensor_name],
            linewidth=1.0,
            alpha=0.8,
            label=f"raw {sensor_name} ({raw_sensor.samplerate} Hz)",
        )
    axes[1].set_ylabel("raw z-score")
    axes[1].set_title("raw pressure, strain, travel")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    for sensor_name in SENSOR_NAMES:
        signal = processed_by_sensor[sensor_name].resampled_signal[:pressure_event.n_samples]
        axes[2].plot(
            time_axis,
            _normalize_for_plot(signal),
            color=SENSOR_COLORS[sensor_name],
            linewidth=1.0,
            alpha=0.8,
            label=sensor_name,
        )
    axes[2].set_ylabel("z-score")
    axes[2].set_title(f"{run_dir.name} | normalized resampled sensors")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    for point_name, color in POINT_COLORS.items():
        pred_index = predictions[point_name]["index"]
        if pred_index is not None:
            axes[2].axvline(
                pred_index / float(samplerate),
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"pred {point_name}",
            )
        heuristic_index = heuristic_points[point_name]["index"]
        if heuristic_index is not None:
            axes[2].axvline(
                heuristic_index / float(samplerate),
                color=color,
                linestyle=":",
                linewidth=1.4,
                label=f"heuristic {point_name}",
            )

    first_prediction = predictions[POINT_NAMES[0]]
    search_start_sec = first_prediction.get("search_start_sec")
    search_end_sec = first_prediction.get("search_end_sec")
    event_center_sec = first_prediction.get("event_center_sec")
    for ax in axes[2:]:
        if search_start_sec is not None and search_end_sec is not None:
            ax.axvspan(
                search_start_sec,
                search_end_sec,
                color="gray",
                alpha=0.08,
                label="model search window",
            )
        if event_center_sec is not None:
            ax.axvline(
                event_center_sec,
                color="gray",
                linestyle="-.",
                alpha=0.65,
                label="pressure event center",
            )

    for channel_index, point_name in enumerate(POINT_NAMES, start=3):
        axes[channel_index].plot(
            time_axis,
            probs[channel_index - 3],
            color=POINT_COLORS[point_name],
            linewidth=1.0,
            label=f"{point_name} probability",
        )
        pred_index = predictions[point_name]["index"]
        if pred_index is not None:
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
    started_at = perf_counter()
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _latest_run_dir()
    pressure_path = _resolve_pressure_path(args).resolve()

    print("Run dir:", run_dir)
    print("Pressure:", pressure_path)

    if args.pressure_only_fill_missing:
        print("Strain:   zeros (--pressure-only-fill-missing)")
        print("Travel:   zeros (--pressure-only-fill-missing)")
        raw_by_sensor, processed_by_sensor = _load_pressure_only_group(pressure_path)
    else:
        group = _sensor_group_from_pressure_path(dataset_root, pressure_path)
        print("Strain:  ", group["strain"])
        print("Travel:  ", group["travel"])
        raw_by_sensor, processed_by_sensor = _load_processed_group(group)
    print(f"Timing load/preprocess: {perf_counter() - started_at:.2f}s", flush=True)

    x, sensor_presence, pressure_event, _ = _prepare_model_input(processed_by_sensor)
    print(f"Timing prepare input: {perf_counter() - started_at:.2f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(run_dir, device)
    print(f"Timing load model: {perf_counter() - started_at:.2f}s", flush=True)

    predictions, probs = _predict_points(
        model,
        x,
        sensor_presence,
        pressure_event.samplerate,
        device,
        min_prob=args.min_prob,
        pressure_signal=pressure_event.resampled_signal,
        travel_signal=processed_by_sensor["travel"].resampled_signal,
        ignore_start_sec=args.ignore_start_sec,
        ignore_end_sec=args.ignore_end_sec,
        event_window_before_sec=args.event_window_before_sec,
        event_window_after_sec=args.event_window_after_sec,
        min_gap_sec=args.min_gap_sec,
        sp_min_after_bp_sec=args.sp_min_after_bp_sec,
        sp_pressure_prior_weight=args.sp_pressure_prior_weight,
        sp_pressure_prior_window_sec=args.sp_pressure_prior_window_sec,
        max_peaks_per_channel=args.max_peaks_per_channel,
    )
    print(f"Timing predict: {perf_counter() - started_at:.2f}s", flush=True)

    heuristic = _heuristic_points(processed_by_sensor, pressure_event)
    print(f"Timing heuristic: {perf_counter() - started_at:.2f}s", flush=True)
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
        if prediction["time_sec"] is None:
            pred_text = f"pred=None conf_max={prediction['confidence']:.4f}"
        else:
            pred_text = (
                f"pred={prediction['index']} "
                f"t={prediction['time_sec']:.3f}s "
                f"conf={prediction['confidence']:.4f}"
            )

        heur_text = (
            f"heuristic={heuristic_index} t={heuristic_time:.3f}s"
            if heuristic_time is not None
            else "heuristic=None"
        )
        mae_position = position_mae[point_name]
        mae_text = f"mae_pos={mae_position:.1f}" if mae_position is not None else "mae_pos=None"
        print(f" {point_name}: {pred_text} | {heur_text} | {mae_text}")

    print(f"Timing total: {perf_counter() - started_at:.2f}s", flush=True)

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
