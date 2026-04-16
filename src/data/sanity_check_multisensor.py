from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.signal import resample_poly

from .targets import BPConfig, PRPConfig, SPConfig, detect_bp, detect_prp, detect_sp, moving_avg


DEFAULT_PRESSURE_ROOTS = [
    Path("data/raw"),
    Path("src/data/raw/raw1"),
    Path("src/data/raw/raw2"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]
DEFAULT_TWO_SENSOR_ROOTS = [
    Path("data/raw"),
    Path("/mnt/d/Bachelor_data/data/raw_2SENSOR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one multisensor event with pressure-derived PRP/BP/SP markers "
            "shown on pressure, strain, and travel side by side."
        ),
    )
    parser.add_argument("--path", help="Path to one pressure event JSON file.")
    parser.add_argument("--eventid", type=int, help="Event ID to search for.")
    parser.add_argument("--site", help="Optional sitename to narrow eventid search.")
    parser.add_argument(
        "--pressure-root",
        action="append",
        default=[],
        help="Pressure data root or pressure sensor directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--strain-root",
        action="append",
        default=[],
        help="Strain data root or strain sensor directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--travel-root",
        action="append",
        default=[],
        help="Travel data root or travel sensor directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--save",
        help="Optional output path for the figure. If omitted, the plot is shown interactively.",
    )
    return parser.parse_args()


def _existing_roots(extra_roots: list[str], default_roots: list[Path]) -> list[Path]:
    roots = [Path(root) for root in extra_roots] if extra_roots else default_roots
    return [root for root in roots if root.exists()]


def _find_event_path(eventid: int, roots: list[Path], site: str | None = None) -> Path:
    for root in roots:
        for path in root.rglob("*.json"):
            if site and not path.name.startswith(f"{site}_"):
                continue
            if path.stem.endswith(f"_{eventid}"):
                return path
    raise FileNotFoundError(f"Could not find event {eventid} in roots: {roots}")


def _resolve_pressure_event_path(args: argparse.Namespace) -> Path:
    if args.path:
        return Path(args.path)

    if args.eventid is None:
        raise ValueError("Provide either --path or --eventid.")

    roots = _existing_roots(args.pressure_root, DEFAULT_PRESSURE_ROOTS)
    if not roots:
        raise FileNotFoundError("No pressure data roots found.")
    return _find_event_path(args.eventid, roots, site=args.site)


def _infer_site_from_pressure_path(path: Path) -> str | None:
    stem_parts = path.stem.split("_", 1)
    if stem_parts:
        return stem_parts[0]
    return None


def _candidate_sensor_paths(
    *,
    roots: list[Path],
    sensor_name: str,
    site: str,
    filename: str,
) -> list[Path]:
    candidates: list[Path] = []

    for root in roots:
        candidates.extend(
            [
                root / filename,
                root / sensor_name / filename,
                root / "events" / sensor_name / filename,
                root / site / "events" / sensor_name / filename,
                root / site / "events" / filename,
            ]
        )

    # Preserve order while removing duplicates.
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        normalized = path.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(path)
    return unique_candidates


def _resolve_sensor_file(
    *,
    sensor_name: str,
    pressure_path: Path,
    roots: list[Path],
    site: str,
) -> Path | None:
    filename = pressure_path.name
    for candidate in _candidate_sensor_paths(
        roots=roots,
        sensor_name=sensor_name,
        site=site,
        filename=filename,
    ):
        if candidate.exists():
            return candidate

    return None


def _load_sensor_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    signal = [float(x) for x in data["eventdata"]]
    if not signal:
        raise ValueError(f"Empty eventdata in {path}")

    samplerate = int(data["samplerate"])
    if samplerate <= 0:
        raise ValueError(f"Invalid samplerate in {path}")

    return {
        "eventid": int(data["eventid"]),
        "valvetag": str(data["valvetag"]),
        "sitename": str(data["sitename"]),
        "samplerate": samplerate,
        "signal": signal,
        "path": str(path),
    }


def _build_multisensor_event(
    pressure_path: Path,
    strain_path: Path | None,
    travel_path: Path | None,
) -> dict:
    payloads = {"pressure": _load_sensor_payload(pressure_path)}
    if strain_path is not None:
        payloads["strain"] = _load_sensor_payload(strain_path)
    if travel_path is not None:
        payloads["travel"] = _load_sensor_payload(travel_path)

    anchor = payloads["pressure"]
    for sensor_name, payload in payloads.items():
        for key in ("eventid", "valvetag", "sitename"):
            if payload[key] != anchor[key]:
                raise ValueError(
                    f"Mismatched {key} between pressure and {sensor_name} for event {pressure_path.name}"
                )

    return {
        "eventid": anchor["eventid"],
        "valvetag": anchor["valvetag"],
        "sitename": anchor["sitename"],
        "path": str(pressure_path),
        "samplerates": {name: int(payload["samplerate"]) for name, payload in payloads.items()},
        "signals": {name: list(payload["signal"]) for name, payload in payloads.items()},
    }


def _validate_signal(signal: list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size == 0:
        raise ValueError("signal is empty")
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or INF")
    return x


def _resample_signal(signal: list[float] | np.ndarray, from_samplerate: int, to_samplerate: int) -> np.ndarray:
    x = _validate_signal(signal)
    if from_samplerate <= 0 or to_samplerate <= 0:
        raise ValueError("samplerates must be above 0")
    if from_samplerate == to_samplerate:
        return x.copy()
    y = resample_poly(x, up=to_samplerate, down=from_samplerate).astype(np.float32)
    if y.size == 0:
        raise ValueError("resampled signal is empty")
    return y


def _build_signal_views(
    signal: np.ndarray,
    samplerate: int,
    *,
    smooth_before_derivative: bool = False,
    smooth_window_samples: int = 1,
) -> dict[str, np.ndarray]:
    dt = 1.0 / float(samplerate)
    derivative_source = signal.astype(np.float32)
    if smooth_before_derivative:
        derivative_source = moving_avg(derivative_source, max(1, smooth_window_samples))

    first = np.gradient(derivative_source, dt).astype(np.float32) if signal.size > 1 else np.zeros(signal.size, dtype=np.float32)
    second = np.gradient(first, dt).astype(np.float32) if first.size > 1 else np.zeros(first.size, dtype=np.float32)
    return {
        "raw": signal.astype(np.float32),
        "first_derivative": first,
        "second_derivative": second,
    }


def _attach_marker_toggle_button(fig: plt.Figure, marker_lines: list) -> None:
    if not marker_lines:
        return

    button_ax = fig.add_axes([0.78, 0.01, 0.19, 0.05])
    button = Button(button_ax, "Toggle PRP/BP/SP")

    def _toggle(_event) -> None:
        make_visible = not marker_lines[0].get_visible()
        for line in marker_lines:
            line.set_visible(make_visible)
        fig.canvas.draw_idle()

    button.on_clicked(_toggle)
    # Keep widget references alive for the lifetime of the figure.
    fig._marker_toggle_button_ax = button_ax
    fig._marker_toggle_button = button


def _attach_strain_smoothing_toggle_button(
    fig: plt.Figure,
    strain_axes: dict[str, plt.Axes],
    strain_signal: np.ndarray,
    samplerate: int,
    time_axis: np.ndarray,
    smooth_window_samples: int,
) -> None:
    if not strain_axes:
        return

    button_ax = fig.add_axes([0.56, 0.01, 0.19, 0.05])
    button = Button(button_ax, "Toggle Strain Smooth")

    smoothed_views = _build_signal_views(
        strain_signal,
        samplerate,
        smooth_before_derivative=True,
        smooth_window_samples=smooth_window_samples,
    )
    raw_views = _build_signal_views(
        strain_signal,
        samplerate,
        smooth_before_derivative=False,
        smooth_window_samples=smooth_window_samples,
    )

    state = {"smoothed": True}

    def _toggle(_event) -> None:
        state["smoothed"] = not state["smoothed"]
        current_views = smoothed_views if state["smoothed"] else raw_views

        for view_key, ax in strain_axes.items():
            line = ax.lines[0]
            line.set_ydata(current_views[view_key])
            if view_key == "raw":
                line.set_label("strain raw")
            elif state["smoothed"]:
                line.set_label(f"strain {view_key.replace('_', ' ')} (smoothed first)")
            else:
                line.set_label(f"strain {view_key.replace('_', ' ')}")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.legend(loc="upper right")

        fig.canvas.draw_idle()

    button.on_clicked(_toggle)
    fig._strain_toggle_button_ax = button_ax
    fig._strain_toggle_button = button


def inspect_multisensor_event(args: argparse.Namespace) -> None:
    pressure_path = _resolve_pressure_event_path(args)
    site = args.site or _infer_site_from_pressure_path(pressure_path)
    if not site:
        raise ValueError(f"Could not infer site from pressure path: {pressure_path}")

    strain_roots = _existing_roots(args.strain_root, DEFAULT_TWO_SENSOR_ROOTS)
    travel_roots = _existing_roots(args.travel_root, DEFAULT_TWO_SENSOR_ROOTS)
    if not strain_roots:
        raise FileNotFoundError("No strain data roots found.")
    if not travel_roots:
        raise FileNotFoundError("No travel data roots found.")

    strain_path = _resolve_sensor_file(
        sensor_name="strain",
        pressure_path=pressure_path,
        roots=strain_roots,
        site=site,
    )
    travel_path = _resolve_sensor_file(
        sensor_name="travel",
        pressure_path=pressure_path,
        roots=travel_roots,
        site=site,
    )

    available_aux_sensors = [
        sensor_name
        for sensor_name, sensor_path in (("strain", strain_path), ("travel", travel_path))
        if sensor_path is not None
    ]
    if not available_aux_sensors:
        raise FileNotFoundError(
            f"Could not find either strain or travel matching {pressure_path.name}"
        )

    event = _build_multisensor_event(pressure_path, strain_path, travel_path)
    target_samplerate = 400
    resampled_signals = {
        name: _resample_signal(
            signal=signal,
            from_samplerate=event["samplerates"][name],
            to_samplerate=target_samplerate,
        )
        for name, signal in event["signals"].items()
    }
    n_samples = min(signal.size for signal in resampled_signals.values())
    resampled_signals = {
        name: signal[:n_samples].astype(np.float32)
        for name, signal in resampled_signals.items()
    }

    pressure_signal = resampled_signals["pressure"]
    samplerate = target_samplerate

    prp_cfg = PRPConfig(samplerate=samplerate)
    bp_cfg = BPConfig()
    sp_cfg = SPConfig()

    prp_result = detect_prp(pressure_signal, prp_cfg)
    bp_result = detect_bp(
        pressure_signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=bp_cfg,
    )
    sp_result = detect_sp(
        pressure_signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=sp_cfg,
    )

    print("Event")
    print(" path       ", event["path"])
    print(" pressure   ", pressure_path)
    print(" strain     ", strain_path if strain_path is not None else "missing")
    print(" travel     ", travel_path if travel_path is not None else "missing")
    print(" site       ", event["sitename"])
    print(" valvetag   ", event["valvetag"])
    print(" eventid    ", event["eventid"])
    print(" samplerate ", samplerate)
    print(" sensors    ", ", ".join(sorted(event["signals"])))
    print()
    print("Pressure-derived markers")
    print(" PRP", prp_result.prp_index, prp_result.reason)
    print(" BP ", bp_result.bp_index, bp_result.reason)
    print(" SP ", sp_result.sp_index, sp_result.reason)

    marker_specs = [
        ("PRP", prp_result.prp_index, "tab:blue"),
        ("BP", bp_result.bp_index, "tab:red"),
        ("SP", sp_result.sp_index, "tab:green"),
    ]

    time_axis = np.arange(n_samples, dtype=np.float32) / float(samplerate)
    view_specs = (
        ("raw", "Raw"),
        ("first_derivative", "1st Derivative"),
        ("second_derivative", "2nd Derivative"),
    )
    sensor_names = ("pressure", "strain", "travel")
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    marker_lines = []
    strain_axes: dict[str, plt.Axes] = {}

    for row_index, sensor_name in enumerate(sensor_names):
        signal = resampled_signals.get(sensor_name)

        if signal is None:
            for col_index, (_, column_title) in enumerate(view_specs):
                ax = axes[row_index, col_index]
                ax.text(
                    0.5,
                    0.5,
                    f"{sensor_name} missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                if row_index == 0:
                    ax.set_title(column_title)
                if col_index == 0:
                    ax.set_ylabel(sensor_name)
                ax.grid(True, alpha=0.3)
            continue

        signal_views = _build_signal_views(
            signal,
            samplerate,
            smooth_before_derivative=(sensor_name == "strain"),
            smooth_window_samples=prp_cfg.smooth_samples,
        )

        for col_index, (view_key, column_title) in enumerate(view_specs):
            ax = axes[row_index, col_index]
            line_label = f"{sensor_name} {column_title.lower()}"
            if sensor_name == "strain" and view_key != "raw":
                line_label = f"{sensor_name} {view_key.replace('_', ' ')} (smoothed first)"
            ax.plot(
                time_axis,
                signal_views[view_key],
                color="tab:purple",
                linewidth=1.0,
                alpha=0.9,
                label=line_label,
            )

            for label, index, color in marker_specs:
                if index is None:
                    continue
                marker_line = ax.axvline(
                    index / float(samplerate),
                    linestyle="--",
                    color=color,
                    linewidth=1.4,
                    label=f"{label} {index}",
                )
                marker_lines.append(marker_line)

            if row_index == 0:
                ax.set_title(column_title)
            if col_index == 0:
                ax.set_ylabel(sensor_name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            if sensor_name == "strain":
                strain_axes[view_key] = ax

    fig.suptitle(
        "Multisensor inspection with PRP/BP/SP derived from pressure\n"
        f"site={event['sitename']} | tag={event['valvetag']} | event={event['eventid']} | samplerate={samplerate}",
        y=0.99,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print("saved figure", output_path)
        plt.close(fig)
        return

    _attach_marker_toggle_button(fig, marker_lines)
    strain_signal = resampled_signals.get("strain")
    if strain_signal is not None:
        _attach_strain_smoothing_toggle_button(
            fig,
            strain_axes,
            strain_signal,
            samplerate,
            time_axis,
            prp_cfg.smooth_samples,
        )
    plt.show()


def main() -> None:
    args = parse_args()
    inspect_multisensor_event(args)


if __name__ == "__main__":
    main()
