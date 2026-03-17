from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import iter_event_files, load_event
from .preprocess import EventProcessor


DEFAULT_ROOTS = [
    Path("src/data/raw/raw1"),
    Path("src/data/raw/raw2"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one event with pressure, first derivative, and second derivative.",
    )
    parser.add_argument("--path", help="Path to one event JSON file.")
    parser.add_argument("--eventid", type=int, help="Event ID to search for.")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Data root to search. Can be passed multiple times.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the raw event instead of preprocessing to 400 Hz first.",
    )
    parser.add_argument(
        "--save",
        help="Optional output path for the figure. If omitted, the plot is shown interactively.",
    )
    return parser.parse_args()


def existing_roots(extra_roots: list[str]) -> list[Path]:
    roots = [Path(root) for root in extra_roots] if extra_roots else DEFAULT_ROOTS
    return [root for root in roots if root.exists()]


def find_event_path(eventid: int, roots: list[Path]) -> Path:
    for path in iter_event_files(*roots):
        if path.stem.endswith(f"_{eventid}"):
            return path
    raise FileNotFoundError(f"Could not find event {eventid} in roots: {roots}")


def build_series(signal: np.ndarray, samplerate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.gradient(signal) if signal.size > 1 else np.zeros(signal.size, dtype=np.float32)
    second = np.gradient(first) if signal.size > 1 else np.zeros(signal.size, dtype=np.float32)
    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)
    return time_axis, first, second


def plot_event(args: argparse.Namespace) -> None:
    if not args.path and args.eventid is None:
        raise ValueError("Provide either --path or --eventid.")

    if args.path:
        event_path = Path(args.path)
    else:
        roots = existing_roots(args.root)
        if not roots:
            raise FileNotFoundError("No data roots found.")
        event_path = find_event_path(args.eventid, roots)

    event = load_event(event_path)

    if args.raw:
        signal = np.asarray(event.signal, dtype=np.float32)
        samplerate = event.samplerate
        source = "raw"
    else:
        processed = EventProcessor().preprocess_event(event)
        signal = processed.signal
        samplerate = processed.samplerate
        source = "preprocessed"

    time_axis, first, second = build_series(signal, samplerate)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(time_axis, signal, linewidth=1.0)
    axes[0].set_ylabel("Pressure")
    axes[0].set_title(
        f"{source} event {event.eventid} | site={event.sitename} | tag={event.valvetag} | samplerate={samplerate}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, first, linewidth=1.0, color="tab:orange")
    axes[1].set_ylabel("1st deriv.")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, second, linewidth=1.0, color="tab:green")
    axes[2].set_ylabel("2nd deriv.")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    print("event path:", event.path)
    print("source:", source)
    print("original samplerate:", event.samplerate)
    print("plotted samplerate:", samplerate)
    print("samples:", signal.size)
    print("duration_sec:", signal.size / float(samplerate))

    fig.tight_layout()

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print("saved figure:", output_path)
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    plot_event(parse_args())
