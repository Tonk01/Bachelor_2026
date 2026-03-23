from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data.data_loader import load_event
from src.data.targets import BPConfig, PRPConfig, detect_bp, detect_prp, moving_avg, ms_to_samples


def _interesting_window(
    signal: np.ndarray,
    smooth_signal: np.ndarray,
    samplerate: int,
    prp_index: int | None,
    bp_index: int | None,
) -> tuple[int, int]:
    marker_indices = [index for index in (prp_index, bp_index) if index is not None]
    if marker_indices:
        pad_before = ms_to_samples(1500.0, samplerate)
        pad_after = ms_to_samples(3000.0, samplerate)
        start = max(0, min(marker_indices) - pad_before)
        end = min(signal.size, max(marker_indices) + pad_after)
        if end - start >= 10:
            return start, end

    slope = np.abs(np.gradient(smooth_signal))
    peak_index = int(np.argmax(slope))
    pad = ms_to_samples(2500.0, samplerate)
    start = max(0, peak_index - pad)
    end = min(signal.size, peak_index + pad)
    return start, max(start + 10, end)


def inspect_signal(signal: np.ndarray, samplerate: int = 400, title: str = "BP inspection") -> None:
    prp_cfg = PRPConfig(
        samplerate=samplerate,
        smooth_ms=15.0,
        pre_window_ms=40.0,
        post_window_ms=40.0,
        noise_window_ms=40.0,
        future_confirm_ms=150.0,
        inner_region_ratio=0.7,
    )
    bp_cfg = BPConfig()

    prp_result = detect_prp(signal, prp_cfg)
    bp_result = detect_bp(
        signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=bp_cfg,
    )

    smooth_n = prp_cfg.smooth_samples
    x_smooth = moving_avg(signal.astype(np.float32), smooth_n)
    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)
    zoom_start, zoom_end = _interesting_window(
        signal,
        x_smooth,
        samplerate,
        prp_result.prp_index,
        bp_result.bp_index,
    )

    print("PRP Result")
    print(f" start_index {prp_result.start_index}")
    print(f" end_index   {prp_result.end_index}")
    print(f" prp_index  {prp_result.prp_index}")
    print(f" confidence {prp_result.confidence:.4f}")
    print(f" reason {prp_result.reason}")
    print()
    print("BP Result")
    print(f" start_index {bp_result.start_index}")
    print(f" end_index   {bp_result.end_index}")
    print(f" bp_index   {bp_result.bp_index}")
    print(f" confidence {bp_result.confidence:.4f}")
    print(f" reason {bp_result.reason}")
    print()

    fig, (overview_ax, zoom_ax) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes = [
        (overview_ax, slice(None), "Full Series"),
        (zoom_ax, slice(zoom_start, zoom_end), "Interesting Window"),
    ]

    marker_specs = [
        (prp_result.prp_index, "tab:blue", "PRP"),
        (bp_result.bp_index, "tab:red", "BP"),
    ]
    for ax, current_slice, subtitle in axes:
        ax.plot(
            time_axis[current_slice],
            signal[current_slice],
            color="0.6",
            linewidth=1.0,
            alpha=0.8,
            label="raw pressure",
        )
        ax.plot(
            time_axis[current_slice],
            x_smooth[current_slice],
            color="tab:orange",
            linewidth=2.0,
            label="smooth pressure",
        )

        if bp_result.start_index is not None and bp_result.end_index is not None:
            ax.axvline(
                bp_result.start_index / float(samplerate),
                color="tab:red",
                alpha=0.25,
                label=f"BP start {bp_result.start_index}",
            )
            ax.axvline(
                bp_result.end_index / float(samplerate),
                color="tab:red",
                alpha=0.25,
                label=f"BP end {bp_result.end_index}",
            )

        for index, color, label in marker_specs:
            if index is None:
                continue
            ax.axvline(index / float(samplerate), linestyle="--", color=color, label=f"{label} {index}")

        ax.set_ylabel("pressure")
        ax.set_title(f"{title} | {subtitle}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    zoom_ax.set_xlabel("time (s)")

    fig.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/AHA/events/AHA_17174.json"),
        Path("src/data/raw/raw1/JSDP/events/JSDP_493.json"),
    ]

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No known event path found.")

    print("Attempt to load:", path.resolve())

    event = load_event(path)
    signal = np.asarray(event.signal, dtype=np.float32)
    title = (
        f"BP inspection | site={event.sitename} | tag={event.valvetag} "
        f"| event={event.eventid} | samplerate={event.samplerate}"
    )
    inspect_signal(signal, samplerate=event.samplerate, title=title)


if __name__ == "__main__":
    main()
