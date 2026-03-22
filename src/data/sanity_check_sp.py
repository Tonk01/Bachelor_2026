from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data.data_loader import load_event
from src.data.targets import SPConfig, detect_sp, find_active_window, moving_avg, ms_to_samples


def _interesting_window(
    signal: np.ndarray,
    smooth_signal: np.ndarray,
    samplerate: int,
    sp_index: int | None,
    smooth_n: int,
) -> tuple[int, int]:
    if sp_index is not None:
        pad_before = ms_to_samples(3000.0, samplerate)
        pad_after = ms_to_samples(1500.0, samplerate)
        start = max(0, sp_index - pad_before)
        end = min(signal.size, sp_index + pad_after)
        if end - start >= 10:
            return start, end

    start, end = find_active_window(
        signal.astype(np.float32),
        samplerate,
        smooth_n,
        active_window_ms=2500.0,
    )
    return start, max(start + 10, end)


def inspect_signal(signal: np.ndarray, samplerate: int = 400, title: str = "SP inspection") -> None:
    sp_cfg = SPConfig()
    sp_result = detect_sp(signal, samplerate=samplerate, config=sp_cfg)

    smooth_n = ms_to_samples(sp_cfg.smooth_ms, samplerate)
    x_smooth = moving_avg(signal.astype(np.float32), smooth_n)
    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)
    zoom_start, zoom_end = _interesting_window(
        signal,
        x_smooth,
        samplerate,
        sp_result.sp_index,
        smooth_n,
    )

    print("SP Result")
    print(f" sp_index   {sp_result.sp_index}")
    print(f" confidence {sp_result.confidence:.4f}")
    print(f" reason {sp_result.reason}")
    print()

    fig, (overview_ax, zoom_ax) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes = [
        (overview_ax, slice(None), "Full Series"),
        (zoom_ax, slice(zoom_start, zoom_end), "Interesting Window"),
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

        if sp_result.sp_index is not None:
            ax.axvline(
                sp_result.sp_index / float(samplerate),
                linestyle="--",
                color="tab:green",
                label=f"SP {sp_result.sp_index}",
            )

        ax.set_ylabel("pressure")
        ax.set_title(f"{title} | {subtitle}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    zoom_ax.set_xlabel("time (s)")

    fig.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/AHA/events/AHA_1003.json"),
        Path("src/data/raw/raw1/JSDP/events/JSDP_493.json"),
    ]

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No known event path found.")

    print("Attempt to load:", path.resolve())

    event = load_event(path)
    signal = np.asarray(event.signal, dtype=np.float32)
    title = (
        f"SP inspection | site={event.sitename} | tag={event.valvetag} "
        f"| event={event.eventid} | samplerate={event.samplerate}"
    )
    inspect_signal(signal, samplerate=event.samplerate, title=title)


if __name__ == "__main__":
    main()
