from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data.data_loader import load_event
from src.data.targets import BPConfig, PRPconfig, detect_bp, detect_prp, moving_avg, ms_to_samples


def inspect_signal(signal: np.ndarray, samplerate: int = 400, title: str = "BP inspection") -> None:
    prp_cfg = PRPconfig(
        search_end_ratio=0.35,
        slope_sigma_mult=0.9,
        min_amplitude_sigma=1.0,
        min_rise_ms=8.0,
    )
    bp_cfg = BPConfig()

    prp_result = detect_prp(signal, samplerate=samplerate, config=prp_cfg)
    bp_result = detect_bp(
        signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=bp_cfg,
    )

    smooth_n = ms_to_samples(prp_cfg.smooth_ms, samplerate)
    x_smooth = moving_avg(signal.astype(np.float32), smooth_n)
    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)

    print("PRP Result")
    print(f" prp_index  {prp_result.prp_index}")
    print(f" confidence {prp_result.confidence:.4f}")
    print(f" reason {prp_result.reason}")
    print()
    print("BP Result")
    print(f" bp_index   {bp_result.bp_index}")
    print(f" confidence {bp_result.confidence:.4f}")
    print(f" reason {bp_result.reason}")
    print()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_axis, signal, color="0.6", linewidth=1.0, alpha=0.8, label="raw pressure")
    ax.plot(time_axis, x_smooth, color="tab:orange", linewidth=2.0, label="smooth pressure")
    ax.set_ylabel("pressure")
    ax.set_xlabel("time (s)")
    ax.set_title(title)

    marker_specs = [
        (prp_result.prp_index, "tab:orange", "PRP"),
        (bp_result.bp_index, "tab:red", "BP"),
    ]
    for index, color, label in marker_specs:
        if index is None:
            continue
        ax.axvline(index / float(samplerate), linestyle="--", color=color, label=f"{label} {index}")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/JSP1/events/JSP1_10005.json"),
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
