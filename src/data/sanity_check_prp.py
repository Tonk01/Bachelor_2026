from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_loader import load_event
from src.data.targets import detect_prp, moving_avg, PRPConfig

DEFAULT_ROOTS = [
    Path("src/data/raw/raw1"),
    Path("src/data/raw/raw2"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]


def inspect_signal(signal: np.ndarray, config: PRPConfig | None = None) -> None:
    cfg = config or PRPConfig()
    result = detect_prp(signal, cfg)

    x_smooth = moving_avg(signal.astype(np.float32), cfg.smooth_samples)

    print("PRP Result")
    print(f" start_index : {result.start_index}")
    print(f" end_index   : {result.end_index}")
    print(f" prp_index   : {result.prp_index}")
    print(f" confidence  : {result.confidence:.4f}")
    print(f" reason      : {result.reason}")

    plt.figure(figsize=(14, 5))
    plt.plot(signal, label="raw pressure", alpha=0.6)
    plt.plot(x_smooth, label="smooth pressure", linewidth=2)

    if result.start_index is not None and result.end_index is not None:
        plt.axvline(result.start_index, result.end_index, alpha=0.2, label=f"PRP region [{result.start_index}, {result.end_index}]"),
    
    if result.prp_index is not None:
        plt.axvline(
            result.prp_index,
            linestyle="--",
            label=f"PRP midpoint {result.prp_index}",
        )

    plt.title("PRP inspection")
    plt.xlabel("sample index")
    plt.ylabel("pressure")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/JSP1/events/JSP1_10005.json"),
        #Path("src/data/raw/raw1/JSDP/events/JSDP_493.json"),
        Path("src/data/raw/raw1/AHA/events/AHA_17174.json"),
    ]

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No known event path found.")

    print("Attempt to load:", path.resolve())

    event = load_event(path)
    signal = np.asarray(event.signal, dtype=np.float32)

    cfg = PRPConfig(
    samplerate= 400,
    smooth_ms= 15.0,
    pre_window_ms = 40.0,
    post_window_ms = 40.0,
    noise_window_ms = 40.0,
    future_confirm_ms= 150.0,

    inner_region_ratio = 0.7,
    )

    inspect_signal(signal, config=cfg)


if __name__ == "__main__":
    main()
