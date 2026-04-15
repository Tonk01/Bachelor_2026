from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_loader import load_event
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import detect_prp, moving_avg, PRPConfig

DEFAULT_ROOTS = [
    Path("src/data/raw"),
    Path("src/data/raw"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]


def inspect_signal(signal: np.ndarray, samplerate: int, config: PRPConfig | None = None) -> None:
    cfg = config or PRPConfig(samplerate=samplerate)
    result = detect_prp(signal, cfg)

    x_smooth = moving_avg(signal.astype(np.float32), cfg.smooth_samples)
    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)

    print("PRP Result")
    print(f" start_index : {result.start_index}")
    print(f" end_index   : {result.end_index}")
    print(f" prp_index   : {result.prp_index}")
    print(f" confidence  : {result.confidence:.4f}")
    print(f" reason      : {result.reason}")

    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, signal, label="resampled pressure", alpha=0.6)
    #plt.plot(time_axis, x_smooth, label="smooth pressure", linewidth=2)

    if result.start_index is not None and result.end_index is not None:
        plt.axvspan(
            result.start_index / float(samplerate), 
            result.end_index / float(samplerate), 
            alpha=0.2, 
            label=f"PRP region [{result.start_index}, {result.end_index}]"),
    
    if result.prp_index is not None:
        plt.axvline(
            result.prp_index / float(samplerate),
            linestyle="--",
            label=f"PRP midpoint {result.prp_index}",
        )

    plt.title("PRP inspection")
    plt.xlabel("time (s)")
    plt.ylabel("pressure")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/AHA/events/AHA_17174.json"),
        Path("src/data/raw/JSDP/events/JSDP_542.json"),
        #Path("src/data/raw/AHA/events/AHA_228.json"),
        #Path("src\\data\\raw\\AHA\\events\\AHA_18682.json"),
        #Path("src/data/raw/JSDP/events/JSDP_315000.json")
    ]

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No known event path found.")

    print("Attempt to load:", path.resolve())

    event = load_event(path)

    processor = EventProcessor(
        PreprocessingConfig(
            allowed_samplerate=(400, 1000),
            target_samplerate=400,
            normalize_for_model=True,
        )
    )

    processed_event = processor.preprocess_event(event)
    signal = processed_event.resampled_signal

    cfg = PRPConfig(
    samplerate=processed_event.samplerate,
    smooth_ms= 15.0,
    pre_window_ms = 40.0,
    post_window_ms = 40.0,
    noise_window_ms = 40.0,
    future_confirm_ms= 150.0,
    inner_region_ratio = 0.7,
    )

    inspect_signal(signal=signal, samplerate=processed_event.samplerate, config=cfg)


if __name__ == "__main__":
    main()
