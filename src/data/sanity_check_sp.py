from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_loader import load_event
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (
    PRPConfig,
    SPConfig,
    detect_prp,
    detect_sp,
    moving_avg,
    ms_to_samples,
)


DEFAULT_ROOTS = [
    Path("src/data/raw"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]


def inspect_signal(
    signal: np.ndarray,
    samplerate: int,
    prp_config: PRPConfig | None = None,
    sp_config: SPConfig | None = None,
    raw_signal: np.ndarray | None = None,
    raw_samplerate: int | None = None,
) -> None:
    prp_cfg = prp_config or PRPConfig(samplerate=samplerate)
    sp_cfg = sp_config or SPConfig()

    prp_result = detect_prp(signal, prp_cfg)

    sp_result = detect_sp(
        signal=signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=sp_cfg,
    )

    smooth_window = ms_to_samples(sp_cfg.smooth_ms, samplerate, minimum=3)
    x_smooth = moving_avg(signal.astype(np.float32), smooth_window)
    d1 = np.gradient(x_smooth).astype(np.float32)

    global_smooth_window = ms_to_samples(180.0, samplerate, minimum=5)
    d1_global = moving_avg(d1, global_smooth_window).astype(np.float32)


    time_axis = np.arange(signal.size, dtype=np.float32) / float(samplerate)

    print("PRP Result")
    print(f" start_index : {prp_result.start_index}")
    print(f" end_index   : {prp_result.end_index}")
    print(f" prp_index   : {prp_result.prp_index}")
    print(f" confidence  : {prp_result.confidence:.4f}")
    print(f" reason      : {prp_result.reason}")
    print()

    print("SP Result")
    print(f" start_index : {sp_result.start_index}")
    print(f" end_index   : {sp_result.end_index}")
    print(f" sp_index    : {sp_result.sp_index}")
    print(f" confidence  : {sp_result.confidence:.4f}")
    print(f" reason      : {sp_result.reason}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ----- Panel 1: raw / resampled signal -----
    if raw_signal is not None and raw_samplerate is not None:
        raw_time_axis = np.arange(raw_signal.size, dtype=np.float32) / float(raw_samplerate)
        axes[0].plot(raw_time_axis, raw_signal, label="raw", alpha=0.35, color="gray")

    axes[0].plot(time_axis, signal, label="resampled", alpha=0.8)

    if prp_result.prp_index is not None:
        axes[0].axvline(
            prp_result.prp_index / float(samplerate),
            linestyle="--",
            label=f"PRP {prp_result.prp_index}",
        )

    if sp_result.start_index is not None and sp_result.end_index is not None:
        axes[0].axvspan(
            sp_result.start_index / float(samplerate),
            sp_result.end_index / float(samplerate),
            alpha=0.2,
            color="green",
            label=f"SP region [{sp_result.start_index}, {sp_result.end_index}]",
        )

    if sp_result.sp_index is not None:
        axes[0].axvline(
            sp_result.sp_index / float(samplerate),
            linestyle="--",
            color="green",
            label=f"SP {sp_result.sp_index}",
        )

    axes[0].set_ylabel("pressure")
    axes[0].legend()
    axes[0].set_title("Raw and resampled signal")

    # ----- Panel 2: d1 -----
    axes[1].plot(time_axis, d1, label="d1")
    axes[1].axhline(0.0, linestyle="--", alpha=0.5, label="zero")

    if prp_result.prp_index is not None:
        axes[1].axvline(
            prp_result.prp_index / float(samplerate),
            linestyle="--",
            label="PRP",
        )

    if sp_result.sp_index is not None:
        axes[1].axvline(
            sp_result.sp_index / float(samplerate),
            linestyle="--",
            color="green",
            label="SP",
        )

    if sp_result.start_index is not None and sp_result.end_index is not None:
        axes[1].axvspan(
            sp_result.start_index / float(samplerate),
            sp_result.end_index / float(samplerate),
            alpha=0.2,
            color="green",
            label="SP region",
        )

    axes[1].set_ylabel("d1")
    axes[1].legend()
    axes[1].set_title("First derivative")

    # ----- Panel 3: smoothed global d1 -----
    axes[2].plot(
        time_axis,
        d1_global,
        label=f"d1 global smoothed ({global_smooth_window} samples)",
    )
    axes[2].axhline(0.0, linestyle="--", alpha=0.5, label="zero")

    if prp_result.prp_index is not None:
        axes[2].axvline(
            prp_result.prp_index / float(samplerate),
            linestyle="--",
            color="black",
            label="PRP",
        )

    if sp_result.sp_index is not None:
        axes[2].axvline(
            sp_result.sp_index / float(samplerate),
            linestyle="--",
            color="green",
            label="SP",
        )

    if sp_result.start_index is not None and sp_result.end_index is not None:
        axes[2].axvspan(
            sp_result.start_index / float(samplerate),
            sp_result.end_index / float(samplerate),
            alpha=0.2,
            color="green",
            label="SP region",
        )

    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("d1 global")
    axes[2].legend()
    axes[2].set_title("First derivative - global smoothed view")

    plt.tight_layout()
    plt.show()


def main() -> None:
    candidate_paths = [
        Path("/mnt/d/Bachelor_data/data/raw_1sensor/AHA/events/AHA_17174.json"),

        # -- drops --
        #Path("src/data/raw/JSDP/events/JSDP_542.json"),
        #Path("src/data/raw/AHA/events/AHA_228.json"),
        #Path("src/data/raw/AHA/events/AHA_18682.json"),
        #Path("src/data/raw/JSDP/events/JSDP_315000.json"),
        #Path("src/data/raw/JCB/events/JCB_26848.json"),


        # -- increases --
        #Path("src/data/raw/GRA/events/GRA_2207.json"),
        #Path("src/data/raw/JSP1/events/JSP1_6233.json"),
        #Path("src/data/raw/JSP1/events/JSP1_6647.json"),
        Path("src/data/raw/JSRP/events/JSRP_870.json"),
        #Path("src/data/raw/AHA/events/AHA_699.json"),
        #Path("src/data/raw/TROA/events/TROA_6286.json"),


    ]

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("No known event path found.")

    print("Attempt to load:", path.resolve())

    event = load_event(path)

    processor = EventProcessor(
        PreprocessingConfig(
            allowed_samplerate=(200, 400, 800, 1000),
            target_samplerate=400,
            normalize_for_model=False,
        )
    )

    processed_event = processor.preprocess_event(event)
    signal = processed_event.resampled_signal

    prp_cfg = PRPConfig(
        samplerate=processed_event.samplerate,
        smooth_ms=15.0,
    )

    sp_cfg = SPConfig(
        smooth_ms=15.0,
    )

    inspect_signal(
        signal=signal,
        samplerate=processed_event.samplerate,
        prp_config=prp_cfg,
        sp_config=sp_cfg,
        raw_signal=np.asarray(event.signal, dtype=np.float32),
        raw_samplerate=event.samplerate,
    )


if __name__ == "__main__":
    main()