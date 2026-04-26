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

        # ---------------------------------------- || drops || ------------------------------------------------------
        #Path("src/data/raw/AHA/events/AHA_228.json"),   # -1
        #Path("src/data/raw/AHA/events/AHA_4037.json"),  # +1
        #Path("src/data/raw/AHA/events/AHA_18682.json"), # +1

        #Path("src/data/raw/GKR/events/GKR_43.json"),   # +1
        #Path("src/data/raw/GKR/events/GKR_821.json"),  # +1
        #Path("src/data/raw/GKR/events/GKR_2169.json"), # +1
        #Path("src/data/raw/GKR/events/GKR_6465.json"), # +1

        #Path("src/data/raw/GRA/events/GRA_121.json"), # +1
        #Path("src/data/raw/GRA/events/GRA_308.json"), # -1 look at closer
        #Path("src/data/raw/GRA/events/GRA_702.json"), # -1 look at closer
        #Path("src/data/raw/GRA/events/GRA_566.json"), # +1
        #Path("src/data/raw/GRA/events/GRA_861.json"), # +1
        
        #Path("src/data/raw/JCB/events/JCB_26848.json"), # +1
        #Path("src/data/raw/JCB/events/JCB_7354.json"),  # -1
        #Path("src/data/raw/JCB/events/JCB_11333.json"), # +1

        #Path("src/data/raw/JSDP/events/JSDP_542.json"),    # +1
        #Path("src/data/raw/JSDP/events/JSDP_773.json"),    # +1
        #Path("src/data/raw/JSDP/events/JSDP_1122.json"),   # +1
        #Path("src/data/raw/JSDP/events/JSDP_37954.json"),  # +1
        #Path("src/data/raw/JSDP/events/JSDP_315000.json"), # +1

        #Path("src/data/raw/JSP1/events/JSP1_7788.json"),  # +1
        #Path("src/data/raw/JSP1/events/JSP1_8225.json"),  # +1
        #Path("src/data/raw/JSP1/events/JSP1_9871.json"),  # +1
        #Path("src/data/raw/JSP1/events/JSP1_18000.json"), # +1
        #Path("src/data/raw/JSP1/events/JSP1_12342.json"), # interesting case  -1

        #Path("src/data/raw/JSP2/events/JSP2_4996.json"),  # +1
        #Path("src/data/raw/JSP2/events/JSP2_4968.json"),  # +1  bruk som FIGUR eksempel i rapport!
        #Path("src/data/raw/JSP2/events/JSP2_39340.json"), # -1

        #Path("src/data/raw/JSRP/events/JSRP_27570.json"), # look at closer  -1 
        #Path("src/data/raw/JSRP/events/JSRP_94094.json"), # look at closer  -1 
        #Path("src/data/raw/JSRP/events/JSRP_303736.json"), # +1
        #Path("src/data/raw/JSRP/events/JSRP_347251.json"), # +1

        #Path("src/data/raw/OSH/events/OSH_4694.json"), # -1  what? how?
        #Path("src/data/raw/OSH/events/OSH_4734.json"), # -1  hmmm
        #Path("src/data/raw/OSH/events/OSH_5144.json"), # +1
        #Path("src/data/raw/OSH/events/OSH_4486.json"), # -1 

        #Path("src/data/raw/OSS/events/OSS_7053.json"), # -1 
        #Path("src/data/raw/OSS/events/OSS_1924.json"), # -1

        #Path("src/data/raw/TROA/events/TROA_6244.json"), # +1
        #Path("src/data/raw/TROA/events/TROA_6248.json"), # +1
        #Path("src/data/raw/TROA/events/TROA_6711.json"), # +0.3


        # ---------------------------------------- || increases || ------------------------------------------------------
        #Path("src/data/raw/AHA/events/AHA_1445.json"),  # +1
        #Path("src/data/raw/AHA/events/AHA_2912.json"),  # low bump +1
        #Path("src/data/raw/AHA/events/AHA_5047.json"),  # +1

        #Path("src/data/raw/GKR/events/GKR_2168.json"),  # -1
        #Path("src/data/raw/GKR/events/GKR_6547.json"),  # -1
        #Path("src/data/raw/GKR/events/GKR_9406.json"),  # Ingen data. 0 Data. 0. 

        #Path("src/data/raw/GRA/events/GRA_41.json"),    # + 0.5
        #Path("src/data/raw/GRA/events/GRA_309.json"),   # +1
        #Path("src/data/raw/GRA/events/GRA_1184.json"),  # +1 
        #Path("src/data/raw/GRA/events/GRA_1901.json"),  # +1
        
        #Path("src/data/raw/JCB/events/JCB_11377.json"), # +1
        #Path("src/data/raw/JCB/events/JCB_12198.json"), # +0.5
        #Path("src/data/raw/JCB/events/JCB_80553.json"), # +1

        #Path("src/data/raw/JSDP/events/JSDP_693.json"),   # +1
        #Path("src/data/raw/JSDP/events/JSDP_3202.json"),  # +1
        #Path("src/data/raw/JSDP/events/JSDP_19843.json"), # +1

        #Path("src/data/raw/JSP1/events/JSP1_6958.json"),   # +1
        #Path("src/data/raw/JSP1/events/JSP1_9636.json"),   # +1
        #Path("src/data/raw/JSP1/events/JSP1_10054.json"),  # +1
        #Path("src/data/raw/JSP1/events/JSP1_218593.json"), # +1

        #Path("src/data/raw/JSP2/events/JSP2_4951.json"),  # +1
        #Path("src/data/raw/JSP2/events/JSP2_5108.json"),  # +0.5
        #Path("src/data/raw/JSP2/events/JSP2_9657.json"),  # +0.5
        #Path("src/data/raw/JSP2/events/JSP2_10157.json"), # +0.5

        #Path("src/data/raw/JSRP/events/JSRP_34479.json"),  # +1
        #Path("src/data/raw/JSRP/events/JSRP_2256.json"),   # -1
        #Path("src/data/raw/JSRP/events/JSRP_122062.json"), # +1

        #Path("src/data/raw/OSH/events/OSH_3489.json"),  # -1
        #Path("src/data/raw/OSH/events/OSH_5970.json"),  # +1
        #Path("src/data/raw/OSH/events/OSH_6261.json"),  # +1

        #Path("src/data/raw/TROA/events/TROA_7021.json"),  # -1
        #Path("src/data/raw/TROA/events/TROA_12310.json"), # +1

        #Path("src/data/raw/JSP2/events/JSP2_28516.json"), # +1
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