from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.data_loader import load_event
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.data.targets import (BPConfig, PRPConfig, SPConfig, detect_bp, detect_prp, detect_sp, moving_avg, ms_to_samples)


def inspect_signal(
    signal: np.ndarray,
    samplerate: int,
    raw_signal: np.ndarray | None = None,
    raw_samplerate: int | None = None,
    title: str = "Critical point sanity check",
) -> None:
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
    sp_cfg = SPConfig(smooth_ms=15.0)

    prp_result = detect_prp(signal, prp_cfg)

    bp_result = detect_bp(
        signal=signal,
        samplerate=samplerate,
        prp_index=prp_result.prp_index,
        config=bp_cfg,
    )

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

    print("BP Result")
    print(f" start_index : {bp_result.start_index}")
    print(f" end_index   : {bp_result.end_index}")
    print(f" bp_index    : {bp_result.bp_index}")
    print(f" confidence  : {bp_result.confidence:.4f}")

    print("SP Result")
    print(f" start_index : {sp_result.start_index}")
    print(f" end_index   : {sp_result.end_index}")
    print(f" sp_index    : {sp_result.sp_index}")
    print(f" confidence  : {sp_result.confidence:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    if raw_signal is not None and raw_samplerate is not None:
        raw_time = np.arange(raw_signal.size, dtype=np.float32) / float(raw_samplerate)
        axes[0].plot(raw_time, raw_signal, label="raw", alpha=0.35, color="gray")

    #axes[0].plot(time_axis, signal, label="resampled", alpha=0.85)

    marker_specs = [
        #("PRP", prp_result.prp_index, "tab:blue"),
        #("BP", bp_result.bp_index, "tab:red"),
        #("SP", sp_result.sp_index, "tab:green"),
    ]

    for label, index, color in marker_specs:
        if index is not None:
            axes[0].axvline(
                index / float(samplerate),
                linestyle="--",
                color=color,
                label=f"{label} {index}",
            )

    #if bp_result.start_index is not None and bp_result.end_index is not None:
        #axes[0].axvspan(
            #bp_result.start_index / float(samplerate),
            #bp_result.end_index / float(samplerate),
            #alpha=0.15,
            #color="red",
            #label="BP region",
        #)

    #if sp_result.start_index is not None and sp_result.end_index is not None:
        #axes[0].axvspan(
            #sp_result.start_index / float(samplerate),
            #sp_result.end_index / float(samplerate),
            #alpha=0.15,
            #color="green",
            #label="SP region",
        #)

    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("pressure")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, d1, label="d1")
    axes[1].axhline(0.0, linestyle="--", alpha=0.5, label="zero")

    for label, index, color in marker_specs:
        if index is not None:
            axes[1].axvline(
                index / float(samplerate),
                linestyle="--",
                color=color,
                label=label,
            )

    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("d1")
    #axes[1].set_title("First derivative")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main() -> None:
    candidate_paths = [

       # ---------------------------------------- || drops || ------------------------------------------------------

        #Path("src/data/raw/AHA/events/AHA_150.json"),   # -1
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

        Path("src/data/raw/JCB/events/JCB_108172.json"),  # +1
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

        #Path("src/data/raw/JSP1/events/JSP1_10599.json"),
        #Path("src/data/raw/JSP1/events/JSP1_236649.json"),
        #Path("src/data/raw/JSP1/events/JSP1_53328.json"),
        #Path("src/data/raw/JSP2/events/JSP2_74355.json"),
        #Path("src/data/raw/AHA/events/AHA_23478.json"),
        
        #JSP1_10599, JSP1_236649, JSP1_53328, JSP2_74355, AHA_23478


    ]

    existing_paths = [candidate for candidate in candidate_paths if candidate.exists()]

    if not existing_paths:
        raise FileNotFoundError("No known event paths found.")

    print(f"Found {len(existing_paths)} existing candidate paths.")

    processor = EventProcessor(
        PreprocessingConfig(
            allowed_samplerate=(200, 400, 800, 1000),
            target_samplerate=400,
            normalize_for_model=False,
        )
    )

    for i, path in enumerate(existing_paths, start=1):
        print(f"\n[{i}/{len(existing_paths)}] Attempt to load: {path.resolve()}")

        event = load_event(path)
        processed_event = processor.preprocess_event(event)
        signal = processed_event.resampled_signal

        title = (
            f"[{i}/{len(existing_paths)}] Critical point inspection | "
            f"site={event.sitename} | "
            f"tag={event.valvetag} | event={event.eventid} | "
            f"samplerate={processed_event.samplerate}"
        )

        inspect_signal(
            signal=signal,
            samplerate=processed_event.samplerate,
            raw_signal=np.asarray(event.signal, dtype=np.float32),
            raw_samplerate=event.samplerate,
            title=title,
        )


if __name__ == "__main__":
    main()