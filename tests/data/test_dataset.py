import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from src.data.dataset import ValveDataset
from src.data.collate import valve_collate
from src.data.data_loader import load_event
from src.data.targets import detect_prp, moving_avg, detect_bp, detect_sp,build_bp_target, build_prp_target, build_sp_target


dataset = ValveDataset("src/data/raw/raw1/AHA/events")

loader = DataLoader(
    dataset, 
    batch_size=4,
    shuffle=True,
    collate_fn=valve_collate,
)

batch = next(iter(loader))

print("x", batch["x"].shape)
print("y", batch["y"].shape)
print("mask", batch["mask"].shape)
print("lengths", batch["lengths"])
print("example meta", batch["meta"][0])

def test_dataset() -> None:
    dataset = ValveDataset("src/data/raw/raw1/AHA/events")

    target_path = "src/data/raw/raw1/AHA/events/AHA_228.json"

    event = load_event(target_path)
    processed = dataset.processor.preprocess_event(event)

    label_signal = processed.resampled_signal
    samplerate = processed.samplerate
    n_samples = processed.n_samples

    prp_result = detect_prp(label_signal, dataset.prp_config)
    bp_result = detect_bp(label_signal, samplerate, prp_result.prp_index, dataset.bp_config)
    sp_result = detect_sp(label_signal, samplerate, prp_result.prp_index, dataset.sp_config)

    prp_target = build_prp_target(
        n_samples=n_samples,
        prp_index=prp_result.prp_index,
        samplerate=samplerate,
    )

    bp_target = build_bp_target(
        n_samples=n_samples,
        bp_index=bp_result.bp_index,
        samplerate=samplerate,
    )

    sp_target = build_sp_target(
        n_samples=n_samples,
        sp_index=sp_result.sp_index,
        samplerate=samplerate,
    )

    print("PATH:", target_path)
    print("PROCESSED_SAMPLERATE:", samplerate)
    print("PROCESSED_N_SAMPLES:", n_samples)
    print("PROCESSED_DURATION:", processed.duration_sec)

    print("PRP_INDEX:", prp_result.prp_index)
    print("PRP_START:", prp_result.start_index)
    print("PRP_END:", prp_result.end_index)
    print("PRP_REASON:", prp_result.reason)

    print("BP_INDEX:", bp_result.bp_index)
    print("BP_START:", bp_result.start_index)
    print("BP_END:", bp_result.end_index)
    print("BP_REASON:", bp_result.reason)

    print("SP_INDEX:", sp_result.sp_index)
    print("SP_START:", sp_result.start_index)
    print("SP_END:", sp_result.end_index)
    print("SP_REASON:", sp_result.reason)

    signal = label_signal
    smooth = moving_avg(signal.astype(np.float32), dataset.prp_config.smooth_samples)
    time_sec = np.arange(len(signal), dtype=np.float32) / float(samplerate)

    plt.figure(figsize=(14, 5))

    plt.plot(time_sec, signal, label="resampled signal", alpha=0.6)
    plt.plot(time_sec, smooth, label="smoothed signal", linewidth=2)

    if prp_result.start_index is not None and prp_result.end_index is not None:
        plt.axvspan(
        prp_result.start_index / samplerate,
        prp_result.end_index / samplerate,
        alpha=0.2,
        label="PRP region",
    )

    if prp_result.prp_index is not None:
        plt.axvline(
        prp_result.prp_index / samplerate,
        linestyle="--",
        color="blue",
        label=f"PRP {prp_result.prp_index}",
    )

    if bp_result.bp_index is not None:
        plt.axvline(
            bp_result.bp_index / samplerate,
            linestyle="--",
            color="red",
            label=f"BP {bp_result.bp_index}",
    )

    if sp_result.sp_index is not None:
        plt.axvline(
            sp_result.sp_index / samplerate,
            linestyle="--",
            color="green",
            label=f"SP {sp_result.sp_index}",
    )

    plt.title(f"Signal with PRP/BP/SP markers: {target_path}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pressure")
    plt.xlim(0, 20)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_dataset()

# python -m tests.data.test_dataset