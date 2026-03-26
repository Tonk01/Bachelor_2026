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
    print("BP_INDEX:", bp_result.bp_index)
    print("SP_INDEX:", sp_result.sp_index)


if __name__ == "__main__":
    test_dataset()

# python -m tests.data.test_dataset