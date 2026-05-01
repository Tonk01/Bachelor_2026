from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.data.data_loader import load_event
from src.data.preprocess import EventProcessor, PreprocessingConfig
from src.models.CNN_model import ValveEventCNN

RUN_NAME = "valve-cnn-20260427-000855"
NUM_RANDOM_EVENTS = 50
RANDOM_SEED = 42
POINT_NAMES = ["PRP", "BP", "SP"]


def load_model(run_dir: Path, device: torch.device) -> ValveEventCNN:
    model_path = run_dir / "best_model.pt"

    if not model_path.exists():
        raise ValueError(f"Missing model file: {model_path}")

    model = ValveEventCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def predict_points(
    model: ValveEventCNN,
    signal: np.ndarray,
    samplerate: int,
    device: torch.device,
) -> dict:
    x = torch.from_numpy(signal).to(torch.float32)
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = {}

    for c, name in enumerate(POINT_NAMES):
        values = probs[c]
        idx = int(np.argmax(values))
        conf = float(values[idx])

        results[name] = {
            "index": idx,
            "time_sec": idx / float(samplerate),
            "confidence": conf,
        }

    return results

def plot_predictions(
    raw_signal: np.ndarray,
    raw_samplerate: int,
    resampled_signal: np.ndarray,
    samplerate: int,
    predictions: dict,
    event_path: Path,
    run_dir: Path,
) -> None:
    raw_time = np.arange(raw_signal.size) / float(raw_samplerate)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    ax.plot(raw_time, raw_signal, label="raw", alpha=0.35)

    colors = {"PRP": "blue", "BP": "orange", "SP": "green"}

    for name in POINT_NAMES:
        if name not in predictions:
            continue

        result = predictions[name]
        t = result["time_sec"]
        idx = result["index"]
        conf = result["confidence"]

        ax.axvline(
            t,
            linestyle="--",
            color=colors[name],
            linewidth=2,
            label=f"{name} idx={idx} conf={conf:.3f}",
        )

    ax.set_title(f"{event_path.name} | Model: {run_dir.name}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    run_dir = PROJECT_ROOT / "artifacts" / RUN_NAME
    raw_dir = PROJECT_ROOT / "src" / "data" / "raw"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("raw_dir:", raw_dir)
    print("raw_dir exists:", raw_dir.exists())
    print("run_dir:", run_dir)
    print("run_dir exists:", run_dir.exists())

    if not run_dir.exists():
        raise ValueError(f"Run not found: {run_dir}")

    all_event_paths = sorted(raw_dir.glob("*/events/*.json"))

    if not all_event_paths:
        raise ValueError(f"No raw event files found in: {raw_dir}")

    rng = np.random.default_rng(RANDOM_SEED)

    n = min(NUM_RANDOM_EVENTS, len(all_event_paths))
    selected_indices = rng.choice(len(all_event_paths), size=n, replace=False)
    selected_paths = [all_event_paths[i] for i in selected_indices]

    print("Using model:", run_dir)
    print(f"Found {len(all_event_paths)} raw events.")
    print(f"Randomly selected {len(selected_paths)} events.")

    processor = EventProcessor(
        PreprocessingConfig(
            allowed_samplerate=(200, 400, 800, 1000),
            target_samplerate=400,
            normalize_for_model=True,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, device)

    for i, event_path in enumerate(selected_paths, start=1):
        print(f"\n[{i}/{len(selected_paths)}] Using file: {event_path}")

        try:
            event = load_event(event_path)
            processed = processor.preprocess_event(event)

            predictions = predict_points(
                model=model,
                signal=processed.normalized_signal,
                samplerate=processed.samplerate,
                device=device,
            )

            print("Predictions:")
            for name, r in predictions.items():
                print(
                    f"{name}: index = {r['index']} "
                    f"time = {r['time_sec']:.3f}s "
                    f"confidence = {r['confidence']:.4f}"
                )

            plot_predictions(
                raw_signal=np.asarray(event.signal, dtype=np.float32),
                raw_samplerate=event.samplerate,
                resampled_signal=processed.resampled_signal,
                samplerate=processed.samplerate,
                predictions=predictions,
                event_path=event_path,
                run_dir=run_dir,
            )

        except Exception as exc:
            print(f"Failed on {event_path}: {exc}")
            continue


if __name__ == "__main__":
    main()