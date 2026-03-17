from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

from src.data.data_loader import load_event
from src.data.targets import detect_prp, moving_avg, ms_to_samples, PRPconfig

def inspect_signal(signal: np.ndarray, samplerate: int = 400, config: PRPconfig | None = None) -> None:
    cfg = config or PRPconfig()
    result = detect_prp(signal, samplerate=samplerate, config = cfg)

    smooth_n = ms_to_samples(cfg.smooth_ms, samplerate)
    x_smooth = moving_avg(signal.astype(np.float32), smooth_n)

    print("PRP Result")
    print(f" prp_index  {result.prp_index}")
    print(f" confidence {result.confidence:.4f}")
    print(f" reason {result.reason}")

    plt.figure(figsize = (14, 5))
    plt.plot(signal, label = "raw pressure", alpha = 0.6)
    plt.plot(x_smooth, label = "smooth pressure", linewidth = 2)

    if result.prp_index is not None:
        plt.axvline(result.prp_index, linestyle="--", label= f"PRP {result.prp_index}")

    plt.title("PRP inspection")
    plt.xlabel("sample index")
    plt.ylabel("pressure")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    
    path = Path("src/data/raw/raw1/JSDP/events/JSDP_493.json")
    print("Attempt to load: ", path.resolve())

    event = load_event(path)
    signal = np.asarray(event.signal, dtype = np.float32)

    cfg = PRPconfig(
        search_end_ratio=0.35,
        slope_sigma_mult=2.0,
        min_amplitude_sigma=1.5,
        min_rise_ms=8.0,
    )

    inspect_signal(signal, samplerate = event.samplerate, config=cfg)

if __name__ == "__main__":
    main()
