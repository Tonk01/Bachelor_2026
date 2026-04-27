import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import valve_collate
from src.data.dataset import ValveDataset


def _write_event(path: Path, *, eventid: int, samplerate: int, signal: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eventid": eventid,
        "valvetag": "valve-1",
        "sitename": "site-1",
        "samplerate": samplerate,
        "eventdata": signal,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _target_builder(processed_event) -> torch.Tensor:
    return torch.zeros((3, processed_event.n_samples), dtype=torch.float32)


def test_valve_dataset_loads_event_and_builds_target(tmp_path: Path) -> None:
    event_path = tmp_path / "AHA" / "events" / "AHA_1.json"
    _write_event(
        event_path,
        eventid=1,
        samplerate=400,
        signal=np.linspace(0.0, 1.0, 400, dtype=np.float32).tolist(),
    )

    dataset = ValveDataset(tmp_path, target_builder=_target_builder)

    assert len(dataset) == 1
    sample = dataset[0]

    assert sample is not None
    assert sample["x"].shape == (1, 400)
    assert sample["y"].shape == (3, 400)
    assert sample["length"] == 400
    assert sample["meta"]["eventid"] == 1


def test_valve_collate_pads_variable_length_samples(tmp_path: Path) -> None:
    _write_event(
        tmp_path / "AHA" / "events" / "AHA_1.json",
        eventid=1,
        samplerate=400,
        signal=np.linspace(0.0, 1.0, 400, dtype=np.float32).tolist(),
    )
    _write_event(
        tmp_path / "AHA" / "events" / "AHA_2.json",
        eventid=2,
        samplerate=400,
        signal=np.linspace(0.0, 1.0, 200, dtype=np.float32).tolist(),
    )

    dataset = ValveDataset(tmp_path, target_builder=_target_builder)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=valve_collate,
    )

    batch = next(iter(loader))

    assert batch is not None
    assert batch["x"].shape == (2, 1, 400)
    assert batch["y"].shape == (2, 3, 400)
    assert batch["mask"].shape == (2, 1, 400)
    lengths = batch["lengths"].tolist()
    assert sorted(lengths) == [200, 400]
    for row, length in enumerate(lengths):
        assert batch["mask"][row, :, :length].all()
        assert not batch["mask"][row, :, length:].any()
