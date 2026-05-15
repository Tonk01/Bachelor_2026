from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F

def pad_tensor(x: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = x.shape[-1]
    pad_amount = target_length - current_length

    if pad_amount < 0:
        raise ValueError(f"Target length {target_length} is smaller than current length {current_length}")
    
    if pad_amount == 0:
        return x
    
    return F.pad(x, (0, pad_amount), value = 0.0)


def valve_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None
    
    lengths = torch.tensor([sample["length"] for sample in batch], dtype = torch.long)
    max_length = int(lengths.max().item())

    x_list = []
    y_list = []
    mask_list = []
    sensor_presence_list = []
    target_valid_mask_list = []
    sample_weight_list = []
    meta_list = []

    for sample in batch:
        x = sample["x"]
        y = sample["y"]
        length = sample["length"]

        x_pad = pad_tensor(x, max_length)
        y_pad = pad_tensor(y, max_length)

        mask = torch.zeros((1, max_length), dtype = torch.bool)
        mask[:, :length] = True

        x_list.append(x_pad)
        y_list.append(y_pad)
        mask_list.append(mask)
        sensor_presence = sample.get("sensor_presence")
        if sensor_presence is None:
            sensor_presence = torch.ones(x.shape[0], dtype=torch.float32)
        else:
            sensor_presence = torch.as_tensor(sensor_presence, dtype=torch.float32)
        sensor_presence_list.append(sensor_presence)

        target_valid_mask = sample.get("target_valid_mask")
        if target_valid_mask is None:
            target_valid_mask = torch.ones(y.shape[0], dtype=torch.bool)
        else:
            target_valid_mask = torch.as_tensor(target_valid_mask, dtype=torch.bool)
        target_valid_mask_list.append(target_valid_mask)

        sample_weight = sample.get("sample_weight")
        if sample_weight is None:
            sample_weight = sample["meta"].get("sample_weight", 1.0)
        sample_weight_list.append(float(sample_weight))
        meta_list.append(sample["meta"])

    return {
        "x": torch.stack(x_list, dim = 0),
        "y": torch.stack(y_list, dim = 0),
        "mask": torch.stack(mask_list, dim = 0),
        "sensor_presence": torch.stack(sensor_presence_list, dim = 0),
        "target_valid_mask": torch.stack(target_valid_mask_list, dim = 0),
        "sample_weights": torch.tensor(sample_weight_list, dtype=torch.float32),
        "lengths": lengths,
        "meta": meta_list,
    }   
