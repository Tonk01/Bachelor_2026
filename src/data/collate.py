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
    print("CUSTOM COLLATE RUNNING")

    if len(batch) == 0:
        raise ValueError("Empty batch")
    
    lengths = torch.tensor([sample["length"] for sample in batch], dtype = torch.long)
    max_length = int(lengths.max().item())

    x_list = []
    y_list = []
    mask_list = []
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
        meta_list.append(sample["meta"])

    return {
        "x": torch.stack(x_list, dim = 0),
        "y": torch.stack(y_list, dim = 0),
        "mask": torch.stack(mask_list, dim = 0),
        "lengths": lengths,
        "meta": meta_list,
    }   