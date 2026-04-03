from __future__ import annotations

import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        channel_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction = reduction

        if channel_weights is not None:
            if channel_weights.ndim != 1:
                raise ValueError("channel_weights must be 1D")
            self.register_buffer("channel_weights", channel_weights.view(1, -1, 1))
        else:
            self.channel_weights = None

    def forward(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        
        if preds.shape != targets.shape:
            raise ValueError(f"Shape missmatch: preds {preds.shape} vs targets {targets.shape}")
        
        if mask.ndim != 3:
            raise ValueError(f"Mask must be (B, 1, T), got {mask.shape}")
        
        loss = self.criterion(preds, targets)
        mask = mask.expand_as(loss)

        loss = loss * mask

        # channel weighting
        if self.channel_weights is not None:
            loss = loss * self.channel_weights

        # Reduction
        if self.reduction == "mean":
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        
        elif self.reduction == "sum":
            return loss.sum()
        
        else:
            return loss
