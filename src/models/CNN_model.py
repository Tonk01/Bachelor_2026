from __future__ import annotations

import torch
import torch.nn as nn


def _append_sensor_mask(
    x: torch.Tensor,
    sensor_presence: torch.Tensor | None,
    sensor_mask_channels: int,
) -> torch.Tensor:
    if sensor_mask_channels <= 0 or sensor_presence is None:
        return x

    if sensor_presence.ndim != 2 or sensor_presence.shape[0] != x.shape[0]:
        raise ValueError(
            "sensor_presence must have shape (B, C) matching the batch dimension"
        )

    mask = sensor_presence.to(device=x.device, dtype=x.dtype)
    if mask.shape[1] < sensor_mask_channels:
        repeats = (sensor_mask_channels + mask.shape[1] - 1) // mask.shape[1]
        mask = mask.repeat(1, repeats)
    mask = mask[:, :sensor_mask_channels].unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.cat((x, mask), dim=1)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class ValveEventCNN(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 3,
            hidden_channels: int = 32,
            dropout: float = 0.1,
            sensor_mask_channels: int = 0,
            ) -> None:
        super().__init__()
        self.sensor_mask_channels = sensor_mask_channels

        self.features = nn.Sequential(
            ConvBlock(
                in_channels + sensor_mask_channels,
                hidden_channels,
                kernel_size=15,
                dropout=dropout,
            ),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=11, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels * 2, kernel_size=9, dropout=dropout),
            ConvBlock(hidden_channels * 2, hidden_channels * 2, kernel_size=7, dropout=dropout),
            ConvBlock(hidden_channels * 2, hidden_channels * 2, kernel_size=5, dropout=dropout)
        )

        self.head = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Excpeted inputs hape (B, C, T), got {tuple(x.shape)}")

        x = _append_sensor_mask(x, sensor_presence, self.sensor_mask_channels)
        y = self.features(x)
        y = self.head(y)
        return y
    
# small sanity check
if __name__ == "__main__":
    model = ValveEventCNN()
    x = torch.randn(2, 1, 1600)
    y = model(x)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
