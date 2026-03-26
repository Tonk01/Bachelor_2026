from __future__ import annotations

import torch
import torch.nn as nn

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
            ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=9, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=9, dropout=dropout),
            ConvBlock(hidden_channels, hidden_channels * 2, kernel_size=7, dropout=dropout),
            ConvBlock(hidden_channels * 2, hidden_channels * 2, kernel_size=7, dropout=dropout),
            ConvBlock(hidden_channels * 2, hidden_channels * 2, kernel_size=5, dropout=dropout)
        )

        self.head = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Excpeted inputs hape (B, C, T), got {tuple(x.shape)}")
        
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