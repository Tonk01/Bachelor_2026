from __future__ import annotations

import torch
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if dilation <= 0:
            raise ValueError("dilation must be greater than 0")

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ValveCnnEncoder(nn.Module):
    """TCN-style dilated 1D-CNN from 04_cnn_architecture.md.

    The forward pass returns logits. Use torch.sigmoid(output) for probabilities
    at inference time. Training should pass these logits directly to
    BCEWithLogitsLoss, as the existing MaskedBCELoss does.
    """

    def __init__(
        self,
        num_sensors: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layer1 = DilatedConvBlock(num_sensors, 64, dilation=1, dropout=dropout)
        self.layer2 = DilatedConvBlock(64, 128, dilation=2, dropout=dropout)
        self.layer3 = DilatedConvBlock(128, 128, dilation=4, dropout=dropout)
        self.layer4 = DilatedConvBlock(128, 64, dilation=8, dropout=dropout)
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


class ValveEventTCN(ValveCnnEncoder):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_sensors=in_channels,
            num_classes=out_channels,
            dropout=dropout,
        )


if __name__ == "__main__":
    model = ValveEventTCN(in_channels=3)
    x = torch.randn(2, 3, 1600)
    y = model(x)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
