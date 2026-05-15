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
   

    def __init__(
        self,
        num_sensors: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
        sensor_mask_channels: int = 0,
    ) -> None:
        super().__init__()
        self.sensor_mask_channels = sensor_mask_channels

        self.layer1 = DilatedConvBlock(
            num_sensors + sensor_mask_channels,
            64,
            dilation=1,
            dropout=dropout,
        )
        self.layer2 = DilatedConvBlock(64, 128, dilation=2, dropout=dropout)
        self.layer3 = DilatedConvBlock(128, 128, dilation=4, dropout=dropout)
        self.layer4 = DilatedConvBlock(128, 64, dilation=8, dropout=dropout)
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        sensor_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")

        x = _append_sensor_mask(x, sensor_presence, self.sensor_mask_channels)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


class ValveEventTCN(nn.Module):
    """
    TCN-arkitektur optimert for ventil-analyse (10 lag).
    Receptive field: 2047 samples (5.1s @ 400Hz).
    Designet for å håndtere 3 sensor-input: Pressure, Strain og Travel.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_layers: int = 10,
        dropout: float = 0.1,
        sensor_mask_channels: int = 0,
    ) -> None:
        super().__init__()
        self.sensor_mask_channels = sensor_mask_channels

        layers = []
        curr_channels = in_channels + sensor_mask_channels

        for i in range(num_layers):
            dilation = 2**i
            # Øker kanalbredden til 128 for å fange komplekse sensor-interaksjoner
            out_ch = 128 if 0 < i < num_layers - 1 else 64
            layers.append(DilatedConvBlock(curr_channels, out_ch, dilation, dropout))
            curr_channels = out_ch

        self.tcn_backbone = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(curr_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        sensor_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input format: (Batch, Sensors, Time)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        x = _append_sensor_mask(x, sensor_presence, self.sensor_mask_channels)
        features = self.tcn_backbone(x)
        logits = self.classifier(features)
        return logits


class ValveEventTCNMedium(ValveEventTCN):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dropout: float = 0.1,
        sensor_mask_channels: int = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=6,
            dropout=dropout,
            sensor_mask_channels=sensor_mask_channels,
        )


class ValveEventTCNSmall(ValveCnnEncoder):
    """
    Gammel TCN-arkitektur (4 lag) - beholdt for referanse.
    Receptive field: ~128 samples (0.32s @ 400Hz).
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dropout: float = 0.1,
        sensor_mask_channels: int = 0,
    ) -> None:
        super().__init__(
            num_sensors=in_channels,
            num_classes=out_channels,
            dropout=dropout,
            sensor_mask_channels=sensor_mask_channels,
        )


if __name__ == "__main__":
    # Test nye 10-lag modell
    model_large = ValveEventTCN(in_channels=3)
    x = torch.randn(2, 3, 1600)
    y = model_large(x)
    print("ValveEventTCN (10-lag):")
    print("  input shape:", x.shape)
    print("  output shape:", y.shape)

    # Test gamle 4-lag modell
    model_small = ValveEventTCNSmall(in_channels=3)
    y_small = model_small(x)
    print("\nValveEventTCNSmall (4-lag):")
    print("  input shape:", x.shape)
    print("  output shape:", y_small.shape)
