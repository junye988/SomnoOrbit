from __future__ import annotations

import math

import torch
from torch import nn


class StatisticsEpochEncoder(nn.Module):
    """Compact raw-waveform CNN used by the contextual baselines."""

    def __init__(self, channels: int, width: int = 64):
        super().__init__()
        self.output_dim = 4 * width
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, 31, stride=5, padding=15, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 15, stride=4, padding=7, groups=width, bias=False),
            nn.Conv1d(width, 2 * width, 1, bias=False),
            nn.BatchNorm1d(2 * width),
            nn.GELU(),
            nn.Conv1d(
                2 * width,
                2 * width,
                9,
                stride=4,
                padding=4,
                groups=2 * width,
                bias=False,
            ),
            nn.BatchNorm1d(2 * width),
            nn.GELU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        feature = self.net(signal)
        return torch.cat(
            [feature.mean(-1), feature.std(-1, unbiased=False)], dim=-1
        )


class EEGNetEpoch(nn.Module):
    """EEGNet-style compact epoch encoder with adaptive temporal pooling."""

    def __init__(
        self,
        channels: int,
        sample_rate: int,
        classes: int = 5,
        f1: int = 8,
        depth: int = 2,
        f2: int = 16,
        dropout: float = 0.35,
    ):
        super().__init__()
        temporal_kernel = max(16, int(round(sample_rate / 2)))
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1
        separable_kernel = max(8, int(round(sample_rate / 8)))
        if separable_kernel % 2 == 0:
            separable_kernel += 1
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                (1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * depth,
                (channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth,
                f1 * depth,
                (1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1 * depth,
                bias=False,
            ),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.head = nn.Linear(f2 * 8, classes)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        feature = self.temporal(signal[:, None])
        feature = self.spatial(feature)
        feature = self.separable(feature).flatten(1)
        return self.head(feature)


class EEGNetSequence(nn.Module):
    def __init__(self, channels: int, sample_rate: int, classes: int = 5, **_):
        super().__init__()
        self.epoch = EEGNetEpoch(channels, sample_rate, classes)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = windows.shape
        logits = self.epoch(windows.reshape(batch * steps, channels, samples))
        return logits.reshape(batch, steps, -1)


class CNNBiGRU(nn.Module):
    def __init__(self, channels: int, width: int = 64, classes: int = 5, **_):
        super().__init__()
        self.encoder = StatisticsEpochEncoder(channels, width)
        self.context = nn.GRU(
            self.encoder.output_dim,
            width,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = windows.shape
        epoch = self.encoder(windows.reshape(batch * steps, channels, samples))
        contextual, _ = self.context(epoch.reshape(batch, steps, -1))
        return self.head(contextual)


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class CNNTCN(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int = 64,
        classes: int = 5,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        self.encoder = StatisticsEpochEncoder(channels, width)
        self.project = nn.Conv1d(self.encoder.output_dim, 2 * width, 1)
        self.context = nn.Sequential(
            TemporalResidualBlock(2 * width, 1, dropout),
            TemporalResidualBlock(2 * width, 2, dropout),
            TemporalResidualBlock(2 * width, 4, dropout),
        )
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = windows.shape
        epoch = self.encoder(windows.reshape(batch * steps, channels, samples))
        feature = self.project(epoch.reshape(batch, steps, -1).transpose(1, 2))
        feature = self.context(feature).transpose(1, 2)
        return self.head(feature)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int = 64,
        classes: int = 5,
        dropout: float = 0.2,
        context: int = 20,
        **_,
    ):
        super().__init__()
        self.encoder = StatisticsEpochEncoder(channels, width)
        dimension = self.encoder.output_dim
        self.position = nn.Parameter(torch.zeros(1, context, dimension))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=8,
            dim_feedforward=4 * dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, classes))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = windows.shape
        epoch = self.encoder(windows.reshape(batch * steps, channels, samples))
        sequence = epoch.reshape(batch, steps, -1)
        if steps > self.position.shape[1]:
            raise ValueError("input sequence exceeds configured context length")
        sequence = self.context(sequence + self.position[:, :steps])
        return self.head(sequence)


MODEL_REGISTRY = {
    "eegnet": EEGNetSequence,
    "tcn": CNNTCN,
    "cnn_transformer": CNNTransformer,
    "cnn_bigru": CNNBiGRU,
}


def build_model(name: str, **kwargs) -> nn.Module:
    try:
        constructor = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from error
    return constructor(**kwargs)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
