from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


class EpochEncoder(nn.Module):
    """Compact raw-waveform encoder used to obtain epoch evidence."""

    def __init__(self, channels: int = 6, width: int = 64):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.cat([x.mean(-1), x.std(-1, unbiased=False)], dim=1)


class ContextBackbone(nn.Module):
    """Window-level contextual evidence model.

    The backbone supplies five-stage evidence to the physiological observer.
    """

    def __init__(
        self,
        channels: int = 6,
        width: int = 64,
        classes: int = 5,
    ):
        super().__init__()
        self.encoder = EpochEncoder(channels, width)
        self.context = nn.GRU(4 * width, width, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4:
            raise ValueError("windows must have shape [batch, epochs, channels, samples]")
        batch, epochs, channels, samples = windows.shape
        encoded = self.encoder(windows.reshape(batch * epochs, channels, samples))
        contextual, _ = self.context(encoded.reshape(batch, epochs, -1))
        return self.head(contextual)

    def loss(
        self,
        windows: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(windows)
        return F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), weight=class_weights
        )

    def load_evidence_state(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load encoder/context/head weights from a compatible checkpoint."""
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own
            and own[key].shape == value.shape
            and key.startswith(("encoder.", "context.", "head."))
        }
        missing = sorted(set(own) - set(compatible))
        unexpected = sorted(set(state_dict) - set(compatible))
        self.load_state_dict(compatible, strict=False)
        return missing, unexpected
