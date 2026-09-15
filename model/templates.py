from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def rem_locked_phase_index(
    logits: torch.Tensor,
    *,
    period: float = 200.0,
    threshold: float = 0.7,
    release_ratio: float = 0.6,
    bins: int = 36,
) -> np.ndarray:
    """Return the legacy REM-locked phase bin for every segment."""
    probability = logits.softmax(-1)
    age, in_rem, output = 0, False, []
    for current in probability:
        high = bool(current[4] > threshold and current.argmax() == 4)
        if high:
            age, in_rem = 0, True
        elif in_rem and bool(current[4] < release_ratio * threshold):
            age, in_rem = 1, False
        elif not in_rem:
            age += 1
        output.append(int((age % period) / period * bins) % bins)
    return np.asarray(output, dtype=np.int64)


def _log_ratio(count: np.ndarray) -> torch.Tensor:
    conditional = count / count.sum(1, keepdims=True)
    marginal = count.sum(0) / count.sum()
    value = np.log(conditional + 1e-10) - np.log(marginal + 1e-10)
    value -= value.mean(1, keepdims=True)
    return torch.tensor(value, dtype=torch.float32)


def estimate_orbit_templates(
    logits: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    clock_bins: int = 81,
    clock_sigma: float = 4.0,
    phase_bins: int = 36,
    phase_sigma: float = 2.0,
    period: float = 200.0,
    rem_threshold: float = 0.7,
    release_ratio: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate reference and slow--fast interaction fields from train nights.

    The first two outputs exactly follow SomnoPhase.  The interaction field is
    the phase-conditioned early/late log-odds contrast and therefore captures
    how the same ultradian phase changes meaning across the night.
    """
    if len(logits) != len(labels):
        raise ValueError("logits and labels must contain the same number of nights")

    clock_count = np.ones((clock_bins, 5), dtype=np.float64)
    phase_count = np.ones((phase_bins, 5), dtype=np.float64) * 5.0
    early_count = np.ones((phase_bins, 5), dtype=np.float64) * 5.0
    late_count = np.ones((phase_bins, 5), dtype=np.float64) * 5.0

    for current_logits, current_labels in zip(logits, labels):
        z = torch.as_tensor(current_logits, dtype=torch.float32).cpu()
        y = torch.as_tensor(current_labels, dtype=torch.long).cpu().numpy()
        if len(z) != len(y):
            raise ValueError("a logit night and its labels have different lengths")
        clock_index = np.round(
            np.arange(len(y)) / max(len(y) - 1, 1) * (clock_bins - 1)
        ).astype(np.int64)
        phase_index = rem_locked_phase_index(
            z,
            period=period,
            threshold=rem_threshold,
            release_ratio=release_ratio,
            bins=phase_bins,
        )
        np.add.at(clock_count, (clock_index, y), 1)
        np.add.at(phase_count, (phase_index, y), 1)
        midpoint = max(len(y) - 1, 1) / 2.0
        early = np.arange(len(y)) <= midpoint
        np.add.at(early_count, (phase_index[early], y[early]), 1)
        np.add.at(late_count, (phase_index[~early], y[~early]), 1)

    clock_count = gaussian_filter1d(
        clock_count, clock_sigma, axis=0, mode="nearest"
    )
    phase_count = gaussian_filter1d(
        phase_count, phase_sigma, axis=0, mode="wrap"
    )
    early_count = gaussian_filter1d(
        early_count, phase_sigma, axis=0, mode="wrap"
    )
    late_count = gaussian_filter1d(
        late_count, phase_sigma, axis=0, mode="wrap"
    )

    early_probability = early_count / early_count.sum(1, keepdims=True)
    late_probability = late_count / late_count.sum(1, keepdims=True)
    interaction = np.log(early_probability + 1e-10) - np.log(
        late_probability + 1e-10
    )
    interaction -= interaction.mean(1, keepdims=True)
    return (
        _log_ratio(clock_count),
        _log_ratio(phase_count),
        torch.tensor(interaction, dtype=torch.float32),
    )


def nights_from_artifact(
    artifact: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    lengths = [int(value) for value in artifact["lengths"]]
    flat_logits = artifact["logits"]
    flat_labels = artifact["labels"]
    if sum(lengths) != len(flat_logits):
        raise ValueError("artifact lengths do not cover its logits")
    logits, labels, offset = [], [], 0
    for length in lengths:
        logits.append(flat_logits[offset : offset + length].float())
        labels.append(flat_labels[offset : offset + length].long())
        offset += length
    return logits, labels
