from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def context_starts(length: int, context: int = 20, stride: int = 2) -> list[int]:
    """Return starts that cover every epoch, including the final boundary."""
    if length < context:
        raise ValueError(f"night has {length} epochs, fewer than context={context}")
    if stride < 1:
        raise ValueError("stride must be positive")
    starts = list(range(0, length - context + 1, stride))
    final = length - context
    if starts[-1] != final:
        starts.append(final)
    return starts


def robust_context_normalize(windows: np.ndarray) -> np.ndarray:
    """Median/MAD normalization per context and EEG channel.

    ``windows`` has shape ``[B, K, C, L]``.  A channel is normalized jointly
    over its ``K`` epochs.
    """
    windows = np.asarray(windows, dtype=np.float32)
    median = np.median(windows, axis=(1, 3), keepdims=True)
    mad = np.median(np.abs(windows - median), axis=(1, 3), keepdims=True)
    return np.clip((windows - median) / (1.4826 * mad + 1e-6), -8.0, 8.0)


def marginalize_context_phase(
    backbone: torch.nn.Module,
    night: np.ndarray | torch.Tensor,
    *,
    context: int = 20,
    stride: int = 2,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    normalized_windows: np.ndarray | None = None,
    normalized_window_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Average every available contextual view of each epoch.

    For epoch :math:`t`, this computes

    .. math::
       \bar z_t = |\mathcal W(t)|^{-1}\sum_{s\in\mathcal W(t)}
       g_\theta(X_{s:s+K})_{t-s}.

    The returned count records how many computational phases contributed to
    each epoch.  Context logits are averaged before orbit observation.
    """
    if isinstance(night, torch.Tensor):
        array = night.detach().cpu().numpy()
    else:
        array = np.asarray(night)
    array = array.astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError("night must have shape [epochs, channels, samples]")
    starts = context_starts(len(array), context=context, stride=stride)
    if (normalized_windows is None) != (normalized_window_indices is None):
        raise ValueError(
            "normalized_windows and normalized_window_indices must be provided together"
        )
    if normalized_window_indices is not None and len(normalized_window_indices) != len(starts):
        raise ValueError("normalized-window indices do not match context starts")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device is None:
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    total = np.zeros((len(array), 5), dtype=np.float64)
    count = np.zeros(len(array), dtype=np.int64)
    training = backbone.training
    backbone.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            current = starts[offset : offset + batch_size]
            if normalized_windows is None:
                windows = np.stack([array[start : start + context] for start in current])
                windows = robust_context_normalize(windows)
            else:
                assert normalized_window_indices is not None
                positions = range(offset, offset + len(current))
                windows = np.stack(
                    [
                        np.asarray(normalized_windows[normalized_window_indices[position]])
                        for position in positions
                    ]
                )
            logits = backbone(torch.from_numpy(windows).to(device)).float().cpu().numpy()
            for start, value in zip(current, logits):
                total[start : start + context] += value
                count[start : start + context] += 1
    backbone.train(training)
    if np.any(count == 0):
        raise RuntimeError("polyphase contexts did not cover the complete night")
    evidence = (total / count[:, None]).astype(np.float32)
    return torch.from_numpy(evidence), torch.from_numpy(count)


def single_origin_context_logits(
    backbone: torch.nn.Module,
    night: np.ndarray | torch.Tensor,
    *,
    context: int = 20,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    normalized_windows: np.ndarray | None = None,
    normalized_window_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate every epoch from one deterministic contextual origin.

    Consecutive non-overlapping contexts are used whenever possible.  A final
    boundary-aligned context supplies only the otherwise uncovered tail, so
    each epoch contributes exactly once.  This is the inference rule of the
    matched contextual-backbone baseline; unlike COM, it never averages
    multiple computational positions of the same epoch.
    """
    if isinstance(night, torch.Tensor):
        array = night.detach().cpu().numpy()
    else:
        array = np.asarray(night)
    array = array.astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError("night must have shape [epochs, channels, samples]")
    if len(array) < context:
        raise ValueError(f"night has {len(array)} epochs, fewer than context={context}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device is None:
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    starts = list(range(0, len(array) - context + 1, context))
    final = len(array) - context
    if starts[-1] != final:
        starts.append(final)
    if (normalized_windows is None) != (normalized_window_indices is None):
        raise ValueError(
            "normalized_windows and normalized_window_indices must be provided together"
        )
    if normalized_window_indices is not None and len(normalized_window_indices) != len(starts):
        raise ValueError("normalized-window indices do not match context starts")
    evidence = np.zeros((len(array), 5), dtype=np.float32)
    assigned = np.zeros(len(array), dtype=bool)
    training = backbone.training
    backbone.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            current = starts[offset : offset + batch_size]
            if normalized_windows is None:
                windows = np.stack([array[start : start + context] for start in current])
                windows = robust_context_normalize(windows)
            else:
                assert normalized_window_indices is not None
                positions = range(offset, offset + len(current))
                windows = np.stack(
                    [
                        np.asarray(normalized_windows[normalized_window_indices[position]])
                        for position in positions
                    ]
                )
            logits = backbone(torch.from_numpy(windows).to(device)).float().cpu().numpy()
            for start, value in zip(current, logits):
                indices = np.arange(start, start + context)
                take = ~assigned[indices]
                evidence[indices[take]] = value[take]
                assigned[indices[take]] = True
    backbone.train(training)
    if not np.all(assigned):
        raise RuntimeError("single-origin contexts did not cover the complete night")
    return torch.from_numpy(evidence), torch.ones(len(array), dtype=torch.int64)


def pad_nights(
    logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """Pad variable-length nights for batched observer evaluation."""
    if not logits:
        raise ValueError("at least one night is required")
    lengths = [int(row.shape[0]) for row in logits]
    steps = max(lengths)
    batch = len(lengths)
    padded = logits[0].new_zeros(batch, steps, 5)
    mask = torch.zeros(batch, steps, dtype=torch.bool)
    padded_labels = None
    if labels is not None:
        padded_labels = torch.zeros(batch, steps, dtype=torch.long)
    for row, length in enumerate(lengths):
        padded[row, :length] = logits[row]
        mask[row, :length] = True
        if padded_labels is not None:
            padded_labels[row, :length] = labels[row]
    result = {"logits": padded, "mask": mask, "lengths": torch.tensor(lengths)}
    if padded_labels is not None:
        result["labels"] = padded_labels
    return result
