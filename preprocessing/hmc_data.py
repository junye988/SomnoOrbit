from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_DIRS = {"train": "train", "val": "eval", "eval": "eval", "test": "test"}
EXPECTED_CHANNELS = ("F4", "C4", "O2", "C3")
EXPECTED_SHAPE = (4, 3000)
EXPECTED_FS = 100


@dataclass(frozen=True)
class EpochRef:
    path: Path
    record_id: str
    epoch_index: int


@dataclass(frozen=True)
class NightRef:
    split: str
    record_id: str
    epochs: tuple[EpochRef, ...]

    def __len__(self) -> int:
        return len(self.epochs)


def _parse_epoch_path(path: Path) -> EpochRef:
    try:
        record_id, raw_index = path.stem.rsplit("-", 1)
        epoch_index = int(raw_index)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid HMC epoch filename: {path.name}") from exc
    return EpochRef(path=path, record_id=record_id, epoch_index=epoch_index)


def index_split(root: str | Path, split: str) -> list[NightRef]:
    root = Path(root)
    split_key = split.lower()
    if split_key not in SPLIT_DIRS:
        raise ValueError(f"unknown split {split!r}; choose train, val/eval, or test")
    split_name = SPLIT_DIRS[split_key]
    split_dir = root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"HMC split directory not found: {split_dir}")

    grouped: dict[str, list[EpochRef]] = defaultdict(list)
    for path in split_dir.glob("*.pkl"):
        ref = _parse_epoch_path(path)
        grouped[ref.record_id].append(ref)
    if not grouped:
        raise FileNotFoundError(f"no HMC epoch pickles found in {split_dir}")

    nights: list[NightRef] = []
    for record_id in sorted(grouped):
        refs = sorted(grouped[record_id], key=lambda ref: ref.epoch_index)
        indices = [ref.epoch_index for ref in refs]
        expected = list(range(len(refs)))
        if indices != expected:
            missing = sorted(set(expected) - set(indices))[:20]
            duplicates = sorted(k for k, v in Counter(indices).items() if v > 1)[:20]
            raise ValueError(
                f"non-contiguous epoch indices for {record_id}: "
                f"count={len(refs)}, first={indices[:3]}, last={indices[-3:]}, "
                f"missing={missing}, duplicates={duplicates}"
            )
        nights.append(NightRef(split=split_name, record_id=record_id, epochs=tuple(refs)))
    return nights


def load_epoch(
    ref: EpochRef,
    *,
    validate: bool = True,
) -> tuple[np.ndarray, int]:
    with ref.path.open("rb") as handle:
        sample = pickle.load(handle)

    required = {"X", "ch_names", "y", "fs", "record_id", "epoch_index"}
    missing = required - set(sample)
    if missing:
        raise KeyError(f"{ref.path} is missing fields {sorted(missing)}")

    signal = np.asarray(sample["X"])
    label = int(sample["y"])
    if validate:
        if str(sample["record_id"]) != ref.record_id:
            raise ValueError(f"record_id mismatch in {ref.path}")
        if int(sample["epoch_index"]) != ref.epoch_index:
            raise ValueError(f"epoch_index mismatch in {ref.path}")
        if signal.shape != EXPECTED_SHAPE:
            raise ValueError(f"expected signal shape {EXPECTED_SHAPE}, got {signal.shape} in {ref.path}")
        if signal.dtype != np.float32:
            raise TypeError(f"expected float32 signal, got {signal.dtype} in {ref.path}")
        if tuple(str(name) for name in sample["ch_names"]) != EXPECTED_CHANNELS:
            raise ValueError(f"unexpected channel order in {ref.path}: {sample['ch_names']}")
        if int(sample["fs"]) != EXPECTED_FS:
            raise ValueError(f"expected {EXPECTED_FS} Hz, got {sample['fs']} in {ref.path}")
        if label not in range(5):
            raise ValueError(f"expected a five-stage label, got {label} in {ref.path}")
        if not np.isfinite(signal).all():
            raise ValueError(f"non-finite signal values in {ref.path}")
    return signal.astype(np.float32, copy=False), label


def load_night(
    night: NightRef,
    *,
    start: int = 0,
    stop: int | None = None,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    selected = night.epochs[slice(start, stop)]
    if not selected:
        raise ValueError(f"empty epoch selection for {night.record_id}")
    signal = np.empty((len(selected), *EXPECTED_SHAPE), dtype=np.float32)
    labels = np.empty(len(selected), dtype=np.int64)
    for offset, ref in enumerate(selected):
        signal[offset], labels[offset] = load_epoch(ref, validate=validate)
    return signal, labels


def robust_context_normalize(signal: np.ndarray) -> np.ndarray:
    """Match SomnoOrbit's context-wise per-channel median/MAD normalization."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 3 or signal.shape[1:] != EXPECTED_SHAPE:
        raise ValueError(
            f"expected [epochs, {EXPECTED_SHAPE[0]}, {EXPECTED_SHAPE[1]}], got {signal.shape}"
        )
    median = np.median(signal, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(signal - median), axis=(0, 2), keepdims=True)
    normalized = np.clip((signal - median) / (1.4826 * mad + 1e-6), -8.0, 8.0)
    return normalized.astype(np.float32, copy=False)


class HMCWindowDataset(Dataset):
    """Lazy, night-safe sequence windows for the contextual evidence model."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        context: int = 20,
        stride: int = 20,
        augment: bool = False,
        max_records: int = 0,
        max_windows: int = 0,
    ) -> None:
        if context < 1 or stride < 1:
            raise ValueError("context and stride must be positive")
        self.context = int(context)
        self.stride = int(stride)
        self.augment = bool(augment)
        self.root = Path(root)
        self.split = SPLIT_DIRS[split.lower()]
        self.nights = index_split(root, split)
        if max_records:
            self.nights = self.nights[:max_records]
        self.items: list[tuple[NightRef, int]] = []
        for night in self.nights:
            if len(night) < self.context:
                continue
            for start in range(0, len(night) - self.context + 1, stride):
                self.items.append((night, start))
                if max_windows and len(self.items) >= max_windows:
                    break
            if max_windows and len(self.items) >= max_windows:
                break
        if not self.items:
            raise ValueError(f"no length-{context} windows available for split {split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        night, start = self.items[index]
        signal, labels = load_night(
            night,
            start=start,
            stop=start + self.context,
            validate=True,
        )
        signal = robust_context_normalize(signal)
        if self.augment:
            scale = np.random.uniform(0.9, 1.1, size=(1, signal.shape[1], 1)).astype(np.float32)
            signal = signal * scale
        return torch.from_numpy(signal.copy()), torch.from_numpy(labels.copy())


def load_meta(root: str | Path) -> dict:
    path = Path(root) / "meta_summary.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"HMC metadata not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)
