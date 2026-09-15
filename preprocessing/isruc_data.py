from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def subject_dirs(root: str | Path) -> dict[int, Path]:
    root = Path(root)
    result = {}
    for path in (root / "seq").glob("ISRUC-group1-*"):
        match = re.fullmatch(r"ISRUC-group1-(\d+)", path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def split_subjects(root: str | Path, seed: int = 2026, limit: int = 0):
    ids = np.asarray(sorted(subject_dirs(root)), dtype=np.int64)
    if limit:
        ids = ids[:limit]
    generator = np.random.default_rng(seed)
    generator.shuffle(ids)
    n_test = max(1, round(0.20 * len(ids)))
    n_val = max(1, round(0.10 * len(ids)))
    return {
        "train": sorted(ids[n_test + n_val :].tolist()),
        "val": sorted(ids[n_test : n_test + n_val].tolist()),
        "test": sorted(ids[:n_test].tolist()),
    }


def chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def load_night(root: str | Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    sequence_dir = root / "seq" / f"ISRUC-group1-{subject}"
    label_dir = root / "labels" / f"ISRUC-group1-{subject}"
    paths = sorted(sequence_dir.glob("*.npy"), key=chunk_index)
    if not paths:
        raise FileNotFoundError(f"no sequence chunks for subject {subject}")
    signal = np.concatenate(
        [np.load(path).astype(np.float32, copy=False) for path in paths]
    )
    labels = np.concatenate(
        [np.load(label_dir / path.name).astype(np.int64, copy=False) for path in paths]
    )
    return signal, labels


class WindowDataset(Dataset):
    """Dataset adapter for training the contextual evidence model."""

    def __init__(
        self, root: str | Path, subjects: list[int], *, augment: bool = False
    ):
        self.root = Path(root)
        self.augment = bool(augment)
        self.items: list[tuple[Path, Path]] = []
        for subject in subjects:
            sequence_dir = self.root / "seq" / f"ISRUC-group1-{subject}"
            label_dir = self.root / "labels" / f"ISRUC-group1-{subject}"
            for signal_path in sequence_dir.glob("*.npy"):
                label_path = label_dir / signal_path.name
                if not label_path.exists():
                    raise FileNotFoundError(f"missing paired label {label_path}")
                self.items.append((signal_path, label_path))
        self.items.sort(
            key=lambda pair: (pair[0].parent.name, chunk_index(pair[0]))
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        signal_path, label_path = self.items[index]
        signal = np.load(signal_path).astype(np.float32, copy=False)
        labels = np.load(label_path).astype(np.int64, copy=False)
        median = np.median(signal, axis=(0, 2), keepdims=True)
        mad = np.median(np.abs(signal - median), axis=(0, 2), keepdims=True)
        signal = np.clip((signal - median) / (1.4826 * mad + 1e-6), -8.0, 8.0)
        if self.augment:
            scale = np.random.uniform(
                0.9, 1.1, size=(1, signal.shape[1], 1)
            ).astype(np.float32)
            signal = signal * scale
        return torch.from_numpy(signal.copy()), torch.from_numpy(labels.copy())
