from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def subject_dirs(root: Path) -> dict[int, Path]:
    out = {}
    for path in (root / "seq").glob("ISRUC-group1-*"):
        match = re.fullmatch(r"ISRUC-group1-(\d+)", path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def split_subjects(root: Path, seed: int = 2026, limit: int = 0):
    ids = np.array(sorted(subject_dirs(root)))
    if limit:
        ids = ids[:limit]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, round(0.20 * len(ids)))
    n_val = max(1, round(0.10 * len(ids)))
    return {
        "train": sorted(ids[n_test + n_val :].tolist()),
        "val": sorted(ids[n_test : n_test + n_val].tolist()),
        "test": sorted(ids[:n_test].tolist()),
    }


class ISRUCSequences(Dataset):
    def __init__(self, root: str | Path, subjects: list[int], augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        self.items: list[tuple[Path, Path]] = []
        for sid in subjects:
            seq_dir = self.root / "seq" / f"ISRUC-group1-{sid}"
            label_dir = self.root / "labels" / f"ISRUC-group1-{sid}"
            for x_path in seq_dir.glob("*.npy"):
                y_path = label_dir / x_path.name
                if not y_path.exists():
                    raise FileNotFoundError(f"Missing paired label: {y_path}")
                self.items.append((x_path, y_path))
        self.items.sort(key=lambda pair: pair[0].name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        x_path, y_path = self.items[index]
        x = np.load(x_path).astype(np.float32, copy=False)
        y = np.load(y_path).astype(np.int64, copy=False)
        # Robust per-recording/channel scaling; clipping limits movement artifacts.
        med = np.median(x, axis=(0, 2), keepdims=True)
        mad = np.median(np.abs(x - med), axis=(0, 2), keepdims=True)
        x = np.clip((x - med) / (1.4826 * mad + 1e-6), -8.0, 8.0)
        if self.augment:
            scale = np.random.uniform(0.9, 1.1, size=(1, x.shape[1], 1)).astype(np.float32)
            x = x * scale
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
