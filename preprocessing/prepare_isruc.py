"""Prepare ISRUC-SLEEP Group I recordings as 20-epoch sequences.

The EDF preparation follows https://github.com/wjq-learning/CBraMod.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from .edf_ import read_raw_edf
except ImportError:
    from edf_ import read_raw_edf


LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}
SAMPLING_RATE = 200
CONTEXT = 20


def process_subject(
    raw_root: Path,
    output_root: Path,
    subject: int,
    *,
    first_chunk: int = 0,
    n_jobs: int = 1,
) -> int:
    """Filter one recording and save paired signal and label sequences."""
    recording = raw_root / str(subject) / f"{subject}.rec"
    label_path = raw_root / str(subject) / f"{subject}_1.txt"
    raw = read_raw_edf(str(recording), preload=True)
    try:
        if float(raw.info["sfreq"]) != SAMPLING_RATE:
            raise ValueError(f"Expected {SAMPLING_RATE} Hz, got {raw.info['sfreq']}")
        raw.filter(0.3, 35, fir_design="firwin", n_jobs=n_jobs)
        raw.notch_filter(50, n_jobs=n_jobs)
        psg_array = raw.to_data_frame().values
        psg_array = psg_array[:, 1:]
        psg_array = psg_array[:, 2:8]
    finally:
        raw.close()

    remainder_samples = psg_array.shape[0] % (30 * SAMPLING_RATE)
    if remainder_samples > 0:
        psg_array = psg_array[:-remainder_samples, :]
    psg_array = psg_array.reshape(-1, 30 * SAMPLING_RATE, 6)
    remainder_epochs = psg_array.shape[0] % CONTEXT
    if remainder_epochs > 0:
        psg_array = psg_array[:-remainder_epochs, :, :]
    psg_array = psg_array.reshape(-1, CONTEXT, 30 * SAMPLING_RATE, 6)
    sequences = psg_array.transpose(0, 1, 3, 2)

    with label_path.open() as handle:
        labels = np.array([LABEL_MAP[line.strip()] for line in handle if line.strip()])
    if remainder_epochs > 0:
        labels = labels[:-remainder_epochs]
    labels = labels.reshape(-1, CONTEXT)
    if len(sequences) != len(labels):
        raise ValueError(f"Signal and label sequence counts differ for subject {subject}")

    name = f"ISRUC-group1-{subject}"
    sequence_dir = output_root / "seq" / name
    label_dir = output_root / "labels" / name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for offset, (signal, target) in enumerate(zip(sequences, labels)):
        filename = f"{name}-{first_chunk + offset}.npy"
        np.save(sequence_dir / filename, signal)
        np.save(label_dir / filename, target)
    return len(sequences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Group I directory containing 1/1.rec, 2/2.rec, etc.")
    parser.add_argument("--processed", type=Path, required=True, help="Output directory for seq/ and labels/")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 101)))
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    next_chunk = 0
    for subject in args.subjects:
        count = process_subject(args.raw, args.processed, subject, first_chunk=next_chunk, n_jobs=args.n_jobs)
        next_chunk += count
        print(f"Subject {subject}: saved {count} sequences")
    print(f"Saved {next_chunk} sequences to {args.processed}")


if __name__ == "__main__":
    main()
