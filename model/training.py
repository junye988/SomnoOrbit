"""CNN--BiGRU training and physiological orbit construction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .backbone import ContextBackbone
from .backbones import build_model
from .checkpoint import load_model
from .metrics import classification_metrics
from .model import SomnoOrbit
from .observer import OrbitConfig, SomnoOrbitObserver
from .polyphase import marginalize_context_phase
from .templates import estimate_orbit_templates
from .training_backbone import SomnoEnergy


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def trajectory_energy_penalty(logits: torch.Tensor) -> torch.Tensor:
    """Expected adjacent-stage cost in the order W, N1, N2, N3, REM."""
    cost = logits.new_tensor(
        [
            [0.0, 0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, 3.0, 0.0],
        ]
    )
    probability = logits.softmax(-1)
    return torch.einsum(
        "bti,ij,btj->bt", probability[:, :-1], cost, probability[:, 1:]
    ).mean()


def make_loader(dataset, *, training: bool, config: dict, device: torch.device) -> DataLoader:
    options = {}
    if config["dataset"] == "hmc":
        options["generator"] = torch.Generator().manual_seed(config["seed"])
        options["worker_init_fn"] = seed_worker if config["workers"] > 0 else None
    return DataLoader(
        dataset,
        batch_size=config["batch_size"] if training else config["eval_batch_size"],
        shuffle=training,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
        **options,
    )


def prepare_data(data_root: Path, config: dict, quick: bool, device: torch.device):
    if config["dataset"] == "isruc":
        from preprocessing.isruc_sequences import ISRUCSequences, split_subjects

        splits = split_subjects(data_root, config["split_seed"])
        if not all(splits[name] for name in ("train", "val", "test")):
            raise ValueError("ISRUC requires enough preprocessed subjects for train/validation/test splits")
        selected = {
            "train": splits["train"][:2] if quick else splits["train"],
            "val": splits["val"][:1] if quick else splits["val"],
        }
        train = ISRUCSequences(data_root, selected["train"], augment=True)
        validation = ISRUCSequences(data_root, selected["val"], augment=False)
        if quick:
            train.items = train.items[: 2 * config["batch_size"]]
            validation.items = validation.items[: config["eval_batch_size"]]
        counts = np.zeros(5, dtype=np.int64)
        for _, label_path in train.items:
            counts += np.bincount(np.load(label_path), minlength=5)
        weights = torch.tensor(
            counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device
        )
        weights = weights / weights.mean()
    else:
        from preprocessing.hmc_data import HMCWindowDataset, load_meta

        train = HMCWindowDataset(
            data_root, "train", context=config["context"], stride=config["train_stride"],
            augment=True, max_records=2 if quick else 0,
            max_windows=2 * config["batch_size"] if quick else 0,
        )
        validation = HMCWindowDataset(
            data_root, "eval", context=config["context"], stride=config["val_stride"],
            augment=False, max_records=1 if quick else 0,
            max_windows=config["eval_batch_size"] if quick else 0,
        )
        selected = {"train": train.nights, "val": validation.nights}
        counts_raw = load_meta(data_root)["train_summary"]["label_counts"]
        counts = np.asarray([int(counts_raw.get(i, counts_raw.get(str(i), 0))) for i in range(5)])
        if np.any(counts <= 0):
            raise ValueError(f"all five stages must occur in HMC training data: {counts.tolist()}")
        weights = counts.sum() / counts
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("no training or validation windows found; run dataset preprocessing first")
    for dataset in (train, validation):
        # Shape validation leaves the augmentation random stream unchanged.
        state = np.random.get_state()
        signal, labels = dataset[0]
        np.random.set_state(state)
        expected = (config["context"], config["channels"], config["samples_per_epoch"])
        if tuple(signal.shape) != expected or tuple(labels.shape) != (config["context"],):
            raise ValueError(f"expected windows {expected} and {config['context']} labels; got {tuple(signal.shape)}, {tuple(labels.shape)}")
        if not torch.isfinite(signal).all() or not ((labels >= 0) & (labels < 5)).all():
            raise ValueError("training data contains non-finite signals or invalid stage labels")
    return train, validation, selected, weights


@torch.inference_mode()
def evaluate_windows(model, loader: DataLoader, device: torch.device, dataset: str) -> dict:
    model.eval()
    targets, predictions = [], []
    for signal, labels in loader:
        signal = signal.to(device, non_blocking=True)
        prediction = model.predict(signal) if dataset == "isruc" else model(signal).argmax(-1)
        targets.append(labels.numpy().reshape(-1))
        predictions.append(prediction.cpu().numpy().reshape(-1))
    return classification_metrics(np.concatenate(targets), np.concatenate(predictions))


def train_backbone(train, validation, weights, config: dict, device: torch.device, output_dir: Path):
    train_loader = make_loader(train, training=True, config=config, device=device)
    val_loader = make_loader(validation, training=False, config=config, device=device)
    if config["dataset"] == "isruc":
        model = SomnoEnergy(
            channels=6, width=config["width"], classes=5, structured=False,
            prior_strength=0.7, local_anchor=1.0, duration_aware=False,
            max_duration=20, trajectory_energy=config["trajectory_energy"],
        ).to(device)
    else:
        model = build_model(
            "cnn_bigru", channels=4, sample_rate=100, width=config["width"],
            classes=5, context=config["context"],
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = None
    if config["dataset"] == "hmc":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config["epochs"], 1), eta_min=config["min_lr"]
        )
    best, bad_epochs, history = -float("inf"), 0, []
    checkpoint_path = output_dir / "backbone.pt"
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for signal, labels in train_loader:
            signal = signal.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                if config["dataset"] == "isruc":
                    loss = model.loss(signal, labels, weights)
                else:
                    logits = model(signal)
                    loss = F.cross_entropy(
                        logits.flatten(0, 1), labels.flatten(), weight=weights,
                        label_smoothing=config["label_smoothing"],
                    )
                    if config["trajectory_energy"] > 0.0:
                        loss = loss + config["trajectory_energy"] * trajectory_energy_penalty(logits)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * signal.shape[0]
            seen += signal.shape[0]
        if scheduler is not None:
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        metrics = evaluate_windows(model, val_loader, device, config["dataset"])
        row = {"epoch": epoch, "train_loss": total_loss / seen, **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["balanced_accuracy"] > best:
            best, bad_epochs = metrics["balanced_accuracy"], 0
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch, "validation": metrics},
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        write_json(output_dir / "history.json", history)
        if bad_epochs >= config["patience"]:
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = ContextBackbone(channels=config["channels"], width=config["width"], classes=5)
    missing, _ = backbone.load_evidence_state(checkpoint["model"])
    if missing:
        raise RuntimeError(f"backbone checkpoint is missing weights: {missing}")
    return backbone.to(device).eval(), checkpoint


def load_record(data_root: Path, dataset: str, record, quick: bool, context: int):
    stop = 3 * context if quick else None
    if dataset == "isruc":
        from preprocessing.isruc_data import load_night

        signal, labels = load_night(data_root, record)
        return signal[:stop], labels[:stop]
    from preprocessing.hmc_data import load_night

    return load_night(record, stop=stop)


def record_name(dataset: str, record) -> str:
    return str(record) if dataset == "isruc" else record.record_id


def fit_and_evaluate(backbone, selected: dict, data_root: Path, config: dict, quick: bool,
                     device: torch.device, output_dir: Path, checkpoint: dict) -> dict:
    train_logits, train_labels = [], []
    for record in selected["train"]:
        signal, labels = load_record(data_root, config["dataset"], record, quick, config["context"])
        evidence, _ = marginalize_context_phase(
            backbone, signal, context=config["context"], stride=config["template_stride"],
            batch_size=config["eval_batch_size"], device=device,
        )
        train_logits.append(evidence)
        train_labels.append(torch.from_numpy(labels))
        print(f"Template evidence: {record_name(config['dataset'], record)} ({len(labels)} epochs)", flush=True)
    orbit_config = OrbitConfig(**config["orbit"])
    templates = estimate_orbit_templates(
        train_logits, train_labels, period=orbit_config.period,
        rem_threshold=orbit_config.rem_threshold, release_ratio=orbit_config.release_ratio,
    )
    observer = SomnoOrbitObserver(*templates, config=orbit_config).to(device).eval()
    model = SomnoOrbit(backbone, observer).to(device).eval()
    targets, evidence_predictions, predictions, records = [], [], [], []
    first_evidence, first_logits = None, None
    with torch.inference_mode():
        for record in selected["val"]:
            signal, labels = load_record(data_root, config["dataset"], record, quick, config["context"])
            evidence, count = marginalize_context_phase(
                backbone, signal, context=config["context"], stride=config["eval_stride"],
                batch_size=config["eval_batch_size"], device=device,
            )
            logits = observer(evidence.to(device)).cpu()
            if not torch.isfinite(logits).all() or int(count.min()) < 1:
                raise FloatingPointError("invalid physiological orbit output or incomplete context coverage")
            targets.append(labels)
            evidence_predictions.append(evidence.argmax(-1).numpy())
            predictions.append(logits.argmax(-1).numpy())
            records.append({"record_id": record_name(config["dataset"], record), "epochs": len(labels)})
            if first_evidence is None:
                first_evidence, first_logits = evidence, logits
            print(f"Validation: {record_name(config['dataset'], record)} ({len(labels)} epochs)", flush=True)
    final_path = output_dir / "somnoorbit.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "training": {"best_epoch": checkpoint["epoch"], "quick": quick},
        },
        final_path,
    )
    restored, _ = load_model(final_path, device=device)
    with torch.inference_mode():
        restored_logits = restored.observe(first_evidence.to(device)).cpu()
    if not torch.equal(restored_logits, first_logits):
        raise RuntimeError("checkpoint reload changed the validation logits")
    targets = np.concatenate(targets)
    metrics = {
        "dataset": config["dataset"], "quick": quick,
        "validation_records": records,
        "evidence": classification_metrics(targets, np.concatenate(evidence_predictions)),
        "somnoorbit": classification_metrics(targets, np.concatenate(predictions)),
        "checkpoint_reload_equal": True,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_demo(dataset: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train SomnoOrbit on {dataset.upper()}.")
    parser.add_argument("--data-root", type=Path, required=True, help="Preprocessed dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / dataset)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / f"{dataset}.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--workers", type=int, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, help="Model initialization and training seed.")
    parser.add_argument("--quick", action="store_true", help="Run one epoch on a small real-data subset and 60-epoch orbit inputs.")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"dataset directory does not exist: {args.data_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset"].lower() != dataset:
        parser.error(f"configuration dataset must be {dataset}")
    if args.quick:
        config["epochs"] = 1
        config["workers"] = 0
    for key in ("epochs", "workers", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["epochs"] < 1 or config["workers"] < 0:
        parser.error("epochs must be positive and workers must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}; choose a new --output-dir")
    value = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    set_seed(config["seed"])
    train, validation, selected, weights = prepare_data(args.data_root, config, args.quick, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config)
    print(json.dumps({"dataset": dataset, "device": str(device), "quick": args.quick,
                      "train_windows": len(train), "validation_windows": len(validation)}), flush=True)
    backbone, checkpoint = train_backbone(train, validation, weights, config, device, args.output_dir)
    metrics = fit_and_evaluate(backbone, selected, args.data_root, config, args.quick,
                               device, args.output_dir, checkpoint)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved model: {args.output_dir / 'somnoorbit.pt'}", flush=True)
