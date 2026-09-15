from __future__ import annotations

from pathlib import Path

import torch

from .backbone import ContextBackbone
from .model import SomnoOrbit
from .observer import OrbitConfig, SomnoOrbitObserver


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[SomnoOrbit, dict]:
    """Restore a self-contained SomnoOrbit checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    state = payload["model"]
    orbit_config = OrbitConfig(**config["orbit"])
    backbone = ContextBackbone(
        channels=int(config["channels"]),
        width=int(config["width"]),
        classes=int(config.get("classes", 5)),
    )
    observer = SomnoOrbitObserver(
        torch.zeros_like(state["observer.legacy.clock_template"]),
        torch.zeros_like(state["observer.legacy.oscillator_template"]),
        torch.zeros_like(state["observer.interaction_template"]),
        orbit_config,
    )
    model = SomnoOrbit(backbone, observer)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
