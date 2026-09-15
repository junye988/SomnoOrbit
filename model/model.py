from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .backbone import ContextBackbone
from .observer import SomnoOrbitObserver
from .polyphase import marginalize_context_phase


class SomnoOrbit(nn.Module):
    """Whole-night model with context marginalization and orbit observation."""

    def __init__(self, backbone: ContextBackbone, observer: SomnoOrbitObserver):
        super().__init__()
        self.backbone = backbone
        self.observer = observer

    def evidence_logits(self, windows: torch.Tensor) -> torch.Tensor:
        return self.backbone(windows)

    def observe(
        self,
        evidence_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.observer(evidence_logits, mask, diagnostics=diagnostics)

    def forward(
        self,
        night: np.ndarray | torch.Tensor,
        *,
        context: int = 20,
        stride: int = 2,
        batch_size: int = 8,
        diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        evidence, count = marginalize_context_phase(
            self.backbone,
            night,
            context=context,
            stride=stride,
            batch_size=batch_size,
            device=next(self.parameters()).device,
        )
        evidence = evidence.to(next(self.parameters()).device)
        result = self.observer(evidence, diagnostics=diagnostics)
        if diagnostics:
            result["polyphase_evidence"] = evidence
            result["context_count"] = count.to(evidence.device)
        return result

    @torch.inference_mode()
    def predict(self, night: np.ndarray | torch.Tensor, **kwargs) -> torch.Tensor:
        output = self.forward(night, **kwargs)
        if isinstance(output, dict):
            output = output["logits"]
        return output.argmax(-1)
