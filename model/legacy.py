from __future__ import annotations

import math

import torch
from torch import nn


class SomnoPhaseObserver(nn.Module):
    r"""Evidence-locked physiological phase observer.

    The state has two physiological coordinates: normalized night progress
    and an ultradian phase reset by a Schmitt trigger on REM evidence.  Their
    class messages are fused in logit space with normalized evidence entropy
    as precision:

    .. math::
       z'_t = \bar z_t + h_t^\eta
       [\alpha c(u_t) + \beta o(\phi_t)].
    """

    def __init__(
        self,
        clock_template: torch.Tensor,
        oscillator_template: torch.Tensor,
        *,
        period: float = 200.0,
        rem_threshold: float = 0.7,
        release_ratio: float = 0.6,
        clock_strength: float = 0.8,
        oscillator_strength: float = 0.15,
        entropy_exponent: float = 1.0,
    ):
        super().__init__()
        clock_template = torch.as_tensor(clock_template, dtype=torch.float32)
        oscillator_template = torch.as_tensor(oscillator_template, dtype=torch.float32)
        if clock_template.ndim != 2 or clock_template.shape[1] != 5:
            raise ValueError("clock_template must have shape [bins, 5]")
        if oscillator_template.ndim != 2 or oscillator_template.shape[1] != 5:
            raise ValueError("oscillator_template must have shape [bins, 5]")
        self.register_buffer("clock_template", self._center(clock_template))
        self.register_buffer("oscillator_template", self._center(oscillator_template))
        self.register_buffer("clock_strength", torch.tensor(float(clock_strength)))
        self.register_buffer(
            "oscillator_strength", torch.tensor(float(oscillator_strength))
        )
        self.period = float(period)
        self.rem_threshold = float(rem_threshold)
        self.release_ratio = float(release_ratio)
        self.entropy_exponent = float(entropy_exponent)

    @staticmethod
    def _center(value: torch.Tensor) -> torch.Tensor:
        return value - value.mean(-1, keepdim=True)

    def _clock_state(self, mask: torch.Tensor) -> torch.Tensor:
        batch, steps = mask.shape
        lengths = mask.sum(-1).clamp_min(1)
        denominator = (lengths - 1).clamp_min(1)
        position = torch.arange(steps, device=mask.device)[None] / denominator[:, None]
        index = torch.round(position * (self.clock_template.shape[0] - 1)).long()
        index = index.clamp(0, self.clock_template.shape[0] - 1)
        state = self.clock_template[index]
        return torch.where(mask[:, :, None], state, torch.zeros_like(state))

    def _oscillator_state(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probability = logits.softmax(-1)
        batch, steps, _ = logits.shape
        age = logits.new_zeros(batch)
        in_rem = torch.zeros(batch, dtype=torch.bool, device=logits.device)
        states, phases = [], []
        release_threshold = self.release_ratio * self.rem_threshold
        for time in range(steps):
            valid = mask[:, time]
            rem = probability[:, time, 4]
            high = (rem > self.rem_threshold) & (probability[:, time].argmax(-1) == 4)
            release = in_rem & (rem < release_threshold)
            proposed_age = torch.where(
                high,
                torch.zeros_like(age),
                torch.where(
                    release,
                    torch.ones_like(age),
                    torch.where(~in_rem, age + 1.0, age),
                ),
            )
            proposed_latch = torch.where(
                high,
                torch.ones_like(in_rem),
                torch.where(release, torch.zeros_like(in_rem), in_rem),
            )
            age = torch.where(valid, proposed_age, age)
            in_rem = torch.where(valid, proposed_latch, in_rem)
            phase = torch.remainder(age, self.period) / self.period
            index = torch.floor(phase * self.oscillator_template.shape[0]).long()
            index = index.clamp(0, self.oscillator_template.shape[0] - 1)
            state = self.oscillator_template[index]
            states.append(torch.where(valid[:, None], state, torch.zeros_like(state)))
            phases.append(torch.where(valid, phase, torch.zeros_like(phase)))
        return torch.stack(states, dim=1), torch.stack(phases, dim=1)

    def forward(
        self,
        evidence_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        squeeze = evidence_logits.ndim == 2
        if squeeze:
            evidence_logits = evidence_logits.unsqueeze(0)
        if evidence_logits.ndim != 3 or evidence_logits.shape[-1] != 5:
            raise ValueError("evidence_logits must have shape [T, 5] or [B, T, 5]")
        evidence_logits = evidence_logits.float()
        if mask is None:
            mask = torch.ones(
                evidence_logits.shape[:2], dtype=torch.bool, device=evidence_logits.device
            )
        else:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(device=evidence_logits.device, dtype=torch.bool)
        clock = self._clock_state(mask)
        oscillator, phase = self._oscillator_state(evidence_logits, mask)
        probability = evidence_logits.softmax(-1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / math.log(5.0)
        gain = entropy.clamp_min(1e-4).pow(self.entropy_exponent)
        physiological_message = (
            self.clock_strength * clock
            + self.oscillator_strength * oscillator
        )
        correction = gain[:, :, None] * physiological_message
        output = evidence_logits + torch.where(
            mask[:, :, None], correction, torch.zeros_like(correction)
        )
        if squeeze:
            output = output[0]
            clock, oscillator, phase, entropy = (
                clock[0], oscillator[0], phase[0], entropy[0]
            )
        if diagnostics:
            return {
                "logits": output,
                "clock": clock,
                "oscillator": oscillator,
                "phase": phase,
                "evidence_entropy": entropy,
            }
        return output
