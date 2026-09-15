from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .legacy import SomnoPhaseObserver


@dataclass(frozen=True)
class OrbitConfig:
    period: float = 200.0
    rem_threshold: float = 0.7
    release_ratio: float = 0.6
    clock_strength: float = 0.8
    oscillator_strength: float = 0.15
    entropy_exponent: float = 1.0
    eeg_withdrawal_exponent: float | None = None
    interaction_strength: float = 0.0
    homeostatic_relaxation: float = 0.02
    radial_restoration: float = 0.05
    slow_fast_coupling: float = 0.0
    homeostatic_gain: float = 0.0
    phase_gain: float = 0.0
    radius_gain: float = 0.0
    residual_strength: float = 1.0
    lock_floor: float = 0.01
    homeostatic_step_bound: float = 0.02
    phase_step_bound: float = 0.02
    radius_step_bound: float = 0.02
    max_homeostatic_shift: float = 0.25
    max_phase_shift: float = math.pi / 3.0
    min_radius: float = 0.25
    max_radius: float = 1.50

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if not 0.0 < self.release_ratio < 1.0:
            raise ValueError("release_ratio must lie in (0, 1)")
        if not 0.0 < self.rem_threshold < 1.0:
            raise ValueError("rem_threshold must lie in (0, 1)")
        if not 0.0 <= self.homeostatic_relaxation <= 1.0:
            raise ValueError("homeostatic_relaxation must lie in [0, 1]")
        if not 0.0 <= self.radial_restoration <= 0.25:
            raise ValueError("radial_restoration must lie in [0, 0.25]")
        if self.entropy_exponent <= 0.0:
            raise ValueError("entropy_exponent must be positive")
        if (
            self.eeg_withdrawal_exponent is not None
            and self.eeg_withdrawal_exponent <= 0.0
        ):
            raise ValueError("eeg_withdrawal_exponent must be positive when set")
        if min(
            self.homeostatic_gain,
            self.phase_gain,
            self.radius_gain,
            self.homeostatic_step_bound,
            self.phase_step_bound,
            self.radius_step_bound,
            self.max_phase_shift,
        ) < 0.0:
            raise ValueError("observer gains and step bounds must be non-negative")
        if not 0.0 < self.min_radius <= 1.0 <= self.max_radius:
            raise ValueError("radius bounds must contain the unit reference orbit")

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    @property
    def zero_deformation(self) -> bool:
        return (
            self.homeostatic_gain == 0.0
            and self.phase_gain == 0.0
            and self.radius_gain == 0.0
            and self.slow_fast_coupling == 0.0
        )


def _rotate(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cosine, sine = angle.cos(), angle.sin()
    first, second = vector[:, 0], vector[:, 1]
    return torch.stack(
        (cosine * first - sine * second, sine * first + cosine * second),
        dim=-1,
    )


class SomnoOrbitObserver(nn.Module):
    """Reference-locked, evidence-coupled physiological orbit observer.

    The SomnoPhase observer supplies the reference path.  The deformation
    states are a slow displacement ``delta_s`` and a two-dimensional
    reference-frame oscillator ``r``.  Radius and phase shift are derived
    from ``r``.
    """

    def __init__(
        self,
        clock_template: torch.Tensor,
        oscillator_template: torch.Tensor,
        interaction_template: torch.Tensor,
        config: OrbitConfig | None = None,
    ):
        super().__init__()
        self.config = config or OrbitConfig()
        self.legacy = SomnoPhaseObserver(
            clock_template,
            oscillator_template,
            period=self.config.period,
            rem_threshold=self.config.rem_threshold,
            release_ratio=self.config.release_ratio,
            clock_strength=self.config.clock_strength,
            oscillator_strength=self.config.oscillator_strength,
            entropy_exponent=self.config.entropy_exponent,
        )
        interaction = torch.as_tensor(interaction_template, dtype=torch.float32)
        if interaction.ndim != 2 or interaction.shape[1] != 5:
            raise ValueError("interaction_template must have shape [bins, 5]")
        interaction = interaction - interaction.mean(-1, keepdim=True)
        self.register_buffer("interaction_template", interaction)

    @staticmethod
    def _clock_position(mask: torch.Tensor) -> torch.Tensor:
        batch, steps = mask.shape
        lengths = mask.sum(-1).clamp_min(1)
        denominator = (lengths - 1).clamp_min(1)
        return torch.arange(steps, device=mask.device)[None] / denominator[:, None]

    @staticmethod
    def _linear_field(
        table: torch.Tensor, coordinate: torch.Tensor, *, periodic: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bins = table.shape[0]
        if periodic:
            normalized = torch.remainder(coordinate, 2.0 * math.pi) / (2.0 * math.pi)
            position = normalized * bins
            lower = torch.floor(position).long() % bins
            upper = (lower + 1) % bins
            fraction = position - torch.floor(position)
            derivative_scale = bins / (2.0 * math.pi)
        else:
            normalized = coordinate.clamp(0.0, 1.0)
            position = normalized * (bins - 1)
            lower = torch.floor(position).long().clamp(0, bins - 1)
            upper = (lower + 1).clamp(0, bins - 1)
            fraction = position - torch.floor(position)
            derivative_scale = bins - 1
        lower_value, upper_value = table[lower], table[upper]
        value = lower_value + fraction[:, None] * (upper_value - lower_value)
        derivative = (upper_value - lower_value) * derivative_scale
        return value, derivative

    def _stage_field(
        self, slow: torch.Tensor, phase: torch.Tensor, radius: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        clock, dclock_du = self._linear_field(
            self.legacy.clock_template, 1.0 - slow, periodic=False
        )
        oscillator, doscillator = self._linear_field(
            self.legacy.oscillator_template, phase, periodic=True
        )
        interaction, dinteraction = self._linear_field(
            self.interaction_template, phase, periodic=True
        )
        centered_slow = slow - 0.5
        field = (
            cfg.clock_strength * clock
            + cfg.oscillator_strength * radius[:, None] * oscillator
            + cfg.interaction_strength
            * radius[:, None]
            * centered_slow[:, None]
            * interaction
        )
        derivative_slow = (
            -cfg.clock_strength * dclock_du
            + cfg.interaction_strength * radius[:, None] * interaction
        )
        derivative_phase = (
            cfg.oscillator_strength * radius[:, None] * doscillator
            + cfg.interaction_strength
            * radius[:, None]
            * centered_slow[:, None]
            * dinteraction
        )
        derivative_radius = (
            cfg.oscillator_strength * oscillator
            + cfg.interaction_strength * centered_slow[:, None] * interaction
        )
        return field, derivative_slow, derivative_phase, derivative_radius

    def forward(
        self,
        evidence_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        diagnostics: bool = False,
        entropy_override: torch.Tensor | float | None = None,
        readout_entropy_override: torch.Tensor | float | None = None,
        correction_posterior: torch.Tensor | None = None,
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

        legacy = self.legacy(evidence_logits, mask, diagnostics=True)
        base_logits = legacy["logits"]
        cfg = self.config
        if cfg.zero_deformation and cfg.eeg_withdrawal_exponent is None:
            if not diagnostics:
                return base_logits[0] if squeeze else base_logits
            result = dict(legacy)
            zeros = evidence_logits.new_zeros(evidence_logits.shape[:2])
            ones = evidence_logits.new_ones(evidence_logits.shape[:2])
            result.update(
                {
                    "delta_s": zeros,
                    "phase_shift": zeros,
                    "radius": ones,
                    "orbit_precision": ones,
                    "orbit_residual": evidence_logits.new_zeros(evidence_logits.shape),
                    "evidence_authority": ones,
                    "orbit_authority": legacy["evidence_entropy"]
                    .clamp_min(1e-4)
                    .pow(cfg.entropy_exponent),
                }
            )
            if squeeze:
                result = {key: value[0] for key, value in result.items()}
            return result

        probability = evidence_logits.softmax(-1)
        if correction_posterior is None:
            correction_probability = probability
        else:
            correction_probability = correction_posterior.to(
                device=probability.device, dtype=probability.dtype
            )
            if correction_probability.ndim == 2 and squeeze:
                correction_probability = correction_probability.unsqueeze(0)
            if correction_probability.shape != probability.shape:
                raise ValueError("correction_posterior must match evidence logits shape")
            correction_probability = correction_probability.clamp_min(0.0)
            correction_probability = correction_probability / correction_probability.sum(
                -1, keepdim=True
            ).clamp_min(1e-8)
        entropy = legacy["evidence_entropy"]
        if entropy_override is not None:
            override = torch.as_tensor(
                entropy_override,
                dtype=entropy.dtype,
                device=entropy.device,
            )
            if override.ndim == 0:
                override = override.expand_as(entropy)
            else:
                override = torch.broadcast_to(override, entropy.shape)
            entropy = override.clamp(0.0, 1.0)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        readout_entropy = entropy
        if readout_entropy_override is not None:
            override = torch.as_tensor(
                readout_entropy_override,
                dtype=entropy.dtype,
                device=entropy.device,
            )
            if override.ndim == 0:
                override = override.expand_as(entropy)
            else:
                override = torch.broadcast_to(override, entropy.shape)
            readout_entropy = override.clamp(0.0, 1.0)
        base_gain = readout_entropy.clamp_min(1e-4).pow(cfg.entropy_exponent)
        if cfg.eeg_withdrawal_exponent is None:
            evidence_authority = torch.ones_like(entropy)
        else:
            evidence_authority = 1.0 - readout_entropy.pow(
                cfg.eeg_withdrawal_exponent
            )
        base_message = (
            cfg.clock_strength * legacy["clock"]
            + cfg.oscillator_strength * legacy["oscillator"]
        )
        position = self._clock_position(mask)
        reference_slow = 1.0 - position
        reference_phase = legacy["phase"] * (2.0 * math.pi)

        batch, steps, _ = evidence_logits.shape
        delta_s = evidence_logits.new_zeros(batch)
        oscillator_state = evidence_logits.new_zeros(batch, 2)
        oscillator_state[:, 0] = 1.0
        (
            output,
            slow_states,
            phase_shifts,
            radii,
            precisions,
            residuals,
            evidence_authorities,
            orbit_authorities,
        ) = (
            [], [], [], [], [], [], [], []
        )

        for time in range(steps):
            valid = mask[:, time]
            predicted_delta_s = (1.0 - cfg.homeostatic_relaxation) * delta_s
            radius_squared = oscillator_state.square().sum(-1)
            predicted_r = oscillator_state + (
                cfg.radial_restoration
                * (1.0 - radius_squared)[:, None]
                * oscillator_state
            )
            predicted_r = _rotate(
                predicted_r, cfg.slow_fast_coupling * predicted_delta_s
            )
            predicted_radius = predicted_r.norm(dim=-1).clamp_min(1e-6)
            predicted_shift = torch.atan2(predicted_r[:, 1], predicted_r[:, 0])
            predicted_slow = (
                reference_slow[:, time] + predicted_delta_s
            ).clamp(0.0, 1.0)
            predicted_phase = reference_phase[:, time] + predicted_shift

            field, dslow, dphase, _ = self._stage_field(
                predicted_slow, predicted_phase, predicted_radius
            )
            physiological_probability = field.softmax(-1)
            innovation = correction_probability[:, time] - physiological_probability
            gradient_slow = (innovation * dslow).sum(-1)
            gradient_phase = (innovation * dphase).sum(-1)
            local_confidence = confidence[:, time]

            # Exponentiated KL divergence measures bounded agreement. Repeated
            # reliable conflicts reduce the phasor radius, while the Hopf
            # prediction restores it slowly.
            log_eeg = correction_probability[:, time].clamp_min(1e-8).log()
            log_physiology = physiological_probability.clamp_min(1e-8).log()
            agreement = torch.exp(
                -(correction_probability[:, time] * (log_eeg - log_physiology)).sum(-1)
            ).clamp(0.0, 1.0)

            slow_step = (
                cfg.homeostatic_gain * local_confidence * gradient_slow
            ).clamp(-cfg.homeostatic_step_bound, cfg.homeostatic_step_bound)
            phase_step = (
                cfg.phase_gain * local_confidence * gradient_phase
            ).clamp(-cfg.phase_step_bound, cfg.phase_step_bound)
            radius_step = (
                cfg.radius_gain
                * local_confidence
                * (agreement - predicted_radius)
            ).clamp(-cfg.radius_step_bound, cfg.radius_step_bound)
            corrected_delta_s = (
                predicted_delta_s + slow_step
            ).clamp(-cfg.max_homeostatic_shift, cfg.max_homeostatic_shift)
            radial_vector = predicted_r + (
                radius_step / predicted_radius
            )[:, None] * predicted_r
            corrected_r = _rotate(radial_vector, phase_step)
            corrected_norm = corrected_r.norm(dim=-1).clamp_min(1e-6)
            bounded_norm = corrected_norm.clamp(cfg.min_radius, cfg.max_radius)
            corrected_shift = torch.atan2(
                corrected_r[:, 1], corrected_r[:, 0]
            ).clamp(-cfg.max_phase_shift, cfg.max_phase_shift)
            corrected_r = bounded_norm[:, None] * torch.stack(
                (corrected_shift.cos(), corrected_shift.sin()), dim=-1
            )

            delta_s = torch.where(valid, corrected_delta_s, delta_s)
            oscillator_state = torch.where(valid[:, None], corrected_r, oscillator_state)
            current_radius = oscillator_state.norm(dim=-1).clamp_min(1e-6)
            current_shift = torch.atan2(oscillator_state[:, 1], oscillator_state[:, 0])
            current_slow = (reference_slow[:, time] + delta_s).clamp(0.0, 1.0)
            current_phase = reference_phase[:, time] + current_shift
            current_field = self._stage_field(
                current_slow, current_phase, current_radius
            )[0]
            reference_field = self._stage_field(
                reference_slow[:, time],
                reference_phase[:, time],
                torch.ones_like(current_radius),
            )[0]
            residual = cfg.residual_strength * (current_field - reference_field)
            precision = cfg.lock_floor + (1.0 - cfg.lock_floor) * current_radius.clamp(
                0.0, 1.0
            )
            gate = base_gain[:, time] * precision
            if cfg.eeg_withdrawal_exponent is None:
                # Exact compatibility path for the original one-sided readout.
                correction = (
                    (gate - base_gain[:, time])[:, None] * base_message[:, time]
                    + gate[:, None] * residual
                )
                current_output = base_logits[:, time] + correction
                current_orbit_authority = gate
            else:
                # Reciprocal uncertainty readout.  EEG uncertainty gradually
                # withdraws evidence log-odds while admitting the calibrated
                # reference-and-deformation orbit message.
                orbit_message = base_message[:, time] + residual
                current_orbit_authority = readout_entropy[:, time].pow(
                    cfg.entropy_exponent
                )
                current_output = (
                    evidence_authority[:, time, None] * evidence_logits[:, time]
                    + current_orbit_authority[:, None] * orbit_message
                )
            current_output = torch.where(
                valid[:, None], current_output, base_logits[:, time]
            )
            output.append(current_output)
            slow_states.append(torch.where(valid, delta_s, torch.zeros_like(delta_s)))
            phase_shifts.append(
                torch.where(valid, current_shift, torch.zeros_like(current_shift))
            )
            radii.append(torch.where(valid, current_radius, torch.ones_like(current_radius)))
            precisions.append(torch.where(valid, precision, torch.ones_like(precision)))
            residuals.append(torch.where(valid[:, None], residual, torch.zeros_like(residual)))
            evidence_authorities.append(
                torch.where(
                    valid,
                    evidence_authority[:, time],
                    torch.ones_like(evidence_authority[:, time]),
                )
            )
            orbit_authorities.append(
                torch.where(
                    valid,
                    current_orbit_authority,
                    torch.zeros_like(current_orbit_authority),
                )
            )

        logits = torch.stack(output, dim=1)
        if not diagnostics:
            return logits[0] if squeeze else logits
        result = dict(legacy)
        result.update(
            {
                "logits": logits,
                "delta_s": torch.stack(slow_states, dim=1),
                "phase_shift": torch.stack(phase_shifts, dim=1),
                "radius": torch.stack(radii, dim=1),
                "orbit_precision": torch.stack(precisions, dim=1),
                "orbit_residual": torch.stack(residuals, dim=1),
                "evidence_authority": torch.stack(evidence_authorities, dim=1),
                "orbit_authority": torch.stack(orbit_authorities, dim=1),
            }
        )
        if squeeze:
            result = {key: value[0] for key, value in result.items()}
        return result
