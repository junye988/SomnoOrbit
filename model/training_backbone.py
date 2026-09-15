from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EpochEncoder(nn.Module):
    """Compact raw-waveform observation encoder."""

    def __init__(self, channels: int = 6, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, 31, stride=5, padding=15, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 15, stride=4, padding=7, groups=width, bias=False),
            nn.Conv1d(width, 2 * width, 1, bias=False),
            nn.BatchNorm1d(2 * width), nn.GELU(),
            nn.Conv1d(2 * width, 2 * width, 9, stride=4, padding=4, groups=2 * width, bias=False),
            nn.BatchNorm1d(2 * width), nn.GELU(),
        )

    def forward(self, x):
        x = self.net(x)
        # Statistics pooling preserves both activation presence and energy.
        return torch.cat([x.mean(-1), x.std(-1, unbiased=False)], dim=1)


def physiological_cost() -> torch.Tensor:
    # Stage order is W, N1, N2, N3, REM. Costs are soft, not hard rules:
    # direct W<->N3 and N3<->REM jumps are least plausible at 30 s resolution.
    cost = torch.tensor([
        [0., 0., 1., 3., 1.],
        [0., 0., 0., 2., 0.],
        [1., 0., 0., 0., 0.],
        [3., 2., 0., 0., 3.],
        [1., 0., 0., 3., 0.],
    ])
    return cost


class TopologyCRF(nn.Module):
    def __init__(self, classes: int = 5, prior_strength: float = 0.7):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(classes, classes))
        self.start = nn.Parameter(torch.zeros(classes))
        self.end = nn.Parameter(torch.zeros(classes))
        self.prior_strength = prior_strength
        self.register_buffer("transition_cost", physiological_cost())

    @property
    def effective_transitions(self):
        return self.transitions - self.prior_strength * self.transition_cost

    def log_partition(self, emissions):
        score = self.start + emissions[:, 0]
        trans = self.effective_transitions
        for t in range(1, emissions.shape[1]):
            score = torch.logsumexp(score.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
        return torch.logsumexp(score + self.end, dim=1)

    def path_score(self, emissions, tags):
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        trans = self.effective_transitions
        for t in range(1, emissions.shape[1]):
            score = score + trans[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
        return score + self.end[tags[:, -1]]

    def nll(self, emissions, tags):
        return (self.log_partition(emissions) - self.path_score(emissions, tags)).mean()

    @torch.no_grad()
    def decode(self, emissions):
        score = self.start + emissions[:, 0]
        back = []
        trans = self.effective_transitions
        for t in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + trans.unsqueeze(0)
            best_score, best_prev = candidates.max(dim=1)
            score = best_score + emissions[:, t]
            back.append(best_prev)
        state = (score + self.end).argmax(dim=1)
        path = [state]
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        for prev in reversed(back):
            state = prev[batch, state]
            path.append(state)
        return torch.stack(path[::-1], dim=1)


class DurationTopologyCRF(nn.Module):
    """Segmental CRF: topology plus non-geometric stage-duration potentials."""

    def __init__(self, classes=5, max_duration=10, prior_strength=0.7):
        super().__init__()
        self.classes, self.max_duration = classes, max_duration
        self.transitions = nn.Parameter(torch.zeros(classes, classes))
        self.start = nn.Parameter(torch.zeros(classes))
        self.end = nn.Parameter(torch.zeros(classes))
        self.duration = nn.Parameter(torch.zeros(classes, max_duration))
        self.prior_strength = prior_strength
        self.register_buffer("transition_cost", physiological_cost())
        self.register_buffer("different_stage", (~torch.eye(classes, dtype=torch.bool)))

    @property
    def effective_transitions(self):
        return self.transitions - self.prior_strength * self.transition_cost

    def _segment_emission(self, prefix, start, end):
        return prefix[:, end] - prefix[:, start]

    def log_partition(self, emissions):
        b, length, classes = emissions.shape
        prefix = torch.cat([emissions.new_zeros(b, 1, classes), emissions.cumsum(1)], 1)
        alpha = [None] * (length + 1)
        trans = self.effective_transitions.masked_fill(~self.different_stage, -1e4)
        for end in range(1, length + 1):
            candidates = []
            for duration in range(1, min(self.max_duration, end) + 1):
                start = end - duration
                seg = self._segment_emission(prefix, start, end) + self.duration[:, duration - 1]
                if start == 0:
                    candidates.append(self.start + seg)
                else:
                    prev = torch.logsumexp(alpha[start].unsqueeze(2) + trans.unsqueeze(0), dim=1)
                    candidates.append(prev + seg)
            alpha[end] = torch.logsumexp(torch.stack(candidates), dim=0)
        return torch.logsumexp(alpha[length] + self.end, dim=1)

    def path_score(self, emissions, tags):
        scores = []
        for sample_emissions, sample_tags in zip(emissions, tags):
            score = sample_emissions.new_zeros(())
            start, previous = 0, None
            for end in range(1, len(sample_tags) + 1):
                boundary = end == len(sample_tags) or sample_tags[end] != sample_tags[start]
                if boundary:
                    label = sample_tags[start]
                    duration = end - start
                    score = score + sample_emissions[start:end, label].sum()
                    # Long runs are represented by adjacent capped segments.
                    remaining = duration
                    while remaining > 0:
                        chunk = min(remaining, self.max_duration)
                        score = score + self.duration[label, chunk - 1]
                        remaining -= chunk
                    score = score + (self.start[label] if previous is None else
                                     self.effective_transitions[previous, label])
                    previous, start = label, end
            scores.append(score + self.end[previous])
        return torch.stack(scores)

    def nll(self, emissions, tags):
        return (self.log_partition(emissions) - self.path_score(emissions, tags)).mean()

    @torch.no_grad()
    def decode(self, emissions):
        results = []
        trans = self.effective_transitions.masked_fill(~self.different_stage, -1e4)
        for e in emissions:
            length, classes = e.shape
            prefix = torch.cat([e.new_zeros(1, classes), e.cumsum(0)], 0)
            dp, back = [None] * (length + 1), [None] * (length + 1)
            for end in range(1, length + 1):
                best = e.new_full((classes,), -1e9); pointers = []
                best_ptr = [None] * classes
                for duration in range(1, min(self.max_duration, end) + 1):
                    start = end - duration
                    seg = prefix[end] - prefix[start] + self.duration[:, duration - 1]
                    if start == 0:
                        candidate, prev = self.start + seg, torch.full((classes,), -1, device=e.device)
                    else:
                        prev_score, prev = (dp[start].unsqueeze(1) + trans).max(0)
                        candidate = prev_score + seg
                    improve = candidate > best
                    for k in improve.nonzero().flatten().tolist():
                        best_ptr[k] = (start, int(prev[k]))
                    best = torch.maximum(best, candidate)
                dp[end], back[end] = best, best_ptr
            label = int((dp[length] + self.end).argmax())
            output = torch.empty(length, dtype=torch.long, device=e.device); end = length
            while end > 0:
                start, previous = back[end][label]
                output[start:end] = label
                end, label = start, previous
            results.append(output)
        return torch.stack(results)


class SomnoCRF(nn.Module):
    def __init__(self, channels=6, width=64, classes=5, structured=True, prior_strength=0.7,
                 local_anchor=1.0, duration_aware=False, max_duration=20,
                 trajectory_energy=0.0):
        super().__init__()
        self.structured = structured
        self.local_anchor = local_anchor
        self.trajectory_energy = trajectory_energy
        self.encoder = EpochEncoder(channels, width)
        self.context = nn.GRU(4 * width, width, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, classes))
        self.crf = TopologyCRF(classes, prior_strength)
        if duration_aware:
            self.crf = DurationTopologyCRF(classes, max_duration, prior_strength)

    def emissions(self, x):
        b, s, c, t = x.shape
        z = self.encoder(x.reshape(b * s, c, t)).reshape(b, s, -1)
        z, _ = self.context(z)
        return self.head(z)

    def loss(self, x, y, class_weights=None):
        emissions = self.emissions(x)
        if self.structured:
            # CRF sequence likelihood plus a small class-balanced local anchor.
            return self.crf.nll(emissions, y) + self.local_anchor * F.cross_entropy(
                emissions.flatten(0, 1), y.flatten(), weight=class_weights
            )
        local = F.cross_entropy(emissions.flatten(0, 1), y.flatten(), weight=class_weights)
        if self.trajectory_energy > 0:
            probabilities = emissions.softmax(-1)
            expected_cost = torch.einsum(
                "bti,ij,btj->bt", probabilities[:, :-1],
                self.crf.transition_cost, probabilities[:, 1:]
            ).mean()
            local = local + self.trajectory_energy * expected_cost
        return local

    @torch.no_grad()
    def predict(self, x):
        emissions = self.emissions(x)
        return self.crf.decode(emissions) if self.structured else emissions.argmax(-1)


# Expose SomnoEnergy while retaining the SomnoCRF class name for checkpoint loading.
SomnoEnergy = SomnoCRF
