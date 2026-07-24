"""Per-timestep particle-filter health diagnostics.

Computed at every time step during the filter pass (not as a post-hoc
approximation), these feed both the adaptive-resampling decision and the
reported diagnostics in :mod:`dnlssm.diagnostics.robustness`: Effective
Sample Size, (normalized) weight entropy, and an explicit degeneracy flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """``ESS = 1 / sum(w_i^2)`` for normalized weights; ranges from 1 (fully degenerate) to N (uniform)."""
    return float(1.0 / np.sum(weights**2))


def weight_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of the normalized weight distribution, in nats."""
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log(positive)))


def normalized_weight_entropy(weights: np.ndarray) -> float:
    """Entropy divided by ``log(N)``: 1.0 for uniform weights, -> 0 as weights collapse onto one particle."""
    n = weights.shape[0]
    if n <= 1:
        return 1.0
    return weight_entropy(weights) / np.log(n)


@dataclass
class ParticleDiagnosticsSummary:
    """Aggregated degeneracy diagnostics over a full filter run, for the experiment report."""

    ess: np.ndarray  # (T,)
    ess_ratio: np.ndarray  # (T,), ess / n_particles
    weight_entropy: np.ndarray  # (T,)
    normalized_weight_entropy: np.ndarray  # (T,)
    resampled_at: np.ndarray  # (T,) bool
    n_resample_events: int
    degeneracy_threshold_ratio: float
    degenerate_timesteps: list[int]

    @property
    def mean_ess_ratio(self) -> float:
        return float(np.mean(self.ess_ratio))

    @property
    def min_ess_ratio(self) -> float:
        return float(np.min(self.ess_ratio))

    @property
    def fraction_degenerate(self) -> float:
        return len(self.degenerate_timesteps) / len(self.ess_ratio) if len(self.ess_ratio) else 0.0

    def summary_text(self) -> str:
        lines = [
            "Particle filter degeneracy diagnostics:",
            f"  mean ESS ratio:        {self.mean_ess_ratio:.3f}",
            f"  min ESS ratio:         {self.min_ess_ratio:.3f}",
            f"  resampling events:     {self.n_resample_events} / {len(self.resampled_at)} timesteps",
            f"  degenerate timesteps:  {len(self.degenerate_timesteps)} "
            f"({100 * self.fraction_degenerate:.1f}%, ESS ratio < {self.degeneracy_threshold_ratio})",
        ]
        if self.fraction_degenerate > 0.1:
            lines.append(
                "  WARNING: >10% of timesteps show severe particle degeneracy. Consider a "
                "larger n_particles, a less peaked observation noise, or a different proposal "
                "(e.g. an auxiliary/guided particle filter in a future extension)."
            )
        return "\n".join(lines)


def summarize_particle_diagnostics(
    ess: np.ndarray,
    weight_entropy_series: np.ndarray,
    normalized_weight_entropy_series: np.ndarray,
    resampled_at: np.ndarray,
    n_particles: int,
    degeneracy_threshold_ratio: float = 0.1,
) -> ParticleDiagnosticsSummary:
    ess_ratio = ess / n_particles
    degenerate_timesteps = [int(t) for t in np.where(ess_ratio < degeneracy_threshold_ratio)[0]]
    return ParticleDiagnosticsSummary(
        ess=ess,
        ess_ratio=ess_ratio,
        weight_entropy=weight_entropy_series,
        normalized_weight_entropy=normalized_weight_entropy_series,
        resampled_at=resampled_at,
        n_resample_events=int(np.sum(resampled_at)),
        degeneracy_threshold_ratio=degeneracy_threshold_ratio,
        degenerate_timesteps=degenerate_timesteps,
    )


__all__ = [
    "effective_sample_size",
    "weight_entropy",
    "normalized_weight_entropy",
    "ParticleDiagnosticsSummary",
    "summarize_particle_diagnostics",
]
