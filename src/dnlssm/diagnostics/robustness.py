"""Robustness analysis: particle-count and observation-noise-structure sensitivity.

Both analyses re-run real computation (never simulate or assert a
conclusion): particle-count sensitivity re-runs the already-fitted model
through the particle filter at each configured particle count; noise-
structure sensitivity re-fits the model by MLE under each alternative
noise-covariance structure and compares the resulting information
criteria. This directly answers the research design's requirement to
empirically test -- not merely assume -- that the chosen ``n_particles``
and noise structure are adequate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from dnlssm.config.schema import (
    ModelConfig,
    NoiseStructureLiteral,
    OptimizationConfig,
    ParticleFilterConfig,
)
from dnlssm.filters.particle_filter import BootstrapParticleFilter
from dnlssm.model_selection.information_criteria import compute_information_criteria
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.models.dnlssm import DNLSSM
from dnlssm.optimization.mle import MLEEstimator
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.random_state import SeedSequence

logger = get_logger(__name__)

_DEFAULT_RELATIVE_LL_STD_WARNING_THRESHOLD = 0.05


@dataclass
class ParticleCountSensitivityResult:
    """Stability of filtered estimates and log-likelihood across particle counts."""

    particle_counts: list[int]
    log_likelihood_by_count: dict[int, float]
    ess_ratio_mean_by_count: dict[int, float]
    filtered_mean_max_abs_diff_from_largest: dict[int, float]
    log_likelihood_std: float
    log_likelihood_relative_std: float
    unstable: bool
    comparison_table: pd.DataFrame

    def summary_text(self) -> str:
        lines = [
            f"Particle-count sensitivity across {self.particle_counts}:",
            f"  log-likelihood relative std: {self.log_likelihood_relative_std:.4f}",
        ]
        if self.unstable:
            lines.append(
                "  WARNING: log-likelihood varies materially with n_particles. Results at the "
                "configured default n_particles may not have converged in the particle-count "
                "sense; consider increasing n_particles for the primary analysis."
            )
        else:
            lines.append("  Results are stable across the tested particle counts.")
        return "\n".join(lines)


def analyze_particle_count_sensitivity(
    model: AbstractStateSpaceModel,
    theta: np.ndarray,
    observations: np.ndarray,
    base_pf_config: ParticleFilterConfig,
    particle_counts: list[int],
    seed_sequence: SeedSequence,
    relative_std_warning_threshold: float = _DEFAULT_RELATIVE_LL_STD_WARNING_THRESHOLD,
) -> ParticleCountSensitivityResult:
    """Re-runs the particle filter at each of ``particle_counts`` and compares results."""
    if not particle_counts:
        raise ValueError("particle_counts must be non-empty.")

    pf = BootstrapParticleFilter(model, base_pf_config)
    log_likelihoods: dict[int, float] = {}
    filtered_means: dict[int, np.ndarray] = {}
    ess_ratio_means: dict[int, float] = {}

    for n in sorted(particle_counts):
        rng = seed_sequence.spawn("particle_count_sensitivity", n_particles=n)
        result = pf.run(observations, theta, rng, n_particles=n)
        log_likelihoods[n] = result.log_likelihood
        filtered_means[n] = result.filtered_mean
        ess_ratio_means[n] = float(np.mean(result.diagnostics.ess_ratio))
        logger.info("Particle-count sensitivity: N=%d -> log-likelihood=%.4f", n, result.log_likelihood)

    reference_n = max(particle_counts)
    reference_mean = filtered_means[reference_n]
    max_abs_diff = {
        n: float(np.max(np.abs(filtered_means[n] - reference_mean))) for n in particle_counts
    }

    ll_values = np.array(list(log_likelihoods.values()))
    ll_std = float(np.std(ll_values))
    ll_mean_abs = float(np.mean(np.abs(ll_values)))
    relative_std = ll_std / ll_mean_abs if ll_mean_abs > 0 else float("inf")

    comparison_table = pd.DataFrame(
        {
            "n_particles": sorted(particle_counts),
            "log_likelihood": [log_likelihoods[n] for n in sorted(particle_counts)],
            "mean_ess_ratio": [ess_ratio_means[n] for n in sorted(particle_counts)],
            "filtered_mean_max_abs_diff_from_largest_n": [max_abs_diff[n] for n in sorted(particle_counts)],
        }
    )

    return ParticleCountSensitivityResult(
        particle_counts=sorted(particle_counts),
        log_likelihood_by_count=log_likelihoods,
        ess_ratio_mean_by_count=ess_ratio_means,
        filtered_mean_max_abs_diff_from_largest=max_abs_diff,
        log_likelihood_std=ll_std,
        log_likelihood_relative_std=relative_std,
        unstable=relative_std > relative_std_warning_threshold,
        comparison_table=comparison_table,
    )


@dataclass
class NoiseStructureSensitivityResult:
    """Comparative MLE fit under each alternative noise-covariance structure."""

    target: Literal["process_noise", "observation_noise"]
    structures_tried: list[NoiseStructureLiteral]
    comparison_table: pd.DataFrame
    best_structure_by_aic: str
    best_structure_by_bic: str
    arch_effects_caveat: str = field(
        default=(
            "None of these structures model time-varying (conditional) variance; if "
            "diagnostics.heteroskedasticity flags significant ARCH effects, no structure "
            "compared here resolves that -- it would require a stochastic-volatility or "
            "time-varying-noise model family, which is a candidate future extension, not "
            "part of the current model family."
        )
    )


def analyze_noise_structure_sensitivity(
    model_config_template: ModelConfig,
    latent_dim: int,
    observation_labels: list[str],
    pf_config: ParticleFilterConfig,
    opt_config: OptimizationConfig,
    observations: np.ndarray,
    target: Literal["process_noise", "observation_noise"],
    structures: list[NoiseStructureLiteral],
    seed_sequence: SeedSequence,
) -> NoiseStructureSensitivityResult:
    """Refits the model by MLE under each candidate noise structure for ``target`` and compares fit."""
    if not structures:
        raise ValueError("structures must be non-empty.")

    n_observed = int(np.sum(~np.isnan(observations)))
    rows = []

    for structure in structures:
        noise_block = getattr(model_config_template, target).model_copy(update={"structure": structure})
        model_config = model_config_template.model_copy(update={target: noise_block})
        model = DNLSSM(latent_dim, observation_labels, model_config)

        struct_seed = seed_sequence.spawn_seed_int("noise_structure_sensitivity", target=target, structure=structure)
        mle_result = MLEEstimator(model, pf_config, opt_config, SeedSequence(struct_seed)).fit(observations)
        ic = compute_information_criteria(mle_result.best_log_likelihood, model.n_params, n_observed)

        logger.info(
            "Noise structure sensitivity (%s=%s): converged=%s, log-lik=%.3f, AIC=%.2f, BIC=%.2f",
            target,
            structure,
            mle_result.converged,
            ic.log_likelihood,
            ic.aic,
            ic.bic,
        )
        rows.append(
            {
                "structure": structure,
                "n_params": model.n_params,
                "converged": mle_result.converged,
                "log_likelihood": ic.log_likelihood,
                "aic": ic.aic,
                "bic": ic.bic,
            }
        )

    comparison_table = pd.DataFrame(rows)
    best_aic = str(comparison_table.loc[comparison_table["aic"].idxmin(), "structure"])
    best_bic = str(comparison_table.loc[comparison_table["bic"].idxmin(), "structure"])

    return NoiseStructureSensitivityResult(
        target=target,
        structures_tried=structures,
        comparison_table=comparison_table,
        best_structure_by_aic=best_aic,
        best_structure_by_bic=best_bic,
    )


__all__ = [
    "ParticleCountSensitivityResult",
    "analyze_particle_count_sensitivity",
    "NoiseStructureSensitivityResult",
    "analyze_noise_structure_sensitivity",
]
