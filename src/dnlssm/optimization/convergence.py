"""Automatic diagnosis of optimization convergence failures.

The research design is explicit that a ``converged = False`` result must
never be reported as a bare fact -- the platform must investigate *why*.
:func:`diagnose_convergence_failure` applies a fixed set of rule-based
checks (iteration budget exhaustion, a high rate of numerically failed
objective evaluations, a large final gradient norm, divergent parameter
magnitude) against a completed optimization run and returns both the
identified causes and concrete, actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dnlssm.optimization.optimizers import SingleOptimizationResult

_LARGE_PARAM_MAGNITUDE_THRESHOLD = 50.0
_GRADIENT_STALL_MULTIPLE = 100.0
_HIGH_FAILURE_RATE_THRESHOLD = 0.05


@dataclass
class ConvergenceDiagnosis:
    """Identified likely causes of a non-converged optimization run, with recommendations."""

    converged: bool
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        if self.converged:
            return "Optimization converged; no failure diagnosis necessary."
        lines = ["Optimization did NOT converge. Automated diagnosis:"]
        for i, reason in enumerate(self.reasons, start=1):
            lines.append(f"  {i}. {reason}")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


def diagnose_convergence_failure(
    result: SingleOptimizationResult, gradient_tolerance: float, max_iterations: int
) -> ConvergenceDiagnosis:
    """Runs all rule-based checks against one non-converged optimization run."""
    if result.success:
        return ConvergenceDiagnosis(converged=True)

    reasons: list[str] = []
    recommendations: list[str] = []

    if result.n_iterations >= max_iterations:
        reasons.append(
            f"The iteration budget ({max_iterations}) was exhausted before the optimizer's "
            "own convergence tolerance was satisfied."
        )
        recommendations.append("Increase optimization.max_iterations in the experiment configuration.")

    if result.objective is not None and result.objective.n_evaluations > 0:
        failure_rate = result.objective.failure_rate
        if failure_rate > _HIGH_FAILURE_RATE_THRESHOLD:
            reasons.append(
                f"{100 * failure_rate:.1f}% of objective evaluations hit a numerical failure "
                "(most likely a near-singular noise covariance or complete particle degeneracy "
                "at that parameter point)."
            )
            recommendations.append(
                "Increase particle_filter.n_particles, increase "
                "model.process_noise.regularization_epsilon / observation_noise.regularization_epsilon, "
                "or reduce model.latent_dim."
            )

    if result.final_gradient_norm is not None and result.final_gradient_norm > _GRADIENT_STALL_MULTIPLE * gradient_tolerance:
        reasons.append(
            f"The final gradient norm ({result.final_gradient_norm:.3g}) is far above the configured "
            f"tolerance ({gradient_tolerance:.3g}), suggesting the optimizer stalled on a flat or "
            "noisy region of the likelihood surface rather than reaching a true stationary point."
        )
        recommendations.append(
            "Try a different optimizer (e.g. Nelder-Mead, which does not rely on a noisy "
            "finite-difference gradient), or increase n_particles to reduce Monte Carlo noise "
            "in the objective."
        )

    if np.max(np.abs(result.theta_final)) > _LARGE_PARAM_MAGNITUDE_THRESHOLD:
        reasons.append(
            "The final parameter vector has unusually large magnitude entries, which can indicate "
            "the optimizer diverged toward a degenerate region (e.g. a noise variance collapsing "
            "toward zero, or an unbounded transition-function weight)."
        )
        recommendations.append(
            "Inspect optimization.initial_param_scale and consider tighter regularization_epsilon "
            "values, or a bounded/constrained reparameterization for future extension."
        )

    if not reasons:
        reasons.append(
            "No specific automated cause was identified from iteration budget, evaluation failure "
            "rate, gradient norm, or parameter magnitude checks."
        )
        recommendations.append(
            "Inspect the per-iteration log-likelihood trace and residual diagnostics manually; "
            "consider more multi-start restarts (optimization.n_multistarts)."
        )

    return ConvergenceDiagnosis(converged=False, reasons=reasons, recommendations=recommendations)


__all__ = ["ConvergenceDiagnosis", "diagnose_convergence_failure"]
