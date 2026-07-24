"""Hessian-based asymptotic confidence intervals for the MLE parameter vector.

At the maximum-likelihood estimate, the inverse Hessian of the negative
log-likelihood is the standard asymptotic approximation to the sampling
covariance of the estimator (the observed-information approximation to the
Cramer-Rao bound). This module computes that Hessian numerically (via
``numdifftools``, which uses Richardson extrapolation for better accuracy
than a single-step finite difference), regularizes and symmetrizes it
before inversion, and reports per-parameter standard errors and confidence
intervals -- or, if the Hessian cannot be reliably inverted (a real
possibility given the particle-filter objective's Monte Carlo noise and
mild non-smoothness), reports that fact explicitly rather than fabricating
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from dnlssm.optimization.objective import ParticleFilterObjective
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.numerical import regularize_covariance

logger = get_logger(__name__)

_FLAT_HESSIAN_ABS_THRESHOLD = 1e-8


@dataclass
class ConfidenceIntervalResult:
    """Standard errors, confidence intervals, and the Hessian they derive from."""

    available: bool
    standard_errors: np.ndarray | None
    confidence_intervals: np.ndarray | None  # (n_params, 2)
    hessian: np.ndarray | None
    confidence_level: float
    failure_reason: str | None = None


def compute_hessian_confidence_intervals(
    objective: ParticleFilterObjective,
    theta_hat: np.ndarray,
    confidence_level: float,
    step_size: float,
) -> ConfidenceIntervalResult:
    """Computes Hessian-based standard errors and confidence intervals at ``theta_hat``."""
    try:
        import numdifftools as nd

        hessian_func = nd.Hessian(objective, step=step_size)
        hessian = np.asarray(hessian_func(theta_hat), dtype=np.float64)
    except Exception as exc:
        message = f"Numerical Hessian computation failed: {exc}"
        logger.warning(message)
        return ConfidenceIntervalResult(False, None, None, None, confidence_level, message)

    if not np.all(np.isfinite(hessian)):
        message = "Numerical Hessian contains non-finite entries; the likelihood surface may be too noisy at this step size."
        logger.warning(message)
        return ConfidenceIntervalResult(False, None, None, hessian, confidence_level, message)

    hessian_sym = 0.5 * (hessian + hessian.T)
    if np.max(np.abs(hessian_sym)) < _FLAT_HESSIAN_ABS_THRESHOLD:
        message = (
            "The numerical Hessian is essentially flat (all entries near zero) at this parameter "
            "point: the objective shows no detectable curvature at the given step_size, so a "
            "diagonal-jitter regularization would produce a finite but scientifically meaningless "
            "confidence interval. This can indicate local non-identifiability, or that step_size "
            "is too small relative to the likelihood surface's curvature scale."
        )
        logger.warning(message)
        return ConfidenceIntervalResult(False, None, None, hessian_sym, confidence_level, message)
    try:
        covariance = np.linalg.inv(regularize_covariance(hessian_sym, epsilon=1e-8))
    except np.linalg.LinAlgError as exc:
        message = f"Hessian is singular and could not be inverted: {exc}"
        logger.warning(message)
        return ConfidenceIntervalResult(False, None, None, hessian_sym, confidence_level, message)

    variances = np.diag(covariance)
    if np.any(variances < 0):
        logger.warning(
            "Hessian-based covariance has %d negative diagonal entries (the Hessian is not "
            "positive definite at this point, suggesting a saddle point or a flat direction in "
            "the likelihood); standard errors for those parameters are reported as NaN.",
            int(np.sum(variances < 0)),
        )
    standard_errors = np.sqrt(np.where(variances >= 0, variances, np.nan))

    z_score = stats.norm.ppf(0.5 + confidence_level / 2)
    lower = theta_hat - z_score * standard_errors
    upper = theta_hat + z_score * standard_errors
    confidence_intervals = np.column_stack([lower, upper])

    return ConfidenceIntervalResult(
        available=True,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        hessian=hessian_sym,
        confidence_level=confidence_level,
    )


__all__ = ["ConfidenceIntervalResult", "compute_hessian_confidence_intervals"]
