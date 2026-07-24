"""Numerically robust primitives shared across the filtering and optimization stack.

These functions exist to protect the platform against the standard failure
modes of state-space estimation: floating-point overflow/underflow in
importance weights, singular or near-singular covariance matrices, and NaN
propagation once a single such event occurs. Every place in the codebase
that would otherwise call ``np.exp``, ``np.linalg.cholesky``, or normalize a
set of weights goes through this module instead.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla


def log_sum_exp(log_values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable ``log(sum(exp(log_values)))``.

    Subtracting the running maximum before exponentiating is the standard
    log-sum-exp trick: it keeps every exponentiated term in ``(0, 1]``,
    eliminating the overflow that a naive ``np.log(np.sum(np.exp(x)))``
    would suffer whenever ``x`` contains large log-likelihood values (as is
    routine once dozens of Gaussian observation densities are multiplied
    together in a particle filter).
    """
    log_values = np.asarray(log_values, dtype=np.float64)
    max_val = np.max(log_values, axis=axis, keepdims=True)
    # If every entry is -inf (all particles have zero likelihood), avoid
    # producing nan from (-inf) - (-inf); the result should stay -inf.
    max_val_safe = np.where(np.isfinite(max_val), max_val, 0.0)
    summed = np.sum(np.exp(log_values - max_val_safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        # log(0) -> -inf is the intended, correctly-handled outcome for an
        # all -inf slice (masked out again on the next line); suppress the
        # spurious RuntimeWarning that would otherwise accompany it.
        result = max_val_safe + np.log(summed)
    result = np.where(np.isfinite(max_val), result, -np.inf)
    return np.squeeze(result) if axis is None else np.squeeze(result, axis=axis)


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalizes log-importance-weights to a proper probability simplex.

    Returns
    -------
    normalized_weights:
        ``exp(log_weights - logsumexp(log_weights))``, summing to 1.
    log_likelihood_increment:
        ``logsumexp(log_weights) - log(N)``, the log of the average
        importance weight -- the standard unbiased contribution of one time
        step to the particle filter's marginal log-likelihood estimate.
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    n_particles = log_weights.shape[0]
    lse = float(log_sum_exp(log_weights))
    if not np.isfinite(lse):
        raise FloatingPointError(
            "All particle log-weights are -inf: every particle received zero "
            "likelihood at this time step. This indicates the proposal (prior "
            "transition) has collapsed away from the observed data -- consider "
            "increasing process/observation noise, increasing n_particles, or "
            "checking the observation for outliers/unit mismatches."
        )
    normalized = np.exp(log_weights - lse)
    normalized /= normalized.sum()  # exact renormalization against residual float error
    log_likelihood_increment = lse - np.log(n_particles)
    return normalized, log_likelihood_increment


def regularize_covariance(cov: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Adds diagonal jitter to guarantee a covariance matrix is positive definite.

    ``epsilon`` is scaled by the matrix's own trace so the jitter is
    meaningful regardless of the natural scale of the variables (relevant
    here because standardized and un-standardized observation blocks can
    coexist during development/testing).
    """
    cov = np.asarray(cov, dtype=np.float64)
    dim = cov.shape[-1]
    trace_scale = max(float(np.trace(cov)) / dim, 1.0) if cov.ndim == 2 else 1.0
    jitter = epsilon * trace_scale
    return cov + jitter * np.eye(dim)


def safe_cholesky(cov: np.ndarray, epsilon: float = 1e-6, max_attempts: int = 6) -> np.ndarray:
    """Cholesky factorization with adaptive jitter escalation.

    Attempts ``scipy.linalg.cholesky`` on the (lightly) regularized matrix;
    if it still fails (matrix not positive definite due to accumulated
    floating point error or a genuinely singular parameter estimate), the
    jitter is geometrically increased up to ``max_attempts`` times before
    raising, so that a single ill-conditioned covariance does not silently
    propagate NaNs through the rest of the filter.
    """
    cov = np.asarray(cov, dtype=np.float64)
    current_epsilon = epsilon
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            regularized = regularize_covariance(cov, current_epsilon)
            return sla.cholesky(regularized, lower=True)
        except sla.LinAlgError as exc:
            last_error = exc
            current_epsilon *= 10.0
    raise sla.LinAlgError(
        f"Cholesky factorization failed after {max_attempts} jitter escalations "
        f"(final epsilon={current_epsilon:.2e}); the covariance matrix is severely "
        f"ill-conditioned. Last error: {last_error}"
    )


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(x))``, used to map unconstrained
    optimizer parameters onto strictly positive noise standard deviations
    without the overflow that a naive implementation exhibits for large ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.logaddexp(0.0, x)


def inverse_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`softplus`, mapping a positive value back to unconstrained space."""
    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0):
        raise ValueError("inverse_softplus requires strictly positive input.")
    return y + np.log(-np.expm1(-y))


__all__ = [
    "log_sum_exp",
    "normalize_log_weights",
    "regularize_covariance",
    "safe_cholesky",
    "softplus",
    "inverse_softplus",
]
