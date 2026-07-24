"""Resampling schemes for the Bootstrap Particle Filter.

All four standard schemes are implemented (rather than only systematic)
because different schemes trade off resampling variance against
computational simplicity differently, and the platform's robustness
analysis (:mod:`dnlssm.diagnostics.robustness`) needs the ability to compare
them empirically rather than assert a single choice is best.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dnlssm.config.schema import ResamplingLiteral


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform draw, evenly spaced offsets. Lowest-variance scheme."""
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point drift leaving cumulative[-1] < 1
    return np.searchsorted(cumulative, positions)


def stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stratified resampling: one uniform draw per stratum of width ``1/n``."""
    n = weights.shape[0]
    positions = (rng.random(n) + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def multinomial_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Plain multinomial resampling: ``n`` i.i.d. draws from ``Categorical(weights)``."""
    n = weights.shape[0]
    return np.asarray(rng.choice(n, size=n, replace=True, p=weights))


def residual_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Residual resampling: deterministic integer copies plus multinomial resampling of the remainder."""
    n = weights.shape[0]
    scaled = n * weights
    n_copies = np.floor(scaled).astype(int)
    deterministic_indices = np.repeat(np.arange(n), n_copies)

    n_residual = n - deterministic_indices.shape[0]
    if n_residual == 0:
        return deterministic_indices

    residual_weights = scaled - n_copies
    residual_weights_sum = residual_weights.sum()
    if residual_weights_sum <= 0:
        # Degenerate edge case (all mass exactly on integer copies); fall back to uniform.
        residual_weights = np.full(n, 1.0 / n)
    else:
        residual_weights = residual_weights / residual_weights_sum
    extra_indices = rng.choice(n, size=n_residual, replace=True, p=residual_weights)
    return np.concatenate([deterministic_indices, extra_indices])


_RESAMPLERS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "multinomial": multinomial_resample,
    "residual": residual_resample,
}


def resample(weights: np.ndarray, method: ResamplingLiteral, rng: np.random.Generator) -> np.ndarray:
    """Dispatches to the configured resampling scheme; returns an ``(n,)`` array of parent indices."""
    if method not in _RESAMPLERS:
        raise ValueError(f"Unknown resampling method: {method!r}; expected one of {list(_RESAMPLERS)}.")
    return _RESAMPLERS[method](weights, rng)


__all__ = [
    "systematic_resample",
    "stratified_resample",
    "multinomial_resample",
    "residual_resample",
    "resample",
]
