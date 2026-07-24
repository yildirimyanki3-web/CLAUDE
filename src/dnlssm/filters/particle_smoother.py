"""Particle smoothing: backward-simulation and fixed-lag methods.

Both operate on the stored history from a completed
:class:`~dnlssm.filters.particle_filter.FilterResult` and produce a
*smoothed* latent-state estimate ``p(x_t | y_1:T)`` (using the entire
observation sequence), which is systematically more accurate than the
*filtered* estimate ``p(x_t | y_1:t)`` for any ``t < T``. Both filtered and
smoothed trajectories are always reported (never only one), as required by
the research design.

``backward_simulation`` (the default) is the FFBSi algorithm of Godsill,
Doucet & West (2004): it resamples full backward trajectories using the
model's transition density, giving asymptotically consistent smoothed
estimates at the cost of ``O(T * M * N)`` transition-density evaluations.
``fixed_lag`` is a much cheaper approximation that reuses the filter's
resampling genealogy, trading some accuracy (it only looks ``lag`` steps
ahead) for speed on very long series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.filters.particle_filter import FilterResult
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.numerical import gaussian_log_pdf, log_sum_exp

logger = get_logger(__name__)


@dataclass
class SmootherResult:
    """Smoothed latent-state posterior estimate over the full observation window."""

    smoothed_mean: np.ndarray  # (T, latent_dim)
    smoothed_cov: np.ndarray  # (T, latent_dim, latent_dim)
    method: str
    n_trajectories_or_lag: int


def _backward_simulation_smoother(
    model: AbstractStateSpaceModel,
    theta: np.ndarray,
    filter_result: FilterResult,
    n_backward_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_time, n_particles, latent_dim = filter_result.particle_history.shape
    n_traj = min(n_backward_samples, n_particles)
    process_noise_cov = model.process_noise_cov(theta)

    trajectories = np.empty((n_time, n_traj, latent_dim))
    final_idx = rng.choice(n_particles, size=n_traj, replace=True, p=filter_result.weight_history[-1])
    trajectories[-1] = filter_result.particle_history[-1, final_idx]

    for t in range(n_time - 2, -1, -1):
        predicted_means = model.transition_mean(filter_result.particle_history[t], theta)  # (N, d)
        residuals = trajectories[t + 1][:, None, :] - predicted_means[None, :, :]  # (M, N, d)
        flat_log_density = gaussian_log_pdf(
            residuals.reshape(n_traj * n_particles, latent_dim), process_noise_cov
        ).reshape(n_traj, n_particles)

        log_filter_weights = np.log(filter_result.weight_history[t] + np.finfo(float).tiny)
        log_back_weights = log_filter_weights[None, :] + flat_log_density  # (M, N)
        log_back_weights -= log_sum_exp(log_back_weights, axis=1)[:, None]

        # Gumbel-max trick: argmax(log_p + Gumbel noise) samples exactly from
        # Categorical(softmax(log_p)), fully vectorized over all M trajectories
        # at once instead of an M-iteration Python loop over rng.choice.
        uniform_noise = rng.random(size=log_back_weights.shape)
        gumbel_noise = -np.log(-np.log(uniform_noise))
        chosen_ancestors = np.argmax(log_back_weights + gumbel_noise, axis=1)

        trajectories[t] = filter_result.particle_history[t, chosen_ancestors]

    smoothed_mean = trajectories.mean(axis=1)
    centered = trajectories - smoothed_mean[:, None, :]
    smoothed_cov = np.einsum("tmi,tmj->tij", centered, centered) / n_traj
    return smoothed_mean, smoothed_cov


def _fixed_lag_smoother(filter_result: FilterResult, lag: int) -> tuple[np.ndarray, np.ndarray]:
    n_time, n_particles, latent_dim = filter_result.particle_history.shape
    smoothed_mean = np.empty((n_time, latent_dim))
    smoothed_cov = np.empty((n_time, latent_dim, latent_dim))

    for t in range(n_time):
        target_t = min(t + lag, n_time - 1)
        indices = np.arange(n_particles)
        for s in range(target_t - 1, t - 1, -1):
            indices = filter_result.ancestor_history[s][indices]

        traced_particles = filter_result.particle_history[t, indices]
        weights = filter_result.weight_history[target_t]
        mean = np.average(traced_particles, axis=0, weights=weights)
        centered = traced_particles - mean
        cov = (centered * weights[:, None]).T @ centered
        smoothed_mean[t] = mean
        smoothed_cov[t] = cov

    return smoothed_mean, smoothed_cov


class ParticleSmoother:
    """Dispatches to the configured smoothing method."""

    def __init__(self, model: AbstractStateSpaceModel, config: ParticleFilterConfig) -> None:
        self._model = model
        self._config = config

    def run(self, filter_result: FilterResult, theta: np.ndarray, rng: np.random.Generator) -> SmootherResult:
        method = self._config.smoother_method
        if method == "backward_simulation":
            mean, cov = _backward_simulation_smoother(
                self._model, theta, filter_result, self._config.smoother_n_backward_samples, rng
            )
            n_used = min(self._config.smoother_n_backward_samples, filter_result.n_particles)
        elif method == "fixed_lag":
            mean, cov = _fixed_lag_smoother(filter_result, self._config.fixed_lag_window)
            n_used = self._config.fixed_lag_window
        else:
            raise ValueError(f"Unknown smoother_method: {method!r}")

        logger.debug("Particle smoother ('%s') complete over %d timesteps.", method, filter_result.n_timesteps)
        return SmootherResult(smoothed_mean=mean, smoothed_cov=cov, method=method, n_trajectories_or_lag=n_used)


__all__ = ["ParticleSmoother", "SmootherResult"]
