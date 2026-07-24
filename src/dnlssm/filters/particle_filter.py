"""Bootstrap Particle Filter, implemented from scratch.

Implements the standard bootstrap (prior-proposal) SMC recursion with
*adaptive* resampling: particles propagate through the model's transition
density, are reweighted by the observation likelihood (correctly masking
any missing observation dimensions), and are resampled only when the
Effective Sample Size drops below the configured threshold -- carrying
weights forward multiplicatively otherwise, exactly as
:func:`dnlssm.utils.numerical.sequential_importance_weight_update`
implements. Every stage required by the research design is present:
initialization, prediction, importance weighting, log-likelihood
evaluation, weight normalization, ESS computation, resampling, and
filtered posterior estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.filters.diagnostics import (
    ParticleDiagnosticsSummary,
    effective_sample_size,
    normalized_weight_entropy,
    summarize_particle_diagnostics,
)
from dnlssm.filters.diagnostics import weight_entropy as weight_entropy_fn
from dnlssm.filters.resampling import resample
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.numerical import (
    gaussian_log_pdf,
    safe_cholesky,
    sequential_importance_weight_update,
)

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Full output of one Bootstrap Particle Filter pass over ``T`` time steps."""

    filtered_mean: np.ndarray  # (T, latent_dim)
    filtered_cov: np.ndarray  # (T, latent_dim, latent_dim)
    log_likelihood: float
    log_likelihood_per_timestep: np.ndarray  # (T,)
    particle_history: np.ndarray  # (T, n_particles, latent_dim) -- pre-resample positions
    weight_history: np.ndarray  # (T, n_particles) -- normalized filtering weights, pre-resample
    ancestor_history: np.ndarray  # (T-1, n_particles) int -- parent index at t for each particle at t+1
    predicted_observation_mean: np.ndarray  # (T, observation_dim) -- E[y_t | y_1:t-1], genuine one-step-ahead forecast
    predicted_observation_cov: np.ndarray  # (T, observation_dim, observation_dim) -- Var[y_t | y_1:t-1]
    diagnostics: ParticleDiagnosticsSummary
    n_particles: int

    @property
    def n_timesteps(self) -> int:
        return self.filtered_mean.shape[0]


class BootstrapParticleFilter:
    """Runs the bootstrap particle filter for a given :class:`AbstractStateSpaceModel`."""

    def __init__(self, model: AbstractStateSpaceModel, config: ParticleFilterConfig) -> None:
        self._model = model
        self._config = config

    def run(
        self,
        observations: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> FilterResult:
        """Filters ``observations`` (shape ``(T, observation_dim)``, ``NaN`` for missing entries)."""
        n_particles = n_particles or self._config.n_particles
        if observations.ndim != 2 or observations.shape[1] != self._model.observation_dim:
            raise ValueError(
                f"observations must have shape (T, {self._model.observation_dim}); "
                f"got {observations.shape}."
            )

        n_time = observations.shape[0]
        latent_dim = self._model.latent_dim
        observation_dim = self._model.observation_dim

        process_noise_cov = self._model.process_noise_cov(theta)
        observation_noise_cov = self._model.observation_noise_cov(theta)
        process_noise_chol = safe_cholesky(process_noise_cov)

        init_mean, init_cov = self._model.initial_state_moments(theta)
        init_chol = safe_cholesky(init_cov)
        particles = init_mean[None, :] + rng.normal(size=(n_particles, latent_dim)) @ init_chol.T
        log_weights = np.full(n_particles, -np.log(n_particles))

        particle_history = np.empty((n_time, n_particles, latent_dim))
        weight_history = np.empty((n_time, n_particles))
        ancestor_history = np.empty((max(n_time - 1, 0), n_particles), dtype=np.int64)
        filtered_mean = np.empty((n_time, latent_dim))
        filtered_cov = np.empty((n_time, latent_dim, latent_dim))
        predicted_observation_mean = np.empty((n_time, observation_dim))
        predicted_observation_cov = np.empty((n_time, observation_dim, observation_dim))
        ess = np.empty(n_time)
        entropy = np.empty(n_time)
        norm_entropy = np.empty(n_time)
        resampled_at = np.zeros(n_time, dtype=bool)
        loglik_per_t = np.empty(n_time)
        threshold = self._config.ess_threshold_ratio * n_particles

        for t in range(n_time):
            if t > 0:
                predicted_mean = self._model.transition_mean(particles, theta)
                particles = predicted_mean + rng.normal(size=(n_particles, latent_dim)) @ process_noise_chol.T

            y_t = observations[t]
            observed_mask = ~np.isnan(y_t)

            # Computed unconditionally (not just when y_t has observed entries): this is the
            # genuine one-step-ahead forecast E[y_t | y_1:t-1], using only the weights carried
            # in from t-1 -- i.e. it does NOT use y_t itself, unlike the post-update filtered
            # estimate. This is what out-of-sample scoring (dnlssm.model_selection) and
            # innovation-residual diagnostics (dnlssm.diagnostics) must condition on.
            predicted_obs = self._model.observation_mean(particles, theta)
            weights_prior = np.exp(log_weights)
            predicted_observation_mean[t] = np.average(predicted_obs, axis=0, weights=weights_prior)
            centered_obs = predicted_obs - predicted_observation_mean[t]
            predicted_observation_cov[t] = (
                (centered_obs * weights_prior[:, None]).T @ centered_obs + observation_noise_cov
            )

            if observed_mask.any():
                residuals = y_t[observed_mask][None, :] - predicted_obs[:, observed_mask]
                cov_obs = observation_noise_cov[np.ix_(observed_mask, observed_mask)]
                log_increment = gaussian_log_pdf(residuals, cov_obs)
            else:
                log_increment = np.zeros(n_particles)

            log_weights, ll_increment = sequential_importance_weight_update(log_weights, log_increment)
            weights = np.exp(log_weights)

            particle_history[t] = particles
            weight_history[t] = weights
            loglik_per_t[t] = ll_increment

            filtered_mean[t] = np.average(particles, axis=0, weights=weights)
            centered = particles - filtered_mean[t]
            filtered_cov[t] = (centered * weights[:, None]).T @ centered

            ess[t] = effective_sample_size(weights)
            entropy[t] = weight_entropy_fn(weights)
            norm_entropy[t] = normalized_weight_entropy(weights)

            do_resample = (not self._config.adaptive_resampling) or (ess[t] < threshold)
            if do_resample:
                ancestor_idx = resample(weights, self._config.resampling_method, rng)
                particles = particles[ancestor_idx]
                log_weights = np.full(n_particles, -np.log(n_particles))
                resampled_at[t] = True
            else:
                ancestor_idx = np.arange(n_particles)

            if t < n_time - 1:
                ancestor_history[t] = ancestor_idx

        diagnostics = summarize_particle_diagnostics(
            ess=ess,
            weight_entropy_series=entropy,
            normalized_weight_entropy_series=norm_entropy,
            resampled_at=resampled_at,
            n_particles=n_particles,
        )
        logger.debug(diagnostics.summary_text())

        return FilterResult(
            filtered_mean=filtered_mean,
            filtered_cov=filtered_cov,
            log_likelihood=float(np.sum(loglik_per_t)),
            log_likelihood_per_timestep=loglik_per_t,
            particle_history=particle_history,
            weight_history=weight_history,
            ancestor_history=ancestor_history,
            predicted_observation_mean=predicted_observation_mean,
            predicted_observation_cov=predicted_observation_cov,
            diagnostics=diagnostics,
            n_particles=n_particles,
        )


__all__ = ["BootstrapParticleFilter", "FilterResult"]
