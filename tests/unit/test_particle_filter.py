"""Unit tests for dnlssm.filters.particle_filter.BootstrapParticleFilter.

The most important test class here (``TestAgainstExactKalmanFilter``)
validates the from-scratch Bootstrap Particle Filter against the closed-form
Kalman filter on a linear-Gaussian special case of the DNLSSM (obtained by
configuring both the transition and observation functions with
``family="linear"``). This is the strongest correctness check available for
a Monte Carlo filtering algorithm: with enough particles, the PF's filtered
mean and marginal log-likelihood must converge to the Kalman filter's exact
values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import (
    build_stable_linear_model as _build_stable_linear_model,
)
from fixtures.linear_gaussian_reference import kalman_filter_linear_gaussian, unpack_linear_function
from fixtures.linear_gaussian_reference import simulate_from_model as _simulate

from dnlssm.config.schema import ModelConfig, NoiseConfig, ParticleFilterConfig
from dnlssm.filters.particle_filter import BootstrapParticleFilter
from dnlssm.models.dnlssm import DNLSSM


class TestAgainstExactKalmanFilter:
    def test_filtered_mean_matches_kalman_filter(self) -> None:
        latent_dim, obs_dim, n_time = 2, 3, 15
        model, theta = _build_stable_linear_model(latent_dim, obs_dim)
        observations = _simulate(model, theta, n_time, seed=123)

        blocks = model.layout.split(theta)
        F, b_f = unpack_linear_function(blocks["transition"], latent_dim, latent_dim)
        H, b_g = unpack_linear_function(blocks["observation"], latent_dim, obs_dim)
        Q, R = model.process_noise_cov(theta), model.observation_noise_cov(theta)
        mean0, cov0 = model.initial_state_moments(theta)
        kf = kalman_filter_linear_gaussian(observations, F, b_f, Q, H, b_g, R, mean0, cov0)

        pf_config = ParticleFilterConfig(n_particles=8000, ess_threshold_ratio=0.5)
        pf_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(7))

        np.testing.assert_allclose(pf_result.filtered_mean, kf.filtered_means, atol=0.15)
        assert abs(pf_result.log_likelihood - kf.log_likelihood) < 1.0

    def test_one_step_ahead_prediction_matches_kalman_predicted_moments(self) -> None:
        """predicted_observation_mean/cov must match H @ predicted_state + b, H P H^T + R --
        the Kalman filter's *prior* (not posterior) predictive moments -- since this quantity
        is used for genuinely out-of-sample scoring and must not leak information from y_t.
        """
        latent_dim, obs_dim, n_time = 2, 3, 15
        model, theta = _build_stable_linear_model(latent_dim, obs_dim)
        observations = _simulate(model, theta, n_time, seed=123)

        blocks = model.layout.split(theta)
        F, b_f = unpack_linear_function(blocks["transition"], latent_dim, latent_dim)
        H, b_g = unpack_linear_function(blocks["observation"], latent_dim, obs_dim)
        Q, R = model.process_noise_cov(theta), model.observation_noise_cov(theta)
        mean0, cov0 = model.initial_state_moments(theta)
        kf = kalman_filter_linear_gaussian(observations, F, b_f, Q, H, b_g, R, mean0, cov0)

        expected_pred_obs_mean = kf.predicted_means @ H.T + b_g
        expected_pred_obs_cov = np.einsum("ij,tjk,lk->til", H, kf.predicted_covs, H) + R[None, :, :]

        pf_config = ParticleFilterConfig(n_particles=8000, ess_threshold_ratio=0.5)
        pf_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(7))

        np.testing.assert_allclose(pf_result.predicted_observation_mean, expected_pred_obs_mean, atol=0.2)
        np.testing.assert_allclose(pf_result.predicted_observation_cov, expected_pred_obs_cov, atol=0.3)

    def test_handles_missing_observation_dimensions_like_kalman_filter(self) -> None:
        latent_dim, obs_dim, n_time = 2, 4, 12
        model, theta = _build_stable_linear_model(latent_dim, obs_dim, seed=1)
        observations = _simulate(model, theta, n_time, seed=321)
        observations[3, 1] = np.nan
        observations[3, 2] = np.nan
        observations[7, :] = np.nan  # fully missing timestep

        blocks = model.layout.split(theta)
        F, b_f = unpack_linear_function(blocks["transition"], latent_dim, latent_dim)
        H, b_g = unpack_linear_function(blocks["observation"], latent_dim, obs_dim)
        Q, R = model.process_noise_cov(theta), model.observation_noise_cov(theta)
        mean0, cov0 = model.initial_state_moments(theta)
        kf = kalman_filter_linear_gaussian(observations, F, b_f, Q, H, b_g, R, mean0, cov0)

        pf_config = ParticleFilterConfig(n_particles=8000, ess_threshold_ratio=0.5)
        pf_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(11))

        np.testing.assert_allclose(pf_result.filtered_mean, kf.filtered_means, atol=0.2)


class TestParticleFilterContract:
    def test_output_shapes(self) -> None:
        latent_dim, obs_dim, n_time, n_particles = 4, 6, 10, 500
        model, theta = _build_stable_linear_model(latent_dim, obs_dim)
        observations = _simulate(model, theta, n_time, seed=1)
        pf_config = ParticleFilterConfig(n_particles=n_particles)
        result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))

        assert result.filtered_mean.shape == (n_time, latent_dim)
        assert result.filtered_cov.shape == (n_time, latent_dim, latent_dim)
        assert result.particle_history.shape == (n_time, n_particles, latent_dim)
        assert result.weight_history.shape == (n_time, n_particles)
        assert result.ancestor_history.shape == (n_time - 1, n_particles)
        assert result.log_likelihood_per_timestep.shape == (n_time,)
        assert result.predicted_observation_mean.shape == (n_time, obs_dim)
        assert result.predicted_observation_cov.shape == (n_time, obs_dim, obs_dim)

    def test_dimension_mismatch_raises(self) -> None:
        model, theta = _build_stable_linear_model(2, 3)
        bad_observations = np.zeros((10, 5))  # wrong observation_dim
        pf_config = ParticleFilterConfig(n_particles=100)
        with pytest.raises(ValueError):
            BootstrapParticleFilter(model, pf_config).run(bad_observations, theta, np.random.default_rng(0))

    def test_weights_normalized_at_every_timestep(self) -> None:
        model, theta = _build_stable_linear_model(2, 3)
        observations = _simulate(model, theta, 10, seed=2)
        pf_config = ParticleFilterConfig(n_particles=500)
        result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))
        sums = result.weight_history.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-8)

    def test_reproducible_with_same_seed(self) -> None:
        model, theta = _build_stable_linear_model(2, 3)
        observations = _simulate(model, theta, 10, seed=3)
        pf_config = ParticleFilterConfig(n_particles=500)
        r1 = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(99))
        r2 = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(99))
        np.testing.assert_array_equal(r1.filtered_mean, r2.filtered_mean)
        assert r1.log_likelihood == r2.log_likelihood

    def test_ess_never_exceeds_n_particles(self) -> None:
        model, theta = _build_stable_linear_model(2, 3)
        observations = _simulate(model, theta, 20, seed=4)
        n_particles = 300
        pf_config = ParticleFilterConfig(n_particles=n_particles)
        result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))
        assert np.all(result.diagnostics.ess <= n_particles + 1e-6)
        assert np.all(result.diagnostics.ess >= 1.0 - 1e-6)

    def test_non_adaptive_resampling_resamples_every_step(self) -> None:
        model, theta = _build_stable_linear_model(2, 3)
        observations = _simulate(model, theta, 8, seed=5)
        pf_config = ParticleFilterConfig(n_particles=200, adaptive_resampling=False)
        result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))
        assert np.all(result.diagnostics.resampled_at)
        assert result.diagnostics.n_resample_events == 8

    def test_all_particles_zero_likelihood_raises_informative_error(self) -> None:
        latent_dim = 2
        config = ModelConfig(
            latent_dim=latent_dim,
            process_noise=NoiseConfig(structure="diagonal", init_std=1e-8, regularization_epsilon=1e-12),
            observation_noise=NoiseConfig(structure="diagonal", init_std=1e-8, regularization_epsilon=1e-12),
        )
        model = DNLSSM(latent_dim, ["a", "b"], config)
        theta = model.init_theta(np.random.default_rng(0))
        # An observation so incompatible with the (almost) noiseless model that
        # the squared Mahalanobis residual overflows float64, driving every
        # particle's log-likelihood to exactly -inf (not merely "very small").
        observations = np.array([[1e200, 1e200]])
        pf_config = ParticleFilterConfig(n_particles=100)
        with pytest.raises(FloatingPointError):
            BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))
