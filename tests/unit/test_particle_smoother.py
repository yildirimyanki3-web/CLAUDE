"""Unit tests for dnlssm.filters.particle_smoother.ParticleSmoother.

As with the filter tests, the strongest check available is comparison
against the closed-form RTS (Rauch-Tung-Striebel) smoother on a
linear-Gaussian special case of the DNLSSM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import (
    build_stable_linear_model,
    kalman_filter_linear_gaussian,
    rts_smoother,
    simulate_from_model,
    unpack_linear_function,
)

from dnlssm.config.schema import ModelConfig, ParticleFilterConfig
from dnlssm.filters.particle_filter import BootstrapParticleFilter
from dnlssm.filters.particle_smoother import ParticleSmoother
from dnlssm.models.dnlssm import DNLSSM


def _run_filter_and_reference(
    latent_dim: int, obs_dim: int, n_time: int, n_particles: int, seed: int
):
    model, theta = build_stable_linear_model(latent_dim, obs_dim, seed=0)
    observations = simulate_from_model(model, theta, n_time, seed=seed)

    blocks = model.layout.split(theta)
    F, b_f = unpack_linear_function(blocks["transition"], latent_dim, latent_dim)
    H, b_g = unpack_linear_function(blocks["observation"], latent_dim, obs_dim)
    Q, R = model.process_noise_cov(theta), model.observation_noise_cov(theta)
    mean0, cov0 = model.initial_state_moments(theta)
    kf = kalman_filter_linear_gaussian(observations, F, b_f, Q, H, b_g, R, mean0, cov0)
    rts_mean, rts_cov = rts_smoother(F, kf)

    pf_config = ParticleFilterConfig(
        n_particles=n_particles, ess_threshold_ratio=0.5, smoother_n_backward_samples=n_particles // 15
    )
    filter_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(7))
    return model, theta, pf_config, filter_result, rts_mean, rts_cov


class TestBackwardSimulationAgainstRTS:
    def test_smoothed_mean_matches_rts_smoother(self) -> None:
        model, theta, pf_config, filter_result, rts_mean, _ = _run_filter_and_reference(
            latent_dim=2, obs_dim=3, n_time=15, n_particles=6000, seed=123
        )
        smoother = ParticleSmoother(model, pf_config)
        result = smoother.run(filter_result, theta, np.random.default_rng(9))

        np.testing.assert_allclose(result.smoothed_mean, rts_mean, atol=0.2)

    def test_final_timestep_matches_filtered_distribution(self) -> None:
        model, theta, pf_config, filter_result, _, _ = _run_filter_and_reference(
            latent_dim=2, obs_dim=3, n_time=10, n_particles=4000, seed=55
        )
        smoother = ParticleSmoother(model, pf_config)
        result = smoother.run(filter_result, theta, np.random.default_rng(1))
        # At t=T-1 the smoothing distribution equals the filtering distribution exactly.
        np.testing.assert_allclose(result.smoothed_mean[-1], filter_result.filtered_mean[-1], atol=0.15)


class TestFixedLagSmoother:
    def test_output_shapes(self) -> None:
        model, theta, pf_config, filter_result, _, _ = _run_filter_and_reference(
            latent_dim=3, obs_dim=4, n_time=20, n_particles=500, seed=42
        )
        fixed_lag_config = pf_config.model_copy(update={"smoother_method": "fixed_lag", "fixed_lag_window": 5})
        smoother = ParticleSmoother(model, fixed_lag_config)
        result = smoother.run(filter_result, theta, np.random.default_rng(3))
        assert result.smoothed_mean.shape == (20, 3)
        assert result.smoothed_cov.shape == (20, 3, 3)
        assert result.method == "fixed_lag"

    def test_reasonably_close_to_rts_smoother(self) -> None:
        model, theta, pf_config, filter_result, rts_mean, _ = _run_filter_and_reference(
            latent_dim=2, obs_dim=3, n_time=15, n_particles=4000, seed=123
        )
        fixed_lag_config = pf_config.model_copy(update={"smoother_method": "fixed_lag", "fixed_lag_window": 8})
        smoother = ParticleSmoother(model, fixed_lag_config)
        result = smoother.run(filter_result, theta, np.random.default_rng(3))
        # Fixed-lag is a coarser approximation than backward simulation; allow a looser tolerance.
        np.testing.assert_allclose(result.smoothed_mean, rts_mean, atol=0.5)


class TestParticleSmootherDispatch:
    def test_unknown_method_raises(self) -> None:
        latent_dim, obs_dim = 2, 3
        config = ModelConfig(latent_dim=latent_dim)
        model = DNLSSM(latent_dim, [f"v{i}" for i in range(obs_dim)], config)
        theta = model.init_theta(np.random.default_rng(0))
        observations = simulate_from_model(model, theta, 5, seed=1)
        pf_config = ParticleFilterConfig(n_particles=200)
        filter_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(0))

        # Bypass schema validation deliberately to exercise the dispatcher's own
        # defensive branch (pydantic does not validate on plain attribute
        # assignment unless validate_assignment=True is set).
        bad_config = pf_config.model_copy()
        bad_config.smoother_method = "not_a_method"
        with pytest.raises(ValueError):
            ParticleSmoother(model, bad_config).run(filter_result, theta, np.random.default_rng(0))
