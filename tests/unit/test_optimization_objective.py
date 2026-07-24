"""Unit tests for dnlssm.optimization.objective.ParticleFilterObjective."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import ModelConfig, NoiseConfig, ParticleFilterConfig
from dnlssm.models.dnlssm import DNLSSM
from dnlssm.optimization.objective import ParticleFilterObjective


def _tiny_setup(seed: int = 0):
    model, theta_true = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=seed)
    observations = simulate_from_model(model, theta_true, n_time=8, seed=100 + seed)
    pf_config = ParticleFilterConfig(n_particles=300)
    return model, theta_true, observations, pf_config


class TestCommonRandomNumbers:
    def test_repeated_calls_same_theta_identical_nll(self) -> None:
        model, theta_true, observations, pf_config = _tiny_setup()
        objective = ParticleFilterObjective(model, pf_config, observations, seed=42)
        nll1 = objective(theta_true)
        nll2 = objective(theta_true)
        assert nll1 == nll2

    def test_different_seeds_give_different_nll(self) -> None:
        model, theta_true, observations, pf_config = _tiny_setup()
        objective_a = ParticleFilterObjective(model, pf_config, observations, seed=1)
        objective_b = ParticleFilterObjective(model, pf_config, observations, seed=2)
        nll_a = objective_a(theta_true)
        nll_b = objective_b(theta_true)
        assert nll_a != nll_b  # different common-random-number streams -> different MC estimate

    def test_evaluation_log_grows_with_calls(self) -> None:
        model, theta_true, observations, pf_config = _tiny_setup()
        objective = ParticleFilterObjective(model, pf_config, observations, seed=1)
        objective(theta_true)
        objective(theta_true * 1.1)
        assert objective.n_evaluations == 2
        assert len(objective.evaluation_log) == 2


class TestFailureHandling:
    def test_numerical_failure_returns_penalty_not_exception(self) -> None:
        latent_dim = 1
        config = ModelConfig(
            latent_dim=latent_dim,
            process_noise=NoiseConfig(structure="diagonal", init_std=1e-8, regularization_epsilon=1e-12),
            observation_noise=NoiseConfig(structure="diagonal", init_std=1e-8, regularization_epsilon=1e-12),
        )
        model = DNLSSM(latent_dim, ["a"], config)
        theta = model.init_theta(np.random.default_rng(0))
        observations = np.array([[1e200]])
        pf_config = ParticleFilterConfig(n_particles=100)
        objective = ParticleFilterObjective(model, pf_config, observations, seed=0, penalty_value=1e10)

        nll = objective(theta)
        assert nll == 1e10
        assert objective.n_failed_evaluations == 1
        assert objective.failure_rate == 1.0

    def test_failure_rate_zero_when_no_failures(self) -> None:
        model, theta_true, observations, pf_config = _tiny_setup()
        objective = ParticleFilterObjective(model, pf_config, observations, seed=1)
        objective(theta_true)
        assert objective.failure_rate == 0.0
