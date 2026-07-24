"""Unit tests for dnlssm.models.dnlssm.DNLSSM."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.config.schema import ModelConfig
from dnlssm.models.dnlssm import DNLSSM


def _labels(n: int) -> list[str]:
    return [f"var_{i}" for i in range(n)]


class TestDNLSSMShapes:
    @pytest.mark.parametrize("latent_dim", [1, 3, 4, 8])
    def test_transition_and_observation_shapes_single_point(self, latent_dim) -> None:
        n_obs = 17
        model = DNLSSM(latent_dim, _labels(n_obs), ModelConfig(latent_dim=latent_dim))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        x = rng.normal(size=latent_dim)
        x_next = model.transition_mean(x, theta)
        y = model.observation_mean(x, theta)
        assert x_next.shape == (latent_dim,)
        assert y.shape == (n_obs,)

    def test_transition_and_observation_shapes_batched(self) -> None:
        latent_dim, n_particles, n_obs = 4, 4000, 17
        model = DNLSSM(latent_dim, _labels(n_obs), ModelConfig(latent_dim=latent_dim))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        x = rng.normal(size=(n_particles, latent_dim))
        x_next = model.transition_mean(x, theta)
        y = model.observation_mean(x, theta)
        assert x_next.shape == (n_particles, latent_dim)
        assert y.shape == (n_particles, n_obs)

    def test_noise_covariances_are_positive_definite(self) -> None:
        latent_dim, n_obs = 5, 17
        model = DNLSSM(latent_dim, _labels(n_obs), ModelConfig(latent_dim=latent_dim))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        Q = model.process_noise_cov(theta)
        R = model.observation_noise_cov(theta)
        assert Q.shape == (latent_dim, latent_dim)
        assert R.shape == (n_obs, n_obs)
        assert np.all(np.linalg.eigvalsh(Q) > 0)
        assert np.all(np.linalg.eigvalsh(R) > 0)

    def test_initial_state_moments_shapes(self) -> None:
        latent_dim = 6
        model = DNLSSM(latent_dim, _labels(17), ModelConfig(latent_dim=latent_dim))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        mean, cov = model.initial_state_moments(theta)
        assert mean.shape == (latent_dim,)
        assert cov.shape == (latent_dim, latent_dim)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    def test_init_theta_length_matches_n_params(self) -> None:
        model = DNLSSM(4, _labels(17), ModelConfig(latent_dim=4))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        assert theta.shape == (model.n_params,)

    def test_latent_state_labels_are_neutral(self) -> None:
        model = DNLSSM(3, _labels(17), ModelConfig(latent_dim=3))
        assert model.latent_state_labels() == ["State 1", "State 2", "State 3"]
        for label in model.latent_state_labels():
            for banned in ("regime", "boom", "crisis", "transition", "repression"):
                assert banned not in label.lower()

    def test_observation_labels_preserve_order(self) -> None:
        labels = ["a", "b", "c"]
        model = DNLSSM(2, labels, ModelConfig(latent_dim=2))
        assert model.observation_labels() == labels

    def test_invalid_latent_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            DNLSSM(0, _labels(17), ModelConfig(latent_dim=1))

    def test_empty_observation_labels_raises(self) -> None:
        with pytest.raises(ValueError):
            DNLSSM(3, [], ModelConfig(latent_dim=3))


class TestDNLSSMDeterminism:
    def test_same_theta_same_input_gives_same_output(self) -> None:
        model = DNLSSM(4, _labels(17), ModelConfig(latent_dim=4))
        rng = np.random.default_rng(0)
        theta = model.init_theta(rng)
        x = np.random.default_rng(1).normal(size=4)
        y1 = model.observation_mean(x, theta)
        y2 = model.observation_mean(x, theta)
        np.testing.assert_array_equal(y1, y2)
