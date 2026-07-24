"""Unit tests for dnlssm.models.parameters."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.config.schema import ModelConfig
from dnlssm.models.functions import LinearFunction
from dnlssm.models.parameters import DNLSSMParameterLayout, NoiseCovarianceParameterization


class TestNoiseCovarianceParameterization:
    @pytest.mark.parametrize("structure", ["scalar", "diagonal", "full"])
    def test_covariance_is_symmetric_positive_definite(self, structure) -> None:
        dim = 4
        param = NoiseCovarianceParameterization(dim, structure)
        rng = np.random.default_rng(0)
        raw = rng.normal(size=param.n_params)
        cov = param.to_covariance(raw, epsilon=1e-6)
        assert cov.shape == (dim, dim)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_n_params_matches_structure(self) -> None:
        dim = 5
        assert NoiseCovarianceParameterization(dim, "scalar").n_params == 1
        assert NoiseCovarianceParameterization(dim, "diagonal").n_params == 5
        assert NoiseCovarianceParameterization(dim, "full").n_params == 15  # 5*6/2

    def test_init_params_yields_target_std_diagonal(self) -> None:
        dim = 3
        param = NoiseCovarianceParameterization(dim, "diagonal")
        raw = param.init_params(init_std=0.7)
        cov = param.to_covariance(raw, epsilon=0.0)
        stds = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(stds, 0.7, atol=1e-6)

    def test_init_params_yields_target_std_scalar(self) -> None:
        param = NoiseCovarianceParameterization(2, "scalar")
        raw = param.init_params(init_std=1.3)
        cov = param.to_covariance(raw, epsilon=0.0)
        np.testing.assert_allclose(np.diag(cov), 1.3**2, atol=1e-6)

    def test_regularization_epsilon_increases_diagonal(self) -> None:
        param = NoiseCovarianceParameterization(2, "diagonal")
        raw = param.init_params(init_std=1.0)
        cov_small_eps = param.to_covariance(raw, epsilon=1e-8)
        cov_large_eps = param.to_covariance(raw, epsilon=1.0)
        assert np.trace(cov_large_eps) > np.trace(cov_small_eps)


class TestDNLSSMParameterLayout:
    def _make_layout(self, latent_dim=3, obs_dim=4) -> DNLSSMParameterLayout:
        transition_fn = LinearFunction(latent_dim, latent_dim)
        observation_fn = LinearFunction(latent_dim, obs_dim)
        process_noise = NoiseCovarianceParameterization(latent_dim, "diagonal")
        observation_noise = NoiseCovarianceParameterization(obs_dim, "diagonal")
        return DNLSSMParameterLayout(latent_dim, obs_dim, transition_fn, observation_fn, process_noise, observation_noise)

    def test_split_pack_round_trip(self) -> None:
        layout = self._make_layout()
        rng = np.random.default_rng(0)
        theta = rng.normal(size=layout.n_total)
        blocks = layout.split(theta)
        theta_reconstructed = layout.pack(**blocks)
        np.testing.assert_array_equal(theta, theta_reconstructed)

    def test_split_wrong_length_raises(self) -> None:
        layout = self._make_layout()
        with pytest.raises(ValueError):
            layout.split(np.zeros(layout.n_total - 1))

    def test_pack_missing_block_raises(self) -> None:
        layout = self._make_layout()
        with pytest.raises(ValueError):
            layout.pack(transition=np.zeros(layout.n_transition))

    def test_init_theta_has_correct_length(self) -> None:
        layout = self._make_layout()
        rng = np.random.default_rng(0)
        theta = layout.init_theta(rng, ModelConfig())
        assert theta.shape == (layout.n_total,)

    def test_n_total_matches_sum_of_blocks(self) -> None:
        layout = self._make_layout(latent_dim=5, obs_dim=17)
        expected = (
            layout.n_transition
            + layout.n_observation
            + layout.n_process_noise
            + layout.n_observation_noise
            + layout.n_initial_mean
            + layout.n_initial_log_std
        )
        assert layout.n_total == expected
