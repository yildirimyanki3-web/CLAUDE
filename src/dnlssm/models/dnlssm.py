"""The Dynamic Nonlinear State Space Model.

.. math::

    x_t &= f(x_{t-1}; \\theta_f) + w_t, \\quad w_t \\sim \\mathcal{N}(0, Q(\\theta_Q)) \\\\
    y_t &= g(x_t; \\theta_g) + v_t, \\quad v_t \\sim \\mathcal{N}(0, R(\\theta_R)) \\\\
    x_0 &\\sim \\mathcal{N}(\\mu_0(\\theta_0), \\Sigma_0(\\theta_0))

``f`` and ``g`` are arbitrary members of the nonlinear function families in
:mod:`dnlssm.models.functions` (neural by default); this is not a
linearized or Kalman-Filter-reducible model except when the researcher
explicitly configures ``family: linear`` for both maps. The latent
dimension is a free integer, not a modeling assumption baked into this
class -- see :mod:`dnlssm.model_selection` for the search over candidate
dimensions.
"""

from __future__ import annotations

import numpy as np

from dnlssm.config.schema import ModelConfig
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.models.functions import build_nonlinear_function
from dnlssm.models.parameters import DNLSSMParameterLayout, NoiseCovarianceParameterization
from dnlssm.utils.numerical import softplus


class DNLSSM(AbstractStateSpaceModel):
    """Concrete Dynamic Nonlinear State Space Model with a configurable latent dimension."""

    def __init__(self, latent_dim: int, observation_labels: list[str], config: ModelConfig) -> None:
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}.")
        if not observation_labels:
            raise ValueError("observation_labels must be non-empty.")

        self._latent_dim = latent_dim
        self._observation_dim = len(observation_labels)
        self._observation_labels = list(observation_labels)
        self._config = config

        self._transition_fn = build_nonlinear_function(
            config.transition_function, input_dim=latent_dim, output_dim=latent_dim
        )
        self._observation_fn = build_nonlinear_function(
            config.observation_function, input_dim=latent_dim, output_dim=self._observation_dim
        )
        self._process_noise_param = NoiseCovarianceParameterization(latent_dim, config.process_noise.structure)
        self._observation_noise_param = NoiseCovarianceParameterization(
            self._observation_dim, config.observation_noise.structure
        )
        self._layout = DNLSSMParameterLayout(
            latent_dim=latent_dim,
            observation_dim=self._observation_dim,
            transition_fn=self._transition_fn,
            observation_fn=self._observation_fn,
            process_noise_param=self._process_noise_param,
            observation_noise_param=self._observation_noise_param,
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def n_params(self) -> int:
        return self._layout.n_total

    @property
    def layout(self) -> DNLSSMParameterLayout:
        return self._layout

    def transition_mean(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        blocks = self._layout.split(theta)
        return self._transition_fn(x, blocks["transition"])

    def process_noise_cov(self, theta: np.ndarray) -> np.ndarray:
        blocks = self._layout.split(theta)
        return self._process_noise_param.to_covariance(
            blocks["process_noise"], self._config.process_noise.regularization_epsilon
        )

    def observation_mean(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        blocks = self._layout.split(theta)
        return self._observation_fn(x, blocks["observation"])

    def observation_noise_cov(self, theta: np.ndarray) -> np.ndarray:
        blocks = self._layout.split(theta)
        return self._observation_noise_param.to_covariance(
            blocks["observation_noise"], self._config.observation_noise.regularization_epsilon
        )

    def initial_state_moments(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blocks = self._layout.split(theta)
        mean = blocks["initial_mean"]
        std = softplus(blocks["initial_log_std"])
        covariance = np.diag(std**2)
        return mean, covariance

    def init_theta(self, rng: np.random.Generator) -> np.ndarray:
        return self._layout.init_theta(rng, self._config)

    def latent_state_labels(self) -> list[str]:
        return [f"State {i + 1}" for i in range(self._latent_dim)]

    def observation_labels(self) -> list[str]:
        return list(self._observation_labels)


__all__ = ["DNLSSM"]
