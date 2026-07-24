"""Abstract interface shared by every latent state-space model family.

:class:`AbstractStateSpaceModel` is deliberately narrow: it exposes only
the moments a Bootstrap Particle Filter and a Maximum Likelihood estimator
need (transition/observation means, noise covariances, initial-state
moments, and a parameter count with a random initializer). The DNLSSM
(:mod:`dnlssm.models.dnlssm`) is the only implementation today, but the
filtering (:mod:`dnlssm.filters`), optimization
(:mod:`dnlssm.optimization`), and model-selection
(:mod:`dnlssm.model_selection`) packages are written entirely against this
interface. A future Dynamic Factor Model, Hidden Markov Model, or
Switching Linear State Space Model becomes usable by the same
infrastructure by implementing this class -- no other package needs to
change.
"""

from __future__ import annotations

import abc

import numpy as np


class AbstractStateSpaceModel(abc.ABC):
    """State-space model interface: ``x_t = f(x_{t-1}) + w_t``, ``y_t = g(x_t) + v_t``."""

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent state vector."""

    @property
    @abc.abstractmethod
    def observation_dim(self) -> int:
        """Dimension of the observation vector."""

    @property
    @abc.abstractmethod
    def n_params(self) -> int:
        """Length of the flat parameter vector ``theta`` accepted by every method below."""

    @abc.abstractmethod
    def transition_mean(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``E[x_t | x_{t-1} = x]``. ``x``: ``(latent_dim,)`` or ``(n_particles, latent_dim)``."""

    @abc.abstractmethod
    def process_noise_cov(self, theta: np.ndarray) -> np.ndarray:
        """Process noise covariance ``Q``, shape ``(latent_dim, latent_dim)``, symmetric PD."""

    @abc.abstractmethod
    def observation_mean(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``E[y_t | x_t = x]``. ``x``: ``(latent_dim,)`` or ``(n_particles, latent_dim)``."""

    @abc.abstractmethod
    def observation_noise_cov(self, theta: np.ndarray) -> np.ndarray:
        """Observation noise covariance ``R``, shape ``(observation_dim, observation_dim)``, symmetric PD."""

    @abc.abstractmethod
    def initial_state_moments(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, covariance)`` of ``x_0``, shapes ``(latent_dim,)`` and ``(latent_dim, latent_dim)``."""

    @abc.abstractmethod
    def init_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Draws a random parameter vector for multi-start optimization, shape ``(n_params,)``."""

    @abc.abstractmethod
    def latent_state_labels(self) -> list[str]:
        """Neutral labels for each latent dimension, e.g. ``["State 1", "State 2", ...]``."""

    @abc.abstractmethod
    def observation_labels(self) -> list[str]:
        """Canonical ids of each observation-space variable, in column order."""


__all__ = ["AbstractStateSpaceModel"]
