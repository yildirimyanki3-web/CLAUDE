"""Flat-vector parameter layout and numerically stable noise-covariance parameterization.

The optimizer (:mod:`dnlssm.optimization`) works with a single flat
``np.ndarray`` of unconstrained real parameters, as required by SciPy's
unconstrained optimizers. :class:`DNLSSMParameterLayout` is the single
place that knows how that flat vector decomposes into named, semantically
meaningful blocks (transition function weights, observation function
weights, process/observation noise, initial-state moments); every other
module accesses parameters through this layout rather than hardcoding
offsets.

:class:`NoiseCovarianceParameterization` guarantees every covariance matrix
produced anywhere in the model is symmetric positive definite by
construction: variances are parameterized through
:func:`~dnlssm.utils.numerical.softplus` (always positive) and, for the
``"full"`` structure, through a Cholesky factor (so ``Q = L L^T`` is PSD by
construction, with a positive diagonal enforced again via softplus) --
this is the "use a Cholesky factorization" numerical-stability
recommendation implemented structurally rather than patched on afterward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnlssm.config.schema import ModelConfig, NoiseStructureLiteral
from dnlssm.models.functions import NonlinearFunction
from dnlssm.utils.numerical import inverse_softplus, regularize_covariance, softplus


class NoiseCovarianceParameterization:
    """Maps an unconstrained parameter block to a ``(dim, dim)`` covariance matrix.

    Parameters
    ----------
    dim:
        Dimension of the noise vector (latent dimension for process noise,
        observation dimension for observation noise).
    structure:
        ``"scalar"``: one shared variance, ``Q = sigma^2 I``.
        ``"diagonal"``: ``dim`` independent variances, ``Q = diag(sigma_1^2, ..., sigma_d^2)``.
        ``"full"``: a full Cholesky factor, ``Q = L L^T``, ``dim * (dim + 1) / 2`` parameters.
    """

    def __init__(self, dim: int, structure: NoiseStructureLiteral) -> None:
        self.dim = dim
        self.structure = structure

    @property
    def n_params(self) -> int:
        if self.structure == "scalar":
            return 1
        if self.structure == "diagonal":
            return self.dim
        if self.structure == "full":
            return self.dim * (self.dim + 1) // 2
        raise ValueError(f"Unknown noise structure: {self.structure!r}")

    def init_params(self, init_std: float) -> np.ndarray:
        """Unconstrained parameters whose implied standard deviation is ``init_std``."""
        raw_std = float(inverse_softplus(np.asarray(init_std)))
        params = np.zeros(self.n_params)
        params[: min(self.dim, self.n_params)] = raw_std  # diagonal block always comes first
        return params

    def to_covariance(self, params: np.ndarray, epsilon: float) -> np.ndarray:
        if self.structure == "scalar":
            std = softplus(params[0])
            cov = float(std) ** 2 * np.eye(self.dim)
        elif self.structure == "diagonal":
            stds = softplus(params)
            cov = np.diag(stds**2)
        elif self.structure == "full":
            cov = self._full_cholesky_covariance(params)
        else:
            raise ValueError(f"Unknown noise structure: {self.structure!r}")
        return regularize_covariance(cov, epsilon)

    def _full_cholesky_covariance(self, params: np.ndarray) -> np.ndarray:
        L = np.zeros((self.dim, self.dim))
        diag_raw = params[: self.dim]
        offdiag_raw = params[self.dim :]
        L[np.diag_indices(self.dim)] = softplus(diag_raw)
        rows, cols = np.tril_indices(self.dim, k=-1)
        L[rows, cols] = offdiag_raw
        return L @ L.T


@dataclass(frozen=True)
class DNLSSMParameterLayout:
    """Named-block decomposition of the DNLSSM's flat parameter vector."""

    latent_dim: int
    observation_dim: int
    transition_fn: NonlinearFunction
    observation_fn: NonlinearFunction
    process_noise_param: NoiseCovarianceParameterization
    observation_noise_param: NoiseCovarianceParameterization

    @property
    def n_transition(self) -> int:
        return self.transition_fn.n_params

    @property
    def n_observation(self) -> int:
        return self.observation_fn.n_params

    @property
    def n_process_noise(self) -> int:
        return self.process_noise_param.n_params

    @property
    def n_observation_noise(self) -> int:
        return self.observation_noise_param.n_params

    @property
    def n_initial_mean(self) -> int:
        return self.latent_dim

    @property
    def n_initial_log_std(self) -> int:
        return self.latent_dim

    @property
    def n_total(self) -> int:
        return (
            self.n_transition
            + self.n_observation
            + self.n_process_noise
            + self.n_observation_noise
            + self.n_initial_mean
            + self.n_initial_log_std
        )

    def _block_bounds(self) -> dict[str, tuple[int, int]]:
        offsets = {}
        cursor = 0
        for name, size in (
            ("transition", self.n_transition),
            ("observation", self.n_observation),
            ("process_noise", self.n_process_noise),
            ("observation_noise", self.n_observation_noise),
            ("initial_mean", self.n_initial_mean),
            ("initial_log_std", self.n_initial_log_std),
        ):
            offsets[name] = (cursor, cursor + size)
            cursor += size
        return offsets

    def split(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        if theta.shape[0] != self.n_total:
            raise ValueError(
                f"Parameter vector has length {theta.shape[0]}, expected {self.n_total} "
                f"for this DNLSSM configuration (latent_dim={self.latent_dim}, "
                f"observation_dim={self.observation_dim})."
            )
        return {name: theta[lo:hi] for name, (lo, hi) in self._block_bounds().items()}

    def pack(self, **blocks: np.ndarray) -> np.ndarray:
        bounds = self._block_bounds()
        if set(blocks) != set(bounds):
            raise ValueError(f"pack() requires exactly blocks {set(bounds)}, got {set(blocks)}.")
        theta = np.empty(self.n_total)
        for name, (lo, hi) in bounds.items():
            block = np.asarray(blocks[name])
            if block.shape[0] != hi - lo:
                raise ValueError(f"Block '{name}' has length {block.shape[0]}, expected {hi - lo}.")
            theta[lo:hi] = block
        return theta

    def init_theta(self, rng: np.random.Generator, model_config: ModelConfig) -> np.ndarray:
        """Draws a random initial parameter vector consistent with the model configuration."""
        return self.pack(
            transition=self.transition_fn.init_params(rng, model_config.transition_function.weight_init_std),
            observation=self.observation_fn.init_params(rng, model_config.observation_function.weight_init_std),
            process_noise=self.process_noise_param.init_params(model_config.process_noise.init_std),
            observation_noise=self.observation_noise_param.init_params(model_config.observation_noise.init_std),
            initial_mean=rng.normal(
                loc=model_config.initial_state_mean, scale=model_config.initial_state_std, size=self.latent_dim
            ),
            initial_log_std=np.full(
                self.latent_dim, float(inverse_softplus(np.asarray(model_config.initial_state_std)))
            ),
        )


__all__ = ["NoiseCovarianceParameterization", "DNLSSMParameterLayout"]
