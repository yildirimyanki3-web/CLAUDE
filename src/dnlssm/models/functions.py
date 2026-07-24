"""Modular nonlinear function families for state transition and observation maps.

Every family here is dimension-agnostic (works for any ``input_dim`` /
``output_dim``, so the observation space can grow and the latent dimension
can change without touching this module) and exposes the same three-method
interface: ``n_params``, ``__call__(x, params)``, ``init_params(rng, std)``.
:func:`build_nonlinear_function` is the only place that maps a
:class:`~dnlssm.config.schema.NonlinearFunctionConfig` to a concrete
instance, so adding a new functional family later means adding one class
and one branch here -- nothing else in the codebase references a family by
name.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import numpy as np

from dnlssm.config.schema import NonlinearFunctionConfig
from dnlssm.utils.numerical import softplus

_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
    "softplus": softplus,
}

_LOGISTIC_EXP_CLIP = 30.0


class NonlinearFunction(abc.ABC):
    """A parameterized map ``R^input_dim -> R^output_dim``, vectorized over particles.

    ``x`` may be a single point of shape ``(input_dim,)`` or a batch of
    shape ``(n_particles, input_dim)``; the return shape mirrors the input
    (``(output_dim,)`` or ``(n_particles, output_dim)`` respectively), which
    is what the Bootstrap Particle Filter relies on to propagate the full
    particle cloud through one call.
    """

    input_dim: int
    output_dim: int

    @property
    @abc.abstractmethod
    def n_params(self) -> int: ...

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, params: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def init_params(self, rng: np.random.Generator, std: float) -> np.ndarray: ...

    def _as_batched(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        was_1d = np.ndim(x) == 1
        return np.atleast_2d(x), was_1d

    def _restore_shape(self, y: np.ndarray, was_1d: bool) -> np.ndarray:
        return y[0] if was_1d else y


class LinearFunction(NonlinearFunction):
    """``y = W x + b``. Included as a configurable option, not the platform default."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def n_params(self) -> int:
        return self.output_dim * (self.input_dim + 1)

    def __call__(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x2d, was_1d = self._as_batched(x)
        w_size = self.output_dim * self.input_dim
        W = params[:w_size].reshape(self.output_dim, self.input_dim)
        b = params[w_size:]
        y = x2d @ W.T + b
        return self._restore_shape(y, was_1d)

    def init_params(self, rng: np.random.Generator, std: float) -> np.ndarray:
        return rng.normal(scale=std, size=self.n_params)


class PolynomialFunction(NonlinearFunction):
    """``y = W [x, x^2, ..., x^degree] + b`` (element-wise powers, no cross terms).

    Cross terms are deliberately omitted: a full multivariate polynomial
    basis grows combinatorially with ``input_dim`` (:math:`\\binom{d+k}{k}`
    terms), which would silently make every future addition to the
    17-variable observation space vastly more expensive. The element-wise
    basis keeps parameter count linear in ``input_dim * degree`` while
    still providing genuine per-coordinate nonlinearity.
    """

    def __init__(self, input_dim: int, output_dim: int, degree: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self._feature_dim = input_dim * degree

    @property
    def n_params(self) -> int:
        return self.output_dim * (self._feature_dim + 1)

    def _features(self, x2d: np.ndarray) -> np.ndarray:
        powers = [x2d**k for k in range(1, self.degree + 1)]
        return np.concatenate(powers, axis=1)

    def __call__(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x2d, was_1d = self._as_batched(x)
        features = self._features(x2d)
        w_size = self.output_dim * self._feature_dim
        W = params[:w_size].reshape(self.output_dim, self._feature_dim)
        b = params[w_size:]
        y = features @ W.T + b
        return self._restore_shape(y, was_1d)

    def init_params(self, rng: np.random.Generator, std: float) -> np.ndarray:
        return rng.normal(scale=std, size=self.n_params)


class GeneralizedLogisticFunction(NonlinearFunction):
    """``y_j = L_j / (1 + exp(-k_j (W x + b)_j - x0_j))``, an S-shaped nonlinearity per output.

    ``L`` (upper asymptote) and ``k`` (growth rate) are passed through
    :func:`~dnlssm.utils.numerical.softplus` to guarantee they stay
    strictly positive regardless of the unconstrained optimizer's raw
    values; the logistic exponent is clipped to avoid floating-point
    overflow for large pre-activations.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._linear_size = output_dim * input_dim

    @property
    def n_params(self) -> int:
        return self._linear_size + 4 * self.output_dim

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, ...]:
        idx = 0
        W = params[idx : idx + self._linear_size].reshape(self.output_dim, self.input_dim)
        idx += self._linear_size
        b = params[idx : idx + self.output_dim]
        idx += self.output_dim
        raw_L = params[idx : idx + self.output_dim]
        idx += self.output_dim
        raw_k = params[idx : idx + self.output_dim]
        idx += self.output_dim
        x0 = params[idx : idx + self.output_dim]
        return W, b, raw_L, raw_k, x0

    def __call__(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x2d, was_1d = self._as_batched(x)
        W, b, raw_L, raw_k, x0 = self._unpack(params)
        z = x2d @ W.T + b
        L = softplus(raw_L) + 1e-6
        k = softplus(raw_k) + 1e-6
        exponent = np.clip(-k * (z - x0), -_LOGISTIC_EXP_CLIP, _LOGISTIC_EXP_CLIP)
        y = L / (1.0 + np.exp(exponent))
        return self._restore_shape(y, was_1d)

    def init_params(self, rng: np.random.Generator, std: float) -> np.ndarray:
        return rng.normal(scale=std, size=self.n_params)


class NeuralFunction(NonlinearFunction):
    """A single-hidden-layer MLP: ``y = W2 activation(W1 x + b1) + b2``.

    This is the platform default (``family: neural`` in
    ``config/experiment_config.yaml``): it is a genuine, universal-
    approximator-class nonlinearity, so the DNLSSM does not degenerate to a
    linear-Gaussian (classic Kalman Filter) model unless the researcher
    explicitly selects ``family: linear``.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, activation: str) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation!r}; expected one of {list(_ACTIVATIONS)}.")
        self._activation = _ACTIVATIONS[activation]
        self._w1_size = hidden_dim * input_dim
        self._w2_size = output_dim * hidden_dim

    @property
    def n_params(self) -> int:
        return self._w1_size + self.hidden_dim + self._w2_size + self.output_dim

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = 0
        W1 = params[idx : idx + self._w1_size].reshape(self.hidden_dim, self.input_dim)
        idx += self._w1_size
        b1 = params[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        W2 = params[idx : idx + self._w2_size].reshape(self.output_dim, self.hidden_dim)
        idx += self._w2_size
        b2 = params[idx : idx + self.output_dim]
        return W1, b1, W2, b2

    def __call__(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x2d, was_1d = self._as_batched(x)
        W1, b1, W2, b2 = self._unpack(params)
        hidden = self._activation(x2d @ W1.T + b1)
        y = hidden @ W2.T + b2
        return self._restore_shape(y, was_1d)

    def init_params(self, rng: np.random.Generator, std: float) -> np.ndarray:
        return rng.normal(scale=std, size=self.n_params)


def build_nonlinear_function(
    config: NonlinearFunctionConfig, input_dim: int, output_dim: int
) -> NonlinearFunction:
    """Instantiates the nonlinear function family selected by ``config``."""
    if config.family == "linear":
        return LinearFunction(input_dim, output_dim)
    if config.family == "polynomial":
        return PolynomialFunction(input_dim, output_dim, degree=config.polynomial_degree)
    if config.family == "generalized_logistic":
        return GeneralizedLogisticFunction(input_dim, output_dim)
    if config.family == "neural":
        return NeuralFunction(input_dim, output_dim, hidden_dim=config.hidden_dim, activation=config.activation)
    raise ValueError(f"Unknown nonlinear function family: {config.family!r}")


__all__ = [
    "NonlinearFunction",
    "LinearFunction",
    "PolynomialFunction",
    "GeneralizedLogisticFunction",
    "NeuralFunction",
    "build_nonlinear_function",
]
