"""Kalman filter and RTS smoother reference implementations for testing.

Not part of the production library: the DNLSSM is deliberately nonlinear by
default. These closed-form references exist solely so that, when a DNLSSM
is configured with ``family: "linear"`` transition/observation functions
(making it exactly a linear-Gaussian state-space model), the from-scratch
Bootstrap Particle Filter and Particle Smoother can be validated against
the analytically exact answer -- the strongest correctness check available
for a Monte Carlo filtering algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnlssm.config.schema import ModelConfig, NoiseConfig, NonlinearFunctionConfig
from dnlssm.models.dnlssm import DNLSSM


@dataclass
class KalmanFilterResult:
    filtered_means: np.ndarray  # (T, d)
    filtered_covs: np.ndarray  # (T, d, d)
    predicted_means: np.ndarray  # (T, d) -- predicted mean entering step t (== filtered_means[0] at t=0)
    predicted_covs: np.ndarray  # (T, d, d)
    log_likelihood: float


def kalman_filter_linear_gaussian(
    y: np.ndarray,
    F: np.ndarray,
    b_f: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    b_g: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> KalmanFilterResult:
    """Exact Kalman filter for ``x_t = F x_{t-1} + b_f + w_t``, ``y_t = H x_t + b_g + v_t``.

    Missing observation dimensions (``NaN`` in ``y``) are masked out of the
    update step at that time, exactly mirroring how the Bootstrap Particle
    Filter under test treats missing data.
    """
    n_time, obs_dim = y.shape
    latent_dim = F.shape[0]

    filtered_means = np.empty((n_time, latent_dim))
    filtered_covs = np.empty((n_time, latent_dim, latent_dim))
    predicted_means = np.empty((n_time, latent_dim))
    predicted_covs = np.empty((n_time, latent_dim, latent_dim))

    x, P = x0.copy(), P0.copy()
    log_likelihood = 0.0

    for t in range(n_time):
        if t > 0:
            x = F @ x + b_f
            P = F @ P @ F.T + Q
        predicted_means[t] = x
        predicted_covs[t] = P

        y_t = y[t]
        mask = ~np.isnan(y_t)
        if mask.any():
            H_obs = H[mask]
            b_obs = b_g[mask]
            R_obs = R[np.ix_(mask, mask)]
            y_obs = y_t[mask]

            y_pred = H_obs @ x + b_obs
            innovation = y_obs - y_pred
            S = H_obs @ P @ H_obs.T + R_obs
            K = P @ H_obs.T @ np.linalg.inv(S)

            x = x + K @ innovation
            P = P - K @ H_obs @ P

            k = mask.sum()
            sign, logdet = np.linalg.slogdet(S)
            log_likelihood += -0.5 * (k * np.log(2 * np.pi) + logdet + innovation @ np.linalg.solve(S, innovation))

        filtered_means[t] = x
        filtered_covs[t] = P

    return KalmanFilterResult(
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=log_likelihood,
    )


def rts_smoother(
    F: np.ndarray, kf_result: KalmanFilterResult
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel fixed-interval smoother given a completed Kalman filter pass."""
    n_time = kf_result.filtered_means.shape[0]
    smoothed_means = kf_result.filtered_means.copy()
    smoothed_covs = kf_result.filtered_covs.copy()

    for t in range(n_time - 2, -1, -1):
        C = kf_result.filtered_covs[t] @ F.T @ np.linalg.inv(kf_result.predicted_covs[t + 1])
        smoothed_means[t] = kf_result.filtered_means[t] + C @ (
            smoothed_means[t + 1] - kf_result.predicted_means[t + 1]
        )
        smoothed_covs[t] = kf_result.filtered_covs[t] + C @ (
            smoothed_covs[t + 1] - kf_result.predicted_covs[t + 1]
        ) @ C.T

    return smoothed_means, smoothed_covs


def unpack_linear_function(params: np.ndarray, input_dim: int, output_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Extracts ``(W, b)`` from a flat parameter vector using the same packing
    convention as :class:`dnlssm.models.functions.LinearFunction` (``y = W x + b``).
    """
    w_size = output_dim * input_dim
    W = params[:w_size].reshape(output_dim, input_dim)
    b = params[w_size:]
    return W, b


def build_stable_linear_model(latent_dim: int, obs_dim: int, seed: int = 0) -> tuple[DNLSSM, np.ndarray]:
    """A linear-Gaussian DNLSSM with a contractive transition matrix (for a stationary test series)."""
    config = ModelConfig(
        latent_dim=latent_dim,
        transition_function=NonlinearFunctionConfig(family="linear"),
        observation_function=NonlinearFunctionConfig(family="linear"),
        process_noise=NoiseConfig(structure="diagonal", init_std=0.5),
        observation_noise=NoiseConfig(structure="diagonal", init_std=0.5),
    )
    model = DNLSSM(latent_dim, [f"v{i}" for i in range(obs_dim)], config)
    rng = np.random.default_rng(seed)
    theta = model.init_theta(rng)

    # Rescale the transition weight matrix to guarantee spectral radius < 1
    # (a stationary, non-explosive latent process), independent of the raw
    # random initialization drawn above.
    blocks = model.layout.split(theta)
    W, _ = unpack_linear_function(blocks["transition"], latent_dim, latent_dim)
    w_size = latent_dim * latent_dim
    blocks["transition"] = blocks["transition"].copy()
    blocks["transition"][:w_size] = (0.5 * W).flatten()
    theta = model.layout.pack(**blocks)
    return model, theta


def simulate_from_model(model: DNLSSM, theta: np.ndarray, n_time: int, seed: int) -> np.ndarray:
    """Simulates one trajectory of observations from a (typically linear-Gaussian) DNLSSM."""
    rng = np.random.default_rng(seed)
    Q = model.process_noise_cov(theta)
    R = model.observation_noise_cov(theta)
    L_Q, L_R = np.linalg.cholesky(Q), np.linalg.cholesky(R)
    mean0, cov0 = model.initial_state_moments(theta)
    x = mean0 + rng.normal(size=model.latent_dim) @ np.linalg.cholesky(cov0).T

    observations = np.empty((n_time, model.observation_dim))
    for t in range(n_time):
        if t > 0:
            x = model.transition_mean(x, theta) + rng.normal(size=model.latent_dim) @ L_Q.T
        observations[t] = model.observation_mean(x, theta) + rng.normal(size=model.observation_dim) @ L_R.T
    return observations


__all__ = [
    "KalmanFilterResult",
    "kalman_filter_linear_gaussian",
    "rts_smoother",
    "unpack_linear_function",
    "build_stable_linear_model",
    "simulate_from_model",
]
