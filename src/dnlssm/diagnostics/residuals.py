"""Innovation (one-step-ahead) residual computation and per-variable error statistics.

Residuals are computed against ``FilterResult.predicted_observation_mean``
-- the particle filter's genuine one-step-ahead forecast, conditioned only
on ``y_1:t-1`` -- rather than the filtered (posterior, ``y_1:t``-informed)
estimate. Using the posterior estimate would understate residual error and
overstate fit quality, since it partly "predicts" ``y_t`` using ``y_t``
itself. Standardization divides by the predicted marginal standard
deviation (``sqrt(diag(predicted_observation_cov))``), which already
combines both particle-cloud spread and observation noise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dnlssm.filters.particle_filter import FilterResult


@dataclass
class ResidualDiagnostics:
    """Raw and standardized one-step-ahead residuals, plus per-variable error statistics."""

    raw_residuals: pd.DataFrame  # (T, N)
    standardized_residuals: pd.DataFrame  # (T, N)
    per_variable_stats: pd.DataFrame  # one row per variable: n_obs, rmse, mae, bias, std


def compute_residual_diagnostics(
    filter_result: FilterResult,
    observations: np.ndarray,
    variable_labels: list[str],
    time_index: pd.DatetimeIndex,
) -> ResidualDiagnostics:
    """Computes innovation residuals and per-variable error statistics.

    Parameters
    ----------
    filter_result:
        A completed :class:`~dnlssm.filters.particle_filter.FilterResult`
        for the *same* ``observations`` (typically the full series, after
        fitting parameters on a train split -- see
        :mod:`dnlssm.model_selection.selector` for the same leak-free
        pattern).
    """
    if observations.shape != filter_result.predicted_observation_mean.shape:
        raise ValueError(
            f"observations shape {observations.shape} does not match "
            f"filter_result.predicted_observation_mean shape {filter_result.predicted_observation_mean.shape}."
        )

    raw = observations - filter_result.predicted_observation_mean
    variances = np.diagonal(filter_result.predicted_observation_cov, axis1=1, axis2=2)
    std = np.sqrt(np.clip(variances, a_min=1e-300, a_max=None))
    standardized = raw / std

    raw_df = pd.DataFrame(raw, index=time_index, columns=variable_labels)
    standardized_df = pd.DataFrame(standardized, index=time_index, columns=variable_labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns -> NaN stats, by design
        stats_rows = []
        for col in variable_labels:
            values = raw_df[col].dropna().to_numpy()
            n_obs = values.shape[0]
            stats_rows.append(
                {
                    "variable": col,
                    "n_obs": n_obs,
                    "rmse": float(np.sqrt(np.mean(values**2))) if n_obs > 0 else np.nan,
                    "mae": float(np.mean(np.abs(values))) if n_obs > 0 else np.nan,
                    "bias": float(np.mean(values)) if n_obs > 0 else np.nan,
                    "std": float(np.std(values, ddof=1)) if n_obs > 1 else np.nan,
                }
            )
    per_variable_stats = pd.DataFrame(stats_rows)

    return ResidualDiagnostics(
        raw_residuals=raw_df, standardized_residuals=standardized_df, per_variable_stats=per_variable_stats
    )


__all__ = ["ResidualDiagnostics", "compute_residual_diagnostics"]
