"""ACF, PACF, and the Ljung-Box test for residual autocorrelation, per variable.

A well-specified state-space model's one-step-ahead standardized residuals
should be approximately serially uncorrelated (white noise); the Ljung-Box
test formalizes this as a joint test across the first ``ljung_box_lags``
autocorrelations, while the full ACF/PACF arrays are retained for plotting
(:mod:`dnlssm.visualization`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)

_MIN_OBS_FOR_TEST = 10


@dataclass
class AutocorrelationProfile:
    """Full ACF/PACF arrays for one variable, for downstream plotting."""

    variable: str
    acf_values: np.ndarray
    pacf_values: np.ndarray
    n_lags: int


def test_autocorrelation(
    standardized_residuals: pd.DataFrame, max_lags: int, ljung_box_lags: int, significance_level: float
) -> tuple[pd.DataFrame, dict[str, AutocorrelationProfile]]:
    """Runs the Ljung-Box test and computes ACF/PACF for every column.

    Returns
    -------
    summary:
        One row per variable: Ljung-Box statistic/p-value at
        ``ljung_box_lags``, and ``significant_autocorrelation``.
    profiles:
        Full ACF/PACF arrays per variable (variable -> :class:`AutocorrelationProfile`),
        for plotting.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = []
    profiles: dict[str, AutocorrelationProfile] = {}

    for column in standardized_residuals.columns:
        values = standardized_residuals[column].dropna().to_numpy()
        n_obs = values.shape[0]

        if n_obs < max(_MIN_OBS_FOR_TEST, ljung_box_lags + 1):
            rows.append(
                {
                    "variable": column,
                    "n_obs": n_obs,
                    "ljung_box_statistic": np.nan,
                    "ljung_box_pvalue": np.nan,
                    "ljung_box_lags": ljung_box_lags,
                    "significant_autocorrelation": None,
                }
            )
            logger.debug("Skipping autocorrelation test for '%s': only %d observations.", column, n_obs)
            continue

        lb_result = acorr_ljungbox(values, lags=[ljung_box_lags], return_df=True)
        lb_stat = float(lb_result["lb_stat"].iloc[0])
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])

        effective_max_lags = min(max_lags, n_obs // 2 - 1)
        acf_values = acf(values, nlags=max(effective_max_lags, 0), fft=True)
        pacf_values = pacf(values, nlags=max(effective_max_lags, 0))
        profiles[column] = AutocorrelationProfile(
            variable=column, acf_values=acf_values, pacf_values=pacf_values, n_lags=effective_max_lags
        )

        rows.append(
            {
                "variable": column,
                "n_obs": n_obs,
                "ljung_box_statistic": lb_stat,
                "ljung_box_pvalue": lb_pvalue,
                "ljung_box_lags": ljung_box_lags,
                "significant_autocorrelation": lb_pvalue < significance_level,
            }
        )

    return pd.DataFrame(rows), profiles


__all__ = ["AutocorrelationProfile", "test_autocorrelation"]
