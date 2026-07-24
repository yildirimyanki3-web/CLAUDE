"""The ARCH-LM test for conditional heteroskedasticity in standardized residuals.

Detects volatility clustering (time-varying conditional variance) left in
the one-step-ahead residuals. This is reported as a diagnostic finding
only: none of the platform's static noise-covariance structures (scalar,
diagonal, full -- see :mod:`dnlssm.models.parameters.NoiseCovarianceParameterization`)
model *time-varying* variance, so detecting ARCH effects here does not by
itself point to a fix available within the current model family. It is
recorded so the experiment report states the limitation explicitly rather
than silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch

from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_heteroskedasticity(
    standardized_residuals: pd.DataFrame, arch_lags: int, significance_level: float
) -> pd.DataFrame:
    """Runs the ARCH-LM test on every column; returns one row per variable."""
    rows = []
    for column in standardized_residuals.columns:
        values = standardized_residuals[column].dropna().to_numpy()
        n_obs = values.shape[0]

        if n_obs <= arch_lags + 2:
            rows.append(
                {
                    "variable": column,
                    "n_obs": n_obs,
                    "arch_lm_statistic": np.nan,
                    "arch_lm_pvalue": np.nan,
                    "significant_heteroskedasticity": None,
                }
            )
            logger.debug("Skipping ARCH-LM test for '%s': only %d observations.", column, n_obs)
            continue

        lm_stat, lm_pvalue, _f_stat, _f_pvalue = het_arch(values, nlags=arch_lags)
        rows.append(
            {
                "variable": column,
                "n_obs": n_obs,
                "arch_lm_statistic": float(lm_stat),
                "arch_lm_pvalue": float(lm_pvalue),
                "significant_heteroskedasticity": bool(lm_pvalue < significance_level),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["test_heteroskedasticity"]
