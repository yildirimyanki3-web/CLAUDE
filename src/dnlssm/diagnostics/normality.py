"""Normality tests on standardized residuals, per observation variable.

Both Shapiro-Wilk (generally the more powerful test for small-to-moderate
samples) and Jarque-Bera (a moment-based test, cheap even for very large
samples) are reported together rather than either alone, since they can
disagree and a single test's result is not sufficient grounds for a
distributional claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_MIN_OBS_FOR_SHAPIRO = 3
_MAX_OBS_FOR_SHAPIRO = 5000  # scipy.stats.shapiro's documented reliable sample-size ceiling
_MIN_OBS_FOR_JARQUE_BERA = 8  # below this the asymptotic chi-squared approximation is unreliable


def test_normality(standardized_residuals: pd.DataFrame, significance_level: float) -> pd.DataFrame:
    """Runs Shapiro-Wilk and Jarque-Bera normality tests on every column.

    Returns a DataFrame with one row per variable: sample size, both test
    statistics/p-values, and ``rejects_normality_at_alpha`` (``True`` if
    *either* test rejects normality at ``significance_level`` -- the
    conservative choice, since either test flagging non-normality is
    grounds to inspect residual histograms/QQ-plots before trusting
    Gaussian-based inference).
    """
    rows = []
    for column in standardized_residuals.columns:
        values = standardized_residuals[column].dropna().to_numpy()
        n_obs = values.shape[0]

        shapiro_stat, shapiro_p = np.nan, np.nan
        if _MIN_OBS_FOR_SHAPIRO <= n_obs <= _MAX_OBS_FOR_SHAPIRO:
            shapiro_result = stats.shapiro(values)
            shapiro_stat, shapiro_p = float(shapiro_result.statistic), float(shapiro_result.pvalue)

        jb_stat, jb_p = np.nan, np.nan
        if n_obs >= _MIN_OBS_FOR_JARQUE_BERA:
            jb_result = stats.jarque_bera(values)
            jb_stat, jb_p = float(jb_result.statistic), float(jb_result.pvalue)

        p_values = [p for p in (shapiro_p, jb_p) if not np.isnan(p)]
        rejects = any(p < significance_level for p in p_values) if p_values else None

        rows.append(
            {
                "variable": column,
                "n_obs": n_obs,
                "shapiro_statistic": shapiro_stat,
                "shapiro_pvalue": shapiro_p,
                "jarque_bera_statistic": jb_stat,
                "jarque_bera_pvalue": jb_p,
                "rejects_normality_at_alpha": rejects,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["test_normality"]
