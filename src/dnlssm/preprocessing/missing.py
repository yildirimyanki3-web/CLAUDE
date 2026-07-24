"""Missing-observation reporting and imputation.

Two concerns are kept strictly separate: :func:`longest_consecutive_gap`
and the missing-data statistics computed here report the *extent* of
missingness before any imputation is attempted; :func:`impute_series` then
applies the researcher-selected policy (interpolation, forward fill, a
Kalman-smoother-based imputation, or leaving gaps untouched for the model
to treat as missing observations). Any residual ``NaN`` after imputation is
not an error -- the DNLSSM observation likelihood
(:mod:`dnlssm.models.dnlssm`) is designed to mask missing dimensions rather
than require a fully dense matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dnlssm.config.schema import MissingDataLiteral
from dnlssm.preprocessing.exceptions import PreprocessingError

_MIN_OBSERVED_FOR_KALMAN_IMPUTE = 8


@dataclass(frozen=True)
class MissingDataSummary:
    """Missing-observation statistics for one variable, before imputation."""

    canonical_id: str
    n_total: int
    n_missing: int
    pct_missing: float
    max_consecutive_gap: int


def longest_consecutive_gap(is_missing: pd.Series) -> int:
    """Length, in observations, of the longest run of consecutive ``True`` values."""
    if not is_missing.any():
        return 0
    run_id = (is_missing != is_missing.shift()).cumsum()
    run_lengths = is_missing.groupby(run_id).transform("sum")
    return int(run_lengths[is_missing].max())


def summarize_missing(series: pd.Series, canonical_id: str) -> MissingDataSummary:
    is_missing = series.isna()
    n_total = len(series)
    n_missing = int(is_missing.sum())
    return MissingDataSummary(
        canonical_id=canonical_id,
        n_total=n_total,
        n_missing=n_missing,
        pct_missing=100.0 * n_missing / n_total if n_total > 0 else 0.0,
        max_consecutive_gap=longest_consecutive_gap(is_missing),
    )


def impute_series(series: pd.Series, method: MissingDataLiteral, max_consecutive: int) -> pd.Series:
    """Fills missing observations per ``method``. See module docstring for policy semantics."""
    if method == "drop":
        return series.copy()
    if method == "ffill":
        return series.ffill(limit=max_consecutive)
    if method == "linear_interpolate":
        return series.interpolate(method="linear", limit=max_consecutive, limit_direction="both")
    if method == "kalman_impute":
        return _kalman_impute(series)
    raise ValueError(f"Unknown missing_data_method: {method!r}")


def _kalman_impute(series: pd.Series) -> pd.Series:
    """Imputes missing values with a local-level Kalman smoother.

    Uses ``statsmodels``' state-space local-level model, which natively
    treats ``NaN`` observations as missing during Kalman filtering/smoothing
    (no ad hoc pre-filling is required); the smoothed level estimate fills
    only the originally-missing positions, leaving observed values
    untouched.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    n_observed = int(series.notna().sum())
    if n_observed < _MIN_OBSERVED_FOR_KALMAN_IMPUTE:
        raise PreprocessingError(
            f"Kalman imputation for '{series.name}' requires at least "
            f"{_MIN_OBSERVED_FOR_KALMAN_IMPUTE} observed points; got {n_observed}."
        )

    model = UnobservedComponents(series.to_numpy(dtype=float), level="local level")
    fitted = model.fit(disp=False)
    smoothed_level = np.asarray(fitted.smoothed_state[0])

    imputed = series.copy()
    missing_mask = series.isna().to_numpy()
    imputed.iloc[missing_mask] = smoothed_level[missing_mask]
    return imputed


__all__ = ["MissingDataSummary", "longest_consecutive_gap", "summarize_missing", "impute_series"]
