"""Deterministic per-variable transforms: functional form, seasonal adjustment, standardization.

Every function here is a pure ``pandas.Series -> pandas.Series`` (or
``-> (Series, metadata)``) map with no hidden state, so the exact sequence
applied to a variable can be replayed from the
:class:`TransformationLedger` alone. Nothing here decides *which*
transform to use for a given variable -- that policy lives entirely in
``config/variables.yaml`` / ``config/experiment_config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dnlssm.config.schema import StandardizationLiteral, TransformLiteral
from dnlssm.preprocessing.exceptions import PreprocessingError

_SEASONAL_PERIOD_MONTHLY = 12
_MIN_CYCLES_FOR_STL = 2


def apply_functional_transform(series: pd.Series, transform: TransformLiteral) -> pd.Series:
    """Applies the configured level/log/diff/log_diff/pct_change transform."""
    name = series.name
    if transform in ("none", "level"):
        return series.copy()
    if transform == "log":
        _require_positive(series, transform)
        return np.log(series)
    if transform == "diff":
        return series.diff()
    if transform == "log_diff":
        _require_positive(series, transform)
        return np.log(series).diff()
    if transform == "pct_change":
        return series.pct_change()
    raise ValueError(f"Unknown transform: {transform!r} (variable={name!r})")


def _require_positive(series: pd.Series, transform: str) -> None:
    observed = series.dropna()
    if (observed <= 0).any():
        n_bad = int((observed <= 0).sum())
        raise PreprocessingError(
            f"'{transform}' transform requires strictly positive values for variable "
            f"'{series.name}', but {n_bad} observed value(s) are <= 0. Check the unit/sign "
            "convention of the source series (e.g. a balance expressed with a sign) or "
            "choose a different transform in config/variables.yaml."
        )


def seasonally_adjust(
    series: pd.Series, period: int = _SEASONAL_PERIOD_MONTHLY
) -> tuple[pd.Series, bool]:
    """Removes the seasonal component via STL decomposition.

    Returns ``(series, False)`` unchanged -- rather than raising -- when
    there is not enough data to fit STL reliably (fewer than
    ``2 * period`` effectively observed points), since seasonal adjustment
    is an optional refinement, not a required step; the caller records the
    ``False`` outcome in the transformation ledger so it is visible in the
    experiment report.
    """
    from statsmodels.tsa.seasonal import STL

    missing_mask = series.isna()
    working = series.interpolate(method="linear", limit_direction="both") if missing_mask.any() else series

    if working.isna().any() or working.dropna().shape[0] < _MIN_CYCLES_FOR_STL * period:
        return series.copy(), False

    try:
        stl_result = STL(working, period=period, robust=True).fit()
    except Exception:
        return series.copy(), False

    adjusted = working - stl_result.seasonal
    adjusted[missing_mask] = np.nan  # do not fabricate values for genuinely missing periods
    adjusted.name = series.name
    return adjusted, True


@dataclass(frozen=True)
class StandardizationParams:
    """Center/scale parameters used to standardize a series, needed to invert the transform."""

    method: StandardizationLiteral
    center: float
    scale: float

    def apply(self, series: pd.Series) -> pd.Series:
        return (series - self.center) / self.scale

    def invert(self, series: pd.Series) -> pd.Series:
        return series * self.scale + self.center


def standardize_series(
    series: pd.Series, method: StandardizationLiteral
) -> tuple[pd.Series, StandardizationParams]:
    """Standardizes ``series`` and returns the parameters needed to invert the transform later."""
    if method == "none":
        return series.copy(), StandardizationParams(method="none", center=0.0, scale=1.0)

    observed = series.dropna()
    if observed.empty:
        raise PreprocessingError(f"Cannot standardize '{series.name}': no observed values.")

    if method == "zscore":
        center, scale = float(observed.mean()), float(observed.std(ddof=0))
    elif method == "minmax":
        center, scale = float(observed.min()), float(observed.max() - observed.min())
    elif method == "robust":
        q1, q3 = observed.quantile(0.25), observed.quantile(0.75)
        center, scale = float(observed.median()), float(q3 - q1)
    else:
        raise ValueError(f"Unknown standardization method: {method!r}")

    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise PreprocessingError(
            f"Cannot standardize '{series.name}' with method '{method}': scale is zero or "
            "non-finite (the series may be constant over the sample)."
        )

    params = StandardizationParams(method=method, center=center, scale=scale)
    return params.apply(series), params


def winsorize_series(
    series: pd.Series, lower_quantile: float | None, upper_quantile: float | None
) -> pd.Series:
    """Clips extreme values to the given quantile bounds; a no-op if either bound is ``None``."""
    if lower_quantile is None or upper_quantile is None:
        return series.copy()
    lo, hi = series.quantile([lower_quantile, upper_quantile])
    return series.clip(lower=lo, upper=hi)


@dataclass
class VariableTransformationRecord:
    """The complete, ordered log of steps applied to one observation variable."""

    canonical_id: str
    alignment_method: str
    n_total: int
    n_missing_before_imputation: int
    pct_missing_before_imputation: float
    max_consecutive_gap: int
    imputation_method: str
    n_missing_after_imputation: int
    seasonal_adjustment_requested: bool
    seasonal_adjustment_applied: bool
    functional_transform: str
    standardization_method: str
    standardization_center: float | None
    standardization_scale: float | None
    winsorized: bool
    n_missing_final: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TransformationLedger:
    """Collects one :class:`VariableTransformationRecord` per observation variable."""

    records: dict[str, VariableTransformationRecord] = field(default_factory=dict)

    def add(self, record: VariableTransformationRecord) -> None:
        self.records[record.canonical_id] = record

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records.values()]).sort_values(
            "canonical_id"
        ).reset_index(drop=True)


__all__ = [
    "apply_functional_transform",
    "seasonally_adjust",
    "standardize_series",
    "winsorize_series",
    "StandardizationParams",
    "VariableTransformationRecord",
    "TransformationLedger",
]
