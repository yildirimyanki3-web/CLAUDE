"""Aligns raw-frequency series onto the platform's common monthly grid.

Higher-than-monthly frequency series (daily, weekly) are *downsampled* by
averaging within each month. Lower-than-monthly frequency series
(quarterly, annual) are *upsampled* by time-weighted linear interpolation
onto the monthly grid. Both directions are deliberate, auditable
transformations -- the method used for each variable is returned alongside
the aligned series so it can be recorded in the transformation ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnlssm.config.schema import FrequencyLiteral

_DOWNSAMPLE_FREQUENCIES = {"daily", "weekly", "monthly"}
_UPSAMPLE_FREQUENCIES = {"quarterly", "annual"}


def align_to_monthly(
    series: pd.Series, native_frequency: FrequencyLiteral, target_index: pd.DatetimeIndex
) -> tuple[pd.Series, str]:
    """Aligns ``series`` onto ``target_index`` (a month-start ``DatetimeIndex``).

    Returns
    -------
    aligned_series:
        A series indexed exactly by ``target_index``, with ``NaN`` for any
        target month the source data does not cover.
    method:
        A short string describing the alignment method used, for the
        transformation ledger.
    """
    valid = series[~series.index.isna()]
    valid = valid.sort_index()

    if valid.dropna().empty:
        return pd.Series(np.nan, index=target_index, name=series.name), "no_valid_observations"

    if native_frequency in _DOWNSAMPLE_FREQUENCIES:
        resampled = valid.resample("MS").mean()
        method = f"resample_mean_from_{native_frequency}"
        aligned = resampled.reindex(target_index)
    elif native_frequency in _UPSAMPLE_FREQUENCIES:
        union_index = target_index.union(valid.index).sort_values()
        upsampled = valid.reindex(union_index).interpolate(method="time", limit_direction="both")
        aligned = upsampled.reindex(target_index)
        method = f"time_interpolation_upsample_from_{native_frequency}"
    else:
        raise ValueError(f"Unsupported native_frequency for alignment: {native_frequency!r}")

    return aligned, method


__all__ = ["align_to_monthly"]
