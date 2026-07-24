"""Exceptions raised by the preprocessing pipeline."""

from __future__ import annotations


class PreprocessingError(RuntimeError):
    """Raised when a preprocessing step cannot be applied safely (e.g. log of
    non-positive values, standardizing a constant series, or insufficient
    data for seasonal decomposition/Kalman imputation). Always carries an
    actionable message identifying the offending variable and reason --
    preprocessing never substitutes a silently wrong value for a step it
    cannot perform correctly.
    """


__all__ = ["PreprocessingError"]
