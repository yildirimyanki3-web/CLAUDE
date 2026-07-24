"""Unit tests for dnlssm.diagnostics.normality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnlssm.diagnostics.normality import test_normality as run_normality_tests


class TestNormalityTests:
    def test_gaussian_data_not_rejected(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=500)})
        result = run_normality_tests(df, significance_level=0.01)
        assert not result.iloc[0]["rejects_normality_at_alpha"]

    def test_heavily_skewed_data_rejected(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.exponential(scale=1.0, size=500)})
        result = run_normality_tests(df, significance_level=0.05)
        assert result.iloc[0]["rejects_normality_at_alpha"]

    def test_output_has_one_row_per_column(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        result = run_normality_tests(df, significance_level=0.05)
        assert len(result) == 2
        assert set(result["variable"]) == {"a", "b"}

    def test_too_few_observations_gives_nan(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = run_normality_tests(df, significance_level=0.05)
        assert np.isnan(result.iloc[0]["shapiro_statistic"])

    def test_handles_nan_values_by_dropping(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=200)
        values[:10] = np.nan
        df = pd.DataFrame({"x": values})
        result = run_normality_tests(df, significance_level=0.05)
        assert result.iloc[0]["n_obs"] == 190
