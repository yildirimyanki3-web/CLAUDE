"""Unit tests for dnlssm.diagnostics.autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnlssm.diagnostics.autocorrelation import test_autocorrelation as run_autocorrelation_tests


class TestAutocorrelationTests:
    def test_white_noise_not_significantly_autocorrelated(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=300)})
        summary, profiles = run_autocorrelation_tests(df, max_lags=20, ljung_box_lags=10, significance_level=0.01)
        assert not summary.iloc[0]["significant_autocorrelation"]

    def test_strongly_autocorrelated_series_detected(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.normal(scale=0.1)
        df = pd.DataFrame({"x": x})
        summary, profiles = run_autocorrelation_tests(df, max_lags=20, ljung_box_lags=10, significance_level=0.05)
        assert summary.iloc[0]["significant_autocorrelation"]

    def test_acf_pacf_profiles_populated(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=300)})
        _, profiles = run_autocorrelation_tests(df, max_lags=15, ljung_box_lags=10, significance_level=0.05)
        assert "x" in profiles
        assert profiles["x"].acf_values[0] == 1.0  # lag 0 autocorrelation is always 1

    def test_too_few_observations_skipped_gracefully(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        summary, profiles = run_autocorrelation_tests(df, max_lags=5, ljung_box_lags=5, significance_level=0.05)
        assert summary.iloc[0]["significant_autocorrelation"] is None
        assert "x" not in profiles
