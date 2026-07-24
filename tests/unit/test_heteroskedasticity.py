"""Unit tests for dnlssm.diagnostics.heteroskedasticity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnlssm.diagnostics.heteroskedasticity import (
    test_heteroskedasticity as run_heteroskedasticity_tests,
)


def _simulate_arch1(n: int, alpha0: float, alpha1: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    sigma2 = alpha0
    for t in range(n):
        sigma2 = alpha0 + alpha1 * (x[t - 1] ** 2 if t > 0 else 0.0)
        x[t] = np.sqrt(sigma2) * rng.normal()
    return x


class TestHeteroskedasticityTests:
    def test_homoskedastic_gaussian_not_flagged(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(size=400)})
        result = run_heteroskedasticity_tests(df, arch_lags=5, significance_level=0.01)
        assert not result.iloc[0]["significant_heteroskedasticity"]

    def test_arch_process_detected(self) -> None:
        x = _simulate_arch1(n=500, alpha0=0.05, alpha1=0.9, seed=0)
        df = pd.DataFrame({"x": x})
        result = run_heteroskedasticity_tests(df, arch_lags=5, significance_level=0.05)
        assert result.iloc[0]["significant_heteroskedasticity"]

    def test_output_has_one_row_per_column(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        result = run_heteroskedasticity_tests(df, arch_lags=5, significance_level=0.05)
        assert len(result) == 2

    def test_too_few_observations_skipped_gracefully(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = run_heteroskedasticity_tests(df, arch_lags=5, significance_level=0.05)
        assert result.iloc[0]["significant_heteroskedasticity"] is None
