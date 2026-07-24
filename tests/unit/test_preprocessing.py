"""Unit tests for dnlssm.preprocessing."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dnlssm.config.schema import (
    ObservationSpaceConfig,
    PreprocessingConfig,
    ProviderSeries,
    VariableSpec,
)
from dnlssm.preprocessing.alignment import align_to_monthly
from dnlssm.preprocessing.exceptions import PreprocessingError
from dnlssm.preprocessing.missing import impute_series, longest_consecutive_gap, summarize_missing
from dnlssm.preprocessing.observation_matrix import build_observation_matrix
from dnlssm.preprocessing.transforms import (
    apply_functional_transform,
    seasonally_adjust,
    standardize_series,
    winsorize_series,
)


class TestAlignment:
    def test_daily_downsampled_by_monthly_mean(self) -> None:
        idx = pd.date_range("2020-01-01", "2020-02-28", freq="D")
        series = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        target = pd.date_range("2020-01-01", "2020-02-01", freq="MS")
        aligned, method = align_to_monthly(series, "daily", target)
        assert len(aligned) == 2
        assert "resample_mean" in method
        jan_values = np.arange(31, dtype=float)
        assert aligned.iloc[0] == pytest.approx(jan_values.mean())

    def test_quarterly_upsampled_by_interpolation(self) -> None:
        idx = pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"])
        series = pd.Series([10.0, 20.0, 30.0], index=idx)
        target = pd.date_range("2020-01-01", "2020-07-01", freq="MS")
        aligned, method = align_to_monthly(series, "quarterly", target)
        assert len(aligned) == 7
        assert "interpolation_upsample" in method
        # Feb should be between Jan (10) and Apr (20).
        assert 10.0 < aligned.loc["2020-02-01"] < 20.0

    def test_empty_series_returns_all_nan(self) -> None:
        series = pd.Series([np.nan, np.nan], index=pd.date_range("2020-01-01", periods=2, freq="MS"))
        target = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
        aligned, method = align_to_monthly(series, "monthly", target)
        assert aligned.isna().all()
        assert method == "no_valid_observations"


class TestMissingDataSummary:
    def test_longest_consecutive_gap(self) -> None:
        mask = pd.Series([False, True, True, False, True, True, True, False])
        assert longest_consecutive_gap(mask) == 3

    def test_no_gap_returns_zero(self) -> None:
        mask = pd.Series([False, False, False])
        assert longest_consecutive_gap(mask) == 0

    def test_summarize_missing_stats(self) -> None:
        series = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan])
        summary = summarize_missing(series, "x")
        assert summary.n_total == 5
        assert summary.n_missing == 3
        assert summary.pct_missing == pytest.approx(60.0)
        assert summary.max_consecutive_gap == 2


class TestImputation:
    def test_linear_interpolate_fills_interior_gap(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_series(series, "linear_interpolate", max_consecutive=3)
        assert result.iloc[1] == pytest.approx(2.0)

    def test_ffill_respects_limit(self) -> None:
        series = pd.Series([1.0, np.nan, np.nan, np.nan])
        result = impute_series(series, "ffill", max_consecutive=1)
        assert result.iloc[1] == 1.0
        assert np.isnan(result.iloc[2])

    def test_drop_leaves_nan_untouched(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        result = impute_series(series, "drop", max_consecutive=5)
        assert np.isnan(result.iloc[1])

    def test_kalman_impute_fills_missing(self) -> None:
        rng = np.random.default_rng(0)
        values = np.cumsum(rng.normal(scale=0.1, size=30)) + 10.0
        series = pd.Series(values)
        series.iloc[10:12] = np.nan
        result = impute_series(series, "kalman_impute", max_consecutive=5)
        assert not result.isna().any()

    def test_kalman_impute_too_few_points_raises(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(PreprocessingError):
            impute_series(series, "kalman_impute", max_consecutive=5)


class TestFunctionalTransforms:
    def test_log_transform(self) -> None:
        series = pd.Series([1.0, np.e, np.e**2])
        result = apply_functional_transform(series, "log")
        np.testing.assert_allclose(result.to_numpy(), [0.0, 1.0, 2.0], atol=1e-10)

    def test_log_rejects_nonpositive(self) -> None:
        series = pd.Series([1.0, -1.0, 2.0])
        with pytest.raises(PreprocessingError):
            apply_functional_transform(series, "log")

    def test_diff_transform(self) -> None:
        series = pd.Series([1.0, 3.0, 6.0])
        result = apply_functional_transform(series, "diff")
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.0

    def test_log_diff_transform(self) -> None:
        series = pd.Series([1.0, np.e, np.e**2])
        result = apply_functional_transform(series, "log_diff")
        assert np.isnan(result.iloc[0])
        np.testing.assert_allclose(result.iloc[1:].to_numpy(), [1.0, 1.0], atol=1e-10)

    def test_none_transform_is_copy(self) -> None:
        series = pd.Series([1.0, 2.0])
        result = apply_functional_transform(series, "none")
        assert result is not series
        np.testing.assert_array_equal(result.to_numpy(), series.to_numpy())


class TestSeasonalAdjustment:
    def test_insufficient_data_returns_unchanged(self) -> None:
        series = pd.Series(np.arange(10, dtype=float))
        adjusted, applied = seasonally_adjust(series, period=12)
        assert applied is False
        np.testing.assert_array_equal(adjusted.to_numpy(), series.to_numpy())

    def test_sufficient_data_applies_adjustment(self) -> None:
        t = np.arange(48)
        seasonal = 2.0 * np.sin(2 * np.pi * t / 12)
        trend = 0.05 * t
        series = pd.Series(trend + seasonal + 10.0)
        adjusted, applied = seasonally_adjust(series, period=12)
        assert applied is True
        # Seasonal amplitude should be substantially reduced.
        assert adjusted.std() < series.std()

    def test_preserves_missing_positions(self) -> None:
        t = np.arange(48)
        series = pd.Series(10.0 + 2.0 * np.sin(2 * np.pi * t / 12))
        series.iloc[5] = np.nan
        adjusted, applied = seasonally_adjust(series, period=12)
        if applied:
            assert np.isnan(adjusted.iloc[5])


class TestStandardization:
    def test_zscore_mean_zero_std_one(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result, params = standardize_series(series, "zscore")
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert params.method == "zscore"

    def test_minmax_bounds_zero_one(self) -> None:
        series = pd.Series([0.0, 5.0, 10.0])
        result, _ = standardize_series(series, "minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_series_raises(self) -> None:
        series = pd.Series([5.0, 5.0, 5.0])
        with pytest.raises(PreprocessingError):
            standardize_series(series, "zscore")

    def test_none_method_is_identity(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0])
        result, params = standardize_series(series, "none")
        np.testing.assert_array_equal(result.to_numpy(), series.to_numpy())
        assert params.center == 0.0 and params.scale == 1.0

    def test_invert_round_trip(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        standardized, params = standardize_series(series, "zscore")
        recovered = params.invert(standardized)
        np.testing.assert_allclose(recovered.to_numpy(), series.to_numpy())


class TestWinsorize:
    def test_clips_extremes(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = winsorize_series(series, 0.0, 0.8)
        assert result.max() < 1000.0

    def test_none_bounds_is_noop(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0])
        result = winsorize_series(series, None, None)
        np.testing.assert_array_equal(result.to_numpy(), series.to_numpy())


def _simple_observation_space(n_months: int = 36) -> ObservationSpaceConfig:
    start = date(2020, 1, 1)
    var1 = VariableSpec(
        canonical_id="var_log",
        description="x",
        unit="index",
        native_frequency="monthly",
        transform="log",
        standardize=True,
        provider_priority=[ProviderSeries(provider="manual")],
    )
    var2 = VariableSpec(
        canonical_id="var_level",
        description="x",
        unit="percent",
        native_frequency="monthly",
        transform="none",
        standardize=True,
        provider_priority=[ProviderSeries(provider="manual")],
    )
    placeholder = VariableSpec(
        canonical_id="var_placeholder",
        description="x",
        unit="index",
        native_frequency="annual",
        is_placeholder=True,
        provider_priority=[],
    )
    return ObservationSpaceConfig(start_date=start, variables=[var1, var2, placeholder])


class TestBuildObservationMatrix:
    def test_shape_matches_declared_variables(self) -> None:
        obs_space = _simple_observation_space()
        idx = pd.date_range("2020-01-01", periods=36, freq="MS")
        rng = np.random.default_rng(0)
        series = {
            "var_log": pd.Series(np.exp(rng.normal(size=36)), index=idx),
            "var_level": pd.Series(rng.normal(size=36), index=idx),
        }
        obs_space = obs_space.model_copy(update={"end_date": idx[-1].date()})
        preprocessing = PreprocessingConfig()
        matrix = build_observation_matrix(series, obs_space, preprocessing)
        assert matrix.n_timesteps == 36
        assert matrix.n_variables == 3
        assert matrix.variable_ids == ["var_log", "var_level", "var_placeholder"]

    def test_placeholder_column_is_all_nan(self) -> None:
        obs_space = _simple_observation_space()
        idx = pd.date_range("2020-01-01", periods=36, freq="MS")
        rng = np.random.default_rng(1)
        series = {
            "var_log": pd.Series(np.exp(rng.normal(size=36)), index=idx),
            "var_level": pd.Series(rng.normal(size=36), index=idx),
        }
        obs_space = obs_space.model_copy(update={"end_date": idx[-1].date()})
        matrix = build_observation_matrix(series, obs_space, PreprocessingConfig())
        placeholder_col = matrix.variable_ids.index("var_placeholder")
        assert np.isnan(matrix.values[:, placeholder_col]).all()

    def test_missing_variable_becomes_all_nan_column_not_shape_shrink(self) -> None:
        obs_space = _simple_observation_space()
        idx = pd.date_range("2020-01-01", periods=36, freq="MS")
        rng = np.random.default_rng(2)
        series = {"var_log": pd.Series(np.exp(rng.normal(size=36)), index=idx)}  # var_level missing entirely
        obs_space = obs_space.model_copy(update={"end_date": idx[-1].date()})
        matrix = build_observation_matrix(series, obs_space, PreprocessingConfig())
        assert matrix.n_variables == 3
        missing_col = matrix.variable_ids.index("var_level")
        assert np.isnan(matrix.values[:, missing_col]).all()

    def test_ledger_has_one_record_per_variable(self) -> None:
        obs_space = _simple_observation_space()
        idx = pd.date_range("2020-01-01", periods=36, freq="MS")
        rng = np.random.default_rng(3)
        series = {
            "var_log": pd.Series(np.exp(rng.normal(size=36)), index=idx),
            "var_level": pd.Series(rng.normal(size=36), index=idx),
        }
        obs_space = obs_space.model_copy(update={"end_date": idx[-1].date()})
        matrix = build_observation_matrix(series, obs_space, PreprocessingConfig())
        assert set(matrix.ledger.records.keys()) == {"var_log", "var_level", "var_placeholder"}

    def test_no_data_at_all_without_end_date_raises(self) -> None:
        obs_space = _simple_observation_space()  # end_date is None
        with pytest.raises(PreprocessingError):
            build_observation_matrix({}, obs_space, PreprocessingConfig())
