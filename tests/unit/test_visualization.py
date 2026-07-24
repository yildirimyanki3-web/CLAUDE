"""Unit tests for dnlssm.visualization.

Plotting functions are tested for the properties that matter for a
non-interactive, file-producing pipeline: they must not raise, they must
produce a non-empty file at the requested path, and they must handle the
documented edge cases (all-NaN columns, single-dimension latent spaces)
without special-casing by the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnlssm.diagnostics.autocorrelation import AutocorrelationProfile
from dnlssm.models.latent import LatentManifold, LatentTrajectory
from dnlssm.optimization.optimizers import IterationRecord, SingleOptimizationResult
from dnlssm.visualization import (
    plot_acf_pacf,
    plot_ess_timeseries,
    plot_information_criteria_comparison,
    plot_latent_trajectories,
    plot_observation_fit,
    plot_optimization_convergence,
    plot_particle_count_sensitivity,
    plot_residual_histogram_qq,
)


def _assert_saved(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


class TestLatentPlots:
    @pytest.mark.parametrize("dim", [1, 2, 5])
    def test_plot_latent_trajectories(self, tmp_path, dim) -> None:
        idx = pd.date_range("2020-01-01", periods=20, freq="MS")
        manifold = LatentManifold.default(dim)
        rng = np.random.default_rng(0)
        filtered = LatentTrajectory(
            manifold=manifold,
            time_index=idx,
            mean=rng.normal(size=(20, dim)),
            kind="filtered",
            covariance=np.stack([np.eye(dim)] * 20),
        )
        smoothed = LatentTrajectory(
            manifold=manifold, time_index=idx, mean=rng.normal(size=(20, dim)), kind="smoothed"
        )
        path = plot_latent_trajectories(filtered, smoothed, tmp_path / "latent.png")
        _assert_saved(path)

    def test_plot_latent_trajectories_without_smoothed(self, tmp_path) -> None:
        idx = pd.date_range("2020-01-01", periods=10, freq="MS")
        manifold = LatentManifold.default(3)
        filtered = LatentTrajectory(
            manifold=manifold, time_index=idx, mean=np.zeros((10, 3)), kind="filtered"
        )
        path = plot_latent_trajectories(filtered, None, tmp_path / "latent_only.png")
        _assert_saved(path)

    def test_plot_ess_timeseries(self, tmp_path) -> None:
        idx = pd.date_range("2020-01-01", periods=15, freq="MS")
        ess_ratio = np.linspace(1.0, 0.2, 15)
        resampled_at = ess_ratio < 0.5
        path = plot_ess_timeseries(idx, ess_ratio, resampled_at, 0.5, tmp_path / "ess.png")
        _assert_saved(path)


class TestObservationPlots:
    def test_plot_observation_fit(self, tmp_path) -> None:
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        rng = np.random.default_rng(0)
        n_vars = 6
        observations = rng.normal(size=(12, n_vars))
        predicted_mean = rng.normal(size=(12, n_vars))
        predicted_cov = np.stack([np.eye(n_vars)] * 12)
        labels = [f"var_{i}" for i in range(n_vars)]
        path = plot_observation_fit(observations, predicted_mean, predicted_cov, labels, idx, tmp_path / "obs.png")
        _assert_saved(path)

    def test_plot_observation_fit_handles_all_nan_column(self, tmp_path) -> None:
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        rng = np.random.default_rng(0)
        n_vars = 3
        observations = rng.normal(size=(12, n_vars))
        observations[:, 1] = np.nan  # placeholder-style column
        predicted_mean = rng.normal(size=(12, n_vars))
        predicted_cov = np.stack([np.eye(n_vars)] * 12)
        labels = [f"var_{i}" for i in range(n_vars)]
        path = plot_observation_fit(observations, predicted_mean, predicted_cov, labels, idx, tmp_path / "obs2.png")
        _assert_saved(path)


class TestDiagnosticPlots:
    def test_plot_residual_histogram_qq(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        residuals = rng.normal(size=200)
        path = plot_residual_histogram_qq("var_x", residuals, tmp_path / "resid.png")
        _assert_saved(path)

    def test_plot_residual_histogram_qq_all_nan(self, tmp_path) -> None:
        residuals = np.full(50, np.nan)
        path = plot_residual_histogram_qq("var_x", residuals, tmp_path / "resid_nan.png")
        _assert_saved(path)

    def test_plot_acf_pacf(self, tmp_path) -> None:
        profile = AutocorrelationProfile(
            variable="var_x", acf_values=np.linspace(1, 0, 10), pacf_values=np.linspace(0.5, 0, 10), n_lags=9
        )
        path = plot_acf_pacf(profile, significance_level=0.05, output_path=tmp_path / "acf.png")
        _assert_saved(path)


def _fake_optimization_run(method: str, start_index: int, success: bool) -> SingleOptimizationResult:
    iteration_log = [
        IterationRecord(iteration=i, negative_log_likelihood=10.0 - i * 0.5, param_norm=1.0, param_delta_norm=0.1, gradient_norm=None)
        for i in range(1, 6)
    ]
    return SingleOptimizationResult(
        method=method,
        start_index=start_index,
        initial_theta=np.zeros(3),
        theta_final=np.ones(3),
        nll_final=7.5,
        success=success,
        termination_message="ok",
        n_iterations=5,
        n_function_evals=20,
        wall_time_seconds=0.1,
        final_gradient_norm=None,
        iteration_log=iteration_log,
    )


class TestConvergencePlots:
    def test_plot_optimization_convergence(self, tmp_path) -> None:
        runs = [
            _fake_optimization_run("L-BFGS-B", 0, True),
            _fake_optimization_run("L-BFGS-B", 1, False),
            _fake_optimization_run("Nelder-Mead", 0, True),
        ]
        path = plot_optimization_convergence(runs, tmp_path / "convergence.png")
        _assert_saved(path)

    def test_plot_optimization_convergence_empty_iteration_logs(self, tmp_path) -> None:
        run = _fake_optimization_run("L-BFGS-B", 0, True)
        run.iteration_log = []
        path = plot_optimization_convergence([run], tmp_path / "convergence_empty.png")
        _assert_saved(path)

    def test_plot_information_criteria_comparison(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "latent_dim": [3, 4, 5],
                "aic": [100.0, 90.0, 95.0],
                "bic": [110.0, 105.0, 115.0],
                "hqic": [105.0, 97.0, 104.0],
            }
        )
        path = plot_information_criteria_comparison(df, tmp_path / "ic.png")
        _assert_saved(path)

    def test_plot_particle_count_sensitivity(self, tmp_path) -> None:
        df = pd.DataFrame(
            {
                "n_particles": [2000, 3000, 4000, 5000],
                "log_likelihood": [-100.0, -99.5, -99.2, -99.1],
                "mean_ess_ratio": [0.6, 0.65, 0.7, 0.72],
            }
        )
        path = plot_particle_count_sensitivity(df, tmp_path / "pc_sensitivity.png")
        _assert_saved(path)
