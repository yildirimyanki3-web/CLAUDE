"""Unit tests for dnlssm.diagnostics.residuals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.diagnostics.residuals import compute_residual_diagnostics
from dnlssm.filters.particle_filter import BootstrapParticleFilter


def _run_filter(latent_dim=2, obs_dim=3, n_time=20, seed=0):
    model, theta = build_stable_linear_model(latent_dim, obs_dim, seed=seed)
    observations = simulate_from_model(model, theta, n_time, seed=100 + seed)
    pf_config = ParticleFilterConfig(n_particles=2000)
    filter_result = BootstrapParticleFilter(model, pf_config).run(observations, theta, np.random.default_rng(1))
    labels = model.observation_labels()
    time_index = pd.date_range("2020-01-01", periods=n_time, freq="MS")
    return filter_result, observations, labels, time_index


class TestComputeResidualDiagnostics:
    def test_output_shapes(self) -> None:
        filter_result, observations, labels, time_index = _run_filter()
        diag = compute_residual_diagnostics(filter_result, observations, labels, time_index)
        assert diag.raw_residuals.shape == observations.shape
        assert diag.standardized_residuals.shape == observations.shape
        assert len(diag.per_variable_stats) == len(labels)

    def test_per_variable_stats_columns(self) -> None:
        filter_result, observations, labels, time_index = _run_filter()
        diag = compute_residual_diagnostics(filter_result, observations, labels, time_index)
        for col in ("variable", "n_obs", "rmse", "mae", "bias", "std"):
            assert col in diag.per_variable_stats.columns

    def test_rmse_is_nonnegative(self) -> None:
        filter_result, observations, labels, time_index = _run_filter()
        diag = compute_residual_diagnostics(filter_result, observations, labels, time_index)
        assert (diag.per_variable_stats["rmse"].dropna() >= 0).all()

    def test_missing_column_produces_nan_stats(self) -> None:
        filter_result, observations, labels, time_index = _run_filter()
        observations = observations.copy()
        observations[:, 0] = np.nan
        diag = compute_residual_diagnostics(filter_result, observations, labels, time_index)
        row = diag.per_variable_stats[diag.per_variable_stats["variable"] == labels[0]].iloc[0]
        assert row["n_obs"] == 0
        assert np.isnan(row["rmse"])

    def test_shape_mismatch_raises(self) -> None:
        filter_result, observations, labels, time_index = _run_filter()
        with pytest.raises(ValueError):
            compute_residual_diagnostics(filter_result, observations[:, :2], labels[:2], time_index)
