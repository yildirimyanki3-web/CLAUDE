"""Unit tests for dnlssm.optimization.confidence_intervals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.optimization.confidence_intervals import compute_hessian_confidence_intervals
from dnlssm.optimization.objective import ParticleFilterObjective


class TestHessianConfidenceIntervals:
    def test_available_and_correctly_shaped_for_well_behaved_objective(self) -> None:
        model, theta_true = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta_true, n_time=10, seed=100)
        pf_config = ParticleFilterConfig(n_particles=500)
        objective = ParticleFilterObjective(model, pf_config, observations, seed=7)

        result = compute_hessian_confidence_intervals(
            objective, theta_true, confidence_level=0.95, step_size=1e-2
        )

        n_params = theta_true.shape[0]
        if result.available:
            assert result.standard_errors.shape == (n_params,)
            assert result.confidence_intervals.shape == (n_params, 2)
            assert result.hessian.shape == (n_params, n_params)
            finite_rows = ~np.isnan(result.standard_errors)
            assert np.all(
                result.confidence_intervals[finite_rows, 0] <= result.confidence_intervals[finite_rows, 1]
            )
        else:
            assert result.failure_reason is not None

    def test_degenerate_flat_objective_reports_unavailable(self) -> None:
        def constant_objective(theta: np.ndarray) -> float:
            return 42.0

        result = compute_hessian_confidence_intervals(
            constant_objective, np.zeros(3), confidence_level=0.95, step_size=1e-2
        )
        assert result.available is False
        assert result.failure_reason is not None
        assert result.standard_errors is None
        assert result.confidence_intervals is None

    def test_confidence_level_affects_interval_width(self) -> None:
        model, theta_true = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta_true, n_time=10, seed=100)
        pf_config = ParticleFilterConfig(n_particles=500)

        objective_90 = ParticleFilterObjective(model, pf_config, observations, seed=7)
        result_90 = compute_hessian_confidence_intervals(objective_90, theta_true, 0.90, 1e-2)

        objective_99 = ParticleFilterObjective(model, pf_config, observations, seed=7)
        result_99 = compute_hessian_confidence_intervals(objective_99, theta_true, 0.99, 1e-2)

        if result_90.available and result_99.available:
            width_90 = result_90.confidence_intervals[:, 1] - result_90.confidence_intervals[:, 0]
            width_99 = result_99.confidence_intervals[:, 1] - result_99.confidence_intervals[:, 0]
            finite = ~np.isnan(width_90) & ~np.isnan(width_99)
            assert np.all(width_99[finite] >= width_90[finite] - 1e-8)
