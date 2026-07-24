"""Unit tests for dnlssm.optimization.optimizers.run_single_optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.optimization.objective import ParticleFilterObjective
from dnlssm.optimization.optimizers import run_single_optimization


def _tiny_objective(seed: int = 0):
    model, theta_true = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=seed)
    observations = simulate_from_model(model, theta_true, n_time=8, seed=100 + seed)
    pf_config = ParticleFilterConfig(n_particles=300)
    objective = ParticleFilterObjective(model, pf_config, observations, seed=7)
    return objective, theta_true, model


@pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead"])
class TestRunSingleOptimization:
    def test_improves_on_initial_point(self, method) -> None:
        objective, theta_true, model = _tiny_objective()
        rng = np.random.default_rng(3)
        theta0 = model.init_theta(rng) * 3.0  # a deliberately poor starting point
        nll_initial = objective(theta0)

        result = run_single_optimization(
            objective, theta0, method, start_index=0, max_iterations=25,
            function_tolerance=1e-6, gradient_tolerance=1e-3,
        )
        assert result.nll_final <= nll_initial + 1e-6

    def test_output_contract(self, method) -> None:
        objective, theta_true, model = _tiny_objective()
        result = run_single_optimization(
            objective, theta_true, method, start_index=2, max_iterations=15,
            function_tolerance=1e-6, gradient_tolerance=1e-3,
        )
        assert result.method == method
        assert result.start_index == 2
        assert result.theta_final.shape == theta_true.shape
        assert isinstance(result.success, bool)
        assert result.n_function_evals > 0
        assert result.wall_time_seconds >= 0.0
        df = result.iteration_log_dataframe()
        assert list(df.columns) or len(result.iteration_log) == 0  # empty log is valid for 0-iteration runs


class TestGradientNormTracking:
    def test_gradient_norm_logged_for_lbfgsb(self) -> None:
        objective, theta_true, model = _tiny_objective()
        result = run_single_optimization(
            objective, theta_true, "L-BFGS-B", start_index=0, max_iterations=10,
            function_tolerance=1e-6, gradient_tolerance=1e-3,
        )
        if result.iteration_log:
            assert any(r.gradient_norm is not None for r in result.iteration_log)

    def test_gradient_norm_not_tracked_for_nelder_mead(self) -> None:
        objective, theta_true, model = _tiny_objective()
        result = run_single_optimization(
            objective, theta_true, "Nelder-Mead", start_index=0, max_iterations=10,
            function_tolerance=1e-6, gradient_tolerance=1e-3,
        )
        assert all(r.gradient_norm is None for r in result.iteration_log)
