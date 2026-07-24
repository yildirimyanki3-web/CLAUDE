"""Unit tests for dnlssm.optimization.mle.MLEEstimator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import OptimizationConfig, ParticleFilterConfig
from dnlssm.optimization.mle import MLEEstimator
from dnlssm.utils.random_state import SeedSequence


def _tiny_problem():
    model, theta_true = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
    observations = simulate_from_model(model, theta_true, n_time=8, seed=100)
    pf_config = ParticleFilterConfig(n_particles=300)
    return model, theta_true, observations, pf_config


class TestMLEEstimatorFit:
    def test_multistart_runs_all_method_start_combinations(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["L-BFGS-B", "Nelder-Mead"],
            n_multistarts=2,
            max_iterations=15,
            compute_hessian_ci=False,
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(123))
        result = estimator.fit(observations)
        assert result.n_runs == 2 * 2

    def test_best_run_has_highest_converged_log_likelihood(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["Nelder-Mead"], n_multistarts=3, max_iterations=30, compute_hessian_ci=False
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(7))
        result = estimator.fit(observations)

        converged_ll = [-r.nll_final for r in result.all_runs if r.success]
        if converged_ll:
            assert result.best_log_likelihood == max(converged_ll)

    def test_method_comparison_table_has_expected_columns(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["L-BFGS-B", "Nelder-Mead"], n_multistarts=2, max_iterations=10, compute_hessian_ci=False
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(1))
        result = estimator.fit(observations)

        df = result.method_comparison
        assert set(df["method"]) == {"L-BFGS-B", "Nelder-Mead"}
        for col in ("n_starts", "n_converged", "convergence_rate", "best_log_likelihood"):
            assert col in df.columns

    def test_confidence_intervals_computed_when_requested(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["L-BFGS-B"], n_multistarts=1, max_iterations=15, compute_hessian_ci=True
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(3))
        result = estimator.fit(observations)
        assert result.confidence_intervals is not None  # a ConfidenceIntervalResult, available or not

    def test_confidence_intervals_skipped_when_not_requested(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["Nelder-Mead"], n_multistarts=1, max_iterations=10, compute_hessian_ci=False
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(3))
        result = estimator.fit(observations)
        assert result.confidence_intervals is None

    def test_diagnosis_populated_only_when_not_converged(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        # An unreasonably tiny iteration budget makes non-convergence likely.
        opt_config = OptimizationConfig(
            methods=["L-BFGS-B"], n_multistarts=1, max_iterations=1, compute_hessian_ci=False
        )
        estimator = MLEEstimator(model, pf_config, opt_config, SeedSequence(3))
        result = estimator.fit(observations)
        if not result.converged:
            assert result.convergence_diagnosis is not None
            assert len(result.convergence_diagnosis.reasons) > 0
        else:
            assert result.convergence_diagnosis is None

    def test_reproducible_across_identical_seed_sequences(self) -> None:
        model, theta_true, observations, pf_config = _tiny_problem()
        opt_config = OptimizationConfig(
            methods=["Nelder-Mead"], n_multistarts=1, max_iterations=10, compute_hessian_ci=False
        )
        result1 = MLEEstimator(model, pf_config, opt_config, SeedSequence(55)).fit(observations)
        result2 = MLEEstimator(model, pf_config, opt_config, SeedSequence(55)).fit(observations)
        np.testing.assert_array_equal(result1.best_theta, result2.best_theta)
        assert result1.best_log_likelihood == result2.best_log_likelihood
