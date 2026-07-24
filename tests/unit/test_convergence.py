"""Unit tests for dnlssm.optimization.convergence."""

from __future__ import annotations

import numpy as np

from dnlssm.optimization.convergence import diagnose_convergence_failure
from dnlssm.optimization.optimizers import SingleOptimizationResult


def _make_result(**overrides) -> SingleOptimizationResult:
    defaults = dict(
        method="L-BFGS-B",
        start_index=0,
        initial_theta=np.zeros(3),
        theta_final=np.ones(3),
        nll_final=1.0,
        success=False,
        termination_message="stopped",
        n_iterations=5,
        n_function_evals=20,
        wall_time_seconds=0.1,
        final_gradient_norm=1e-6,
        iteration_log=[],
        objective=None,
    )
    defaults.update(overrides)
    return SingleOptimizationResult(**defaults)


class TestDiagnoseConvergenceFailure:
    def test_converged_run_needs_no_diagnosis(self) -> None:
        result = _make_result(success=True)
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=100)
        assert diagnosis.converged
        assert diagnosis.reasons == []

    def test_iteration_budget_exhausted_flagged(self) -> None:
        result = _make_result(success=False, n_iterations=50)
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=50)
        assert not diagnosis.converged
        assert any("iteration budget" in r.lower() for r in diagnosis.reasons)

    def test_high_failure_rate_flagged(self) -> None:
        class FakeObjective:
            n_evaluations = 100
            failure_rate = 0.5

        result = _make_result(success=False, n_iterations=5, objective=FakeObjective())
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=100)
        assert any("numerical failure" in r.lower() for r in diagnosis.reasons)

    def test_large_gradient_norm_flagged(self) -> None:
        result = _make_result(success=False, n_iterations=5, final_gradient_norm=10.0)
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=100)
        assert any("gradient norm" in r.lower() for r in diagnosis.reasons)

    def test_large_parameter_magnitude_flagged(self) -> None:
        result = _make_result(success=False, n_iterations=5, theta_final=np.array([1000.0, 0.0, 0.0]))
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=100)
        assert any("magnitude" in r.lower() for r in diagnosis.reasons)

    def test_recommendations_always_accompany_reasons(self) -> None:
        result = _make_result(success=False, n_iterations=50)
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=50)
        assert len(diagnosis.recommendations) > 0

    def test_summary_text_nonempty(self) -> None:
        result = _make_result(success=False, n_iterations=50)
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=50)
        text = diagnosis.summary_text()
        assert "did NOT converge" in text

    def test_no_specific_cause_fallback(self) -> None:
        # A "failed" run with nothing anomalous: low iterations relative to budget, no
        # objective failures, tiny gradient, small parameters.
        result = _make_result(success=False, n_iterations=2, final_gradient_norm=1e-8, theta_final=np.zeros(3))
        diagnosis = diagnose_convergence_failure(result, gradient_tolerance=1e-5, max_iterations=100)
        assert len(diagnosis.reasons) == 1
        assert "no specific" in diagnosis.reasons[0].lower()
