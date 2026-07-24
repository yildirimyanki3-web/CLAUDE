"""Multi-start Maximum Likelihood Estimation of DNLSSM parameters.

:class:`MLEEstimator` is the top-level entry point: it runs every
configured optimizer method (``optimization.methods``) from every
multi-start initial point (``optimization.n_multistarts``), each with its
own independent random initialization and its own fixed common-random-number
seed for the particle-filter objective, selects the best converged run,
builds a cross-method comparison table (to answer "which optimizer is more
reliable for this model"), and -- if requested -- computes Hessian-based
confidence intervals at the selected optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dnlssm.config.schema import OptimizationConfig, ParticleFilterConfig
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.optimization.confidence_intervals import (
    ConfidenceIntervalResult,
    compute_hessian_confidence_intervals,
)
from dnlssm.optimization.convergence import ConvergenceDiagnosis, diagnose_convergence_failure
from dnlssm.optimization.objective import ParticleFilterObjective
from dnlssm.optimization.optimizers import SingleOptimizationResult, run_single_optimization
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.random_state import SeedSequence

logger = get_logger(__name__)


@dataclass
class MLEResult:
    """The full outcome of a multi-start, multi-method MLE run."""

    best_theta: np.ndarray
    best_log_likelihood: float
    best_method: str
    best_start_index: int
    converged: bool
    all_runs: list[SingleOptimizationResult] = field(default_factory=list)
    convergence_diagnosis: ConvergenceDiagnosis | None = None
    confidence_intervals: ConfidenceIntervalResult | None = None
    method_comparison: pd.DataFrame | None = None
    total_wall_time_seconds: float = 0.0

    @property
    def n_runs(self) -> int:
        return len(self.all_runs)

    @property
    def n_converged_runs(self) -> int:
        return sum(1 for r in self.all_runs if r.success)


class MLEEstimator:
    """Fits an :class:`AbstractStateSpaceModel` to ``observations`` via multi-start MLE."""

    def __init__(
        self,
        model: AbstractStateSpaceModel,
        pf_config: ParticleFilterConfig,
        opt_config: OptimizationConfig,
        seed_sequence: SeedSequence,
    ) -> None:
        self._model = model
        self._pf_config = pf_config
        self._opt_config = opt_config
        self._seeds = seed_sequence

    def fit(self, observations: np.ndarray) -> MLEResult:
        start_time = time.time()
        all_runs: list[SingleOptimizationResult] = []

        for method in self._opt_config.methods:
            for start_index in range(self._opt_config.n_multistarts):
                run = self._run_one_attempt(observations, method, start_index)
                all_runs.append(run)

        if not all_runs:
            raise ValueError("optimization.methods and/or optimization.n_multistarts produced zero runs to try.")

        converged_runs = [r for r in all_runs if r.success]
        if converged_runs:
            best_run = min(converged_runs, key=lambda r: r.nll_final)
            converged = True
            diagnosis = None
        else:
            best_run = min(all_runs, key=lambda r: r.nll_final)
            converged = False
            diagnosis = diagnose_convergence_failure(
                best_run, self._opt_config.gradient_tolerance, self._opt_config.max_iterations
            )
            logger.error(
                "MLE did not converge in any of %d (method x multistart) attempts. %s",
                len(all_runs),
                diagnosis.summary_text(),
            )

        confidence_result = None
        if self._opt_config.compute_hessian_ci:
            ci_objective = ParticleFilterObjective(
                self._model,
                self._pf_config,
                observations,
                seed=self._seeds.spawn_seed_int(
                    "mle_pf_common_random_numbers", method=best_run.method, start=best_run.start_index
                ),
            )
            confidence_result = compute_hessian_confidence_intervals(
                ci_objective,
                best_run.theta_final,
                self._opt_config.confidence_level,
                self._opt_config.hessian_step_size,
            )

        total_wall_time = time.time() - start_time
        method_comparison = self._build_method_comparison(all_runs)

        logger.info(
            "MLE complete: best method=%s, start=%d, converged=%s, log-likelihood=%.4f, "
            "total wall time=%.1fs across %d runs.",
            best_run.method,
            best_run.start_index,
            converged,
            -best_run.nll_final,
            total_wall_time,
            len(all_runs),
        )

        return MLEResult(
            best_theta=best_run.theta_final,
            best_log_likelihood=-best_run.nll_final,
            best_method=best_run.method,
            best_start_index=best_run.start_index,
            converged=converged,
            all_runs=all_runs,
            convergence_diagnosis=diagnosis,
            confidence_intervals=confidence_result,
            method_comparison=method_comparison,
            total_wall_time_seconds=total_wall_time,
        )

    def _run_one_attempt(self, observations: np.ndarray, method: str, start_index: int) -> SingleOptimizationResult:
        init_rng = self._seeds.spawn("mle_init", method=method, start=start_index)
        theta0 = self._model.init_theta(init_rng)

        pf_seed = self._seeds.spawn_seed_int("mle_pf_common_random_numbers", method=method, start=start_index)
        objective = ParticleFilterObjective(self._model, self._pf_config, observations, seed=pf_seed)

        return run_single_optimization(
            objective=objective,
            theta0=theta0,
            method=method,
            start_index=start_index,
            max_iterations=self._opt_config.max_iterations,
            function_tolerance=self._opt_config.function_tolerance,
            gradient_tolerance=self._opt_config.gradient_tolerance,
        )

    @staticmethod
    def _build_method_comparison(all_runs: list[SingleOptimizationResult]) -> pd.DataFrame:
        rows = []
        methods = sorted({r.method for r in all_runs})
        for method in methods:
            runs = [r for r in all_runs if r.method == method]
            converged = [r for r in runs if r.success]
            rows.append(
                {
                    "method": method,
                    "n_starts": len(runs),
                    "n_converged": len(converged),
                    "convergence_rate": len(converged) / len(runs) if runs else 0.0,
                    "best_log_likelihood": max((-r.nll_final for r in converged), default=np.nan),
                    "mean_log_likelihood_converged": (
                        float(np.mean([-r.nll_final for r in converged])) if converged else np.nan
                    ),
                    "mean_wall_time_seconds": float(np.mean([r.wall_time_seconds for r in runs])),
                    "mean_n_function_evals": float(np.mean([r.n_function_evals for r in runs])),
                }
            )
        return pd.DataFrame(rows).sort_values("best_log_likelihood", ascending=False).reset_index(drop=True)


__all__ = ["MLEEstimator", "MLEResult"]
