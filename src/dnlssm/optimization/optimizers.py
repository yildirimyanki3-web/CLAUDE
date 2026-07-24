"""A uniform, fully-logged wrapper around ``scipy.optimize.minimize``.

Every configured method (``L-BFGS-B``, ``BFGS``, ``Nelder-Mead``, ...) is
run through the same code path, producing directly comparable
:class:`SingleOptimizationResult` objects -- this is what lets
:mod:`dnlssm.optimization.mle` report which method converges more reliably
for a given model, rather than asserting one is best a priori.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from dnlssm.optimization.objective import ParticleFilterObjective
from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)

_GRADIENT_BASED_METHODS = {"BFGS", "L-BFGS-B", "trust-constr"}
_GRADIENT_FD_EPSILON = 1.5e-8


@dataclass
class IterationRecord:
    """One logged optimizer iteration."""

    iteration: int
    negative_log_likelihood: float
    param_norm: float
    param_delta_norm: float
    gradient_norm: float | None


@dataclass
class SingleOptimizationResult:
    """The full, inspectable outcome of one (method, starting point) optimization attempt."""

    method: str
    start_index: int
    initial_theta: np.ndarray
    theta_final: np.ndarray
    nll_final: float
    success: bool
    termination_message: str
    n_iterations: int
    n_function_evals: int
    wall_time_seconds: float
    final_gradient_norm: float | None
    iteration_log: list[IterationRecord] = field(default_factory=list)
    objective: ParticleFilterObjective | None = None

    def iteration_log_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.iteration_log])


def run_single_optimization(
    objective: ParticleFilterObjective,
    theta0: np.ndarray,
    method: str,
    start_index: int,
    max_iterations: int,
    function_tolerance: float,
    gradient_tolerance: float,
    track_gradient_norm: bool = True,
) -> SingleOptimizationResult:
    """Runs one optimization attempt, logging every accepted iteration."""
    iteration_log: list[IterationRecord] = []
    previous_theta = theta0.copy()

    def callback(xk: np.ndarray) -> None:
        nll = objective(xk)
        delta_norm = float(np.linalg.norm(xk - previous_theta))
        gradient_norm = None
        if track_gradient_norm and method in _GRADIENT_BASED_METHODS:
            try:
                gradient = optimize.approx_fprime(xk, objective, _GRADIENT_FD_EPSILON)
                gradient_norm = float(np.linalg.norm(gradient))
            except Exception as exc:  # gradient probing must never abort the run
                logger.debug("Gradient-norm probe failed at iteration %d: %s", len(iteration_log) + 1, exc)
        iteration_log.append(
            IterationRecord(
                iteration=len(iteration_log) + 1,
                negative_log_likelihood=nll,
                param_norm=float(np.linalg.norm(xk)),
                param_delta_norm=delta_norm,
                gradient_norm=gradient_norm,
            )
        )
        previous_theta[:] = xk

    options: dict[str, object] = {"maxiter": max_iterations}
    if method in ("L-BFGS-B", "BFGS"):
        options["gtol"] = gradient_tolerance
    if method == "Nelder-Mead":
        options["fatol"] = function_tolerance
        options["xatol"] = 1e-6
        options["adaptive"] = True

    start_time = time.time()
    scipy_result = optimize.minimize(
        objective, theta0, method=method, callback=callback, options=options
    )
    wall_time = time.time() - start_time

    final_gradient_norm = None
    jac = getattr(scipy_result, "jac", None)
    if jac is not None:
        final_gradient_norm = float(np.linalg.norm(jac))

    logger.info(
        "[%s | start %d] converged=%s, nll=%.4f, n_iter=%d, n_evals=%d, wall_time=%.2fs",
        method,
        start_index,
        scipy_result.success,
        scipy_result.fun,
        scipy_result.nit,
        scipy_result.nfev,
        wall_time,
    )

    return SingleOptimizationResult(
        method=method,
        start_index=start_index,
        initial_theta=theta0,
        theta_final=np.asarray(scipy_result.x, dtype=np.float64),
        nll_final=float(scipy_result.fun),
        success=bool(scipy_result.success),
        termination_message=str(scipy_result.message),
        n_iterations=int(scipy_result.nit),
        n_function_evals=int(scipy_result.nfev),
        wall_time_seconds=wall_time,
        final_gradient_norm=final_gradient_norm,
        iteration_log=iteration_log,
        objective=objective,
    )


__all__ = ["IterationRecord", "SingleOptimizationResult", "run_single_optimization"]
