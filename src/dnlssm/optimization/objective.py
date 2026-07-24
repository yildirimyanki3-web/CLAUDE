"""The particle-filter-based negative log-likelihood objective.

A Bootstrap Particle Filter's log-likelihood estimate is a Monte Carlo
quantity: evaluating it twice at the *same* parameter vector with
independently-seeded randomness gives two different numbers. Naively
handing such a noisy objective to a deterministic optimizer (gradient-based
or not) causes exactly the instability the research design warns about:
apparent non-convergence that is really optimizer noise, not a genuine
optimization difficulty.

The standard remedy -- used throughout the particle-MCMC / particle-MLE
literature -- is *common random numbers* (CRN): fix the particle filter's
random seed once per optimization attempt and reuse it for every objective
evaluation within that attempt. The objective becomes a deterministic
(though not everywhere-smooth, due to the discrete resampling step)
function of the parameters, which is what any of ``scipy.optimize``'s
methods actually require to behave sensibly.
"""

from __future__ import annotations

import numpy as np

from dnlssm.config.schema import ParticleFilterConfig
from dnlssm.filters.particle_filter import BootstrapParticleFilter
from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_PENALTY_VALUE = 1e10


class ParticleFilterObjective:
    """Negative log-likelihood of ``observations`` under the DNLSSM, as a function of ``theta``.

    Numerical failures (singular covariance after regularization, complete
    particle degeneracy) are caught and mapped to a large finite penalty
    value rather than propagated as an exception, so that a single bad
    parameter region does not crash an entire multi-start optimization run
    -- the optimizer instead sees a steep but finite "wall" and is pushed
    away from it. Every evaluation is logged for post-hoc inspection.
    """

    def __init__(
        self,
        model: AbstractStateSpaceModel,
        pf_config: ParticleFilterConfig,
        observations: np.ndarray,
        seed: int,
        penalty_value: float = DEFAULT_PENALTY_VALUE,
    ) -> None:
        self._model = model
        self._particle_filter = BootstrapParticleFilter(model, pf_config)
        self._observations = observations
        self._seed = seed
        self._penalty_value = penalty_value

        self.n_evaluations = 0
        self.n_failed_evaluations = 0
        self.evaluation_log: list[tuple[int, float, float]] = []  # (eval_index, nll, ||theta||)

    def __call__(self, theta: np.ndarray) -> float:
        self.n_evaluations += 1
        theta = np.asarray(theta, dtype=np.float64)
        rng = np.random.default_rng(self._seed)  # reset every call: common random numbers
        try:
            result = self._particle_filter.run(self._observations, theta, rng)
            nll = -result.log_likelihood
            if not np.isfinite(nll):
                raise FloatingPointError(f"Non-finite negative log-likelihood: {nll}.")
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            self.n_failed_evaluations += 1
            logger.debug("Objective evaluation %d failed (%s); returning penalty value.", self.n_evaluations, exc)
            nll = self._penalty_value

        self.evaluation_log.append((self.n_evaluations, nll, float(np.linalg.norm(theta))))
        return nll

    @property
    def failure_rate(self) -> float:
        return self.n_failed_evaluations / self.n_evaluations if self.n_evaluations else 0.0


__all__ = ["ParticleFilterObjective", "DEFAULT_PENALTY_VALUE"]
