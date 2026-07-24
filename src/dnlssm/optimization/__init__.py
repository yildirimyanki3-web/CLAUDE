"""Maximum Likelihood Estimation of DNLSSM parameters via multi-start optimization.

``objective.py``
    :class:`~dnlssm.optimization.objective.ParticleFilterObjective`: the
    negative log-likelihood objective, using *common random numbers* (a
    fixed particle-filter random seed reused across every evaluation within
    one optimization attempt) so that the Monte Carlo estimator becomes a
    deterministic function of the parameters for that attempt -- essential
    for the stability of any gradient-based or gradient-free optimizer
    applied to a particle-filter likelihood.
``optimizers.py``
    A uniform wrapper around ``scipy.optimize.minimize`` (L-BFGS-B, BFGS,
    Nelder-Mead, ...) that logs every iteration's negative log-likelihood,
    parameter norm, parameter step size, and (for gradient-based methods)
    gradient norm.
``convergence.py``
    Automatic, rule-based diagnosis of *why* an optimization run failed to
    converge (iteration budget, degenerate evaluations, large final
    gradient, divergent parameter magnitude), each with an actionable
    recommendation -- never a silent "converged=False".
``mle.py``
    :class:`~dnlssm.optimization.mle.MLEEstimator`: runs every configured
    optimizer method from every multi-start initial point, selects the best
    converged run, cross-method comparison, and Hessian-based asymptotic
    confidence intervals.
"""

from dnlssm.optimization.mle import MLEEstimator, MLEResult
from dnlssm.optimization.objective import ParticleFilterObjective

__all__ = ["MLEEstimator", "MLEResult", "ParticleFilterObjective"]
