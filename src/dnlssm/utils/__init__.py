"""Cross-cutting utilities: logging, reproducible randomness, and numerical stability.

Nothing in this package encodes economic or modeling assumptions; it exists
so that every other sub-package can share one logging configuration, one
seed-management strategy, and one set of numerically robust primitives
(log-sum-exp, covariance regularization, safe Cholesky) instead of
reimplementing them ad hoc.
"""

from dnlssm.utils.logging_config import get_logger, configure_logging
from dnlssm.utils.random_state import SeedSequence, spawn_generator
from dnlssm.utils.numerical import (
    log_sum_exp,
    regularize_covariance,
    safe_cholesky,
    softplus,
    normalize_log_weights,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "SeedSequence",
    "spawn_generator",
    "log_sum_exp",
    "regularize_covariance",
    "safe_cholesky",
    "softplus",
    "normalize_log_weights",
]
