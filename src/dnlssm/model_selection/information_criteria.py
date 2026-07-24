"""Information criteria for comparing DNLSSM specifications of different latent dimension.

All three standard criteria are computed together (never just one), since
AIC, BIC, and HQIC penalize model complexity differently (BIC most
strongly, tracking ``log(n)``; HQIC, ``log(log(n))``; AIC, a constant) and
can disagree about the preferred dimension -- exactly the kind of
disagreement :mod:`dnlssm.model_selection.selector` is designed to surface
rather than paper over with a single number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InformationCriteria:
    """AIC, BIC, and HQIC for one fitted model, plus the inputs used to compute them."""

    log_likelihood: float
    n_params: int
    n_observations: int
    aic: float
    bic: float
    hqic: float


def compute_information_criteria(
    log_likelihood: float, n_params: int, n_observations: int
) -> InformationCriteria:
    """Computes AIC, BIC, and HQIC from a maximized log-likelihood.

    ``n_observations`` should be the count of *actually observed* (non-missing)
    scalar data points contributing to the likelihood, not merely the number
    of time steps -- otherwise BIC/HQIC's sample-size penalty would be
    systematically wrong for a panel with substantial missingness.
    """
    if n_observations <= 1:
        raise ValueError(f"n_observations must be > 1 to compute BIC/HQIC penalties, got {n_observations}.")
    if n_params < 0:
        raise ValueError(f"n_params must be >= 0, got {n_params}.")

    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_observations) - 2 * log_likelihood
    hqic = 2 * n_params * np.log(np.log(n_observations)) - 2 * log_likelihood

    return InformationCriteria(
        log_likelihood=log_likelihood,
        n_params=n_params,
        n_observations=n_observations,
        aic=float(aic),
        bic=float(bic),
        hqic=float(hqic),
    )


__all__ = ["InformationCriteria", "compute_information_criteria"]
