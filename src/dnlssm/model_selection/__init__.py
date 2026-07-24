"""Latent-dimension model selection.

The research design requires that latent dimensionality never be assumed
fixed: :class:`~dnlssm.model_selection.selector.ModelSelector` fits a
DNLSSM independently at every candidate ``latent_dim`` in
``model_selection.latent_dim_min .. latent_dim_max`` (3-8 by default),
computes in-sample information criteria (AIC, BIC, HQIC) and genuinely
out-of-sample predictive error (RMSE, MAE from one-step-ahead forecasts on
a held-out tail of the series), and selects the best dimension by combining
*all* of these criteria via configurable weights -- never a single metric
in isolation.
"""

from dnlssm.model_selection.information_criteria import (
    InformationCriteria,
    compute_information_criteria,
)
from dnlssm.model_selection.selector import ModelSelectionResult, ModelSelector

__all__ = [
    "InformationCriteria",
    "compute_information_criteria",
    "ModelSelectionResult",
    "ModelSelector",
]
