"""Systematic latent-dimension search.

:class:`ModelSelector` fits an independent DNLSSM at every candidate
``latent_dim``, using a fixed train/test split of the observation matrix:
parameters are estimated by MLE on the training segment only, and
predictive performance (RMSE, MAE) is scored on the held-out tail using
genuinely out-of-sample one-step-ahead forecasts (see
``FilterResult.predicted_observation_mean``, which conditions only on
``y_1:t-1``) -- never on the in-sample filtered/smoothed estimate, which
would leak information from the scored observation itself.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dnlssm.config.schema import ModelConfig, OptimizationConfig, ParticleFilterConfig
from dnlssm.filters.particle_filter import BootstrapParticleFilter, FilterResult
from dnlssm.model_selection.information_criteria import (
    InformationCriteria,
    compute_information_criteria,
)
from dnlssm.models.dnlssm import DNLSSM
from dnlssm.optimization.mle import MLEEstimator, MLEResult
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.random_state import SeedSequence

logger = get_logger(__name__)


@dataclass
class DimensionResult:
    """The full outcome of fitting and evaluating one candidate latent dimension."""

    latent_dim: int
    model: DNLSSM
    mle_result: MLEResult
    information_criteria: InformationCriteria
    oos_rmse: float
    oos_mae: float
    oos_rmse_per_variable: dict[str, float]
    oos_mae_per_variable: dict[str, float]
    full_series_filter_result: FilterResult
    wall_time_seconds: float


@dataclass
class ModelSelectionResult:
    """The full comparison across all candidate latent dimensions, plus the final selection."""

    per_dimension: list[DimensionResult] = field(default_factory=list)
    comparison_table: pd.DataFrame | None = None
    ranked_selection_table: pd.DataFrame | None = None
    selected_dim: int = 0

    def result_for(self, latent_dim: int) -> DimensionResult:
        for r in self.per_dimension:
            if r.latent_dim == latent_dim:
                return r
        raise KeyError(f"No result for latent_dim={latent_dim}.")

    @property
    def selected(self) -> DimensionResult:
        return self.result_for(self.selected_dim)


def _compute_oos_errors(
    actual: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    error = actual - predicted
    with warnings.catch_warnings():
        # An entirely-missing (placeholder or unresolved) column legitimately produces
        # NaN here ("Mean of empty slice"); this is the correct answer, not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rmse_per_var = np.sqrt(np.nanmean(error**2, axis=0))
        mae_per_var = np.nanmean(np.abs(error), axis=0)
        overall_rmse = float(np.sqrt(np.nanmean(error**2)))
        overall_mae = float(np.nanmean(np.abs(error)))
    return (
        overall_rmse,
        overall_mae,
        dict(zip(labels, rmse_per_var, strict=True)),
        dict(zip(labels, mae_per_var, strict=True)),
    )


class ModelSelector:
    """Fits and compares a DNLSSM at every configured candidate latent dimension."""

    def __init__(
        self,
        observation_labels: list[str],
        model_config_template: ModelConfig,
        pf_config: ParticleFilterConfig,
        opt_config: OptimizationConfig,
        candidate_dims: list[int],
        train_fraction: float,
        criteria_weights: dict[str, float],
        seed_sequence: SeedSequence,
    ) -> None:
        self._observation_labels = observation_labels
        self._model_config_template = model_config_template
        self._pf_config = pf_config
        self._opt_config = opt_config
        self._candidate_dims = candidate_dims
        self._train_fraction = train_fraction
        self._criteria_weights = criteria_weights
        self._seeds = seed_sequence

    def select(self, observation_matrix: np.ndarray) -> ModelSelectionResult:
        n_time = observation_matrix.shape[0]
        n_train = int(round(self._train_fraction * n_time))
        if not (0 < n_train < n_time):
            raise ValueError(
                f"train_fraction={self._train_fraction} yields an empty train or test split "
                f"for T={n_time} timesteps (n_train={n_train}). Choose a train_fraction strictly "
                "between 0 and 1 that leaves at least one observation in each split."
            )
        train_obs = observation_matrix[:n_train]

        per_dim_results: list[DimensionResult] = []
        for dim in self._candidate_dims:
            logger.info(
                "Model selection: fitting latent_dim=%d (%d/%d candidates).",
                dim,
                len(per_dim_results) + 1,
                len(self._candidate_dims),
            )
            per_dim_results.append(self._fit_and_evaluate_one_dim(dim, observation_matrix, train_obs, n_train))

        ranked_table = self._rank_by_combined_score(per_dim_results)
        selected_dim = int(ranked_table.iloc[0]["latent_dim"])
        comparison_table = self._build_comparison_table(per_dim_results)

        logger.info(
            "Model selection complete: selected latent_dim=%d (of candidates %s).",
            selected_dim,
            self._candidate_dims,
        )
        return ModelSelectionResult(
            per_dimension=per_dim_results,
            comparison_table=comparison_table,
            ranked_selection_table=ranked_table,
            selected_dim=selected_dim,
        )

    def _fit_and_evaluate_one_dim(
        self, dim: int, full_obs: np.ndarray, train_obs: np.ndarray, n_train: int
    ) -> DimensionResult:
        start_time = time.time()
        dim_master_seed = self._seeds.spawn_seed_int("model_selection_dim", latent_dim=dim)
        dim_seeds = SeedSequence(dim_master_seed)

        model_config = self._model_config_template.model_copy(update={"latent_dim": dim})
        model = DNLSSM(dim, self._observation_labels, model_config)

        mle_estimator = MLEEstimator(model, self._pf_config, self._opt_config, dim_seeds)
        mle_result = mle_estimator.fit(train_obs)

        n_train_observed = int(np.sum(~np.isnan(train_obs)))
        info_criteria = compute_information_criteria(
            mle_result.best_log_likelihood, model.n_params, n_train_observed
        )

        oos_rng = dim_seeds.spawn("model_selection_oos_eval")
        full_filter_result = BootstrapParticleFilter(model, self._pf_config).run(
            full_obs, mle_result.best_theta, oos_rng
        )
        test_actual = full_obs[n_train:]
        test_predicted = full_filter_result.predicted_observation_mean[n_train:]
        oos_rmse, oos_mae, rmse_per_var, mae_per_var = _compute_oos_errors(
            test_actual, test_predicted, self._observation_labels
        )

        wall_time = time.time() - start_time
        logger.info(
            "latent_dim=%d: converged=%s, log-lik=%.3f, AIC=%.2f, BIC=%.2f, OOS RMSE=%.4f, OOS MAE=%.4f (%.1fs)",
            dim,
            mle_result.converged,
            mle_result.best_log_likelihood,
            info_criteria.aic,
            info_criteria.bic,
            oos_rmse,
            oos_mae,
            wall_time,
        )

        return DimensionResult(
            latent_dim=dim,
            model=model,
            mle_result=mle_result,
            information_criteria=info_criteria,
            oos_rmse=oos_rmse,
            oos_mae=oos_mae,
            oos_rmse_per_variable=rmse_per_var,
            oos_mae_per_variable=mae_per_var,
            full_series_filter_result=full_filter_result,
            wall_time_seconds=wall_time,
        )

    def _rank_by_combined_score(self, results: list[DimensionResult]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "latent_dim": [r.latent_dim for r in results],
                "converged": [r.mle_result.converged for r in results],
                "log_likelihood": [r.information_criteria.log_likelihood for r in results],
                "aic": [r.information_criteria.aic for r in results],
                "bic": [r.information_criteria.bic for r in results],
                "hqic": [r.information_criteria.hqic for r in results],
                "oos_rmse": [r.oos_rmse for r in results],
                "oos_mae": [r.oos_mae for r in results],
            }
        )
        # Rank-based aggregation (lower value = better = lower rank) is robust to the very
        # different numeric scales of AIC/BIC vs. RMSE/MAE, and to outlier magnitude.
        combined_score = pd.Series(0.0, index=df.index)
        valid_columns = set(df.columns) - {"latent_dim", "converged"}
        for criterion, weight in self._criteria_weights.items():
            if criterion not in valid_columns:
                raise ValueError(
                    f"model_selection.criteria_weights references unknown criterion '{criterion}'; "
                    f"expected one of {sorted(valid_columns)}."
                )
            combined_score = combined_score + weight * df[criterion].rank(method="average")
        df["combined_score"] = combined_score
        return df.sort_values("combined_score").reset_index(drop=True)

    @staticmethod
    def _build_comparison_table(results: list[DimensionResult]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "latent_dim": r.latent_dim,
                    "n_params": r.model.n_params,
                    "converged": r.mle_result.converged,
                    "best_optimizer": r.mle_result.best_method,
                    "log_likelihood": r.information_criteria.log_likelihood,
                    "aic": r.information_criteria.aic,
                    "bic": r.information_criteria.bic,
                    "hqic": r.information_criteria.hqic,
                    "oos_rmse": r.oos_rmse,
                    "oos_mae": r.oos_mae,
                    "wall_time_seconds": r.wall_time_seconds,
                }
                for r in results
            ]
        )


__all__ = ["DimensionResult", "ModelSelectionResult", "ModelSelector"]
