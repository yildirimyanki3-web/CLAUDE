"""Unit tests for dnlssm.model_selection.selector.ModelSelector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import (
    ModelConfig,
    NonlinearFunctionConfig,
    OptimizationConfig,
    ParticleFilterConfig,
)
from dnlssm.model_selection.selector import ModelSelector
from dnlssm.utils.random_state import SeedSequence


def _tiny_observation_matrix() -> tuple[np.ndarray, list[str]]:
    model, theta_true = build_stable_linear_model(latent_dim=2, obs_dim=3, seed=0)
    observations = simulate_from_model(model, theta_true, n_time=24, seed=100)
    return observations, model.observation_labels()


def _tiny_selector(candidate_dims: list[int], criteria_weights: dict[str, float] | None = None) -> ModelSelector:
    _, labels = _tiny_observation_matrix()
    model_config_template = ModelConfig(
        transition_function=NonlinearFunctionConfig(family="linear"),
        observation_function=NonlinearFunctionConfig(family="linear"),
    )
    pf_config = ParticleFilterConfig(n_particles=200)
    opt_config = OptimizationConfig(
        methods=["Nelder-Mead"], n_multistarts=1, max_iterations=10, compute_hessian_ci=False
    )
    weights = criteria_weights or {"aic": 0.5, "oos_rmse": 0.5}
    return ModelSelector(
        observation_labels=labels,
        model_config_template=model_config_template,
        pf_config=pf_config,
        opt_config=opt_config,
        candidate_dims=candidate_dims,
        train_fraction=0.75,
        criteria_weights=weights,
        seed_sequence=SeedSequence(11),
    )


class TestModelSelectorContract:
    def test_one_result_per_candidate_dimension(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2])
        result = selector.select(observations)
        assert {r.latent_dim for r in result.per_dimension} == {1, 2}
        assert len(result.per_dimension) == 2

    def test_selected_dim_is_among_candidates(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2])
        result = selector.select(observations)
        assert result.selected_dim in (1, 2)
        assert result.selected.latent_dim == result.selected_dim

    def test_comparison_table_has_one_row_per_dim_and_expected_columns(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2])
        result = selector.select(observations)
        assert len(result.comparison_table) == 2
        for col in ("latent_dim", "n_params", "converged", "aic", "bic", "hqic", "oos_rmse", "oos_mae"):
            assert col in result.comparison_table.columns

    def test_ranked_table_top_row_matches_selected_dim(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2])
        result = selector.select(observations)
        assert int(result.ranked_selection_table.iloc[0]["latent_dim"]) == result.selected_dim

    def test_result_for_unknown_dim_raises(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2])
        result = selector.select(observations)
        with pytest.raises(KeyError):
            result.result_for(99)

    def test_unknown_criterion_in_weights_raises(self) -> None:
        observations, _ = _tiny_observation_matrix()
        selector = _tiny_selector([1, 2], criteria_weights={"not_a_real_criterion": 1.0})
        with pytest.raises(ValueError):
            selector.select(observations)

    def test_invalid_train_fraction_raises(self) -> None:
        observations, labels = _tiny_observation_matrix()
        model_config_template = ModelConfig(
            transition_function=NonlinearFunctionConfig(family="linear"),
            observation_function=NonlinearFunctionConfig(family="linear"),
        )
        pf_config = ParticleFilterConfig(n_particles=200)
        opt_config = OptimizationConfig(methods=["Nelder-Mead"], n_multistarts=1, max_iterations=5, compute_hessian_ci=False)
        selector = ModelSelector(
            observation_labels=labels,
            model_config_template=model_config_template,
            pf_config=pf_config,
            opt_config=opt_config,
            candidate_dims=[1],
            train_fraction=1.0,  # leaves nothing for the test split
            criteria_weights={"aic": 1.0},
            seed_sequence=SeedSequence(1),
        )
        with pytest.raises(ValueError):
            selector.select(observations)

    def test_oos_error_dicts_cover_all_variables(self) -> None:
        observations, labels = _tiny_observation_matrix()
        selector = _tiny_selector([1])
        result = selector.select(observations)
        dim_result = result.result_for(1)
        assert set(dim_result.oos_rmse_per_variable.keys()) == set(labels)
        assert set(dim_result.oos_mae_per_variable.keys()) == set(labels)
