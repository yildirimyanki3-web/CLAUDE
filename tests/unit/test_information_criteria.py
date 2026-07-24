"""Unit tests for dnlssm.model_selection.information_criteria."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.model_selection.information_criteria import compute_information_criteria


class TestComputeInformationCriteria:
    def test_matches_manual_formula(self) -> None:
        log_lik, n_params, n_obs = -100.0, 10, 500
        result = compute_information_criteria(log_lik, n_params, n_obs)
        assert result.aic == pytest.approx(2 * n_params - 2 * log_lik)
        assert result.bic == pytest.approx(n_params * np.log(n_obs) - 2 * log_lik)
        assert result.hqic == pytest.approx(2 * n_params * np.log(np.log(n_obs)) - 2 * log_lik)

    def test_bic_penalizes_more_than_aic_for_large_sample(self) -> None:
        result = compute_information_criteria(log_likelihood=-100.0, n_params=10, n_observations=10000)
        assert result.bic > result.aic

    def test_more_params_increases_all_criteria_at_fixed_likelihood(self) -> None:
        small = compute_information_criteria(-100.0, n_params=5, n_observations=500)
        large = compute_information_criteria(-100.0, n_params=20, n_observations=500)
        assert large.aic > small.aic
        assert large.bic > small.bic
        assert large.hqic > small.hqic

    def test_higher_likelihood_decreases_all_criteria(self) -> None:
        worse = compute_information_criteria(-200.0, n_params=10, n_observations=500)
        better = compute_information_criteria(-100.0, n_params=10, n_observations=500)
        assert better.aic < worse.aic
        assert better.bic < worse.bic
        assert better.hqic < worse.hqic

    def test_rejects_insufficient_observations(self) -> None:
        with pytest.raises(ValueError):
            compute_information_criteria(-100.0, n_params=5, n_observations=1)

    def test_rejects_negative_params(self) -> None:
        with pytest.raises(ValueError):
            compute_information_criteria(-100.0, n_params=-1, n_observations=100)
