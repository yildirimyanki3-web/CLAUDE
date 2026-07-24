"""Unit tests for dnlssm.utils.numerical."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp

from dnlssm.utils.numerical import (
    inverse_softplus,
    log_sum_exp,
    normalize_log_weights,
    regularize_covariance,
    safe_cholesky,
    softplus,
)


class TestLogSumExp:
    def test_matches_scipy_reference(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=50) * 100
        assert log_sum_exp(x) == pytest.approx(scipy_logsumexp(x))

    def test_handles_large_values_without_overflow(self) -> None:
        x = np.array([1e5, 1e5 + 1.0, 1e5 - 1.0])
        result = log_sum_exp(x)
        assert np.isfinite(result)

    def test_all_neg_inf_returns_neg_inf(self) -> None:
        x = np.full(5, -np.inf)
        assert log_sum_exp(x) == -np.inf

    def test_axis_reduction(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = log_sum_exp(x, axis=1)
        expected = scipy_logsumexp(x, axis=1)
        np.testing.assert_allclose(result, expected)


class TestNormalizeLogWeights:
    def test_normalized_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(1)
        log_w = rng.normal(size=200, scale=10.0)
        w, _ = normalize_log_weights(log_w)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0.0)

    def test_uniform_log_weights_give_uniform_weights(self) -> None:
        log_w = np.zeros(10)
        w, ll_increment = normalize_log_weights(log_w)
        np.testing.assert_allclose(w, np.full(10, 0.1))
        assert ll_increment == pytest.approx(0.0)

    def test_raises_on_total_degeneracy(self) -> None:
        with pytest.raises(FloatingPointError):
            normalize_log_weights(np.full(10, -np.inf))


class TestRegularizeCovariance:
    def test_adds_positive_jitter_to_diagonal(self) -> None:
        cov = np.eye(3)
        reg = regularize_covariance(cov, epsilon=1e-3)
        np.testing.assert_allclose(np.diag(reg), 1.0 + 1e-3)

    def test_off_diagonal_untouched(self) -> None:
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        reg = regularize_covariance(cov, epsilon=1e-3)
        assert reg[0, 1] == pytest.approx(0.3)


class TestSafeCholesky:
    def test_recovers_original_for_well_conditioned_matrix(self) -> None:
        cov = np.array([[2.0, 0.1], [0.1, 3.0]])
        chol = safe_cholesky(cov, epsilon=1e-10)
        np.testing.assert_allclose(chol @ chol.T, cov, atol=1e-6)

    def test_repairs_near_singular_matrix(self) -> None:
        cov = np.array([[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]])
        chol = safe_cholesky(cov, epsilon=1e-8)
        assert np.all(np.isfinite(chol))

    def test_repairs_slightly_indefinite_matrix(self) -> None:
        # A symmetric matrix with a small negative eigenvalue, as can arise
        # from floating-point error accumulation in covariance propagation.
        cov = np.array([[1.0, 1.01], [1.01, 1.0]])
        chol = safe_cholesky(cov, epsilon=1e-6, max_attempts=10)
        assert np.all(np.isfinite(chol))


class TestSoftplusRoundTrip:
    def test_round_trip(self) -> None:
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        y = softplus(x)
        assert np.all(y > 0.0)
        x_recovered = inverse_softplus(y)
        np.testing.assert_allclose(x_recovered, x, atol=1e-6)

    def test_softplus_no_overflow_for_large_input(self) -> None:
        x = np.array([1e4])
        y = softplus(x)
        assert np.isfinite(y[0])
        np.testing.assert_allclose(y, x, atol=1e-6)  # softplus(x) ~= x for large x

    def test_inverse_softplus_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError):
            inverse_softplus(np.array([0.0]))
