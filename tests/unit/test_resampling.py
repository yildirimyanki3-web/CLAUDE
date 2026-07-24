"""Unit tests for dnlssm.filters.resampling."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.filters.resampling import (
    multinomial_resample,
    resample,
    residual_resample,
    stratified_resample,
    systematic_resample,
)

ALL_METHODS = [systematic_resample, stratified_resample, multinomial_resample, residual_resample]


@pytest.mark.parametrize("method", ALL_METHODS, ids=lambda f: f.__name__)
class TestResamplingContract:
    def test_output_shape_and_range(self, method) -> None:
        rng = np.random.default_rng(0)
        n = 500
        weights = rng.dirichlet(np.ones(n))
        indices = method(weights, rng)
        assert indices.shape == (n,)
        assert indices.min() >= 0
        assert indices.max() < n

    def test_degenerate_weights_selects_single_particle(self, method) -> None:
        rng = np.random.default_rng(0)
        n = 20
        weights = np.zeros(n)
        weights[7] = 1.0
        indices = method(weights, rng)
        assert np.all(indices == 7)

    def test_uniform_weights_approximately_uniform_selection(self, method) -> None:
        rng = np.random.default_rng(0)
        n = 200
        weights = np.full(n, 1.0 / n)
        counts = np.zeros(n)
        for _ in range(50):
            indices = method(weights, rng)
            counts += np.bincount(indices, minlength=n)
        # Each particle selected roughly 50 times on average (50 trials * n draws / n particles).
        assert counts.mean() == pytest.approx(50.0, rel=0.05)
        assert counts.std() / counts.mean() < 0.5

    def test_reproducible_with_same_seed(self, method) -> None:
        n = 50
        weights = np.random.default_rng(1).dirichlet(np.ones(n))
        idx1 = method(weights, np.random.default_rng(42))
        idx2 = method(weights, np.random.default_rng(42))
        np.testing.assert_array_equal(idx1, idx2)


class TestResampleDispatch:
    def test_dispatches_by_name(self) -> None:
        rng = np.random.default_rng(0)
        weights = np.full(10, 0.1)
        indices = resample(weights, "systematic", rng)
        assert indices.shape == (10,)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            resample(np.full(10, 0.1), "not_a_method", np.random.default_rng(0))
