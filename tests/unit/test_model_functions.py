"""Unit tests for dnlssm.models.functions."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.config.schema import NonlinearFunctionConfig
from dnlssm.models.functions import (
    GeneralizedLogisticFunction,
    LinearFunction,
    NeuralFunction,
    PolynomialFunction,
    build_nonlinear_function,
)

FAMILIES = [
    LinearFunction(3, 5),
    PolynomialFunction(3, 5, degree=2),
    GeneralizedLogisticFunction(3, 5),
    NeuralFunction(3, 5, hidden_dim=4, activation="tanh"),
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: type(f).__name__)
class TestNonlinearFunctionContract:
    def test_init_params_has_correct_length(self, fn) -> None:
        rng = np.random.default_rng(0)
        params = fn.init_params(rng, std=0.1)
        assert params.shape == (fn.n_params,)

    def test_single_point_output_shape(self, fn) -> None:
        rng = np.random.default_rng(0)
        params = fn.init_params(rng, std=0.1)
        x = rng.normal(size=fn.input_dim)
        y = fn(x, params)
        assert y.shape == (fn.output_dim,)

    def test_batched_output_shape(self, fn) -> None:
        rng = np.random.default_rng(0)
        params = fn.init_params(rng, std=0.1)
        x = rng.normal(size=(37, fn.input_dim))
        y = fn(x, params)
        assert y.shape == (37, fn.output_dim)

    def test_batched_matches_single_point_rowwise(self, fn) -> None:
        rng = np.random.default_rng(0)
        params = fn.init_params(rng, std=0.1)
        x = rng.normal(size=(5, fn.input_dim))
        y_batched = fn(x, params)
        y_rowwise = np.stack([fn(x[i], params) for i in range(5)])
        np.testing.assert_allclose(y_batched, y_rowwise, atol=1e-10)

    def test_no_nan_or_inf_for_reasonable_input(self, fn) -> None:
        rng = np.random.default_rng(1)
        params = fn.init_params(rng, std=0.5)
        x = rng.normal(size=(100, fn.input_dim), scale=2.0)
        y = fn(x, params)
        assert np.all(np.isfinite(y))


class TestGeneralizedLogisticBounds:
    def test_output_bounded_by_softplus_L(self) -> None:
        fn = GeneralizedLogisticFunction(2, 3)
        rng = np.random.default_rng(0)
        params = fn.init_params(rng, std=1.0)
        x = rng.normal(size=(50, 2), scale=10.0)
        y = fn(x, params)
        _, _, raw_L, _, _ = fn._unpack(params)
        from dnlssm.utils.numerical import softplus

        L = softplus(raw_L) + 1e-6
        assert np.all(y <= L[None, :] + 1e-6)
        assert np.all(y >= 0.0)


class TestBuildNonlinearFunction:
    @pytest.mark.parametrize(
        "family,expected_type",
        [
            ("linear", LinearFunction),
            ("polynomial", PolynomialFunction),
            ("generalized_logistic", GeneralizedLogisticFunction),
            ("neural", NeuralFunction),
        ],
    )
    def test_dispatch(self, family, expected_type) -> None:
        config = NonlinearFunctionConfig(family=family)
        fn = build_nonlinear_function(config, input_dim=4, output_dim=6)
        assert isinstance(fn, expected_type)
        assert fn.input_dim == 4
        assert fn.output_dim == 6
