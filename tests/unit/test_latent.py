"""Unit tests for dnlssm.models.latent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnlssm.models.latent import LatentManifold, LatentTrajectory


class TestLatentManifold:
    def test_labels_are_neutral_and_one_indexed(self) -> None:
        manifold = LatentManifold(manifold_id="primary", dim=4)
        assert manifold.labels == ["State 1", "State 2", "State 3", "State 4"]

    def test_default_factory(self) -> None:
        manifold = LatentManifold.default(dim=3)
        assert manifold.manifold_id == "primary"
        assert manifold.dim == 3


class TestLatentTrajectory:
    def test_valid_construction(self) -> None:
        manifold = LatentManifold.default(dim=2)
        idx = pd.date_range("2020-01-01", periods=5, freq="MS")
        mean = np.zeros((5, 2))
        traj = LatentTrajectory(manifold=manifold, time_index=idx, mean=mean, kind="filtered")
        assert traj.as_dataframe().shape == (5, 2)
        assert list(traj.as_dataframe().columns) == ["State 1", "State 2"]

    def test_mean_shape_mismatch_raises(self) -> None:
        manifold = LatentManifold.default(dim=2)
        idx = pd.date_range("2020-01-01", periods=5, freq="MS")
        with pytest.raises(ValueError):
            LatentTrajectory(manifold=manifold, time_index=idx, mean=np.zeros((5, 3)), kind="filtered")

    def test_covariance_shape_mismatch_raises(self) -> None:
        manifold = LatentManifold.default(dim=2)
        idx = pd.date_range("2020-01-01", periods=5, freq="MS")
        with pytest.raises(ValueError):
            LatentTrajectory(
                manifold=manifold,
                time_index=idx,
                mean=np.zeros((5, 2)),
                kind="filtered",
                covariance=np.zeros((5, 3, 3)),
            )

    def test_marginal_std_none_without_covariance(self) -> None:
        manifold = LatentManifold.default(dim=2)
        idx = pd.date_range("2020-01-01", periods=3, freq="MS")
        traj = LatentTrajectory(manifold=manifold, time_index=idx, mean=np.zeros((3, 2)), kind="smoothed")
        assert traj.marginal_std() is None

    def test_marginal_std_computed_from_covariance(self) -> None:
        manifold = LatentManifold.default(dim=2)
        idx = pd.date_range("2020-01-01", periods=3, freq="MS")
        cov = np.stack([np.diag([4.0, 9.0])] * 3)
        traj = LatentTrajectory(
            manifold=manifold, time_index=idx, mean=np.zeros((3, 2)), kind="smoothed", covariance=cov
        )
        std = traj.marginal_std()
        np.testing.assert_allclose(std[0], [2.0, 3.0])
