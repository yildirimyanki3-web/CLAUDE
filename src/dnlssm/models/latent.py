"""Neutral, economically-uninterpreted latent-state data structures.

:class:`LatentManifold` identifies one latent coordinate system by an
opaque ``manifold_id`` and a dimension; :class:`LatentTrajectory` holds a
time series of estimates (mean, and optionally covariance) on one such
manifold. Today exactly one manifold is ever instantiated (the domestic
DNLSSM's latent space, conventionally ``manifold_id="primary"``), but
nothing in either class assumes that: a future second manifold (e.g. a
global/international latent system) is just another
``LatentManifold`` instance, and a morphism between two manifolds is a
function ``LatentTrajectory -> LatentTrajectory`` that this module does not
need to change to accommodate.

Latent dimensions are always labelled ``State 1``, ``State 2``, ... --
never given an economic name -- so that no interpretive claim is ever
embedded in code; interpretation is the researcher's downstream task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TrajectoryKind = Literal["filtered", "smoothed"]

_DEFAULT_MANIFOLD_ID = "primary"


@dataclass(frozen=True)
class LatentManifold:
    """Identifies one latent coordinate system: an opaque id plus its dimension."""

    manifold_id: str
    dim: int

    @property
    def labels(self) -> list[str]:
        return [f"State {i + 1}" for i in range(self.dim)]

    @classmethod
    def default(cls, dim: int) -> "LatentManifold":
        return cls(manifold_id=_DEFAULT_MANIFOLD_ID, dim=dim)


@dataclass
class LatentTrajectory:
    """A time series of latent-state estimates on one :class:`LatentManifold`.

    ``covariance`` is optional: particle-based estimates may report only
    empirical mean and (separately, via
    :mod:`dnlssm.diagnostics`) empirical spread, while a parametric
    summary can populate the full ``(T, dim, dim)`` covariance path when
    available.
    """

    manifold: LatentManifold
    time_index: pd.DatetimeIndex
    mean: np.ndarray  # (T, dim)
    kind: TrajectoryKind
    covariance: np.ndarray | None = None  # (T, dim, dim) or None

    def __post_init__(self) -> None:
        n_time = len(self.time_index)
        if self.mean.shape != (n_time, self.manifold.dim):
            raise ValueError(
                f"LatentTrajectory.mean has shape {self.mean.shape}, expected "
                f"({n_time}, {self.manifold.dim})."
            )
        if self.covariance is not None and self.covariance.shape != (n_time, self.manifold.dim, self.manifold.dim):
            raise ValueError(
                f"LatentTrajectory.covariance has shape {self.covariance.shape}, expected "
                f"({n_time}, {self.manifold.dim}, {self.manifold.dim})."
            )

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.mean, index=self.time_index, columns=self.manifold.labels)

    def marginal_std(self) -> np.ndarray | None:
        """Per-dimension marginal standard deviation over time, ``(T, dim)``, if covariance is known."""
        if self.covariance is None:
            return None
        return np.sqrt(np.diagonal(self.covariance, axis1=1, axis2=2))


__all__ = ["LatentManifold", "LatentTrajectory", "TrajectoryKind"]
