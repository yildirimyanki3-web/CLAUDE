"""State-space model interfaces and the Dynamic Nonlinear State Space Model (DNLSSM).

This package deliberately separates three concerns:

``base.py``
    :class:`~dnlssm.models.base.AbstractStateSpaceModel`, the interface any
    latent state-space model family must implement to be usable by the
    filtering, optimization, model-selection, and diagnostics stack. A
    future Dynamic Factor Model, Hidden Markov Model, or Switching Linear
    State Space Model plugs in here without touching any other package.
``functions.py``
    Modular, dimension-agnostic nonlinear function families (neural,
    polynomial, generalized-logistic, linear) used for both the state
    transition and observation maps -- selected entirely through
    :class:`~dnlssm.config.schema.NonlinearFunctionConfig`.
``parameters.py``
    The flat-vector <-> named-block parameter layout, and numerically
    stable, always-positive-definite noise-covariance parameterizations.
``latent.py``
    :class:`~dnlssm.models.latent.LatentManifold` and
    :class:`~dnlssm.models.latent.LatentTrajectory`: neutral,
    economically-uninterpreted containers for latent state estimates,
    structured so a second manifold and inter-manifold morphisms can be
    added later without changing this data model.
``dnlssm.py``
    The concrete DNLSSM implementation.

Latent states are always labelled neutrally (``State 1``, ``State 2``, ...);
no economic or historical meaning is attached anywhere in this package.
"""

from dnlssm.models.base import AbstractStateSpaceModel
from dnlssm.models.dnlssm import DNLSSM
from dnlssm.models.latent import LatentManifold, LatentTrajectory

__all__ = ["AbstractStateSpaceModel", "DNLSSM", "LatentManifold", "LatentTrajectory"]
