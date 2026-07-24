"""Centralized, reproducible random-number management.

A single master seed (``ExperimentConfig.runtime.global_seed``) is the only
source of randomness-affecting configuration in the whole platform. Every
component that needs randomness (multi-start initial parameter draws,
particle initialization, resampling, bootstrap confidence intervals, ...)
receives an independent :class:`numpy.random.Generator` spawned from that
single seed via :class:`numpy.random.SeedSequence`. This gives two
guarantees simultaneously:

1. **Full reproducibility**: re-running with the same ``global_seed`` and
   the same spawn order reproduces bit-identical results.
2. **Statistical independence between components**: unlike naively reusing
   one global ``np.random.seed(...)`` call (which silently correlates
   "independent" stochastic components such as multi-start restarts and
   particle-filter resampling), each spawned child stream is independent by
   construction (NumPy's ``SeedSequence.spawn`` uses distinct entropy per
   child).

Usage
-----
>>> from dnlssm.utils.random_state import SeedSequence
>>> seeds = SeedSequence(global_seed=20240101)
>>> rng_pf = seeds.spawn("particle_filter")
>>> rng_ms_dim4_start0 = seeds.spawn("multistart", latent_dim=4, start_index=0)
"""

from __future__ import annotations

import hashlib

import numpy as np


class SeedSequence:
    """Deterministic, named child-stream factory built on ``np.random.SeedSequence``.

    Child streams are keyed by a human-readable label plus arbitrary
    keyword tags (e.g. ``latent_dim=5, start_index=2``); the same label and
    tags always yield the same child generator for a fixed
    ``global_seed``, regardless of call order, which is essential for
    reproducibility when stages are re-run independently (e.g. resuming a
    model-selection sweep).
    """

    def __init__(self, global_seed: int) -> None:
        self._global_seed = int(global_seed)
        self._root = np.random.SeedSequence(self._global_seed)

    @property
    def global_seed(self) -> int:
        return self._global_seed

    def _child_seed_sequence(self, label: str, **tags: object) -> np.random.SeedSequence:
        key = label + "|" + "|".join(f"{k}={v}" for k, v in sorted(tags.items()))
        # Derive a deterministic 128-bit entropy value from the label so that
        # spawn order (which SeedSequence.spawn() depends on if used directly)
        # cannot silently change results when call order changes between runs.
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        entropy = int.from_bytes(digest[:16], byteorder="big")
        return np.random.SeedSequence([self._global_seed, entropy])

    def spawn(self, label: str, **tags: object) -> np.random.Generator:
        """Returns a fresh, independent, deterministic ``Generator`` for ``label``."""
        return np.random.default_rng(self._child_seed_sequence(label, **tags))

    def spawn_seed_int(self, label: str, **tags: object) -> int:
        """Returns a plain integer seed, for APIs (e.g. some SciPy routines) that need one."""
        return int(self._child_seed_sequence(label, **tags).generate_state(1)[0])


def spawn_generator(global_seed: int, label: str, **tags: object) -> np.random.Generator:
    """Convenience one-shot wrapper around :class:`SeedSequence` for a single draw."""
    return SeedSequence(global_seed).spawn(label, **tags)


__all__ = ["SeedSequence", "spawn_generator"]
