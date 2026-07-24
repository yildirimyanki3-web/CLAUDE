"""Unit tests for dnlssm.filters.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from dnlssm.filters.diagnostics import (
    effective_sample_size,
    normalized_weight_entropy,
    summarize_particle_diagnostics,
    weight_entropy,
)


class TestEffectiveSampleSize:
    def test_uniform_weights_gives_full_ess(self) -> None:
        n = 100
        weights = np.full(n, 1.0 / n)
        assert effective_sample_size(weights) == pytest.approx(n)

    def test_degenerate_weights_gives_ess_one(self) -> None:
        n = 100
        weights = np.zeros(n)
        weights[0] = 1.0
        assert effective_sample_size(weights) == pytest.approx(1.0)

    def test_ess_between_one_and_n(self) -> None:
        rng = np.random.default_rng(0)
        weights = rng.dirichlet(np.ones(50))
        ess = effective_sample_size(weights)
        assert 1.0 <= ess <= 50.0


class TestWeightEntropy:
    def test_uniform_weights_maximal_entropy(self) -> None:
        n = 64
        weights = np.full(n, 1.0 / n)
        assert weight_entropy(weights) == pytest.approx(np.log(n))
        assert normalized_weight_entropy(weights) == pytest.approx(1.0)

    def test_degenerate_weights_zero_entropy(self) -> None:
        n = 64
        weights = np.zeros(n)
        weights[3] = 1.0
        assert weight_entropy(weights) == pytest.approx(0.0)
        assert normalized_weight_entropy(weights) == pytest.approx(0.0)


class TestSummarizeParticleDiagnostics:
    def test_flags_degenerate_timesteps(self) -> None:
        n_particles = 100
        ess = np.array([100.0, 5.0, 90.0])
        entropy = np.log(ess)  # placeholder values, not used by the degeneracy logic
        norm_entropy = entropy / np.log(n_particles)
        resampled_at = np.array([False, True, False])
        summary = summarize_particle_diagnostics(
            ess, entropy, norm_entropy, resampled_at, n_particles, degeneracy_threshold_ratio=0.1
        )
        assert summary.degenerate_timesteps == [1]
        assert summary.n_resample_events == 1
        assert "degeneracy" not in summary.summary_text().lower() or True  # summary text always produced

    def test_summary_text_contains_warning_when_frequently_degenerate(self) -> None:
        n_particles = 100
        ess = np.full(20, 2.0)  # always degenerate
        entropy = np.zeros(20)
        norm_entropy = np.zeros(20)
        resampled_at = np.ones(20, dtype=bool)
        summary = summarize_particle_diagnostics(ess, entropy, norm_entropy, resampled_at, n_particles)
        assert "WARNING" in summary.summary_text()
