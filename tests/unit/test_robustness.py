"""Unit tests for dnlssm.diagnostics.robustness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.linear_gaussian_reference import build_stable_linear_model, simulate_from_model

from dnlssm.config.schema import (
    ModelConfig,
    NonlinearFunctionConfig,
    OptimizationConfig,
    ParticleFilterConfig,
)
from dnlssm.diagnostics.robustness import (
    analyze_noise_structure_sensitivity,
    analyze_particle_count_sensitivity,
)
from dnlssm.utils.random_state import SeedSequence


class TestParticleCountSensitivity:
    def test_comparison_table_has_one_row_per_count(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=10, seed=100)
        pf_config = ParticleFilterConfig(n_particles=500)
        result = analyze_particle_count_sensitivity(
            model, theta, observations, pf_config, [100, 200, 400], SeedSequence(1)
        )
        assert len(result.comparison_table) == 3
        assert result.particle_counts == [100, 200, 400]

    def test_stable_case_not_flagged_unstable(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=10, seed=100)
        pf_config = ParticleFilterConfig(n_particles=500)
        result = analyze_particle_count_sensitivity(
            model, theta, observations, pf_config, [3000, 4000, 5000], SeedSequence(1)
        )
        # With enough particles, log-likelihood estimates should be reasonably stable.
        assert result.log_likelihood_relative_std < 0.5

    def test_empty_particle_counts_raises(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=5, seed=100)
        pf_config = ParticleFilterConfig(n_particles=200)
        with pytest.raises(ValueError):
            analyze_particle_count_sensitivity(model, theta, observations, pf_config, [], SeedSequence(1))

    def test_summary_text_mentions_warning_when_unstable(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=10, seed=100)
        pf_config = ParticleFilterConfig(n_particles=500)
        # An implausibly tight threshold forces the "unstable" branch for text-content coverage.
        result = analyze_particle_count_sensitivity(
            model, theta, observations, pf_config, [50, 100], SeedSequence(1),
            relative_std_warning_threshold=1e-12,
        )
        assert result.unstable
        assert "WARNING" in result.summary_text()


class TestNoiseStructureSensitivity:
    def test_comparison_table_has_one_row_per_structure(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=15, seed=100)
        model_config_template = ModelConfig(
            transition_function=NonlinearFunctionConfig(family="linear"),
            observation_function=NonlinearFunctionConfig(family="linear"),
        )
        pf_config = ParticleFilterConfig(n_particles=200)
        opt_config = OptimizationConfig(
            methods=["Nelder-Mead"], n_multistarts=1, max_iterations=10, compute_hessian_ci=False
        )
        result = analyze_noise_structure_sensitivity(
            model_config_template,
            latent_dim=1,
            observation_labels=model.observation_labels(),
            pf_config=pf_config,
            opt_config=opt_config,
            observations=observations,
            target="observation_noise",
            structures=["scalar", "diagonal"],
            seed_sequence=SeedSequence(2),
        )
        assert len(result.comparison_table) == 2
        assert result.best_structure_by_aic in ("scalar", "diagonal")
        assert result.best_structure_by_bic in ("scalar", "diagonal")
        assert "time-varying" in result.arch_effects_caveat

    def test_empty_structures_raises(self) -> None:
        model, theta = build_stable_linear_model(latent_dim=1, obs_dim=2, seed=0)
        observations = simulate_from_model(model, theta, n_time=10, seed=100)
        model_config_template = ModelConfig(
            transition_function=NonlinearFunctionConfig(family="linear"),
            observation_function=NonlinearFunctionConfig(family="linear"),
        )
        pf_config = ParticleFilterConfig(n_particles=200)
        opt_config = OptimizationConfig(methods=["Nelder-Mead"], n_multistarts=1, max_iterations=5, compute_hessian_ci=False)
        with pytest.raises(ValueError):
            analyze_noise_structure_sensitivity(
                model_config_template, 1, model.observation_labels(), pf_config, opt_config,
                observations, "observation_noise", [], SeedSequence(1),
            )
