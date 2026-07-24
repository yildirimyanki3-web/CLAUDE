"""Unit tests for dnlssm.config: schema validation and the YAML loader."""

from __future__ import annotations

import pydantic
import pytest

from dnlssm.config import load_experiment_config, load_variable_registry
from dnlssm.config.loader import ConfigError
from dnlssm.config.schema import ModelSelectionConfig, ObservationSpaceConfig, VariableSpec


class TestShippedConfigLoads:
    def test_default_config_loads_and_validates(self) -> None:
        cfg = load_experiment_config()
        assert cfg.observation_space.n_observed == 17
        assert cfg.particle_filter.n_particles == 4000

    def test_variable_registry_has_no_duplicate_ids(self) -> None:
        variables = load_variable_registry()
        ids = [v.canonical_id for v in variables]
        assert len(ids) == len(set(ids))

    def test_placeholder_variables_flagged(self) -> None:
        variables = load_variable_registry()
        placeholders = {v.canonical_id for v in variables if v.is_placeholder}
        assert placeholders == {"economic_complexity_index", "structural_orientation_index"}


class TestVariableSpecValidation:
    def test_non_placeholder_requires_provider(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            VariableSpec(
                canonical_id="x",
                description="x",
                unit="percent",
                native_frequency="monthly",
                provider_priority=[],
                is_placeholder=False,
            )

    def test_placeholder_may_omit_provider(self) -> None:
        spec = VariableSpec(
            canonical_id="x",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[],
            is_placeholder=True,
        )
        assert spec.is_placeholder


class TestObservationSpaceConfig:
    def test_duplicate_ids_rejected(self) -> None:
        spec_kwargs = dict(
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[{"provider": "manual"}],
        )
        with pytest.raises(pydantic.ValidationError):
            ObservationSpaceConfig(
                start_date="2020-01-01",
                variables=[
                    VariableSpec(canonical_id="dup", **spec_kwargs),
                    VariableSpec(canonical_id="dup", **spec_kwargs),
                ],
            )

    def test_empty_variable_list_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ObservationSpaceConfig(start_date="2020-01-01", variables=[])


class TestModelSelectionConfig:
    def test_dim_range_validated(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ModelSelectionConfig(latent_dim_min=8, latent_dim_max=3)

    def test_candidate_dims_inclusive(self) -> None:
        cfg = ModelSelectionConfig(latent_dim_min=3, latent_dim_max=5)
        assert cfg.candidate_dims == [3, 4, 5]


class TestLoaderErrorHandling:
    def test_missing_file_raises_config_error(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            load_experiment_config(config_path=tmp_path / "nonexistent.yaml")
