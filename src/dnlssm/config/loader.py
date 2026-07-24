"""Loads and validates the experiment configuration from YAML files.

The platform reads exactly two human-edited files to determine every
hyperparameter of a run:

``config/experiment_config.yaml``
    Runtime, preprocessing, model, optimization, particle filter, model
    selection, and diagnostics settings.
``config/variables.yaml``
    The observation-space variable registry (the 17 series described in the
    research design, plus any future additions).

Both are merged into a single validated :class:`~dnlssm.config.schema.ExperimentConfig`.
Secrets (API keys) are never read from these files -- they are resolved
separately from environment variables / a local ``.env`` file by
:mod:`dnlssm.data.credentials` so that config files remain safe to commit
and to cite in a reproducibility appendix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnlssm.config.schema import ExperimentConfig, ObservationSpaceConfig, VariableSpec

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class ConfigError(RuntimeError):
    """Raised when the configuration files are missing or fail validation."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(content, dict):
        raise ConfigError(f"Configuration file must contain a mapping at the top level: {path}")
    return content


def load_variable_registry(path: Path | None = None) -> list[VariableSpec]:
    """Loads the observation-variable registry from ``variables.yaml``.

    Parameters
    ----------
    path:
        Path to the YAML file. Defaults to ``config/variables.yaml`` at the
        project root.
    """
    resolved = path or (_DEFAULT_CONFIG_DIR / "variables.yaml")
    content = _read_yaml(resolved)
    raw_variables = content.get("variables")
    if not raw_variables:
        raise ConfigError(f"No 'variables' key found in {resolved}")
    try:
        return [VariableSpec.model_validate(entry) for entry in raw_variables]
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigError(f"Invalid variable specification in {resolved}: {exc}") from exc


def load_experiment_config(
    config_path: Path | None = None,
    variables_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Builds a fully validated :class:`ExperimentConfig` from disk.

    Parameters
    ----------
    config_path:
        Path to ``experiment_config.yaml``. Defaults to the project's
        ``config/experiment_config.yaml``.
    variables_path:
        Path to ``variables.yaml``. Defaults to the project's
        ``config/variables.yaml``.
    overrides:
        Optional nested-dict overrides applied after loading (e.g. from CLI
        flags), merged shallowly at the top level of each section. Intended
        for programmatic use (tests, notebooks); interactive users should
        prefer editing the YAML files directly so the run stays traceable
        to a single committed configuration.
    """
    resolved_config_path = config_path or (_DEFAULT_CONFIG_DIR / "experiment_config.yaml")
    content = _read_yaml(resolved_config_path)

    variables = load_variable_registry(variables_path)
    observation_space_raw = dict(content.get("observation_space", {}))
    observation_space_raw["variables"] = [v.model_dump(mode="json") for v in variables]
    content["observation_space"] = observation_space_raw

    if overrides:
        content = _deep_merge(content, overrides)

    try:
        return ExperimentConfig.model_validate(content)
    except Exception as exc:
        raise ConfigError(
            f"Experiment configuration failed validation ({resolved_config_path}): {exc}"
        ) from exc


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = ["ConfigError", "load_experiment_config", "load_variable_registry", "ObservationSpaceConfig"]
