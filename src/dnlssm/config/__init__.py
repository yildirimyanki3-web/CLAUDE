"""Central configuration system.

Every numerical, structural, or operational choice made anywhere in the
DNLSSM platform (particle counts, optimizer tolerances, latent dimension
search range, variable definitions, transformation choices, ...) is a field
on one of the models in :mod:`dnlssm.config.schema` and is resolved through
:func:`dnlssm.config.loader.load_experiment_config`. No module outside this
package should define a numeric hyperparameter as a bare literal.
"""

from dnlssm.config.loader import load_experiment_config, load_variable_registry
from dnlssm.config.schema import (
    DiagnosticsConfig,
    ExperimentConfig,
    ModelConfig,
    ModelSelectionConfig,
    ObservationSpaceConfig,
    OptimizationConfig,
    ParticleFilterConfig,
    PreprocessingConfig,
    RuntimeConfig,
    VariableSpec,
)

__all__ = [
    "load_experiment_config",
    "load_variable_registry",
    "DiagnosticsConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ModelSelectionConfig",
    "ObservationSpaceConfig",
    "OptimizationConfig",
    "ParticleFilterConfig",
    "PreprocessingConfig",
    "RuntimeConfig",
    "VariableSpec",
]
