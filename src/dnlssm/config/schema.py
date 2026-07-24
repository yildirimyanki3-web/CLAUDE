"""Typed configuration schema for the DNLSSM research platform.

Every structural or numerical choice used anywhere downstream (particle
counts, optimizer tolerances, latent-dimension search range, per-variable
transformations, noise covariance structure, ...) is a validated field on
one of the models defined here. Business logic modules must read these
values through an :class:`ExperimentConfig` instance -- they must never
embed the same information as a bare literal.

The schema is intentionally agnostic to *how many* observation variables or
*what* latent dimension is used: ``ObservationSpaceConfig.variables`` is a
list of arbitrary length, and ``ModelConfig.latent_dim`` is a plain integer
with no privileged default baked into any consumer. This is what allows the
observation space to grow (e.g. once the Economic Complexity Index or the
Structural Orientation Index stop being placeholders) or the latent
dimensionality to change without touching any code outside this file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FrequencyLiteral = Literal["daily", "weekly", "monthly", "quarterly", "annual"]
TransformLiteral = Literal["none", "level", "log", "diff", "log_diff", "pct_change"]
ProviderLiteral = Literal[
    "fred", "world_bank", "imf_ifs", "oecd_sdmx", "bis", "evds", "manual"
]
OptimizerLiteral = Literal["L-BFGS-B", "BFGS", "Nelder-Mead", "Powell", "trust-constr"]
ResamplingLiteral = Literal["systematic", "stratified", "multinomial", "residual"]
SmootherLiteral = Literal["backward_simulation", "fixed_lag"]
NonlinearFamilyLiteral = Literal["neural", "polynomial", "generalized_logistic", "linear"]
NoiseStructureLiteral = Literal["diagonal", "full", "scalar"]
SeasonalAdjustLiteral = Literal["stl", "moving_average", "none"]
StandardizationLiteral = Literal["zscore", "minmax", "robust", "none"]
MissingDataLiteral = Literal["linear_interpolate", "ffill", "kalman_impute", "drop"]


class RuntimeConfig(BaseModel):
    """Global run identity, reproducibility, and I/O settings."""

    experiment_name: str = Field(
        default="dnlssm_baseline", description="Human-readable prefix for the experiment ID."
    )
    global_seed: int = Field(
        default=20240101,
        ge=0,
        description="Master seed. All NumPy Generators used in the pipeline are spawned from it.",
    )
    output_root: Path = Field(default=Path("experiments_output"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    n_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "Worker processes for embarrassingly parallel stages (e.g. robustness sweeps). "
            "Numerical results are independent of this value; it only affects wall-clock time."
        ),
    )


class ProviderSeries(BaseModel):
    """One candidate data source for a single observation variable."""

    provider: ProviderLiteral
    series_code: str | None = Field(
        default=None, description="Provider-native series identifier, e.g. a FRED series ID."
    )
    notes: str | None = None


class VariableSpec(BaseModel):
    """Full specification of one column of the observation matrix."""

    canonical_id: str = Field(description="Stable snake_case identifier used throughout the code.")
    description: str = Field(description="Documentation-only human-readable label.")
    unit: str
    native_frequency: FrequencyLiteral
    transform: TransformLiteral = "none"
    standardize: bool = True
    seasonal_adjust: bool = False
    is_placeholder: bool = Field(
        default=False,
        description="True for series with no available provider yet (e.g. ECI, Structural "
        "Orientation Index). Placeholder columns are carried through the full observation "
        "matrix as NaN-safe columns so the architecture requires no changes when data arrives.",
    )
    provider_priority: list[ProviderSeries] = Field(default_factory=list)

    @model_validator(mode="after")
    def _placeholder_has_no_providers_or_is_explicit(self) -> VariableSpec:
        if not self.is_placeholder and not self.provider_priority:
            raise ValueError(
                f"Variable '{self.canonical_id}' is not a placeholder but declares no "
                "provider_priority; either supply at least one provider or set is_placeholder=true."
            )
        return self


class ObservationSpaceConfig(BaseModel):
    """The full, order-preserving list of observed macroeconomic series."""

    target_frequency: Literal["MS"] = Field(
        default="MS", description="Pandas offset alias; MS = month-start, the platform's common grid."
    )
    start_date: date
    end_date: date | None = None
    variables: list[VariableSpec]

    @property
    def n_observed(self) -> int:
        return len(self.variables)

    @field_validator("variables")
    @classmethod
    def _unique_ids(cls, v: list[VariableSpec]) -> list[VariableSpec]:
        ids = [spec.canonical_id for spec in v]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate canonical_id(s) in observation space: {duplicates}")
        if not v:
            raise ValueError("Observation space must declare at least one variable.")
        return v


class PreprocessingConfig(BaseModel):
    """Frequency alignment, missing-data, and transformation policy."""

    missing_data_method: MissingDataLiteral = "linear_interpolate"
    max_consecutive_missing_interpolate: int = Field(default=3, ge=0)
    seasonal_adjustment_method: SeasonalAdjustLiteral = "stl"
    standardization_method: StandardizationLiteral = "zscore"
    winsorize_lower_quantile: float | None = Field(default=None, ge=0.0, le=0.5)
    winsorize_upper_quantile: float | None = Field(default=None, ge=0.5, le=1.0)

    @model_validator(mode="after")
    def _winsorize_bounds_consistent(self) -> PreprocessingConfig:
        lo, hi = self.winsorize_lower_quantile, self.winsorize_upper_quantile
        if (lo is None) != (hi is None):
            raise ValueError("winsorize_lower_quantile and winsorize_upper_quantile must both be set or both be null.")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("winsorize_lower_quantile must be < winsorize_upper_quantile.")
        return self


class NonlinearFunctionConfig(BaseModel):
    """Specifies the functional family used for a transition or observation map.

    The concrete callable is instantiated by
    :func:`dnlssm.models.functions.build_nonlinear_function` purely from
    this configuration -- no function family is privileged in code.
    """

    family: NonlinearFamilyLiteral = "neural"
    hidden_dim: int = Field(default=8, ge=1, description="Used when family == 'neural'.")
    polynomial_degree: int = Field(default=2, ge=1, description="Used when family == 'polynomial'.")
    activation: Literal["tanh", "relu", "softplus"] = "tanh"
    weight_init_std: float = Field(
        default=0.1, gt=0.0, description="Std of the Gaussian used to initialize free weights."
    )


class NoiseConfig(BaseModel):
    """Covariance structure and numerical-stabilization policy for a noise term."""

    structure: NoiseStructureLiteral = "diagonal"
    init_std: float = Field(default=0.5, gt=0.0)
    regularization_epsilon: float = Field(
        default=1e-6,
        gt=0.0,
        description="Diagonal jitter added before every Cholesky factorization to guard "
        "against near-singular covariance matrices.",
    )


class ModelConfig(BaseModel):
    """Structural specification of the Dynamic Nonlinear State Space Model."""

    latent_dim: int = Field(default=4, ge=1, description="Overridden by model_selection sweeps.")
    transition_function: NonlinearFunctionConfig = Field(default_factory=NonlinearFunctionConfig)
    observation_function: NonlinearFunctionConfig = Field(default_factory=NonlinearFunctionConfig)
    process_noise: NoiseConfig = Field(default_factory=NoiseConfig)
    observation_noise: NoiseConfig = Field(default_factory=NoiseConfig)
    initial_state_mean: float = 0.0
    initial_state_std: float = Field(default=1.0, gt=0.0)


class OptimizationConfig(BaseModel):
    """Multi-start Maximum Likelihood Estimation policy."""

    methods: list[OptimizerLiteral] = Field(default_factory=lambda: ["L-BFGS-B", "Nelder-Mead"])
    n_multistarts: int = Field(default=8, ge=1)
    max_iterations: int = Field(default=500, ge=1)
    function_tolerance: float = Field(default=1e-8, gt=0.0)
    gradient_tolerance: float = Field(default=1e-5, gt=0.0)
    initial_param_scale: float = Field(
        default=0.5, gt=0.0, description="Std of the Gaussian used to jitter multi-start initial points."
    )
    compute_hessian_ci: bool = True
    hessian_step_size: float = Field(default=1e-4, gt=0.0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class ParticleFilterConfig(BaseModel):
    """Bootstrap Particle Filter and Particle Smoother settings."""

    n_particles: int = Field(default=4000, ge=100)
    resampling_method: ResamplingLiteral = "systematic"
    adaptive_resampling: bool = True
    ess_threshold_ratio: float = Field(
        default=0.5, gt=0.0, le=1.0, description="Resample when ESS < ratio * n_particles."
    )
    smoother_method: SmootherLiteral = "backward_simulation"
    smoother_n_backward_samples: int = Field(default=200, ge=1)
    fixed_lag_window: int = Field(default=10, ge=1)
    robustness_particle_counts: list[int] = Field(default_factory=lambda: [2000, 3000, 4000, 5000])

    @field_validator("robustness_particle_counts")
    @classmethod
    def _positive_counts(cls, v: list[int]) -> list[int]:
        if any(c <= 0 for c in v):
            raise ValueError("robustness_particle_counts must all be positive.")
        return v


class ModelSelectionConfig(BaseModel):
    """Latent-dimension search policy."""

    latent_dim_min: int = Field(default=3, ge=1)
    latent_dim_max: int = Field(default=8, ge=1)
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    criteria_weights: dict[str, float] = Field(
        default_factory=lambda: {"aic": 0.25, "bic": 0.25, "oos_rmse": 0.25, "oos_mae": 0.25}
    )

    @model_validator(mode="after")
    def _dim_range_consistent(self) -> ModelSelectionConfig:
        if self.latent_dim_min > self.latent_dim_max:
            raise ValueError("latent_dim_min must be <= latent_dim_max.")
        return self

    @property
    def candidate_dims(self) -> list[int]:
        return list(range(self.latent_dim_min, self.latent_dim_max + 1))


class DiagnosticsConfig(BaseModel):
    """Statistical test and robustness-analysis policy."""

    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    ljung_box_lags: int = Field(default=12, ge=1)
    acf_pacf_max_lags: int = Field(default=24, ge=1)
    arch_test_lags: int = Field(default=12, ge=1)
    observation_noise_alternatives: list[NoiseStructureLiteral] = Field(
        default_factory=lambda: ["diagonal", "scalar", "full"]
    )


class ExperimentConfig(BaseModel):
    """Root configuration object: the single source of truth for one run."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    observation_space: ObservationSpaceConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    model_config = {"extra": "forbid", "protected_namespaces": ()}
