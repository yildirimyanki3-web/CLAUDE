"""Assembles the final, dimension-checked observation matrix.

:func:`build_observation_matrix` is the single entry point that turns the
raw series returned by :class:`dnlssm.data.manager.DataAcquisitionResult`
into an :class:`ObservationMatrix` ready for the DNLSSM: it aligns every
variable onto a common monthly grid, reports and imputes missing
observations, optionally seasonally adjusts and transforms each series,
standardizes it, and stacks the results into a ``(T, N)`` array with every
shape invariant checked explicitly rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dnlssm.config.schema import ObservationSpaceConfig, PreprocessingConfig
from dnlssm.preprocessing.alignment import align_to_monthly
from dnlssm.preprocessing.exceptions import PreprocessingError
from dnlssm.preprocessing.missing import impute_series, summarize_missing
from dnlssm.preprocessing.transforms import (
    StandardizationParams,
    TransformationLedger,
    VariableTransformationRecord,
    apply_functional_transform,
    seasonally_adjust,
    standardize_series,
    winsorize_series,
)
from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ObservationMatrix:
    """The fully processed observation panel: a ``(T, N)`` array plus its index metadata.

    ``values`` may contain ``NaN`` -- placeholder variables, unresolved
    data-acquisition gaps, and structural leading NaNs from differencing
    are all represented this way. Downstream, the DNLSSM's observation
    likelihood masks missing dimensions per time step rather than requiring
    a dense matrix (see :mod:`dnlssm.models.dnlssm`).
    """

    values: np.ndarray
    time_index: pd.DatetimeIndex
    variable_ids: list[str]
    ledger: TransformationLedger

    def __post_init__(self) -> None:
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.values.ndim != 2:
            raise PreprocessingError(f"Observation matrix must be 2D, got ndim={self.values.ndim}.")
        n_time, n_vars = self.values.shape
        if n_time != len(self.time_index):
            raise PreprocessingError(
                f"Observation matrix time dimension mismatch: {n_time} rows vs "
                f"{len(self.time_index)} time_index entries."
            )
        if n_vars != len(self.variable_ids):
            raise PreprocessingError(
                f"Observation matrix variable dimension mismatch: {n_vars} columns vs "
                f"{len(self.variable_ids)} variable_ids entries."
            )
        if len(set(self.variable_ids)) != len(self.variable_ids):
            raise PreprocessingError("Observation matrix has duplicate variable_ids.")

    @property
    def n_timesteps(self) -> int:
        return self.values.shape[0]

    @property
    def n_variables(self) -> int:
        return self.values.shape[1]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.time_index, columns=self.variable_ids)

    def missing_mask(self) -> np.ndarray:
        """Boolean ``(T, N)`` array, ``True`` where an observation is missing."""
        return np.isnan(self.values)

    def fraction_observed_per_timestep(self) -> np.ndarray:
        return 1.0 - self.missing_mask().mean(axis=1)


def _resolve_target_index(
    observation_space: ObservationSpaceConfig, series_by_variable: dict[str, pd.Series]
) -> pd.DatetimeIndex:
    end_date = observation_space.end_date
    if end_date is None:
        candidate_ends = [
            s.dropna().index.max() for s in series_by_variable.values() if not s.dropna().empty
        ]
        if not candidate_ends:
            raise PreprocessingError(
                "No usable series were supplied and observation_space.end_date is unset; "
                "cannot determine the observation window. Resolve the data-acquisition gaps "
                "reported by DataManager first, or set an explicit end_date in "
                "config/experiment_config.yaml."
            )
        end_date = max(candidate_ends)
    else:
        end_date = pd.Timestamp(end_date)

    start = pd.Timestamp(observation_space.start_date).to_period("M").to_timestamp()
    end = pd.Timestamp(end_date).to_period("M").to_timestamp()
    if end < start:
        raise PreprocessingError(
            f"Resolved observation window is empty: start={start.date()} > end={end.date()}."
        )
    return pd.date_range(start=start, end=end, freq=observation_space.target_frequency)


def _placeholder_record(canonical_id: str, n_total: int, seasonal_requested: bool) -> VariableTransformationRecord:
    return VariableTransformationRecord(
        canonical_id=canonical_id,
        alignment_method="no_data_available",
        n_total=n_total,
        n_missing_before_imputation=n_total,
        pct_missing_before_imputation=100.0,
        max_consecutive_gap=n_total,
        imputation_method="none",
        n_missing_after_imputation=n_total,
        seasonal_adjustment_requested=seasonal_requested,
        seasonal_adjustment_applied=False,
        functional_transform="none",
        standardization_method="none",
        standardization_center=None,
        standardization_scale=None,
        winsorized=False,
        n_missing_final=n_total,
    )


def build_observation_matrix(
    series_by_variable: dict[str, pd.Series],
    observation_space: ObservationSpaceConfig,
    preprocessing: PreprocessingConfig,
) -> ObservationMatrix:
    """Builds the full observation matrix from raw, provider-native series.

    Per-variable pipeline: align to the common monthly grid -> report
    missingness -> impute -> (optionally) seasonally adjust -> apply the
    configured functional transform -> (optionally) standardize -> (optionally)
    winsorize. Variables with no data available (placeholders, or
    unresolved acquisition gaps) become all-``NaN`` columns rather than
    shrinking the matrix, so ``N`` always equals the declared observation
    space size regardless of data availability.
    """
    if observation_space.n_observed == 0:
        raise PreprocessingError("Observation space declares no variables.")

    target_index = _resolve_target_index(observation_space, series_by_variable)
    ledger = TransformationLedger()
    columns: list[np.ndarray] = []
    variable_ids: list[str] = []

    for variable in observation_space.variables:
        variable_ids.append(variable.canonical_id)
        raw_series = series_by_variable.get(variable.canonical_id)

        if variable.is_placeholder or raw_series is None or raw_series.dropna().empty:
            columns.append(np.full(len(target_index), np.nan))
            ledger.add(_placeholder_record(variable.canonical_id, len(target_index), variable.seasonal_adjust))
            if not variable.is_placeholder:
                logger.warning(
                    "'%s' has no usable data; column will be entirely NaN. Resolve via "
                    "config/variables.yaml provider settings or manual upload.",
                    variable.canonical_id,
                )
            continue

        aligned, alignment_method = align_to_monthly(raw_series, variable.native_frequency, target_index)
        missing_before = summarize_missing(aligned, variable.canonical_id)

        imputed = impute_series(
            aligned, preprocessing.missing_data_method, preprocessing.max_consecutive_missing_interpolate
        )
        n_missing_after_imputation = int(imputed.isna().sum())

        if variable.seasonal_adjust:
            adjusted, seasonal_applied = seasonally_adjust(imputed)
        else:
            adjusted, seasonal_applied = imputed, False

        transformed = apply_functional_transform(adjusted, variable.transform)

        if variable.standardize:
            standardized, std_params = standardize_series(transformed, preprocessing.standardization_method)
        else:
            standardized = transformed
            std_params = StandardizationParams(method="none", center=0.0, scale=1.0)

        winsorize_requested = preprocessing.winsorize_lower_quantile is not None
        final_series = winsorize_series(
            standardized, preprocessing.winsorize_lower_quantile, preprocessing.winsorize_upper_quantile
        )

        ledger.add(
            VariableTransformationRecord(
                canonical_id=variable.canonical_id,
                alignment_method=alignment_method,
                n_total=missing_before.n_total,
                n_missing_before_imputation=missing_before.n_missing,
                pct_missing_before_imputation=missing_before.pct_missing,
                max_consecutive_gap=missing_before.max_consecutive_gap,
                imputation_method=preprocessing.missing_data_method,
                n_missing_after_imputation=n_missing_after_imputation,
                seasonal_adjustment_requested=variable.seasonal_adjust,
                seasonal_adjustment_applied=seasonal_applied,
                functional_transform=variable.transform,
                standardization_method=std_params.method,
                standardization_center=std_params.center,
                standardization_scale=std_params.scale,
                winsorized=winsorize_requested,
                n_missing_final=int(final_series.isna().sum()),
            )
        )
        columns.append(final_series.reindex(target_index).to_numpy(dtype=float))

    values = np.column_stack(columns) if columns else np.empty((len(target_index), 0))
    expected_shape = (len(target_index), observation_space.n_observed)
    if values.shape != expected_shape:
        raise PreprocessingError(
            f"Observation matrix shape mismatch after assembly: built {values.shape}, "
            f"expected {expected_shape}."
        )

    matrix = ObservationMatrix(values=values, time_index=target_index, variable_ids=variable_ids, ledger=ledger)
    logger.info(
        "Built observation matrix: T=%d timesteps, N=%d variables, %.1f%% missing overall.",
        matrix.n_timesteps,
        matrix.n_variables,
        100.0 * matrix.missing_mask().mean(),
    )
    return matrix


__all__ = ["ObservationMatrix", "build_observation_matrix"]
