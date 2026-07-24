"""Orchestrates data acquisition across providers with full provenance tracking.

:class:`DataManager` is the only place in the platform that knows how to
turn an :class:`~dnlssm.config.schema.ObservationSpaceConfig` into actual
time series. For each variable it walks ``provider_priority`` in the order
declared in ``config/variables.yaml`` (which, by convention, lists
``manual`` last as the universal fallback), skipping -- and recording
*why* -- any entry with no configured series code or missing credentials.
The first successful fetch wins; every attempt, not just the winning one,
is recorded in the returned :class:`~dnlssm.data.provenance.ProvenanceLog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dnlssm.config.schema import ObservationSpaceConfig, VariableSpec
from dnlssm.data.base import DataConnector, DataFetchError
from dnlssm.data.connectors import (
    BISConnector,
    EVDSConnector,
    FREDConnector,
    IMFIFSConnector,
    ManualUploadConnector,
    OECDConnector,
    WorldBankConnector,
)
from dnlssm.data.credentials import ProviderCredentials
from dnlssm.data.provenance import (
    ProvenanceLog,
    ProvenanceRecord,
    ProvenanceStatus,
    ProviderAttempt,
    date_range_str,
)
from dnlssm.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataAcquisitionResult:
    """The output of a full data-acquisition run."""

    series_by_variable: dict[str, pd.Series] = field(default_factory=dict)
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)


def build_default_connectors(
    credentials: ProviderCredentials, manual_upload_dir: Path
) -> dict[str, DataConnector]:
    """Instantiates the standard connector registry used by :class:`DataManager`."""
    return {
        "fred": FREDConnector(api_key=credentials.fred_api_key),
        "world_bank": WorldBankConnector(),
        "imf_ifs": IMFIFSConnector(),
        "oecd_sdmx": OECDConnector(),
        "bis": BISConnector(),
        "evds": EVDSConnector(api_key=credentials.evds_api_key),
        "manual": ManualUploadConnector(upload_dir=manual_upload_dir),
    }


class DataManager:
    """Fetches every variable in an observation space, with graceful, logged fallback."""

    def __init__(
        self,
        connectors: dict[str, DataConnector],
        manual_upload_dir: Path,
    ) -> None:
        self._connectors = connectors
        self._manual_upload_dir = Path(manual_upload_dir)

    def fetch_all(self, observation_space: ObservationSpaceConfig) -> DataAcquisitionResult:
        result = DataAcquisitionResult()
        for variable in observation_space.variables:
            record, series = self._fetch_one(variable, observation_space)
            result.provenance.add(record)
            if series is not None:
                result.series_by_variable[variable.canonical_id] = series

        logger.info(result.provenance.summary_text())
        return result

    def _fetch_one(
        self, variable: VariableSpec, observation_space: ObservationSpaceConfig
    ) -> tuple[ProvenanceRecord, pd.Series | None]:
        if variable.is_placeholder:
            record = ProvenanceRecord(canonical_id=variable.canonical_id, status="placeholder")
            logger.info("'%s' is a placeholder variable; no data source attempted.", variable.canonical_id)
            return record, None

        attempts: list[ProviderAttempt] = []
        start_date = observation_space.start_date
        end_date = observation_space.end_date

        for provider_series in variable.provider_priority:
            provider = provider_series.provider
            connector = self._connectors.get(provider)
            if connector is None:
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        series_code=provider_series.series_code,
                        status="failed",
                        error_message=f"No connector registered for provider '{provider}'.",
                    )
                )
                continue

            series_code: str | None
            if provider == "manual":
                series_code = variable.canonical_id  # manual files are named by canonical_id
            else:
                series_code = provider_series.series_code

            if series_code is None:
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        series_code=None,
                        status="skipped_no_series_code",
                    )
                )
                continue

            if not connector.is_available():
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        series_code=series_code,
                        status="skipped_credentials_missing",
                    )
                )
                continue

            try:
                series = connector.fetch_series(series_code, start_date, end_date)
                if series.dropna().empty:
                    raise DataFetchError(
                        f"Provider '{provider}' returned a series for '{series_code}' "
                        "that is entirely missing after parsing."
                    )
            except DataFetchError as exc:
                logger.warning(
                    "Fetch failed for '%s' via provider '%s' (series '%s'): %s",
                    variable.canonical_id,
                    provider,
                    series_code,
                    exc,
                )
                attempts.append(
                    ProviderAttempt(
                        provider=provider, series_code=series_code, status="failed", error_message=str(exc)
                    )
                )
                continue

            attempts.append(ProviderAttempt(provider=provider, series_code=series_code, status="success"))
            status: ProvenanceStatus = "manual_upload" if provider == "manual" else "success"
            record = ProvenanceRecord(
                canonical_id=variable.canonical_id,
                status=status,
                attempts=attempts,
                resolved_provider=provider,
                resolved_series_code=series_code,
                n_observations=int(series.notna().sum()),
                date_range=date_range_str(series),
            )
            logger.info(
                "'%s' resolved via provider '%s' (series '%s'), %d observations.",
                variable.canonical_id,
                provider,
                series_code,
                record.n_observations,
            )
            return record, series

        # Every configured provider (and the implicit manual fallback) failed or was skipped.
        record = ProvenanceRecord(
            canonical_id=variable.canonical_id,
            status="failed_all_providers",
            attempts=attempts,
        )
        logger.error(
            "'%s' could not be resolved from any configured provider or manual upload "
            "(%d attempt(s)); see provenance log for details.",
            variable.canonical_id,
            len(attempts),
        )
        return record, None


__all__ = ["DataManager", "DataAcquisitionResult", "build_default_connectors"]
