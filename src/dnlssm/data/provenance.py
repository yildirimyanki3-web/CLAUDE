"""Provenance tracking for the data-acquisition stage.

Every variable in the observation space produces exactly one
:class:`ProvenanceRecord`, regardless of whether acquisition succeeded,
failed, or was skipped. The aggregated :class:`ProvenanceLog` is written
verbatim into every experiment's ``results/`` directory (see
:mod:`dnlssm.experiments.experiment`), giving each run an auditable answer
to "which source did each column of the observation matrix come from".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

import pandas as pd

ProvenanceStatus = Literal[
    "success",
    "failed_all_providers",
    "skipped_no_series_code",
    "skipped_credentials_missing",
    "placeholder",
    "manual_upload",
]


@dataclass
class ProviderAttempt:
    """One attempted (provider, series_code) pair for a single variable."""

    provider: str
    series_code: str | None
    status: Literal["success", "failed", "skipped_no_series_code", "skipped_credentials_missing"]
    error_message: str | None = None


@dataclass
class ProvenanceRecord:
    """The final outcome for one observation-space variable."""

    canonical_id: str
    status: ProvenanceStatus
    attempts: list[ProviderAttempt] = field(default_factory=list)
    resolved_provider: str | None = None
    resolved_series_code: str | None = None
    n_observations: int = 0
    date_range: tuple[str, str] | None = None
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "status": self.status,
            "resolved_provider": self.resolved_provider,
            "resolved_series_code": self.resolved_series_code,
            "n_observations": self.n_observations,
            "date_range": self.date_range,
            "fetched_at": self.fetched_at,
            "attempts": [
                {
                    "provider": a.provider,
                    "series_code": a.series_code,
                    "status": a.status,
                    "error_message": a.error_message,
                }
                for a in self.attempts
            ],
        }


@dataclass
class ProvenanceLog:
    """The complete per-run provenance report across all observation variables."""

    records: dict[str, ProvenanceRecord] = field(default_factory=dict)

    def add(self, record: ProvenanceRecord) -> None:
        self.records[record.canonical_id] = record

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "canonical_id": r.canonical_id,
                "status": r.status,
                "resolved_provider": r.resolved_provider,
                "resolved_series_code": r.resolved_series_code,
                "n_observations": r.n_observations,
                "date_range_start": r.date_range[0] if r.date_range else None,
                "date_range_end": r.date_range[1] if r.date_range else None,
                "n_provider_attempts": len(r.attempts),
            }
            for r in self.records.values()
        ]
        return pd.DataFrame(rows).sort_values("canonical_id").reset_index(drop=True)

    def unresolved_variables(self) -> list[str]:
        """Variables with no usable data after every provider and manual upload were tried."""
        return [
            cid
            for cid, r in self.records.items()
            if r.status in ("failed_all_providers", "skipped_no_series_code", "skipped_credentials_missing")
        ]

    def summary_text(self) -> str:
        n_total = len(self.records)
        n_success = sum(1 for r in self.records.values() if r.status == "success")
        n_manual = sum(1 for r in self.records.values() if r.status == "manual_upload")
        n_placeholder = sum(1 for r in self.records.values() if r.status == "placeholder")
        n_missing = len(self.unresolved_variables())
        lines = [
            f"Data acquisition summary: {n_total} variables total.",
            f"  success (API):     {n_success}",
            f"  success (manual):  {n_manual}",
            f"  placeholder:       {n_placeholder}",
            f"  UNRESOLVED:        {n_missing}",
        ]
        if n_missing:
            lines.append("  Unresolved variables require a provider fix in config/variables.yaml")
            lines.append("  or a manual CSV/Excel upload:")
            for cid in self.unresolved_variables():
                lines.append(f"    - {cid}")
        return "\n".join(lines)


def date_range_str(series: pd.Series) -> tuple[str, str] | None:
    if series.empty:
        return None
    idx = series.dropna().index
    if len(idx) == 0:
        return None
    first, last = idx.min(), idx.max()
    fmt = "%Y-%m-%d"
    if isinstance(first, (pd.Timestamp, date, datetime)):
        return first.strftime(fmt), last.strftime(fmt)
    return str(first), str(last)


__all__ = [
    "ProviderAttempt",
    "ProvenanceRecord",
    "ProvenanceLog",
    "ProvenanceStatus",
    "date_range_str",
]
