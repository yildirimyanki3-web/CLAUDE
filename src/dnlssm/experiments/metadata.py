"""Experiment metadata: everything needed to audit and reproduce one run.

The research design requires that every run record, at minimum, the data
sources used, hyperparameters, latent dimension, particle count, iteration
counts, optimizer, runtime, hardware, and convergence/performance metrics.
:class:`ExperimentMetadata` captures the fields that are always meaningful
(identity, reproducibility, environment) and leaves an open ``extra`` dict
for pipeline-stage-specific results (model-selection summary, robustness
findings, ...) -- so adding a new diagnostic to the platform never requires
changing this schema.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from dnlssm import __version__ as _dnlssm_version


def collect_hardware_info() -> dict[str, Any]:
    """Environment/hardware fingerprint, for reproducibility auditing (not for tuning)."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "system_version": platform.version(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
    }


@dataclass
class ExperimentMetadata:
    """Reproducibility and provenance record for one experiment run."""

    experiment_id: str
    created_at_utc: str
    software_version: str
    global_seed: int
    hardware: dict[str, Any] = field(default_factory=collect_hardware_info)
    runtime_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, experiment_id: str, global_seed: int) -> ExperimentMetadata:
        return cls(
            experiment_id=experiment_id,
            created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            software_version=_dnlssm_version,
            global_seed=global_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ExperimentMetadata", "collect_hardware_info"]
