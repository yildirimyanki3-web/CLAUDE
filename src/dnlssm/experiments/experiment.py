"""Experiment identity and directory management.

Every run gets a unique, timestamped ``experiment_id`` and a standardized
directory layout (``config.yaml``, ``metadata.json``, ``run.log``,
``results/``, ``figures/``, ``reports/``), so any two runs can be compared
or reproduced without ambiguity about which files belong to which run.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from dnlssm.config.schema import ExperimentConfig
from dnlssm.experiments.metadata import ExperimentMetadata
from dnlssm.reports.table_export import export_table
from dnlssm.utils.logging_config import configure_logging


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    config_file: Path
    metadata_file: Path
    log_file: Path
    results_dir: Path
    figures_dir: Path
    reports_dir: Path
    tables_dir: Path


class Experiment:
    """A single, uniquely-identified, reproducible experiment run."""

    def __init__(self, experiment_id: str, paths: ExperimentPaths) -> None:
        self.experiment_id = experiment_id
        self.paths = paths

    @classmethod
    def create(cls, config: ExperimentConfig, output_root: Path | None = None) -> Experiment:
        """Creates a fresh, uniquely-named experiment directory and persists ``config``."""
        root_base = Path(output_root) if output_root is not None else Path(config.runtime.output_root)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:8]
        experiment_id = f"{config.runtime.experiment_name}_{timestamp}_{short_id}"

        root = root_base / experiment_id
        results_dir = root / "results"
        figures_dir = root / "figures"
        reports_dir = root / "reports"
        tables_dir = reports_dir / "tables"
        for directory in (results_dir, figures_dir, reports_dir, tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = ExperimentPaths(
            root=root,
            config_file=root / "config.yaml",
            metadata_file=root / "metadata.json",
            log_file=root / "run.log",
            results_dir=results_dir,
            figures_dir=figures_dir,
            reports_dir=reports_dir,
            tables_dir=tables_dir,
        )
        experiment = cls(experiment_id, paths)
        experiment._save_config(config)
        configure_logging(level=config.runtime.log_level, log_file=paths.log_file)
        return experiment

    def _save_config(self, config: ExperimentConfig) -> None:
        with self.paths.config_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)

    def save_metadata(self, metadata: ExperimentMetadata) -> Path:
        self.paths.metadata_file.write_text(
            json.dumps(metadata.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        return self.paths.metadata_file

    def save_table(self, df: pd.DataFrame, name: str, caption: str | None = None) -> tuple[Path, Path]:
        """Exports a DataFrame as both CSV and LaTeX under ``reports/tables/``."""
        csv_path = self.paths.tables_dir / f"{name}.csv"
        tex_path = self.paths.tables_dir / f"{name}.tex"
        return export_table(df, csv_path, tex_path, caption=caption)

    def save_results_json(self, name: str, data: dict) -> Path:
        path = self.paths.results_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def figure_path(self, name: str) -> Path:
        return self.paths.figures_dir / name

    def save_report_markdown(self, markdown_text: str, name: str = "report.md") -> Path:
        path = self.paths.reports_dir / name
        path.write_text(markdown_text, encoding="utf-8")
        return path


__all__ = ["Experiment", "ExperimentPaths"]
