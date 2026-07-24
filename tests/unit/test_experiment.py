"""Unit tests for dnlssm.experiments.experiment.Experiment."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from dnlssm.config import load_experiment_config
from dnlssm.experiments.experiment import Experiment
from dnlssm.experiments.metadata import ExperimentMetadata


def _config(tmp_path):
    return load_experiment_config(overrides={"runtime": {"output_root": str(tmp_path)}})


class TestExperimentCreate:
    def test_creates_expected_directory_layout(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        assert experiment.paths.root.exists()
        assert experiment.paths.results_dir.exists()
        assert experiment.paths.figures_dir.exists()
        assert experiment.paths.reports_dir.exists()
        assert experiment.paths.tables_dir.exists()
        assert experiment.paths.config_file.exists()

    def test_config_file_round_trips(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        loaded = yaml.safe_load(experiment.paths.config_file.read_text())
        assert loaded["particle_filter"]["n_particles"] == config.particle_filter.n_particles
        assert loaded["runtime"]["experiment_name"] == config.runtime.experiment_name

    def test_unique_ids_across_calls(self, tmp_path) -> None:
        config = _config(tmp_path)
        e1 = Experiment.create(config, output_root=tmp_path)
        e2 = Experiment.create(config, output_root=tmp_path)
        assert e1.experiment_id != e2.experiment_id

    def test_experiment_id_starts_with_experiment_name(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        assert experiment.experiment_id.startswith(config.runtime.experiment_name)


class TestExperimentPersistence:
    def test_save_metadata(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        metadata = ExperimentMetadata.create(experiment.experiment_id, config.runtime.global_seed)
        path = experiment.save_metadata(metadata)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["experiment_id"] == experiment.experiment_id
        assert loaded["global_seed"] == config.runtime.global_seed

    def test_save_table_writes_csv_and_tex(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        df = pd.DataFrame({"latent_dim": [3, 4], "aic": [100.0, 95.0]})
        csv_path, tex_path = experiment.save_table(df, "model_comparison", caption="Model comparison")
        assert csv_path.exists() and tex_path.exists()
        assert "aic" in csv_path.read_text()
        assert "Model comparison" in tex_path.read_text()

    def test_save_results_json(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        path = experiment.save_results_json("summary", {"n_particles": 4000, "converged": True})
        assert path.exists()
        assert json.loads(path.read_text())["n_particles"] == 4000

    def test_figure_path_inside_figures_dir(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        path = experiment.figure_path("latent_trajectories.png")
        assert path.parent == experiment.paths.figures_dir

    def test_save_report_markdown(self, tmp_path) -> None:
        config = _config(tmp_path)
        experiment = Experiment.create(config, output_root=tmp_path)
        path = experiment.save_report_markdown("# Report\n\nSome content.")
        assert path.exists()
        assert "Some content" in path.read_text()
