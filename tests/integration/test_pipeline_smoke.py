"""End-to-end smoke test of the full pipeline (dnlssm.cli.run_pipeline).

Exercises every stage for real -- data acquisition (via manual CSV upload,
so no network is required), preprocessing, model fitting (with and without
model selection), particle filtering/smoothing, diagnostics, robustness,
figure generation, and report generation -- on a small synthetic dataset
used strictly for testing, never presented as a real research result (see
module docstring of dnlssm.cli). This is the test that catches import
errors, dimension mismatches, and cross-module wiring problems that unit
tests targeting one module at a time cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from dnlssm.cli import build_arg_parser, run_pipeline

_N_MONTHS = 30
_START_DATE = "2020-01-01"


def _write_variables_yaml(path) -> None:
    idx = pd.date_range(_START_DATE, periods=_N_MONTHS, freq="MS")
    rng = np.random.default_rng(0)

    content = {
        "variables": [
            {
                "canonical_id": "var_a",
                "description": "synthetic test series A",
                "unit": "index",
                "native_frequency": "monthly",
                "transform": "log",
                "standardize": True,
                "seasonal_adjust": False,
                "provider_priority": [{"provider": "manual", "series_code": None}],
            },
            {
                "canonical_id": "var_b",
                "description": "synthetic test series B",
                "unit": "percent",
                "native_frequency": "monthly",
                "transform": "none",
                "standardize": True,
                "seasonal_adjust": False,
                "provider_priority": [{"provider": "manual", "series_code": None}],
            },
            {
                "canonical_id": "var_c_placeholder",
                "description": "future series, not yet available",
                "unit": "index",
                "native_frequency": "annual",
                "is_placeholder": True,
                "provider_priority": [],
            },
        ]
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))

    manual_dir = path.parent / "manual_uploads"
    manual_dir.mkdir(exist_ok=True)
    pd.DataFrame({"date": idx, "value": np.exp(rng.normal(size=_N_MONTHS))}).to_csv(
        manual_dir / "var_a.csv", index=False
    )
    pd.DataFrame({"date": idx, "value": rng.normal(size=_N_MONTHS)}).to_csv(manual_dir / "var_b.csv", index=False)
    return manual_dir


def _write_experiment_config_yaml(path, *, latent_dim_min: int, latent_dim_max: int) -> None:
    content = {
        "runtime": {
            "experiment_name": "smoke_test",
            "global_seed": 2024,
            "output_root": str(path.parent / "experiments_output"),
            "log_level": "WARNING",
            "n_jobs": 1,
        },
        "observation_space": {
            "target_frequency": "MS",
            "start_date": _START_DATE,
            "end_date": None,
        },
        "preprocessing": {
            "missing_data_method": "linear_interpolate",
            "max_consecutive_missing_interpolate": 3,
            "seasonal_adjustment_method": "stl",
            "standardization_method": "zscore",
        },
        "model": {
            "latent_dim": latent_dim_min,
            "transition_function": {"family": "linear"},
            "observation_function": {"family": "linear"},
            "process_noise": {"structure": "diagonal", "init_std": 0.5, "regularization_epsilon": 1.0e-6},
            "observation_noise": {"structure": "diagonal", "init_std": 0.5, "regularization_epsilon": 1.0e-6},
            "initial_state_mean": 0.0,
            "initial_state_std": 1.0,
        },
        "optimization": {
            "methods": ["Nelder-Mead"],
            "n_multistarts": 1,
            "max_iterations": 15,
            "compute_hessian_ci": True,
            "hessian_step_size": 1.0e-2,
        },
        "particle_filter": {
            "n_particles": 300,
            "resampling_method": "systematic",
            "adaptive_resampling": True,
            "ess_threshold_ratio": 0.5,
            "smoother_method": "backward_simulation",
            "smoother_n_backward_samples": 100,
            "robustness_particle_counts": [100, 150],
        },
        "model_selection": {
            "latent_dim_min": latent_dim_min,
            "latent_dim_max": latent_dim_max,
            "train_fraction": 0.75,
            "criteria_weights": {"aic": 0.5, "oos_rmse": 0.5},
        },
        "diagnostics": {
            "significance_level": 0.05,
            "ljung_box_lags": 5,
            "acf_pacf_max_lags": 8,
            "arch_test_lags": 5,
            "observation_noise_alternatives": ["diagonal", "scalar"],
        },
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))


@pytest.mark.integration
class TestPipelineSmoke:
    def test_full_pipeline_with_model_selection_and_robustness(self, tmp_path) -> None:
        config_path = tmp_path / "experiment_config.yaml"
        variables_path = tmp_path / "variables.yaml"
        manual_dir = _write_variables_yaml(variables_path)
        _write_experiment_config_yaml(config_path, latent_dim_min=1, latent_dim_max=2)

        args = build_arg_parser().parse_args(
            [
                "run",
                "--config", str(config_path),
                "--variables", str(variables_path),
                "--manual-upload-dir", str(manual_dir),
            ]
        )
        experiment_root = run_pipeline(args)

        assert (experiment_root / "config.yaml").exists()
        assert (experiment_root / "metadata.json").exists()
        assert (experiment_root / "reports" / "report.md").exists()
        assert (experiment_root / "reports" / "tables" / "model_selection_comparison.csv").exists()
        assert (experiment_root / "reports" / "tables" / "data_provenance.csv").exists()
        assert (experiment_root / "figures" / "latent_trajectories.png").exists()
        assert (experiment_root / "figures" / "observation_fit.png").exists()
        assert (experiment_root / "figures" / "information_criteria_comparison.png").exists()
        assert (experiment_root / "figures" / "particle_count_sensitivity.png").exists()
        assert (experiment_root / "figures" / "residuals" / "var_a.png").exists()

        report_text = (experiment_root / "reports" / "report.md").read_text()
        assert "Selected latent dimension" in report_text
        assert "var_c_placeholder" not in report_text.lower().replace("_", " ")  # no economic/series narrative added

    def test_full_pipeline_skip_model_selection_and_robustness(self, tmp_path) -> None:
        config_path = tmp_path / "experiment_config.yaml"
        variables_path = tmp_path / "variables.yaml"
        manual_dir = _write_variables_yaml(variables_path)
        _write_experiment_config_yaml(config_path, latent_dim_min=2, latent_dim_max=2)

        args = build_arg_parser().parse_args(
            [
                "run",
                "--config", str(config_path),
                "--variables", str(variables_path),
                "--manual-upload-dir", str(manual_dir),
                "--skip-model-selection",
                "--skip-robustness",
            ]
        )
        experiment_root = run_pipeline(args)

        assert (experiment_root / "reports" / "report.md").exists()
        assert not (experiment_root / "figures" / "information_criteria_comparison.png").exists()
        assert not (experiment_root / "figures" / "particle_count_sensitivity.png").exists()
        report_text = (experiment_root / "reports" / "report.md").read_text()
        assert "Model selection was not run" in report_text
        assert "Robustness analysis was not run" in report_text

    def test_missing_data_without_allow_partial_raises(self, tmp_path) -> None:
        from dnlssm.cli import PipelineDataError

        config_path = tmp_path / "experiment_config.yaml"
        variables_path = tmp_path / "variables.yaml"
        _write_variables_yaml(variables_path)
        _write_experiment_config_yaml(config_path, latent_dim_min=1, latent_dim_max=1)

        empty_manual_dir = tmp_path / "no_manual_files_here"
        empty_manual_dir.mkdir()

        args = build_arg_parser().parse_args(
            [
                "run",
                "--config", str(config_path),
                "--variables", str(variables_path),
                "--manual-upload-dir", str(empty_manual_dir),
            ]
        )
        with pytest.raises(PipelineDataError):
            run_pipeline(args)
