"""Single-command pipeline entry point.

``dnlssm-run run [--config PATH] [--variables PATH] [options]`` executes the
full research pipeline end to end: data acquisition -> preprocessing ->
observation-matrix assembly -> latent-dimension model selection -> MLE ->
particle filtering/smoothing -> statistical diagnostics -> robustness
analysis -> figure generation -> report generation. Every stage is driven
entirely by the resolved :class:`~dnlssm.config.schema.ExperimentConfig`;
no stage-specific hyperparameter is hardcoded here.

By default, the pipeline refuses to proceed past data acquisition if any
non-placeholder observation variable could not be resolved from any
provider or manual upload -- this is the platform's explicit fulfillment
of "the system must state clearly when real data is unavailable and ask
the researcher for the missing series," rather than silently substituting
placeholder or synthetic data into what claims to be a real research run.
Pass ``--allow-partial-data`` to proceed anyway (the resulting report still
states exactly which variables are missing).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dnlssm.config import load_experiment_config
from dnlssm.config.loader import ConfigError
from dnlssm.data.credentials import load_credentials
from dnlssm.data.manager import DataManager, build_default_connectors
from dnlssm.diagnostics.autocorrelation import test_autocorrelation
from dnlssm.diagnostics.heteroskedasticity import test_heteroskedasticity
from dnlssm.diagnostics.normality import test_normality
from dnlssm.diagnostics.residuals import compute_residual_diagnostics
from dnlssm.diagnostics.robustness import (
    analyze_noise_structure_sensitivity,
    analyze_particle_count_sensitivity,
)
from dnlssm.experiments.experiment import Experiment
from dnlssm.experiments.metadata import ExperimentMetadata
from dnlssm.filters.particle_filter import BootstrapParticleFilter
from dnlssm.filters.particle_smoother import ParticleSmoother
from dnlssm.model_selection.selector import ModelSelector
from dnlssm.models.dnlssm import DNLSSM
from dnlssm.models.latent import LatentManifold, LatentTrajectory
from dnlssm.optimization.mle import MLEEstimator
from dnlssm.preprocessing.observation_matrix import build_observation_matrix
from dnlssm.reports.report_builder import ReportContext, build_experiment_report
from dnlssm.utils.logging_config import get_logger
from dnlssm.utils.random_state import SeedSequence
from dnlssm.visualization.convergence_plots import (
    plot_information_criteria_comparison,
    plot_optimization_convergence,
    plot_particle_count_sensitivity,
)
from dnlssm.visualization.diagnostic_plots import plot_acf_pacf, plot_residual_histogram_qq
from dnlssm.visualization.latent_plots import plot_ess_timeseries, plot_latent_trajectories
from dnlssm.visualization.observation_plots import plot_observation_fit

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnlssm-run", description="DNLSSM research platform pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline end to end.")
    run_parser.add_argument("--config", type=Path, default=None, help="Path to experiment_config.yaml.")
    run_parser.add_argument("--variables", type=Path, default=None, help="Path to variables.yaml.")
    run_parser.add_argument("--output-root", type=Path, default=None, help="Override runtime.output_root.")
    run_parser.add_argument(
        "--manual-upload-dir", type=Path, default=Path("data/manual_uploads"),
        help="Directory scanned for manual CSV/Excel fallback series.",
    )
    run_parser.add_argument(
        "--allow-partial-data", action="store_true",
        help="Proceed even if some non-placeholder observation variables could not be resolved.",
    )
    run_parser.add_argument(
        "--skip-model-selection", action="store_true",
        help="Fit only model.latent_dim instead of sweeping model_selection.latent_dim_min..max.",
    )
    run_parser.add_argument("--skip-robustness", action="store_true", help="Skip the robustness analysis stage.")
    run_parser.add_argument("--log-level", default=None, help="Override runtime.log_level.")
    return parser


class PipelineDataError(RuntimeError):
    """Raised when data acquisition leaves unresolved variables and --allow-partial-data was not passed."""


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 0
    try:
        run_pipeline(args)
    except (ConfigError, PipelineDataError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    return 0


def run_pipeline(args: argparse.Namespace) -> Path:
    pipeline_start = time.time()

    overrides: dict = {}
    if args.output_root is not None:
        overrides.setdefault("runtime", {})["output_root"] = str(args.output_root)
    if args.log_level is not None:
        overrides.setdefault("runtime", {})["log_level"] = args.log_level

    config = load_experiment_config(config_path=args.config, variables_path=args.variables, overrides=overrides)
    experiment = Experiment.create(config, output_root=args.output_root)
    seeds = SeedSequence(config.runtime.global_seed)
    logger.info("Experiment '%s' created at %s", experiment.experiment_id, experiment.paths.root)

    # -- 1. Data acquisition -------------------------------------------------
    credentials = load_credentials()
    connectors = build_default_connectors(credentials, manual_upload_dir=args.manual_upload_dir)
    data_manager = DataManager(connectors, manual_upload_dir=args.manual_upload_dir)
    acquisition = data_manager.fetch_all(config.observation_space)
    experiment.save_table(acquisition.provenance.to_dataframe(), "data_provenance", caption="Data source provenance")

    unresolved = acquisition.provenance.unresolved_variables()
    if unresolved and not args.allow_partial_data:
        raise PipelineDataError(
            f"{len(unresolved)} observation variable(s) could not be resolved from any configured "
            f"provider or manual upload: {unresolved}. Fix this by (a) filling in the correct "
            f"series_code for these variables in config/variables.yaml, (b) setting the required "
            f"API key(s) in .env (see .env.example), or (c) placing a CSV/Excel file for each in "
            f"{args.manual_upload_dir} (see data/manual_uploads/README.md). "
            "Pass --allow-partial-data to proceed anyway with these columns left as missing data."
        )
    if unresolved:
        logger.warning("Proceeding with %d unresolved variable(s) (--allow-partial-data): %s", len(unresolved), unresolved)

    # -- 2. Preprocessing ------------------------------------------------------
    obs_matrix = build_observation_matrix(acquisition.series_by_variable, config.observation_space, config.preprocessing)
    experiment.save_table(
        obs_matrix.ledger.to_dataframe(), "transformation_ledger", caption="Preprocessing transformation ledger"
    )

    # -- 3. Model fitting (model selection or single fixed dimension) ----------
    if args.skip_model_selection:
        model = DNLSSM(config.model.latent_dim, obs_matrix.variable_ids, config.model)
        mle_result = MLEEstimator(model, config.particle_filter, config.optimization, seeds).fit(obs_matrix.values)
        filter_rng = seeds.spawn("final_filter")
        filter_result = BootstrapParticleFilter(model, config.particle_filter).run(
            obs_matrix.values, mle_result.best_theta, filter_rng
        )
        model_selection_result = None
    else:
        selector = ModelSelector(
            observation_labels=obs_matrix.variable_ids,
            model_config_template=config.model,
            pf_config=config.particle_filter,
            opt_config=config.optimization,
            candidate_dims=config.model_selection.candidate_dims,
            train_fraction=config.model_selection.train_fraction,
            criteria_weights=config.model_selection.criteria_weights,
            seed_sequence=seeds,
        )
        model_selection_result = selector.select(obs_matrix.values)
        experiment.save_table(
            model_selection_result.comparison_table, "model_selection_comparison",
            caption="Latent dimension comparison",
        )
        experiment.save_table(
            model_selection_result.ranked_selection_table, "model_selection_ranking",
            caption="Latent dimension ranking (combined score)",
        )
        selected = model_selection_result.selected
        model = selected.model
        mle_result = selected.mle_result
        filter_result = selected.full_series_filter_result

    theta = mle_result.best_theta

    # -- 4. Particle smoothing --------------------------------------------------
    smoother_rng = seeds.spawn("final_smoother")
    smoother_result = ParticleSmoother(model, config.particle_filter).run(filter_result, theta, smoother_rng)

    # -- 5. Residual diagnostics -------------------------------------------------
    residual_diag = compute_residual_diagnostics(filter_result, obs_matrix.values, obs_matrix.variable_ids, obs_matrix.time_index)
    experiment.save_table(residual_diag.per_variable_stats, "residual_error_stats", caption="Per-variable forecast error")

    normality_table = test_normality(residual_diag.standardized_residuals, config.diagnostics.significance_level)
    autocorr_table, acf_pacf_profiles = test_autocorrelation(
        residual_diag.standardized_residuals, config.diagnostics.acf_pacf_max_lags,
        config.diagnostics.ljung_box_lags, config.diagnostics.significance_level,
    )
    heteroskedasticity_table = test_heteroskedasticity(
        residual_diag.standardized_residuals, config.diagnostics.arch_test_lags, config.diagnostics.significance_level
    )
    experiment.save_table(normality_table, "normality_tests", caption="Residual normality tests")
    experiment.save_table(autocorr_table, "autocorrelation_tests", caption="Residual autocorrelation (Ljung-Box)")
    experiment.save_table(heteroskedasticity_table, "heteroskedasticity_tests", caption="Residual heteroskedasticity (ARCH-LM)")

    # -- 6. Robustness analysis --------------------------------------------------
    particle_sensitivity = None
    noise_sensitivity = None
    if not args.skip_robustness:
        particle_sensitivity = analyze_particle_count_sensitivity(
            model, theta, obs_matrix.values, config.particle_filter,
            config.particle_filter.robustness_particle_counts, seeds,
        )
        experiment.save_table(
            particle_sensitivity.comparison_table, "particle_count_sensitivity", caption="Particle-count sensitivity"
        )
        noise_sensitivity = [
            analyze_noise_structure_sensitivity(
                config.model, model.latent_dim, obs_matrix.variable_ids, config.particle_filter,
                config.optimization, obs_matrix.values, "observation_noise",
                config.diagnostics.observation_noise_alternatives, seeds,
            )
        ]
        for result in noise_sensitivity:
            experiment.save_table(
                result.comparison_table, f"noise_structure_sensitivity_{result.target}",
                caption=f"Noise structure sensitivity ({result.target})",
            )

    # -- 7. Figures ---------------------------------------------------------------
    manifold = LatentManifold.default(model.latent_dim)
    filtered_traj = LatentTrajectory(manifold, obs_matrix.time_index, filter_result.filtered_mean, "filtered", filter_result.filtered_cov)
    smoothed_traj = LatentTrajectory(manifold, obs_matrix.time_index, smoother_result.smoothed_mean, "smoothed", smoother_result.smoothed_cov)
    plot_latent_trajectories(filtered_traj, smoothed_traj, experiment.figure_path("latent_trajectories.png"))
    plot_ess_timeseries(
        obs_matrix.time_index, filter_result.diagnostics.ess_ratio, filter_result.diagnostics.resampled_at,
        config.particle_filter.ess_threshold_ratio, experiment.figure_path("ess_timeseries.png"),
    )
    plot_observation_fit(
        obs_matrix.values, filter_result.predicted_observation_mean, filter_result.predicted_observation_cov,
        obs_matrix.variable_ids, obs_matrix.time_index, experiment.figure_path("observation_fit.png"),
    )
    for variable in obs_matrix.variable_ids:
        plot_residual_histogram_qq(
            variable, residual_diag.standardized_residuals[variable].to_numpy(),
            experiment.paths.figures_dir / "residuals" / f"{variable}.png",
        )
        if variable in acf_pacf_profiles:
            plot_acf_pacf(
                acf_pacf_profiles[variable], config.diagnostics.significance_level,
                experiment.paths.figures_dir / "acf_pacf" / f"{variable}.png",
            )
    plot_optimization_convergence(mle_result.all_runs, experiment.figure_path("optimization_convergence.png"))
    if model_selection_result is not None:
        plot_information_criteria_comparison(
            model_selection_result.comparison_table, experiment.figure_path("information_criteria_comparison.png")
        )
    if particle_sensitivity is not None:
        plot_particle_count_sensitivity(
            particle_sensitivity.comparison_table, experiment.figure_path("particle_count_sensitivity.png")
        )

    # -- 8. Report ------------------------------------------------------------------
    total_runtime = time.time() - pipeline_start
    report_context = ReportContext(
        experiment_id=experiment.experiment_id,
        config=config,
        provenance=acquisition.provenance,
        transformation_ledger=obs_matrix.ledger,
        observation_matrix_shape=(obs_matrix.n_timesteps, obs_matrix.n_variables),
        model_selection_result=model_selection_result,
        selected_mle_result=mle_result,
        selected_filter_result=filter_result,
        confidence_intervals=mle_result.confidence_intervals,
        residual_stats=residual_diag.per_variable_stats,
        normality_table=normality_table,
        autocorrelation_table=autocorr_table,
        heteroskedasticity_table=heteroskedasticity_table,
        particle_count_sensitivity=particle_sensitivity,
        noise_structure_sensitivity=noise_sensitivity,
        total_runtime_seconds=total_runtime,
    )
    report_text = build_experiment_report(report_context)
    experiment.save_report_markdown(report_text)

    metadata = ExperimentMetadata.create(experiment.experiment_id, config.runtime.global_seed)
    metadata.runtime_seconds = total_runtime
    metadata.extra = {
        "selected_latent_dim": model.latent_dim,
        "mle_converged": mle_result.converged,
        "best_log_likelihood": mle_result.best_log_likelihood,
        "n_timesteps": obs_matrix.n_timesteps,
        "n_variables": obs_matrix.n_variables,
        "n_unresolved_data_sources": len(unresolved),
    }
    experiment.save_metadata(metadata)

    logger.info("Pipeline complete in %.1fs. Experiment directory: %s", total_runtime, experiment.paths.root)
    print(f"\nExperiment complete: {experiment.experiment_id}")
    print(f"  Selected latent_dim: {model.latent_dim}")
    print(f"  MLE converged: {mle_result.converged}")
    print(f"  Results: {experiment.paths.root}\n")
    return experiment.paths.root


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "run_pipeline", "build_arg_parser", "PipelineDataError"]
