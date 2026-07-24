"""Automated, numeric-only experiment report generation.

The generated report states data coverage, model specification, the
optimization/convergence outcome, model-selection results, particle-filter
health, and residual diagnostics -- as numbers and their direct
statistical meaning only. It never names an economic regime, historical
period, or policy narrative; that interpretation is left entirely to the
researcher working from these numbers afterward. A failed or unavailable
stage is always stated explicitly ("MLE did not converge", "Hessian CI
unavailable: ...") rather than omitted silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dnlssm.config.schema import ExperimentConfig
from dnlssm.data.provenance import ProvenanceLog
from dnlssm.diagnostics.robustness import (
    NoiseStructureSensitivityResult,
    ParticleCountSensitivityResult,
)
from dnlssm.filters.particle_filter import FilterResult
from dnlssm.model_selection.selector import ModelSelectionResult
from dnlssm.optimization.confidence_intervals import ConfidenceIntervalResult
from dnlssm.optimization.mle import MLEResult
from dnlssm.preprocessing.transforms import TransformationLedger


@dataclass
class ReportContext:
    """Everything the report can draw on. Every field beyond identity is optional:
    the report renders whatever stages actually ran, stating plainly what did not.
    """

    experiment_id: str
    config: ExperimentConfig
    provenance: ProvenanceLog | None = None
    transformation_ledger: TransformationLedger | None = None
    observation_matrix_shape: tuple[int, int] | None = None
    model_selection_result: ModelSelectionResult | None = None
    selected_mle_result: MLEResult | None = None
    selected_filter_result: FilterResult | None = None
    confidence_intervals: ConfidenceIntervalResult | None = None
    residual_stats: pd.DataFrame | None = None
    normality_table: pd.DataFrame | None = None
    autocorrelation_table: pd.DataFrame | None = None
    heteroskedasticity_table: pd.DataFrame | None = None
    particle_count_sensitivity: ParticleCountSensitivityResult | None = None
    noise_structure_sensitivity: list[NoiseStructureSensitivityResult] | None = None
    total_runtime_seconds: float | None = None


def _fmt(value, spec: str = ".4f") -> str:
    if value is None:
        return "N/A"
    try:
        if isinstance(value, bool):
            return str(value)
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _section(title: str) -> str:
    return f"\n## {title}\n"


def build_experiment_report(context: ReportContext) -> str:
    """Builds the full Markdown report text for one experiment."""
    lines: list[str] = []
    lines.append(f"# DNLSSM Experiment Report -- {context.experiment_id}")
    lines.append("")
    lines.append(
        "This report presents numerical results only. No economic, historical, or policy "
        "interpretation is produced by this software; that interpretation is left to the "
        "researcher working from the results below."
    )

    lines.append(_section("1. Data Coverage and Provenance"))
    if context.provenance is not None:
        lines.append(context.provenance.summary_text().replace("\n", "  \n"))
        unresolved = context.provenance.unresolved_variables()
        if unresolved:
            lines.append("")
            lines.append(f"**Unresolved variables ({len(unresolved)}):** " + ", ".join(unresolved))
    else:
        lines.append("Data acquisition was not run in this experiment (provenance unavailable).")

    lines.append(_section("2. Preprocessing"))
    if context.transformation_ledger is not None:
        n_vars = len(context.transformation_ledger.records)
        lines.append(f"{n_vars} observation variables processed. Per-variable transformation ledger:")
        lines.append("")
        lines.append(context.transformation_ledger.to_dataframe().to_markdown(index=False))
    else:
        lines.append("Preprocessing was not run in this experiment.")
    if context.observation_matrix_shape is not None:
        t, n = context.observation_matrix_shape
        lines.append(f"\nFinal observation matrix shape: T={t} timesteps, N={n} variables.")

    lines.append(_section("3. Model Specification"))
    model_cfg = context.config.model
    lines.append(f"- Latent dimension (default/config): {model_cfg.latent_dim}")
    lines.append(f"- Transition function family: {model_cfg.transition_function.family}")
    lines.append(f"- Observation function family: {model_cfg.observation_function.family}")
    lines.append(f"- Process noise structure: {model_cfg.process_noise.structure}")
    lines.append(f"- Observation noise structure: {model_cfg.observation_noise.structure}")
    lines.append(f"- Particle filter: n_particles={context.config.particle_filter.n_particles}, "
                 f"resampling={context.config.particle_filter.resampling_method}, "
                 f"adaptive={context.config.particle_filter.adaptive_resampling}")

    lines.append(_section("4. Model Selection"))
    if context.model_selection_result is not None:
        res = context.model_selection_result
        lines.append(f"Candidate latent dimensions evaluated: {[r.latent_dim for r in res.per_dimension]}")
        lines.append(f"**Selected latent dimension: {res.selected_dim}**")
        lines.append("")
        lines.append(res.comparison_table.to_markdown(index=False))
    else:
        lines.append("Model selection was not run in this experiment (a fixed latent_dim was used).")

    lines.append(_section("5. Optimization and Convergence"))
    if context.selected_mle_result is not None:
        mle = context.selected_mle_result
        lines.append(f"- Converged: **{mle.converged}**")
        lines.append(f"- Best method: {mle.best_method} (start index {mle.best_start_index})")
        lines.append(f"- Best log-likelihood: {_fmt(mle.best_log_likelihood)}")
        lines.append(f"- Total runs (methods x multistarts): {mle.n_runs} ({mle.n_converged_runs} converged)")
        lines.append(f"- Total wall time: {_fmt(mle.total_wall_time_seconds, '.1f')}s")
        if not mle.converged and mle.convergence_diagnosis is not None:
            lines.append("")
            lines.append("**Automated non-convergence diagnosis:**")
            for reason in mle.convergence_diagnosis.reasons:
                lines.append(f"  - {reason}")
            lines.append("**Recommendations:**")
            for rec in mle.convergence_diagnosis.recommendations:
                lines.append(f"  - {rec}")
        if mle.method_comparison is not None:
            lines.append("")
            lines.append("Cross-method comparison:")
            lines.append(mle.method_comparison.to_markdown(index=False))
    else:
        lines.append("MLE was not run in this experiment.")

    lines.append(_section("6. Parameter Confidence Intervals (Hessian-based)"))
    ci = context.confidence_intervals
    if ci is not None and ci.available:
        lines.append(f"Confidence level: {ci.confidence_level:.0%}. Standard errors and intervals computed for "
                     f"{len(ci.standard_errors)} parameters.")
        lines.append(f"Median standard error: {_fmt(pd.Series(ci.standard_errors).median())}")
    elif ci is not None:
        lines.append(f"Confidence intervals **unavailable**: {ci.failure_reason}")
    else:
        lines.append("Hessian-based confidence intervals were not computed in this experiment.")

    lines.append(_section("7. Particle Filter Diagnostics"))
    if context.selected_filter_result is not None:
        diag = context.selected_filter_result.diagnostics
        lines.append(diag.summary_text().replace("\n", "  \n"))
    else:
        lines.append("Particle filter diagnostics unavailable.")

    lines.append(_section("8. Residual Diagnostics"))
    if context.residual_stats is not None:
        lines.append("Per-variable one-step-ahead forecast error:")
        lines.append("")
        lines.append(context.residual_stats.to_markdown(index=False))
    if context.normality_table is not None:
        lines.append("\n**Normality tests (Shapiro-Wilk, Jarque-Bera):**")
        lines.append(context.normality_table.to_markdown(index=False))
    if context.autocorrelation_table is not None:
        lines.append("\n**Autocorrelation tests (Ljung-Box):**")
        lines.append(context.autocorrelation_table.to_markdown(index=False))
    if context.heteroskedasticity_table is not None:
        lines.append("\n**Heteroskedasticity tests (ARCH-LM):**")
        lines.append(context.heteroskedasticity_table.to_markdown(index=False))
        lines.append(
            "\nNote: none of the platform's static noise-covariance structures model "
            "time-varying variance; ARCH effects flagged above are a stated limitation, "
            "not one resolved by any structure compared in Section 9."
        )
    if all(
        x is None
        for x in (
            context.residual_stats,
            context.normality_table,
            context.autocorrelation_table,
            context.heteroskedasticity_table,
        )
    ):
        lines.append("Residual diagnostics were not run in this experiment.")

    lines.append(_section("9. Robustness Analysis"))
    if context.particle_count_sensitivity is not None:
        lines.append("**Particle-count sensitivity:**")
        lines.append("")
        lines.append(context.particle_count_sensitivity.summary_text().replace("\n", "  \n"))
        lines.append("")
        lines.append(context.particle_count_sensitivity.comparison_table.to_markdown(index=False))
    if context.noise_structure_sensitivity:
        for result in context.noise_structure_sensitivity:
            lines.append(f"\n**Noise structure sensitivity ({result.target}):**")
            lines.append(f"Best by AIC: {result.best_structure_by_aic}; best by BIC: {result.best_structure_by_bic}")
            lines.append("")
            lines.append(result.comparison_table.to_markdown(index=False))
            lines.append(f"\n{result.arch_effects_caveat}")
    if context.particle_count_sensitivity is None and not context.noise_structure_sensitivity:
        lines.append("Robustness analysis was not run in this experiment.")

    lines.append(_section("10. Reproducibility"))
    lines.append(f"- Global seed: {context.config.runtime.global_seed}")
    if context.total_runtime_seconds is not None:
        lines.append(f"- Total experiment wall time: {_fmt(context.total_runtime_seconds, '.1f')}s")
    lines.append("- Full resolved configuration: see `config.yaml` in this experiment's root directory.")

    return "\n".join(lines) + "\n"


__all__ = ["ReportContext", "build_experiment_report"]
