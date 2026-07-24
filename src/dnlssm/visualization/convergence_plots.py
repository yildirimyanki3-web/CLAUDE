"""Optimization convergence, model-selection comparison, and robustness plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnlssm.optimization.optimizers import SingleOptimizationResult
from dnlssm.visualization.style import PALETTE, publication_style, save_figure


def plot_optimization_convergence(runs: list[SingleOptimizationResult], output_path: Path) -> Path:
    """Overlays the negative-log-likelihood trace of every (method, start) optimization attempt."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        methods = sorted({r.method for r in runs})
        color_by_method = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
        seen_methods = set()

        for run in runs:
            if not run.iteration_log:
                continue
            iterations = [rec.iteration for rec in run.iteration_log]
            nll = [rec.negative_log_likelihood for rec in run.iteration_log]
            label = run.method if run.method not in seen_methods else None
            seen_methods.add(run.method)
            ax.plot(
                iterations,
                nll,
                color=color_by_method[run.method],
                alpha=0.7,
                linewidth=1.0,
                linestyle="-" if run.success else "--",
                label=label,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Negative log-likelihood")
        ax.set_title("Optimization convergence (dashed = did not converge)")
        if seen_methods:
            ax.legend(loc="upper right")
        fig.tight_layout()
        return save_figure(fig, output_path)


def plot_information_criteria_comparison(comparison_table: pd.DataFrame, output_path: Path) -> Path:
    """Grouped bar chart of AIC/BIC/HQIC across candidate latent dimensions."""
    criteria = ["aic", "bic", "hqic"]
    dims = comparison_table["latent_dim"].to_numpy()
    x = np.arange(len(dims))
    width = 0.8 / len(criteria)

    with publication_style():
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(dims)), 4.5))
        for i, criterion in enumerate(criteria):
            ax.bar(
                x + i * width - width,
                comparison_table[criterion].to_numpy(),
                width=width,
                label=criterion.upper(),
                color=PALETTE[i],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Information criterion value (lower is better)")
        ax.set_title("Model selection: information criteria by latent dimension")
        ax.legend()
        fig.tight_layout()
        return save_figure(fig, output_path)


def plot_particle_count_sensitivity(comparison_table: pd.DataFrame, output_path: Path) -> Path:
    """Log-likelihood and mean ESS ratio as a function of the particle count."""
    with publication_style():
        fig, (ax_ll, ax_ess) = plt.subplots(1, 2, figsize=(9, 3.5))
        ax_ll.plot(
            comparison_table["n_particles"], comparison_table["log_likelihood"], marker="o", color=PALETTE[0]
        )
        ax_ll.set_xlabel("n_particles")
        ax_ll.set_ylabel("Log-likelihood")
        ax_ll.set_title("Log-likelihood stability")

        ax_ess.plot(comparison_table["n_particles"], comparison_table["mean_ess_ratio"], marker="o", color=PALETTE[1])
        ax_ess.set_xlabel("n_particles")
        ax_ess.set_ylabel("Mean ESS / n_particles")
        ax_ess.set_ylim(0, 1.05)
        ax_ess.set_title("Effective Sample Size stability")

        fig.tight_layout()
        return save_figure(fig, output_path)


__all__ = ["plot_optimization_convergence", "plot_information_criteria_comparison", "plot_particle_count_sensitivity"]
