"""Observation-fit plots: actual vs. one-step-ahead predicted values, per variable."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnlssm.visualization.style import PALETTE, format_time_axis, publication_style, save_figure

_MAX_COLS = 4
_CONFIDENCE_Z = 1.96


def plot_observation_fit(
    observations: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_cov: np.ndarray,
    variable_labels: list[str],
    time_index: pd.DatetimeIndex,
    output_path: Path,
) -> Path:
    """Grid of subplots (one per observation variable): actual value vs. one-step-ahead forecast.

    All-``NaN`` variables (placeholders, or unresolved data-acquisition
    gaps) still get a subplot, explicitly annotated "no data", rather than
    being silently dropped from the grid -- so the figure's panel count
    always matches the declared observation space.
    """
    n_vars = len(variable_labels)
    ncols = min(_MAX_COLS, n_vars)
    nrows = math.ceil(n_vars / ncols)
    predicted_std = np.sqrt(np.diagonal(predicted_cov, axis1=1, axis2=2))

    with publication_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for i, label in enumerate(variable_labels):
            ax = axes_flat[i]
            actual = observations[:, i]
            if np.all(np.isnan(actual)):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(label, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.plot(time_index, actual, color=PALETTE[0], marker="o", markersize=1.5, linewidth=0.8, label="Actual")
            ax.plot(time_index, predicted_mean[:, i], color=PALETTE[1], linewidth=1.0, label="One-step-ahead forecast")
            ax.fill_between(
                time_index,
                predicted_mean[:, i] - _CONFIDENCE_Z * predicted_std[:, i],
                predicted_mean[:, i] + _CONFIDENCE_Z * predicted_std[:, i],
                color=PALETTE[1],
                alpha=0.15,
            )
            ax.set_title(label, fontsize=9)
            format_time_axis(ax)
            ax.tick_params(axis="x", labelsize=7)

        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles, legend_labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("Observation fit: actual vs. one-step-ahead forecast (standardized scale)", y=1.05)
        fig.tight_layout()
        return save_figure(fig, output_path)


__all__ = ["plot_observation_fit"]
