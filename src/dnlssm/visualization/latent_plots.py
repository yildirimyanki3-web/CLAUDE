"""Latent-state trajectory plots.

Every axis is labelled with the neutral ``State i`` identifier only --
never an economic name -- consistent with the platform-wide rule that
interpretation of the latent states is left entirely to the researcher.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dnlssm.models.latent import LatentTrajectory
from dnlssm.visualization.style import PALETTE, publication_style, save_figure

_MAX_COLS = 3
_CONFIDENCE_Z = 1.96  # ~95% band under the (approximate) Gaussian marginal


def plot_latent_trajectories(
    filtered: LatentTrajectory,
    smoothed: LatentTrajectory | None,
    output_path: Path,
) -> Path:
    """One subplot per latent dimension, showing filtered (and optionally smoothed) trajectories.

    Bands show ``mean +/- 1.96 * std`` where a covariance path is available.
    """
    dim = filtered.manifold.dim
    ncols = min(_MAX_COLS, dim)
    nrows = math.ceil(dim / ncols)

    with publication_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        filtered_std = filtered.marginal_std()
        smoothed_std = smoothed.marginal_std() if smoothed is not None else None

        for i, label in enumerate(filtered.manifold.labels):
            ax = axes_flat[i]
            ax.plot(filtered.time_index, filtered.mean[:, i], color=PALETTE[0], label="Filtered")
            if filtered_std is not None:
                ax.fill_between(
                    filtered.time_index,
                    filtered.mean[:, i] - _CONFIDENCE_Z * filtered_std[:, i],
                    filtered.mean[:, i] + _CONFIDENCE_Z * filtered_std[:, i],
                    color=PALETTE[0],
                    alpha=0.15,
                )
            if smoothed is not None:
                ax.plot(smoothed.time_index, smoothed.mean[:, i], color=PALETTE[1], label="Smoothed")
                if smoothed_std is not None:
                    ax.fill_between(
                        smoothed.time_index,
                        smoothed.mean[:, i] - _CONFIDENCE_Z * smoothed_std[:, i],
                        smoothed.mean[:, i] + _CONFIDENCE_Z * smoothed_std[:, i],
                        color=PALETTE[1],
                        alpha=0.15,
                    )
            ax.set_title(label)
            ax.set_xlabel("Time")
            ax.set_ylabel("Latent value (unitless)")
            ax.legend(loc="upper right")

        for j in range(dim, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Latent state trajectories (neutral, uninterpreted)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return save_figure(fig, output_path)


def plot_ess_timeseries(
    time_index,
    ess_ratio: np.ndarray,
    resampled_at: np.ndarray,
    ess_threshold_ratio: float,
    output_path: Path,
) -> Path:
    """Effective-Sample-Size ratio over time, with resampling events marked."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(time_index, ess_ratio, color=PALETTE[0], label="ESS / N")
        ax.axhline(ess_threshold_ratio, color=PALETTE[1], linestyle="--", label="Resampling threshold")
        resample_times = np.asarray(time_index)[resampled_at]
        if len(resample_times) > 0:
            ax.scatter(
                resample_times,
                ess_ratio[resampled_at],
                color=PALETTE[2],
                marker="o",
                s=18,
                zorder=5,
                label="Resampling triggered",
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Effective Sample Size / n_particles")
        ax.set_ylim(0, 1.05)
        ax.set_title("Particle filter Effective Sample Size")
        ax.legend(loc="lower left")
        fig.tight_layout()
        return save_figure(fig, output_path)


__all__ = ["plot_latent_trajectories", "plot_ess_timeseries"]
