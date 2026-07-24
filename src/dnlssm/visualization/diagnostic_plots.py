"""Per-variable residual diagnostic plots: histograms, QQ-plots, ACF/PACF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dnlssm.diagnostics.autocorrelation import AutocorrelationProfile
from dnlssm.visualization.style import PALETTE, publication_style, save_figure


def plot_residual_histogram_qq(variable: str, standardized_residuals: np.ndarray, output_path: Path) -> Path:
    """Standardized-residual histogram (vs. the standard normal density) and a normal QQ-plot."""
    values = standardized_residuals[~np.isnan(standardized_residuals)]

    with publication_style():
        fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(8, 3.5))

        if values.size == 0:
            for ax in (ax_hist, ax_qq):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
        else:
            ax_hist.hist(values, bins=min(30, max(5, values.size // 5)), density=True, color=PALETTE[0], alpha=0.7)
            grid = np.linspace(values.min() - 1, values.max() + 1, 200)
            ax_hist.plot(grid, stats.norm.pdf(grid), color=PALETTE[1], label="N(0,1) density")
            ax_hist.set_xlabel("Standardized residual")
            ax_hist.set_ylabel("Density")
            ax_hist.legend()

            stats.probplot(values, dist="norm", plot=ax_qq)
            ax_qq.get_lines()[0].set_color(PALETTE[0])
            ax_qq.get_lines()[0].set_markersize(3)
            ax_qq.get_lines()[1].set_color(PALETTE[1])

        ax_hist.set_title(f"{variable}: residual distribution")
        ax_qq.set_title(f"{variable}: normal QQ-plot")
        fig.tight_layout()
        return save_figure(fig, output_path)


def plot_acf_pacf(profile: AutocorrelationProfile, significance_level: float, output_path: Path) -> Path:
    """Bar plots of the ACF and PACF, with approximate significance bounds."""
    n_obs = max(len(profile.acf_values), 2)
    bound = stats.norm.ppf(1 - significance_level / 2) / np.sqrt(n_obs)

    with publication_style():
        fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(8, 3.2))
        lags = np.arange(len(profile.acf_values))
        ax_acf.bar(lags, profile.acf_values, color=PALETTE[0], width=0.6)
        ax_acf.axhline(bound, color=PALETTE[1], linestyle="--", linewidth=1)
        ax_acf.axhline(-bound, color=PALETTE[1], linestyle="--", linewidth=1)
        ax_acf.set_title(f"{profile.variable}: ACF")
        ax_acf.set_xlabel("Lag")

        lags_pacf = np.arange(len(profile.pacf_values))
        ax_pacf.bar(lags_pacf, profile.pacf_values, color=PALETTE[0], width=0.6)
        ax_pacf.axhline(bound, color=PALETTE[1], linestyle="--", linewidth=1)
        ax_pacf.axhline(-bound, color=PALETTE[1], linestyle="--", linewidth=1)
        ax_pacf.set_title(f"{profile.variable}: PACF")
        ax_pacf.set_xlabel("Lag")

        fig.tight_layout()
        return save_figure(fig, output_path)


__all__ = ["plot_residual_histogram_qq", "plot_acf_pacf"]
