"""Publication-quality figure generation.

Every function here is a pure ``(data, output_path) -> saved_path`` map
built on the shared style profile in :mod:`dnlssm.visualization.style`; no
plot produces or implies economic interpretation -- latent-state panels are
always labelled with the neutral ``State i`` identifiers from
:mod:`dnlssm.models.latent`.
"""

from dnlssm.visualization.convergence_plots import (
    plot_information_criteria_comparison,
    plot_optimization_convergence,
    plot_particle_count_sensitivity,
)
from dnlssm.visualization.diagnostic_plots import plot_acf_pacf, plot_residual_histogram_qq
from dnlssm.visualization.latent_plots import plot_ess_timeseries, plot_latent_trajectories
from dnlssm.visualization.observation_plots import plot_observation_fit
from dnlssm.visualization.style import PALETTE, publication_style, save_figure

__all__ = [
    "plot_information_criteria_comparison",
    "plot_optimization_convergence",
    "plot_particle_count_sensitivity",
    "plot_acf_pacf",
    "plot_residual_histogram_qq",
    "plot_ess_timeseries",
    "plot_latent_trajectories",
    "plot_observation_fit",
    "publication_style",
    "save_figure",
    "PALETTE",
]
