"""Centralized, publication-quality matplotlib configuration.

Every plotting function in this package draws through
:func:`publication_style` (a context manager applying one shared
``rcParams`` profile) and saves through :func:`save_figure`, so figure
size, resolution, font sizes, and file-handling are never duplicated or
silently inconsistent across the platform's ~10 plotting functions. The
``Agg`` backend is selected explicitly because the platform is meant to run
headless (CLI / container / CI), never requiring a display.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot import)

from dnlssm.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

FIGURE_DPI = 300
FIGURE_FACECOLOR = "white"

_RC_PARAMS = {
    "figure.dpi": 100,  # on-screen/preview resolution; save-time resolution is set by FIGURE_DPI
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": FIGURE_FACECOLOR,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
}

# Colorblind-safe, print-safe qualitative palette (Wong, 2011); used for every
# multi-series plot (filtered vs smoothed, per-method optimization traces, ...).
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


@contextlib.contextmanager
def publication_style():
    """Context manager applying the shared rcParams profile for the duration of one figure."""
    with plt.rc_context(rc=_RC_PARAMS):
        yield


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Saves ``fig`` to ``path`` (creating parent directories as needed) and closes it.

    Every plotting function in this package returns the saved path via this
    helper rather than leaving figures open, which matters for long batch
    runs (e.g. one diagnostic figure per observation variable) where an
    unclosed ``Figure`` would otherwise accumulate and leak memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.debug("Saved figure: %s", path)
    return path


__all__ = ["publication_style", "save_figure", "PALETTE", "FIGURE_DPI"]
