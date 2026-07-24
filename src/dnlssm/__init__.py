"""DNLSSM Research Platform.

A reproducible academic research platform for estimating Dynamic Nonlinear
State Space Models (DNLSSM) on macroeconomic observation panels via Maximum
Likelihood Estimation and a from-scratch Bootstrap Particle Filter /
Particle Smoother.

The package is strictly theory-neutral: it discovers latent mathematical
structure from data and reports numerical results only. No economic,
historical, or policy interpretation is produced anywhere in this codebase.
Latent states are always labelled neutrally (``State 1``, ``State 2``, ...).

Sub-packages
------------
config          Central, single-source-of-truth configuration system.
data            Automated multi-provider data acquisition with provenance.
preprocessing   Frequency alignment, missing-data handling, transforms.
models          Abstract state-space interfaces and the DNLSSM implementation.
filters         Bootstrap Particle Filter and Particle Smoother.
optimization    Multi-start Maximum Likelihood Estimation.
model_selection Latent-dimension search via information criteria.
diagnostics     Residual, autocorrelation, heteroskedasticity, robustness.
visualization   Publication-quality figure generation.
experiments     Experiment identity, metadata, and directory management.
reports         Numeric-only automated report generation (Markdown/LaTeX/CSV).
utils           Logging, seeding, and numerical-stability helpers.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("dnlssm-research")
except _metadata.PackageNotFoundError:  # pragma: no cover - editable/uninstalled use
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
