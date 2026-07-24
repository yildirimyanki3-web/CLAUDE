"""Bootstrap Particle Filter and Particle Smoother, implemented from scratch.

``resampling.py``
    Systematic, stratified, multinomial, and residual resampling schemes.
``diagnostics.py``
    Effective Sample Size, weight entropy, and particle-degeneracy metrics
    computed at every time step.
``particle_filter.py``
    :class:`~dnlssm.filters.particle_filter.BootstrapParticleFilter`:
    initialization, prediction, importance weighting, log-likelihood
    evaluation, adaptive/systematic resampling, and filtered posterior
    estimation.
``particle_smoother.py``
    Backward-simulation and fixed-lag particle smoothers, run on the
    filter's stored particle history to produce smoothed (as opposed to
    filtered) latent-state estimates.

Nothing in this package interprets the latent state economically; it
operates purely on the :class:`~dnlssm.models.base.AbstractStateSpaceModel`
interface and neutral ``State i`` labels.
"""

from dnlssm.filters.particle_filter import BootstrapParticleFilter, FilterResult
from dnlssm.filters.particle_smoother import ParticleSmoother, SmootherResult

__all__ = ["BootstrapParticleFilter", "FilterResult", "ParticleSmoother", "SmootherResult"]
