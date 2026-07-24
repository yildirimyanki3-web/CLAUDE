"""Experiment identity, metadata, and reproducible output-directory management.

Every run of the pipeline (:mod:`dnlssm.cli`) creates one
:class:`~dnlssm.experiments.experiment.Experiment`: a uniquely-identified
directory holding the exact resolved configuration, a metadata record
(software version, seed, hardware, runtime), and standardized
``results/``, ``figures/``, and ``reports/`` subdirectories. This is what
makes a run auditable and reproducible after the fact -- re-running with
the same ``config.yaml`` and the same ``global_seed`` recovers the same
numerical results.
"""

from dnlssm.experiments.experiment import Experiment, ExperimentPaths
from dnlssm.experiments.metadata import ExperimentMetadata, collect_hardware_info

__all__ = ["Experiment", "ExperimentPaths", "ExperimentMetadata", "collect_hardware_info"]
