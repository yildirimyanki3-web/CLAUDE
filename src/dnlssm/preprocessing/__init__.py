"""Frequency alignment, missing-data handling, transformation, and observation-matrix assembly.

The pipeline implemented here turns the heterogeneous, raw-frequency series
produced by :mod:`dnlssm.data` into a single, dimension-checked observation
matrix ready for :mod:`dnlssm.models`. Every step -- alignment method,
missing-data statistics, imputation method, seasonal adjustment,
transformation, and standardization parameters -- is recorded per variable
in a :class:`~dnlssm.preprocessing.transforms.TransformationLedger` so the
exact numeric provenance of every column is reproducible and reportable,
never an implicit side effect of the code.
"""

from dnlssm.preprocessing.exceptions import PreprocessingError
from dnlssm.preprocessing.observation_matrix import ObservationMatrix, build_observation_matrix

__all__ = ["PreprocessingError", "ObservationMatrix", "build_observation_matrix"]
