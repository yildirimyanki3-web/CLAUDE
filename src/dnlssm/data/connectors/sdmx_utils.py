"""Shared parsing helper for the SDMX-JSON response format.

SDMX-JSON (used by OECD, and by several other statistical agencies) encodes
observations sparsely: a ``dataSets[0].series`` mapping keyed by a
dimension-index tuple, whose ``observations`` field maps a *time-dimension
index* (not a date) to a value array. The corresponding date for each index
is looked up positionally in ``structure.dimensions.observation[0].values``.
This module implements that decoding once, correctly, so every SDMX-JSON
based connector reuses it instead of re-deriving the index arithmetic.
"""

from __future__ import annotations

import pandas as pd

from dnlssm.data.base import DataFetchError


def parse_sdmx_json_single_series(payload: dict, series_code: str) -> pd.Series:
    """Extracts the first (and expected-only) series from an SDMX-JSON compact response.

    Raises
    ------
    DataFetchError
        If the payload does not have the expected SDMX-JSON structure, or
        contains no series / no observations.
    """
    try:
        observation_dim = payload["structure"]["dimensions"]["observation"][0]
        time_values = observation_dim["values"]
        data_sets = payload["dataSets"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DataFetchError(
            f"SDMX-JSON response for '{series_code}' is missing the expected "
            f"structure.dimensions.observation / dataSets fields."
        ) from exc

    if not data_sets or not data_sets[0].get("series"):
        raise DataFetchError(f"SDMX-JSON response for '{series_code}' contains no series data.")

    series_map = data_sets[0]["series"]
    # Exactly one dimension-key combination is expected because the caller
    # constrains the request to a single country/subject/measure/frequency.
    first_key = next(iter(series_map))
    observations = series_map[first_key].get("observations", {})
    if not observations:
        raise DataFetchError(f"SDMX-JSON response for '{series_code}' has an empty observations map.")

    dates: list[str] = []
    values: list[float] = []
    for time_index_str, obs_array in observations.items():
        time_index = int(time_index_str)
        if time_index >= len(time_values):
            raise DataFetchError(
                f"SDMX-JSON observation index {time_index} out of range for '{series_code}' "
                f"(only {len(time_values)} time periods declared)."
            )
        dates.append(time_values[time_index]["id"])
        values.append(obs_array[0] if obs_array else None)

    index = pd.to_datetime(dates)
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.Series(numeric_values.values, index=index, name=series_code).sort_index()


__all__ = ["parse_sdmx_json_single_series"]
