"""BIS Statistics Warehouse connector.

Public REST API, no API key required. Documented at
https://www.bis.org/statistics/dsd_hist_availability.htm and the BIS SDMX
RESTful API guide. ``series_code`` in ``variables.yaml`` may encode
``"{dataflow}:{key}"`` (e.g. ``"WS_TC:Q.TR.P.A.M.770.A"``) to select a
non-default dataflow; if no colon is present, ``series_code`` is used as
the SDMX key against this connector's default dataflow.
"""

from __future__ import annotations

import io
from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError

_REQUEST_TIMEOUT_SECONDS = 30


class BISConnector(DataConnector):
    provider_name: ClassVar[str] = "bis"
    BASE_URL_TEMPLATE: ClassVar[str] = "https://stats.bis.org/api/v1/data/{flow}/{key}/all"
    DEFAULT_DATAFLOW: ClassVar[str] = "WS_TC"  # BIS "Total credit to the non-financial sector"

    def __init__(self, default_dataflow: str = DEFAULT_DATAFLOW) -> None:
        self._default_dataflow = default_dataflow

    def is_available(self) -> bool:
        return True

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        dataflow, key = (
            series_code.split(":", 1) if ":" in series_code else (self._default_dataflow, series_code)
        )
        url = self.BASE_URL_TEMPLATE.format(flow=dataflow, key=key)
        params = {"format": "csv", "startPeriod": str(start_date.year)}
        if end_date is not None:
            params["endPeriod"] = str(end_date.year)

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(f"BIS request failed for key '{key}' (dataflow={dataflow}): {exc}") from exc

        try:
            frame = pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            raise DataFetchError(f"BIS CSV response could not be parsed for key '{key}': {exc}") from exc

        if {"TIME_PERIOD", "OBS_VALUE"}.issubset(frame.columns):
            dates = pd.to_datetime(frame["TIME_PERIOD"], errors="coerce")
            values = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
            series = pd.Series(values.values, index=dates, name=key).dropna(how="all")
            if series.empty:
                raise DataFetchError(f"BIS returned an empty series for key '{key}'.")
            return series.sort_index()

        raise DataFetchError(
            f"Unexpected BIS CSV schema for key '{key}': columns={list(frame.columns)}; "
            "expected a long-format response with TIME_PERIOD/OBS_VALUE columns."
        )


__all__ = ["BISConnector"]
