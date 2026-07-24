"""OECD SDMX-JSON connector (Main Economic Indicators and related datasets).

Public REST API, no API key required. Documented at
https://data.oecd.org/api/sdmx-json-documentation/. ``series_code`` supplies
the OECD "SUBJECT" dimension (e.g. ``PRINTO01``); the country dimension is
fixed by the connector instance.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError
from dnlssm.data.connectors.sdmx_utils import parse_sdmx_json_single_series

_REQUEST_TIMEOUT_SECONDS = 30


class OECDConnector(DataConnector):
    provider_name: ClassVar[str] = "oecd_sdmx"
    BASE_URL_TEMPLATE: ClassVar[str] = "https://stats.oecd.org/SDMX-JSON/data/{dataset}/{key}/all"

    def __init__(self, country_code: str = "TUR", dataset: str = "MEI") -> None:
        self._country_code = country_code
        self._dataset = dataset

    def is_available(self) -> bool:
        return True

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        key = f"{series_code}.{self._country_code}"
        url = self.BASE_URL_TEMPLATE.format(dataset=self._dataset, key=key)
        params = {"contentType": "json", "startTime": start_date.strftime("%Y-%m")}
        if end_date is not None:
            params["endTime"] = end_date.strftime("%Y-%m")

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(
                f"OECD request failed for key '{key}' in dataset '{self._dataset}': {exc}"
            ) from exc

        payload = response.json()
        return parse_sdmx_json_single_series(payload, series_code=series_code)


__all__ = ["OECDConnector"]
