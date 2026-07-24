"""TCMB (Central Bank of the Republic of Turkiye) EVDS connector.

Public REST API gated by a free API key. Documented at
https://evds2.tcmb.gov.tr/index.php?/evds/userGuide. EVDS encodes the value
column of a series using the series code with dots replaced by
underscores (e.g. series ``TP.DK.USD.A.YTL`` -> JSON key
``TP_DK_USD_A_YTL``); this connector looks for that column first and falls
back to a defensive numeric-column search if the naming convention does
not match, rather than guessing silently.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError

_REQUEST_TIMEOUT_SECONDS = 30
_NON_VALUE_COLUMNS = {"Tarih", "UNIXTIME"}


class EVDSConnector(DataConnector):
    provider_name: ClassVar[str] = "evds"
    BASE_URL: ClassVar[str] = "https://evds2.tcmb.gov.tr/service/evds/"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return self._api_key is not None

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        if not self.is_available():
            raise DataFetchError(
                "EVDS_API_KEY is not configured (see .env.example); cannot fetch "
                f"series '{series_code}'."
            )
        end = end_date or date.today()
        params = {
            "series": series_code,
            "startDate": start_date.strftime("%d-%m-%Y"),
            "endDate": end.strftime("%d-%m-%Y"),
            "type": "json",
            "key": self._api_key,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(f"EVDS request failed for series '{series_code}': {exc}") from exc

        payload = response.json()
        items = payload.get("items", [])
        if not items:
            raise DataFetchError(
                f"EVDS returned no observations for series '{series_code}'; verify the "
                "series code in the EVDS interactive query system."
            )

        value_column = self._resolve_value_column(items[0], series_code)
        dates = pd.to_datetime([item["Tarih"] for item in items], format="%d-%m-%Y", errors="coerce")
        values = pd.to_numeric([item.get(value_column) for item in items], errors="coerce")
        return pd.Series(values, index=dates, name=series_code).sort_index()

    @staticmethod
    def _resolve_value_column(sample_item: dict, series_code: str) -> str:
        expected = series_code.replace(".", "_")
        if expected in sample_item:
            return expected

        candidates = [k for k in sample_item if k not in _NON_VALUE_COLUMNS]
        numeric_candidates = []
        for key in candidates:
            try:
                float(sample_item[key])
                numeric_candidates.append(key)
            except (TypeError, ValueError):
                continue

        if len(numeric_candidates) == 1:
            return numeric_candidates[0]

        raise DataFetchError(
            f"Could not identify the value column in the EVDS response for series "
            f"'{series_code}'; expected key '{expected}', candidate columns: {candidates}."
        )


__all__ = ["EVDSConnector"]
