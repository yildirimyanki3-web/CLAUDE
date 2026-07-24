"""World Bank Open Data (WDI) connector.

Public REST API, no API key required. Documented at
https://datahelpdesk.worldbank.org/knowledgebase/articles/889392. World
Bank WDI indicators are annual; the resulting series is deliberately left
at annual frequency here -- alignment onto the platform's common monthly
grid is the responsibility of :mod:`dnlssm.preprocessing.alignment`, so
that the interpolation choice is visible and centrally configured rather
than buried inside a connector.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError

_REQUEST_TIMEOUT_SECONDS = 30


class WorldBankConnector(DataConnector):
    provider_name: ClassVar[str] = "world_bank"
    BASE_URL_TEMPLATE: ClassVar[str] = (
        "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    )

    def __init__(self, country_code: str = "TUR") -> None:
        self._country_code = country_code

    def is_available(self) -> bool:
        return True

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        url = self.BASE_URL_TEMPLATE.format(country=self._country_code, indicator=series_code)
        end_year = end_date.year if end_date is not None else date.today().year
        params: dict[str, str | int] = {
            "format": "json",
            "per_page": 20000,
            "date": f"{start_date.year}:{end_year}",
        }

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(
                f"World Bank request failed for indicator '{series_code}' "
                f"(country={self._country_code}): {exc}"
            ) from exc

        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise DataFetchError(
                f"World Bank returned no data for indicator '{series_code}' "
                f"(country={self._country_code}); the indicator code may be invalid."
            )

        records = payload[1]
        dates = pd.to_datetime([f"{rec['date']}-01-01" for rec in records])
        values = pd.to_numeric([rec["value"] for rec in records], errors="coerce")
        return pd.Series(values, index=dates, name=series_code).sort_index()


__all__ = ["WorldBankConnector"]
