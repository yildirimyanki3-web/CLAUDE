"""FRED (Federal Reserve Economic Data) connector.

API documented at https://fred.stlouisfed.org/docs/api/fred/series_observations.html.
Requires a free API key (``FRED_API_KEY``); see ``.env.example``.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError

_REQUEST_TIMEOUT_SECONDS = 30


class FREDConnector(DataConnector):
    provider_name: ClassVar[str] = "fred"
    BASE_URL: ClassVar[str] = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return self._api_key is not None

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        if not self.is_available():
            raise DataFetchError(
                "FRED_API_KEY is not configured (see .env.example); cannot fetch "
                f"series '{series_code}'."
            )
        params = {
            "series_id": series_code,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
        }
        if end_date is not None:
            params["observation_end"] = end_date.isoformat()

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(f"FRED request failed for series '{series_code}': {exc}") from exc

        payload = response.json()
        observations = payload.get("observations", [])
        if not observations:
            raise DataFetchError(f"FRED returned no observations for series '{series_code}'.")

        dates = pd.to_datetime([obs["date"] for obs in observations])
        # FRED encodes missing observations as the literal string "."; coercing
        # to numeric turns these into NaN rather than raising or silently
        # dropping the row, preserving the time index for later alignment.
        values = pd.to_numeric([obs["value"] for obs in observations], errors="coerce")
        return pd.Series(values, index=dates, name=series_code).sort_index()


__all__ = ["FREDConnector"]
