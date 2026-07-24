"""IMF International Financial Statistics (IFS) connector.

Public SDMX-JSON REST API, no API key required. Documented at
https://datahelp.imf.org/knowledgebase/articles/667681. The request key is
``{frequency}.{reference_area}.{indicator}``; ``series_code`` in
``variables.yaml`` supplies the ``indicator`` component.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import pandas as pd
import requests

from dnlssm.data.base import DataConnector, DataFetchError

_REQUEST_TIMEOUT_SECONDS = 30


class IMFIFSConnector(DataConnector):
    provider_name: ClassVar[str] = "imf_ifs"
    BASE_URL: ClassVar[str] = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"

    def __init__(self, reference_area: str = "TR", frequency_code: str = "M") -> None:
        self._reference_area = reference_area
        self._frequency_code = frequency_code

    def is_available(self) -> bool:
        return True

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        key = f"{self._frequency_code}.{self._reference_area}.{series_code}"
        url = f"{self.BASE_URL}/{key}"
        params = {"startPeriod": str(start_date.year)}
        if end_date is not None:
            params["endPeriod"] = str(end_date.year)

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataFetchError(f"IMF IFS request failed for key '{key}': {exc}") from exc

        payload = response.json()
        try:
            series_node = payload["CompactData"]["DataSet"]["Series"]
        except (KeyError, TypeError) as exc:
            raise DataFetchError(
                f"IMF IFS returned an unexpected payload shape for key '{key}'; "
                f"the indicator code may be invalid."
            ) from exc

        observations = series_node.get("Obs", [])
        if isinstance(observations, dict):
            observations = [observations]
        if not observations:
            raise DataFetchError(f"IMF IFS returned no observations for key '{key}'.")

        dates = pd.to_datetime([obs["@TIME_PERIOD"] for obs in observations])
        values = pd.to_numeric([obs.get("@OBS_VALUE") for obs in observations], errors="coerce")
        return pd.Series(values, index=dates, name=series_code).sort_index()


__all__ = ["IMFIFSConnector"]
