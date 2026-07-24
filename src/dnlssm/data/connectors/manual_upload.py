"""Manual CSV/Excel upload connector.

Serves as the universal fallback for any variable that could not be
retrieved automatically: place a ``<canonical_id>.csv`` or
``<canonical_id>.xlsx`` file with ``date`` and ``value`` columns in the
configured upload directory (default: ``data/manual_uploads/``) and it will
be picked up by :class:`~dnlssm.data.manager.DataManager` exactly like an
API-sourced series, including full provenance logging.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd

from dnlssm.data.base import DataConnector, DataFetchError

_REQUIRED_COLUMNS = {"date", "value"}
_READERS = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".xls": pd.read_excel}


class ManualUploadConnector(DataConnector):
    """Reads a locally supplied series file.

    Here ``series_code`` (as passed by :class:`DataConnector`'s interface)
    is interpreted as the *file stem*, which the :class:`~dnlssm.data.manager.DataManager`
    always sets to the variable's ``canonical_id`` for this provider.
    """

    provider_name: ClassVar[str] = "manual"

    def __init__(self, upload_dir: Path) -> None:
        self._upload_dir = Path(upload_dir)

    def is_available(self) -> bool:
        return True  # the directory is created on demand; absence of a file is per-series

    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        for suffix, reader in _READERS.items():
            path = self._upload_dir / f"{series_code}{suffix}"
            if not path.exists():
                continue
            try:
                frame = reader(path)
            except Exception as exc:
                raise DataFetchError(f"Failed to read manual upload '{path}': {exc}") from exc

            frame.columns = [str(c).strip().lower() for c in frame.columns]
            if not _REQUIRED_COLUMNS.issubset(frame.columns):
                raise DataFetchError(
                    f"Manual upload '{path}' must have columns {_REQUIRED_COLUMNS}, "
                    f"found {set(frame.columns)}."
                )
            dates = pd.to_datetime(frame["date"], errors="coerce")
            values = pd.to_numeric(frame["value"], errors="coerce")
            if dates.isna().all():
                raise DataFetchError(f"Manual upload '{path}' has no parseable dates in the 'date' column.")
            series = pd.Series(values.values, index=dates, name=series_code)
            series = series[~series.index.isna()].sort_index()
            if start_date is not None:
                series = series[series.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                series = series[series.index <= pd.Timestamp(end_date)]
            return series

        raise DataFetchError(
            f"No manual upload file found for '{series_code}' in {self._upload_dir} "
            f"(expected one of {[series_code + s for s in _READERS]}); "
            "supply a CSV/Excel file with 'date' and 'value' columns."
        )


__all__ = ["ManualUploadConnector"]
