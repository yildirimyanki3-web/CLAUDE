"""Abstract data-connector interface.

Every provider-specific connector (FRED, World Bank, IMF IFS, OECD SDMX,
BIS, TCMB EVDS, manual upload) implements this same small interface, so
:class:`dnlssm.data.manager.DataManager` can iterate over a variable's
``provider_priority`` list without any provider-specific branching. Adding
a new provider in the future means writing one new subclass, not touching
the orchestration logic.
"""

from __future__ import annotations

import abc
from datetime import date
from typing import ClassVar

import pandas as pd


class DataFetchError(RuntimeError):
    """Raised by a connector when a series could not be retrieved.

    Always carries a human-readable, actionable message (missing
    credential, malformed provider response, network failure, ...) --
    connectors must never return an empty or synthetic series in place of
    raising this exception.
    """


class DataConnector(abc.ABC):
    """Base class for a single data-provider connector."""

    provider_name: ClassVar[str]

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this connector currently has what it needs to attempt a fetch.

        For key-gated providers this checks credential presence; for public
        providers it should simply return ``True``. This is checked by the
        :class:`~dnlssm.data.manager.DataManager` *before* attempting
        ``fetch_series`` so that a missing credential is reported as
        ``skipped_credentials_missing`` rather than as a generic failure.
        """

    @abc.abstractmethod
    def fetch_series(
        self, series_code: str, start_date: date, end_date: date | None = None
    ) -> pd.Series:
        """Fetches one series and returns it as a ``pandas.Series``.

        Parameters
        ----------
        series_code:
            Provider-native series identifier, taken verbatim from
            ``VariableSpec.provider_priority[i].series_code``.
        start_date, end_date:
            Inclusive date bounds. ``end_date=None`` means "up to the most
            recent observation available from the provider".

        Returns
        -------
        A ``pandas.Series`` indexed by a ``DatetimeIndex`` at the series'
        native frequency, named ``series_code``. Values that the provider
        marks as missing must be represented as ``NaN``, never dropped
        silently (dropping would misalign the index during frequency
        alignment).

        Raises
        ------
        DataFetchError
            If the request fails, the provider returns no data, or the
            response cannot be parsed into the expected shape. Connectors
            must never catch-and-suppress these conditions.
        """


__all__ = ["DataConnector", "DataFetchError"]
