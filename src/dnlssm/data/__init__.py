"""Automated, provenance-tracked data acquisition.

This sub-package downloads the observation-space series defined in
``config/variables.yaml`` from official providers (TCMB EVDS, World Bank,
IMF IFS, OECD SDMX, BIS, FRED) with a uniform, extensible connector
interface, falling back to researcher-supplied CSV/Excel files where no
API source is configured or reachable. Every attempt -- success, failure,
or skip -- is recorded by :mod:`dnlssm.data.provenance` so that every
column of the final observation matrix can be traced to exactly one
source, exactly as the research design requires.

No network call is ever silently retried into a different series or
allowed to fail silently: a missing credential, an unset series code, or
an HTTP error all produce an explicit, itemized report from
:class:`dnlssm.data.manager.DataManager`.
"""

from dnlssm.data.base import DataConnector, DataFetchError
from dnlssm.data.manager import DataAcquisitionResult, DataManager

__all__ = ["DataConnector", "DataFetchError", "DataAcquisitionResult", "DataManager"]
