"""Resolves provider API credentials from the environment.

Credentials are intentionally kept out of :class:`~dnlssm.config.schema.ExperimentConfig`
so that experiment configuration (which is logged verbatim into every
experiment's metadata for reproducibility) never contains secrets. They are
read once from the process environment (populated from a local ``.env`` via
``python-dotenv`` if present) and passed explicitly to the connectors that
need them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ProviderCredentials:
    """Holds the API keys used by key-gated connectors.

    ``None`` means "not configured"; connectors treat this as
    ``is_available() -> False`` rather than attempting a request that would
    fail with an authentication error.
    """

    evds_api_key: str | None
    fred_api_key: str | None

    def summary(self) -> dict[str, bool]:
        """Reports which credentials are present, without ever exposing the values."""
        return {
            "evds_api_key": self.evds_api_key is not None,
            "fred_api_key": self.fred_api_key is not None,
        }


def load_credentials(dotenv_path: str | None = None) -> ProviderCredentials:
    """Loads provider credentials from the environment (and ``.env`` if present)."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return ProviderCredentials(
        evds_api_key=os.environ.get("EVDS_API_KEY") or None,
        fred_api_key=os.environ.get("FRED_API_KEY") or None,
    )


__all__ = ["ProviderCredentials", "load_credentials"]
