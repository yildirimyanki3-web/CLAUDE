"""Unit tests for dnlssm.data.manager.DataManager, using fake in-memory connectors."""

from __future__ import annotations

from datetime import date

import pandas as pd

from dnlssm.config.schema import ObservationSpaceConfig, ProviderSeries, VariableSpec
from dnlssm.data.base import DataConnector, DataFetchError
from dnlssm.data.manager import DataManager


class FakeConnector(DataConnector):
    """In-memory connector: returns a fixed series, raises, or is unavailable, by construction."""

    def __init__(self, provider_name: str, *, available: bool = True, series: pd.Series | None = None):
        self.provider_name = provider_name
        self._available = available
        self._series = series

    def is_available(self) -> bool:
        return self._available

    def fetch_series(self, series_code, start_date, end_date=None) -> pd.Series:
        if self._series is None:
            raise DataFetchError(f"FakeConnector[{self.provider_name}] configured to fail.")
        return self._series


def _dummy_series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, name="dummy")


def _observation_space(variables: list[VariableSpec]) -> ObservationSpaceConfig:
    return ObservationSpaceConfig(start_date=date(2020, 1, 1), end_date=None, variables=variables)


class TestDataManagerFallback:
    def test_first_available_provider_wins(self) -> None:
        good_series = _dummy_series()
        var = VariableSpec(
            canonical_id="x",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[
                ProviderSeries(provider="fred", series_code="ABC"),
                ProviderSeries(provider="world_bank", series_code="DEF"),
            ],
        )
        manager = DataManager(
            connectors={
                "fred": FakeConnector("fred", series=good_series),
                "world_bank": FakeConnector("world_bank", series=good_series),
            },
            manual_upload_dir="/nonexistent",
        )
        result = manager.fetch_all(_observation_space([var]))
        assert result.provenance.records["x"].status == "success"
        assert result.provenance.records["x"].resolved_provider == "fred"
        pd.testing.assert_series_equal(result.series_by_variable["x"], good_series, check_names=False)

    def test_falls_back_to_second_provider_on_failure(self) -> None:
        good_series = _dummy_series()
        var = VariableSpec(
            canonical_id="x",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[
                ProviderSeries(provider="fred", series_code="ABC"),
                ProviderSeries(provider="world_bank", series_code="DEF"),
            ],
        )
        manager = DataManager(
            connectors={
                "fred": FakeConnector("fred", series=None),  # fails
                "world_bank": FakeConnector("world_bank", series=good_series),
            },
            manual_upload_dir="/nonexistent",
        )
        result = manager.fetch_all(_observation_space([var]))
        record = result.provenance.records["x"]
        assert record.status == "success"
        assert record.resolved_provider == "world_bank"
        assert len(record.attempts) == 2
        assert record.attempts[0].status == "failed"

    def test_missing_credentials_recorded_and_skipped(self) -> None:
        var = VariableSpec(
            canonical_id="x",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[ProviderSeries(provider="fred", series_code="ABC")],
        )
        manager = DataManager(
            connectors={"fred": FakeConnector("fred", available=False)},
            manual_upload_dir="/nonexistent",
        )
        result = manager.fetch_all(_observation_space([var]))
        record = result.provenance.records["x"]
        assert record.status == "failed_all_providers"
        assert record.attempts[0].status == "skipped_credentials_missing"
        assert "x" in result.provenance.unresolved_variables()

    def test_null_series_code_skipped_explicitly(self) -> None:
        var = VariableSpec(
            canonical_id="x",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[ProviderSeries(provider="evds", series_code=None)],
        )
        manager = DataManager(connectors={"evds": FakeConnector("evds")}, manual_upload_dir="/nonexistent")
        result = manager.fetch_all(_observation_space([var]))
        record = result.provenance.records["x"]
        assert record.status == "failed_all_providers"
        assert record.attempts[0].status == "skipped_no_series_code"

    def test_placeholder_variable_never_attempts_fetch(self) -> None:
        var = VariableSpec(
            canonical_id="eci",
            description="placeholder",
            unit="index",
            native_frequency="annual",
            provider_priority=[],
            is_placeholder=True,
        )
        manager = DataManager(connectors={}, manual_upload_dir="/nonexistent")
        result = manager.fetch_all(_observation_space([var]))
        assert result.provenance.records["eci"].status == "placeholder"
        assert "eci" not in result.series_by_variable

    def test_manual_provider_uses_canonical_id_as_series_code(self) -> None:
        good_series = _dummy_series()

        class RecordingFakeConnector(FakeConnector):
            def __init__(self, series):
                super().__init__("manual", series=series)
                self.requested_codes: list[str] = []

            def fetch_series(self, series_code, start_date, end_date=None):
                self.requested_codes.append(series_code)
                return super().fetch_series(series_code, start_date, end_date)

        manual_connector = RecordingFakeConnector(series=good_series)
        var = VariableSpec(
            canonical_id="policy_rate",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[ProviderSeries(provider="manual", series_code=None)],
        )
        manager = DataManager(connectors={"manual": manual_connector}, manual_upload_dir="/nonexistent")
        result = manager.fetch_all(_observation_space([var]))
        assert manual_connector.requested_codes == ["policy_rate"]
        assert result.provenance.records["policy_rate"].status == "manual_upload"


class TestProvenanceLogSummary:
    def test_summary_reports_unresolved(self) -> None:
        var_ok = VariableSpec(
            canonical_id="ok",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[ProviderSeries(provider="fred", series_code="X")],
        )
        var_missing = VariableSpec(
            canonical_id="missing",
            description="x",
            unit="percent",
            native_frequency="monthly",
            provider_priority=[ProviderSeries(provider="fred", series_code="Y")],
        )
        manager = DataManager(
            connectors={
                "fred": FakeConnector(
                    "fred",
                    series=_dummy_series(),
                )
            },
            manual_upload_dir="/nonexistent",
        )
        # Force the second variable to fail by using a connector that always fails
        # regardless of series_code, mimicking a provider outage.
        class SometimesFails(FakeConnector):
            def fetch_series(self, series_code, start_date, end_date=None):
                if series_code == "Y":
                    raise DataFetchError("simulated outage")
                return super().fetch_series(series_code, start_date, end_date)

        manager = DataManager(
            connectors={"fred": SometimesFails("fred", series=_dummy_series())},
            manual_upload_dir="/nonexistent",
        )
        result = manager.fetch_all(_observation_space([var_ok, var_missing]))
        assert result.provenance.unresolved_variables() == ["missing"]
        assert "UNRESOLVED" in result.provenance.summary_text()
