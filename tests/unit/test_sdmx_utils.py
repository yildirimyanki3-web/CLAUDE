"""Unit tests for dnlssm.data.connectors.sdmx_utils."""

from __future__ import annotations

import pytest

from dnlssm.data.base import DataFetchError
from dnlssm.data.connectors.sdmx_utils import parse_sdmx_json_single_series


def _sample_payload() -> dict:
    return {
        "structure": {
            "dimensions": {
                "observation": [
                    {
                        "id": "TIME_PERIOD",
                        "values": [{"id": "2020-01"}, {"id": "2020-02"}, {"id": "2020-03"}],
                    }
                ]
            }
        },
        "dataSets": [
            {
                "series": {
                    "0:0:0:0": {
                        "observations": {
                            "0": [1.5],
                            "1": [2.5],
                            "2": [3.5],
                        }
                    }
                }
            }
        ],
    }


class TestParseSdmxJson:
    def test_extracts_values_in_time_order(self) -> None:
        series = parse_sdmx_json_single_series(_sample_payload(), series_code="TEST")
        assert list(series.values) == [1.5, 2.5, 3.5]
        assert series.index.is_monotonic_increasing

    def test_missing_structure_raises(self) -> None:
        with pytest.raises(DataFetchError):
            parse_sdmx_json_single_series({}, series_code="TEST")

    def test_empty_series_map_raises(self) -> None:
        payload = _sample_payload()
        payload["dataSets"][0]["series"] = {}
        with pytest.raises(DataFetchError):
            parse_sdmx_json_single_series(payload, series_code="TEST")

    def test_out_of_range_index_raises(self) -> None:
        payload = _sample_payload()
        payload["dataSets"][0]["series"]["0:0:0:0"]["observations"]["99"] = [9.9]
        with pytest.raises(DataFetchError):
            parse_sdmx_json_single_series(payload, series_code="TEST")
