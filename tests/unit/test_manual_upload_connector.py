"""Unit tests for dnlssm.data.connectors.manual_upload.ManualUploadConnector."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from dnlssm.data.base import DataFetchError
from dnlssm.data.connectors.manual_upload import ManualUploadConnector


class TestManualUploadConnector:
    def test_reads_csv_with_date_value_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "policy_rate.csv"
        csv_path.write_text("date,value\n2020-01-01,7.5\n2020-02-01,8.0\n")
        connector = ManualUploadConnector(upload_dir=tmp_path)
        series = connector.fetch_series("policy_rate", date(2020, 1, 1))
        assert list(series.values) == [7.5, 8.0]
        assert len(series) == 2

    def test_column_names_case_insensitive(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("Date,Value\n2020-01-01,1.0\n")
        connector = ManualUploadConnector(upload_dir=tmp_path)
        series = connector.fetch_series("x", date(2020, 1, 1))
        assert len(series) == 1

    def test_missing_file_raises_data_fetch_error(self, tmp_path: Path) -> None:
        connector = ManualUploadConnector(upload_dir=tmp_path)
        with pytest.raises(DataFetchError):
            connector.fetch_series("nonexistent", date(2020, 1, 1))

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("a,b\n1,2\n")
        connector = ManualUploadConnector(upload_dir=tmp_path)
        with pytest.raises(DataFetchError):
            connector.fetch_series("bad", date(2020, 1, 1))

    def test_date_range_filtering(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("date,value\n2019-01-01,1\n2020-06-01,2\n2021-01-01,3\n")
        connector = ManualUploadConnector(upload_dir=tmp_path)
        series = connector.fetch_series("x", date(2020, 1, 1), date(2020, 12, 31))
        assert len(series) == 1
        assert series.iloc[0] == 2

    def test_excel_file_supported(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "value": [1.1, 2.2]})
        xlsx_path = tmp_path / "wage.xlsx"
        df.to_excel(xlsx_path, index=False)
        connector = ManualUploadConnector(upload_dir=tmp_path)
        series = connector.fetch_series("wage", date(2020, 1, 1))
        assert len(series) == 2
