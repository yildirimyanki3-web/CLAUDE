"""Unit tests for the EVDS catalogue-ranking helpers in scripts/evds_excel_export.py.

Exercises only the pure functions (``norm``, ``puanla``, ``kod_sec``): no
network access and no ``EVDS_API_KEY`` required, since the module defers key
resolution to first use inside ``get()`` (see ``_api_key``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evds_excel_export.py"
_spec = importlib.util.spec_from_file_location("evds_excel_export", _MODULE_PATH)
evds_excel_export = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = evds_excel_export
_spec.loader.exec_module(evds_excel_export)


class TestNorm:
    def test_lowercases_and_strips_turkish_diacritics(self) -> None:
        assert evds_excel_export.norm("İşsizlik Oranı") == "issizlik orani"

    def test_none_becomes_empty_string(self) -> None:
        assert evds_excel_export.norm(None) == ""


class TestPuanla:
    _degisken = dict(
        ad="Issizlik Orani",
        agg="avg",
        aranan=["issizlik oran"],
        tercih=["genel", "toplam"],
        haric=["genc", "tarim disi"],
    )

    def test_missing_required_keyword_is_rejected(self) -> None:
        seri = {"ad": "Sanayi Uretim Endeksi", "frekans": "Aylik", "baslangic": "1990"}
        assert evds_excel_export.puanla(seri, self._degisken) == -1

    def test_excluded_keyword_is_rejected_even_if_required_keyword_present(self) -> None:
        seri = {"ad": "Genc Issizlik Orani", "frekans": "Aylik", "baslangic": "1990"}
        assert evds_excel_export.puanla(seri, self._degisken) == -1

    def test_preferred_keyword_and_monthly_frequency_score_higher(self) -> None:
        base = {"ad": "Issizlik Orani", "frekans": "", "baslangic": ""}
        genel = {"ad": "Genel Issizlik Orani", "frekans": "Aylik", "baslangic": "2000"}
        assert evds_excel_export.puanla(genel, self._degisken) > evds_excel_export.puanla(base, self._degisken)

    def test_early_start_date_adds_score(self) -> None:
        early = {"ad": "Issizlik Orani", "frekans": "", "baslangic": "1995"}
        late = {"ad": "Issizlik Orani", "frekans": "", "baslangic": "2015"}
        assert evds_excel_export.puanla(early, self._degisken) > evds_excel_export.puanla(late, self._degisken)


class TestKodSec:
    _degisken = dict(
        ad="Issizlik Orani",
        agg="avg",
        aranan=["issizlik oran"],
        tercih=["genel"],
        haric=["genc"],
    )

    def test_filters_out_non_matching_and_excluded_series(self) -> None:
        katalog = [
            {"kod": "A1", "ad": "Genel Issizlik Orani", "frekans": "Aylik", "baslangic": "2000"},
            {"kod": "A2", "ad": "Genc Issizlik Orani", "frekans": "Aylik", "baslangic": "2000"},
            {"kod": "A3", "ad": "Sanayi Uretim Endeksi", "frekans": "Aylik", "baslangic": "2000"},
            {"kod": None, "ad": "Kodsuz Seri", "frekans": "Aylik", "baslangic": "2000"},
        ]
        adaylar = evds_excel_export.kod_sec(katalog, self._degisken)
        kodlar = [s["kod"] for _, s in adaylar]
        assert kodlar == ["A1"]

    def test_results_are_sorted_by_descending_score(self) -> None:
        katalog = [
            {"kod": "B1", "ad": "Issizlik Orani", "frekans": "", "baslangic": ""},
            {"kod": "B2", "ad": "Genel Issizlik Orani", "frekans": "Aylik", "baslangic": "2000"},
        ]
        adaylar = evds_excel_export.kod_sec(katalog, self._degisken)
        scores = [p for p, _ in adaylar]
        assert scores == sorted(scores, reverse=True)
        assert adaylar[0][1]["kod"] == "B2"

    def test_caps_at_eight_candidates(self) -> None:
        katalog = [
            {"kod": f"C{i}", "ad": "Issizlik Orani", "frekans": "", "baslangic": ""} for i in range(12)
        ]
        adaylar = evds_excel_export.kod_sec(katalog, self._degisken)
        assert len(adaylar) == 8


class TestApiKeyResolution:
    def test_raises_clear_error_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        evds_excel_export._api_key.cache_clear()
        monkeypatch.delenv("EVDS_API_KEY", raising=False)
        monkeypatch.setattr(evds_excel_export, "load_dotenv", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="EVDS_API_KEY"):
            evds_excel_export._api_key()
        evds_excel_export._api_key.cache_clear()
