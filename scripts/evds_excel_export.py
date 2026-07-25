#!/usr/bin/env python3
"""EVDS (TCMB) macro series discovery -> Excel workbook.

Standalone companion to the main pipeline (``dnlssm-run`` / ``scripts/run_pipeline.py``).
It does **not** feed data into the pipeline or write to ``config/variables.yaml``
directly: ``variables.yaml`` deliberately ships with TCMB EVDS ``series_code``
fields left ``null`` (see the "On provider series codes" note in README.md),
because EVDS codes are periodically renumbered and a silently guessed code is
worse than an explicit gap. This script is the discovery aid for that manual
step -- it searches the EVDS series catalogue, ranks candidates per variable,
pulls the top-ranked series for a first look, and writes everything (chosen
series, alternatives, and their scores) to an Excel workbook so the
researcher can verify each pick against the EVDS interactive query system
before copying a code into ``variables.yaml``.

15 macro variables, 2002-02 .. 2026-05, monthly frequency.

Setup:
    pip install -r requirements.txt   # requests, pandas, openpyxl already included
    cp .env.example .env              # then fill in EVDS_API_KEY
                                       # (obtain at https://evds2.tcmb.gov.tr/index.php?/evds/userLogin)

Usage:
    python scripts/evds_excel_export.py                 # download catalogue, rank candidates, fetch data, write Excel
    python scripts/evds_excel_export.py --ara issizlik   # search the catalogue for a keyword (to fix a code by hand)

Output:
    data/raw/evds_veri.xlsx
        - Veri         : Tarih + 15 monthly columns
        - Seri_Bilgisi : which series code was picked per variable, frequency, aggregation
        - Aday_Seriler : ranked alternative candidates per variable (for manual override)
    data/cache/evds_katalog.json   (catalogue cache; delete to force a re-download)
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
BASLANGIC = "01-02-2002"
BITIS = "31-05-2026"
CIKTI = REPO_ROOT / "data" / "raw" / "evds_veri.xlsx"
KATALOG = REPO_ROOT / "data" / "cache" / "evds_katalog.json"

BASE = "https://evds2.tcmb.gov.tr/service/evds"

# frequency: 5 = aylik, 6 = ceyreklik, 8 = yillik
# aggregation: avg / last / sum  (stok/fiyat -> last veya avg, akim -> sum)
# aranan  : seri adinda mutlaka gecmesi gereken kelimeler (VE)
# haric   : gecerse elenir
# tercih  : bu kelimeler varsa puan artar
DEGISKENLER = [
    dict(ad="Gecelik Borc Verme Faizi (TCMB)", agg="avg",
         kodlar=["TP.APIFON2", "TP.APIFON4"],
         aranan=["gecelik", "borc verme"], tercih=["merkez", "faiz"], haric=["euro", "dolar"]),
    dict(ad="Kredi Hacmi", agg="last",
         kodlar=["TP.KREDI.L001"],
         aranan=["kredi"], tercih=["toplam", "yurt ici krediler"], haric=["faiz", "oran", "takip"]),
    dict(ad="TUFE", agg="last",
         kodlar=["TP.FG.J0"],
         aranan=["tuketici fiyat"], tercih=["genel", "endeks"], haric=["yurt disi", "beklenti"]),
    dict(ad="USD/TRY", agg="last",
         kodlar=["TP.DK.USD.A.YTL"],
         aranan=["abd dolari"], tercih=["alis", "doviz"], haric=["efektif", "sepet"]),
    dict(ad="Cari Islemler Dengesi", agg="sum",
         kodlar=[],
         aranan=["cari islemler"], tercih=["denge", "hesabi"], haric=["yillik", "gsyh"]),
    dict(ad="Butce Dengesi", agg="sum",
         kodlar=[],
         aranan=["butce", "denge"], tercih=["merkezi yonetim"], haric=["faiz disi"]),
    dict(ad="Kamu Borcu / GSYH", agg="last", freq=6,
         kodlar=[],
         aranan=["borc", "gsyh"], tercih=["kamu", "genel yonetim", "brut"], haric=["ozel"]),
    dict(ad="Ozel Sektor Borcu / GSYH", agg="last", freq=6,
         kodlar=[],
         aranan=["ozel sektor", "borc"], tercih=["gsyh", "dis borc"], haric=["kamu"]),
    dict(ad="Sanayi Uretim Endeksi", agg="avg",
         kodlar=[],
         aranan=["sanayi uretim"], tercih=["endeks", "takvim"], haric=["beklenti", "ciro"]),
    dict(ad="Issizlik Orani", agg="avg",
         kodlar=[],
         aranan=["issizlik oran"], tercih=["genel", "toplam"], haric=["genc", "tarim disi"]),
    dict(ad="Ihracat", agg="sum",
         kodlar=[],
         aranan=["ihracat"], tercih=["toplam", "fob"], haric=["birim deger", "endeks", "beklenti"]),
    dict(ad="Ithalat", agg="sum",
         kodlar=[],
         aranan=["ithalat"], tercih=["toplam", "cif"], haric=["birim deger", "endeks", "beklenti"]),
    dict(ad="Vergi Yuku / GSYH", agg="last", freq=6,
         kodlar=[],
         aranan=["vergi"], tercih=["gelir", "gsyh", "merkezi yonetim"], haric=[]),
    dict(ad="Reel Ucret Endeksi", agg="avg",
         kodlar=[],
         aranan=["reel", "ucret"], tercih=["endeks", "imalat"], haric=["asgari"]),
    dict(ad="Imalat Sanayi Kapasite Kullanim Orani", agg="avg",
         kodlar=[],
         aranan=["kapasite kullanim"], tercih=["imalat", "genel"], haric=[]),
]

TR = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosucgosu")


def norm(s):
    return (s or "").translate(TR).lower()


@functools.lru_cache(maxsize=1)
def _api_key() -> str:
    """Resolves the EVDS API key from the environment (loaded from .env if present).

    Deferred to first use (rather than a module-level constant) so the pure
    catalogue-ranking helpers below stay importable -- and testable -- without
    a key configured.
    """
    load_dotenv()
    key = os.environ.get("EVDS_API_KEY")
    if not key:
        raise RuntimeError(
            "EVDS_API_KEY ortam degiskeni tanimli degil. .env.example dosyasini "
            ".env olarak kopyalayip EVDS_API_KEY degerini doldurun "
            "(anahtar: https://evds2.tcmb.gov.tr/index.php?/evds/userLogin)."
        )
    return key


def get(path, **params):
    params.setdefault("type", "json")
    headers = {"key": _api_key(), "User-Agent": "Mozilla/5.0"}
    for deneme in range(4):
        try:
            r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=60)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (deneme + 1))
                continue
            return None
        except Exception:
            time.sleep(2 * (deneme + 1))
    return None


# ---------------------------------------------------------------- katalog
def katalog_yukle():
    if KATALOG.exists():
        with open(KATALOG, encoding="utf-8") as f:
            return json.load(f)

    print("Katalog indiriliyor (ilk calistirmada 2-5 dakika surebilir)...")
    gruplar = get("/datagroups/", mode=0) or []
    seriler = []
    for i, g in enumerate(gruplar, 1):
        kod = g.get("DATAGROUP_CODE")
        if not kod:
            continue
        sl = get("/serieList/", code=kod) or []
        for s in sl:
            seriler.append({
                "kod": s.get("SERIE_CODE"),
                "ad": s.get("SERIE_NAME") or "",
                "frekans": s.get("FREQUENCY_STR") or s.get("FREQUENCY") or "",
                "birim": s.get("DEFAULT_AGG_METHOD_STR") or s.get("BIRIMI") or "",
                "baslangic": s.get("START_DATE") or "",
                "grup": g.get("DATAGROUP_NAME") or kod,
            })
        if i % 25 == 0:
            print(f"  {i}/{len(gruplar)} grup, {len(seriler)} seri")
        time.sleep(0.12)

    KATALOG.parent.mkdir(parents=True, exist_ok=True)
    with open(KATALOG, "w", encoding="utf-8") as f:
        json.dump(seriler, f, ensure_ascii=False)
    print(f"Katalog hazir: {len(seriler)} seri -> {KATALOG}")
    return seriler


def puanla(seri, d):
    ad = norm(seri["ad"])
    for k in d["aranan"]:
        if norm(k) not in ad:
            return -1
    for k in d.get("haric", []):
        if norm(k) in ad:
            return -1
    p = 100 - len(ad) / 10.0                      # kisa/genel isimler daha iyi
    for k in d.get("tercih", []):
        if norm(k) in ad:
            p += 15
    fr = norm(seri["frekans"])
    if "ayl" in fr or "month" in fr:
        p += 30
    elif "ceyr" in fr or "quarter" in fr:
        p += 10
    b = str(seri.get("baslangic", ""))
    if b[-4:].isdigit() and int(b[-4:]) <= 2002:
        p += 10
    return p


def kod_sec(katalog, d):
    adaylar = sorted(
        ((puanla(s, d), s) for s in katalog if s["kod"]),
        key=lambda t: -t[0])
    adaylar = [(p, s) for p, s in adaylar if p > 0][:8]
    return adaylar


# ---------------------------------------------------------------- veri
def seri_cek(kod, freq, agg):
    j = get("/", series=kod, startDate=BASLANGIC, endDate=BITIS,
            frequency=str(freq), aggregationTypes=agg)
    if not j or not j.get("items"):
        return None
    df = pd.DataFrame(j["items"])
    kol = [c for c in df.columns if c not in ("Tarih", "UNIXTIME", "YEARWEEK")]
    if not kol:
        return None
    s = pd.to_numeric(df[kol[0]], errors="coerce")
    t = df["Tarih"].astype(str)
    # "2002-2" / "2002-Q1" / "2002" formatlarini normalize et
    idx = []
    for v in t:
        v = v.replace("Q", "").strip()
        parts = v.split("-")
        if len(parts) == 1:
            idx.append(pd.Timestamp(int(parts[0]), 1, 1))
        else:
            y, p = int(parts[0]), int(parts[1])
            ay = p if freq == 5 else (p - 1) * 3 + 1
            idx.append(pd.Timestamp(y, min(max(ay, 1), 12), 1))
    out = pd.Series(s.values, index=pd.DatetimeIndex(idx)).sort_index()
    return out[~out.index.duplicated(keep="last")]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ara":
        if len(sys.argv) < 3:
            print("Kullanim: python scripts/evds_excel_export.py --ara <arama kelimesi>")
            return
        kat = katalog_yukle()
        q = norm(" ".join(sys.argv[2:]))
        n = 0
        for s in kat:
            if q in norm(s["ad"]):
                print(f"{s['kod']:<28} | {s['frekans']:<12} | {s['ad'][:90]}")
                n += 1
                if n >= 60:
                    break
        print(f"\n{n} sonuc.")
        return

    kat = katalog_yukle()
    aylik = pd.date_range("2002-02-01", "2026-05-01", freq="MS")
    veri, bilgi, aday_kayit = pd.DataFrame(index=aylik), [], []

    for d in DEGISKENLER:
        freq = d.get("freq", 5)
        adaylar = kod_sec(kat, d)
        kod_listesi = list(d["kodlar"]) + [s["kod"] for _, s in adaylar]

        s = None
        secilen = None
        for kod in kod_listesi:
            s = seri_cek(kod, freq, d["agg"])
            if s is not None and s.notna().sum() > 6:
                secilen = kod
                break
            time.sleep(0.2)

        if secilen is None:
            print(f"[BULUNAMADI] {d['ad']}  -> --ara ile elle bakin")
            veri[d["ad"]] = pd.NA
            bilgi.append(dict(Degisken=d["ad"], Seri_Kodu="-", Seri_Adi="BULUNAMADI",
                              Frekans="-", Toplama="-", Not="--ara ile kod bulup DEGISKENLER'e ekleyin"))
            continue

        if freq != 5:                     # ceyreklik/yillik -> aylik ileri doldurma
            s = s.reindex(aylik.union(s.index)).ffill().reindex(aylik)
            notu = "Kaynak ceyreklik/yillik; aylik satirlara ileri dolduruldu"
        else:
            s = s.reindex(aylik)
            notu = ""

        veri[d["ad"]] = s.values
        meta = next((x for _, x in adaylar if x["kod"] == secilen), {})
        print(f"[OK] {d['ad']:<40} {secilen:<28} {int(pd.Series(s.values).notna().sum())} gozlem")
        bilgi.append(dict(Degisken=d["ad"], Seri_Kodu=secilen,
                          Seri_Adi=meta.get("ad", ""), Frekans=meta.get("frekans", ""),
                          Toplama=d["agg"], Not=notu))
        for p, x in adaylar:
            aday_kayit.append(dict(Degisken=d["ad"], Puan=round(p, 1),
                                   Kod=x["kod"], Adi=x["ad"], Frekans=x["frekans"]))

    veri.index.name = "Tarih"
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(CIKTI, engine="openpyxl") as w:
        veri.to_excel(w, sheet_name="Veri")
        pd.DataFrame(bilgi).to_excel(w, sheet_name="Seri_Bilgisi", index=False)
        pd.DataFrame(aday_kayit).to_excel(w, sheet_name="Aday_Seriler", index=False)

    print(f"\nBitti -> {CIKTI}")
    print("Seri_Bilgisi sayfasini kontrol edin; dogrulanan kodlari config/variables.yaml "
          "icindeki ilgili degiskenin evds provider_priority.series_code alanina elle yazin.")


if __name__ == "__main__":
    main()
