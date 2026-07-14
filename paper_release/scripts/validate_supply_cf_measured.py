#!/usr/bin/env python3
"""Reviewer-2 concern 5: validate the MODELLED hourly wind/solar capacity-factor
shapes (ERA5 -> PVWatts/IEC) against MEASURED hourly generation, over the paper's
own window (2025-05-02 .. 2026-05-01), for representative countries.

Measured sources (open, no API key):
  - Germany (DE): SMARD / Bundesnetzagentur  (solar 4068, wind onshore 4067,
    wind offshore 4066), hourly MWh.
Modelled source: data/global_renewable_sites/country_hourly_cf.parquet (the
paper's country-level hourly CF used for R1/R2).

Metrics the reviewer asked for, per technology (PV, WIND), on unit-mean-normalised
shapes so only TIMING is compared (scale divided out):
  - hourly Pearson correlation over the full year
  - diurnal (24 h, local) mean profile: correlation + peak-hour offset
  - seasonality (12 monthly means): correlation
  - continuous-deficit-relevant: nothing here; this validates the supply shape.

Output: data_globalsites_stations_expanded/supply_validation_measured.csv
"""
from __future__ import annotations
import json
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CF = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites" / "country_hourly_cf.parquet"
OUT = (ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai"
       / "data_globalsites_stations_expanded")
W0 = pd.Timestamp("2025-05-02 00:00", tz="UTC")
W1 = pd.Timestamp("2026-05-01 23:00", tz="UTC")
SMARD = "https://www.smard.de/app/chart_data/{f}/DE/{f}_DE_hour_{ts}.json"
SMARD_IDX = "https://www.smard.de/app/chart_data/{f}/DE/index_hour.json"
ELEXON = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type?from={a}&to={b}"
TZ = {"DE": "Europe/Berlin", "GB": "Europe/London"}


def get_json(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if k == tries - 1:
                raise
    return None


def smard_series(filter_id):
    """Return a UTC-indexed hourly MWh Series for one SMARD filter over the window."""
    idx = get_json(SMARD_IDX.format(f=filter_id))["timestamps"]
    lo = int((W0.timestamp() - 8 * 86400) * 1000)
    hi = int((W1.timestamp() + 8 * 86400) * 1000)
    vals = {}
    for ts in idx:
        if ts < lo or ts > hi:
            continue
        for row in get_json(SMARD.format(f=filter_id, ts=ts))["series"]:
            t, v = row[0], row[1]
            if v is not None:
                vals[t] = v
    s = pd.Series(vals)
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.sort_index()


def elexon_gb():
    """Return (solar, wind) UTC-indexed hourly Series for GB from Elexon per-type."""
    solar, wind = {}, {}
    a = W0 - pd.Timedelta(days=2)
    end = W1 + pd.Timedelta(days=2)
    while a < end:
        b = min(a + pd.Timedelta(days=7), end)
        d = get_json(ELEXON.format(a=a.strftime("%Y-%m-%d"), b=b.strftime("%Y-%m-%d")))
        for rec in d.get("data", []):
            t = pd.Timestamp(rec["startTime"])
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            sv = wv = 0.0
            for e in rec.get("data", []):
                p, q = e.get("psrType"), (e.get("quantity") or 0.0)
                if p == "Solar":
                    sv += q
                elif p in ("Wind Onshore", "Wind Offshore"):
                    wv += q
            solar[t], wind[t] = sv, wv
        a = b
    s = pd.Series(solar).sort_index()
    w = pd.Series(wind).sort_index()
    return (s.resample("1h").mean(), w.resample("1h").mean())


def modelled(iso2, tech):
    df = pd.read_parquet(CF)
    d = df[(df.iso2 == iso2) & (df.tech == tech)].copy()
    s = pd.Series(d["cf"].to_numpy(float), index=pd.to_datetime(d["timestamp_utc"], utc=True))
    return s.sort_index()


def norm(s):
    m = s.mean()
    return s / m if m and np.isfinite(m) else s


def compare(model, meas, label, tz):
    """Align on common hourly UTC index, drop NaN, compute timing metrics."""
    df = pd.concat([norm(model).rename("model"), norm(meas).rename("meas")], axis=1, sort=True)
    df = df.loc[W0:W1].dropna()
    if len(df) < 1000:
        return {"series": label, "n_hours": len(df), "note": "insufficient overlap"}
    r = float(np.corrcoef(df["model"], df["meas"])[0, 1])
    loc = df.index.tz_convert(tz)
    hod = loc.hour
    di_m = df["model"].groupby(hod).mean()
    di_e = df["meas"].groupby(hod).mean()
    r_diurnal = float(np.corrcoef(di_m.reindex(range(24)).fillna(0),
                                  di_e.reindex(range(24)).fillna(0))[0, 1])
    peak_m, peak_e = int(di_m.idxmax()), int(di_e.idxmax())
    mon = df.index.month
    mo_m = df["model"].groupby(mon).mean()
    mo_e = df["meas"].groupby(mon).mean()
    r_season = float(np.corrcoef(mo_m.reindex(range(1, 13)).fillna(mo_m.mean()),
                                 mo_e.reindex(range(1, 13)).fillna(mo_e.mean()))[0, 1])
    return {"series": label, "n_hours": int(len(df)),
            "hourly_r": round(r, 3), "diurnal_r": round(r_diurnal, 3),
            "peak_hour_model_local": peak_m, "peak_hour_meas_local": peak_e,
            "peak_hour_offset_h": peak_m - peak_e, "season_r": round(r_season, 3)}


def main():
    print("fetching SMARD measured generation (DE) ...")
    solar = smard_series(4068)
    wind_on = smard_series(4067)
    wind_off = smard_series(4066)
    wind = wind_on.add(wind_off, fill_value=0.0)
    print(f"  solar hrs={len(solar)}  wind_on={len(wind_on)}  wind_off={len(wind_off)}")

    print("loading modelled CF (DE) ...")
    m_pv = modelled("DE", "PV")
    m_wd = modelled("DE", "WIND")

    rows = [compare(m_pv, solar, "DE PV", TZ["DE"]), compare(m_wd, wind, "DE WIND", TZ["DE"])]

    print("fetching Elexon measured generation (GB) ...")
    try:
        gb_solar, gb_wind = elexon_gb()
        print(f"  GB solar hrs={len(gb_solar)}  wind hrs={len(gb_wind)}")
        rows += [compare(modelled("GB", "PV"), gb_solar, "GB PV", TZ["GB"]),
                 compare(modelled("GB", "WIND"), gb_wind, "GB WIND", TZ["GB"])]
    except Exception as e:
        print("  GB/Elexon failed:", repr(e)[:160])

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "supply_validation_measured.csv", index=False)
    print("\n=== SUPPLY-SHAPE VALIDATION: modelled (ERA5->PVWatts/IEC) vs measured (SMARD), DE, 2025-05..2026-05 ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(res.to_string(index=False))
    print(f"\noutput -> {OUT / 'supply_validation_measured.csv'}")


if __name__ == "__main__":
    main()
