# -*- coding: utf-8 -*-
"""Interannual robustness of the matching gap U (reviewer minor item).

U is an equal-energy (normalised) within-day SHAPE metric, so interannual weather
variability — mostly a MAGNITUDE effect (~+-10% annual energy) — divides out. We
test this on representative countries: hold the demand shape fixed and rebuild each
country's wind+solar supply from N years of Open-Meteo ERA5, using the SAME
construction as the main analysis (national PV/WIND capacity-factor from WRI-weighted
sites, combined by IRENA PV/WIND capacity; build_global_supply_cf + load_inputs), then
recompute U per year. The 2025/26 window reproduces the headline per-country U, so the
interannual spread is reported on the exact paper caliber.

8 countries x FULL site set x 7 May-Apr windows (2019/20..2025/26).
"""
from __future__ import annotations
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests

import build_global_supply_cf as B
import analyze_cfe_global_sites as G

SITE_DIR = B.SITE_DIR
BASE_CACHE = SITE_DIR / "weather_cache"
MY_CACHE = SITE_DIR / "weather_cache_multiyear"
MY_CACHE.mkdir(parents=True, exist_ok=True)
API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "shortwave_radiation,temperature_2m,wind_speed_100m"
RELEASE_ROOT = Path(__file__).resolve().parents[1]
PAPER_U_CSV = RELEASE_ROOT / "data" / "Supplementary_Data_1_country_level_domestic_mismatch.csv"
OUT = RELEASE_ROOT / "data" / "station_expanded" / "interannual_matching_gap.csv"

PICK = ["US", "DE", "GB", "SA", "IN", "AU", "BR", "ZA"]
START_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
L = 8760


def cid(lat, lon):
    return f"{lat:.3f}_{lon:.3f}".replace("-", "m")


def get_weather(lat, lon, Y):
    if Y == 2025:
        f = BASE_CACHE / f"w_{cid(lat, lon)}.npz"
        if f.exists():
            d = np.load(f)
            return d["ghi"], d["t2m"], d["ws100"]
    out = MY_CACHE / f"w_{Y}_{cid(lat, lon)}.npz"
    if out.exists():
        d = np.load(out)
        return d["ghi"], d["t2m"], d["ws100"]
    params = dict(latitude=lat, longitude=lon, start_date=f"{Y}-05-02", end_date=f"{Y+1}-05-01",
                  hourly=HOURLY, timezone="UTC", windspeed_unit="ms", cell_selection="nearest")
    for att in range(5):
        try:
            r = requests.get(API, params=params, timeout=(15, 120))
            if r.status_code == 200:
                h = r.json()["hourly"]
                def a(k):
                    return np.asarray([np.nan if v is None else v for v in h[k]], dtype="float32")
                ghi, t2m, ws100 = a("shortwave_radiation"), a("temperature_2m"), a("wind_speed_100m")
                np.savez_compressed(out, ghi=ghi, t2m=t2m, ws100=ws100)
                return ghi, t2m, ws100
            if r.status_code == 429:
                time.sleep(8 * (att + 1)); continue
            time.sleep(2 * (att + 1))
        except Exception:
            time.sleep(3 * (att + 1))
    return None


def wri_weighted(grp, cf_of):
    w = grp["wri_capacity_mw"].to_numpy(float)
    if w.sum() <= 0:
        w = np.ones(len(grp))
    w = w / w.sum()
    nat = np.zeros(L)
    for wi, (_, s) in zip(w, grp.iterrows()):
        nat += wi * cf_of(s)
    return nat


def main():
    print("loading demand ...", flush=True)
    _, demand, _, _, _ = G.load_inputs_expanded()
    sites = pd.read_csv(B.SITES, keep_default_na=False, na_values=[""])
    cap = pd.read_csv(SITE_DIR / "country_capacity.csv", keep_default_na=False, na_values=[""])
    cap_used = {(r.iso2, r.tech): float(r.capacity_mw_used)
                for r in cap.itertuples() if str(r.capacity_mw_used) not in ("", "nan")}
    paper = pd.read_csv(PAPER_U_CSV).set_index("country_iso2")["matching_gap_U_percent"]

    csites_all = sites[sites.iso2.isin(PICK)]
    coords = csites_all[["lat", "lon"]].drop_duplicates()
    tasks = [(float(r.lat), float(r.lon), Y) for _, r in coords.iterrows() for Y in START_YEARS]
    print(f"prefetch {len(tasks)} (coord x year) ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(get_weather, la, lo, Y): 1 for la, lo, Y in tasks}
        ok = fail = 0
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); ok += r is not None; fail += r is None
            if n % 120 == 0:
                print(f"  {n}/{len(tasks)} ok={ok} fail={fail}", flush=True)
    print(f"prefetch done ok={ok} fail={fail}", flush=True)

    def cf_pv(Y):
        return lambda s: B.pv_cf(np.nan_to_num(get_weather(float(s.lat), float(s.lon), Y)[0].astype(float)),
                                 np.nan_to_num(get_weather(float(s.lat), float(s.lon), Y)[1].astype(float), nan=15.0))[:L]

    def cf_wd(Y):
        return lambda s: B.wind_cf(np.nan_to_num(get_weather(float(s.lat), float(s.lon), Y)[2].astype(float)))[:L]

    rows = []
    for c in PICK:
        g = sites[sites.iso2 == c]
        pv_g, wd_g = g[g.tech == "PV"], g[g.tech == "WIND"]
        d = demand[c]["d"].to_numpy(float)[:L]
        d_hat = d / d.mean()
        cpv = cap_used.get((c, "PV"), float(pv_g.wri_capacity_mw.sum()))
        cwd = cap_used.get((c, "WIND"), float(wd_g.wri_capacity_mw.sum()))
        for Y in START_YEARS:
            nat_pv = wri_weighted(pv_g if not pv_g.empty else wd_g, cf_pv(Y))
            nat_wd = wri_weighted(wd_g if not wd_g.empty else pv_g, cf_wd(Y))
            tot = cpv + cwd
            vre = (cpv * nat_pv + cwd * nat_wd) / tot if tot > 0 else 0.5 * (nat_pv + nat_wd)
            if vre.mean() <= 0:
                continue
            U = float(np.maximum(d_hat - vre / vre.mean(), 0.0).sum() / d_hat.sum()) * 100
            rows.append({"country": c, "window": f"{Y}/{str(Y+1)[2:]}", "U_pct": round(U, 2)})
            print(f"  {c} {Y}/{str(Y+1)[2:]}  U={U:.2f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    piv = df.pivot(index="country", columns="window", values="U_pct")
    print("\n=== interannual U (%) — full site set + IRENA weighting ===", flush=True)
    print(piv.to_string(), flush=True)
    sp = df.groupby("country").U_pct.agg(["min", "max", "mean", "std"])
    sp["range_pp"] = (sp["max"] - sp["min"]).round(2)
    sp["paper_U"] = [paper.get(c, np.nan) for c in sp.index]
    sp["base_2025_26"] = [df[(df.country == c) & (df.window == "2025/26")].U_pct.iloc[0] for c in sp.index]
    sp["base_minus_paper"] = (sp["base_2025_26"] - sp["paper_U"]).round(2)
    print("\n=== spread + baseline vs headline validation ===", flush=True)
    print(sp.round(2).to_string(), flush=True)
    print(f"\nmedian interannual range = {sp['range_pp'].median():.2f} pp", flush=True)
    print(f"mean interannual std      = {sp['std'].mean():.2f} pp", flush=True)
    print(f"max |baseline - headline| = {sp['base_minus_paper'].abs().max():.2f} pp", flush=True)


if __name__ == "__main__":
    main()
