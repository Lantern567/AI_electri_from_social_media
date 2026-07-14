#!/usr/bin/env python3
"""Stage 3: convert per-site ERA5 weather into hourly PV and wind capacity
factors, then aggregate to capacity-weighted NATIONAL and GLOBAL supply curves.

CF models (standard, literature-anchored; equal-energy normalization downstream
makes the analysis insensitive to absolute-level bias):
  PV   : cf = clip( GHI/1000 * [1 + gamma*(Tcell-25)], 0, 1 )
         Tcell = T2M + (NOCT-20)/800 * GHI   (PVWatts / GSEE first-order model)
  Wind : single-turbine IEC-II cubic power curve (cut-in 3.5, rated 12,
         cut-out 25 m/s) on wind_speed_100m, convolved with a Gaussian over wind
         speed (sigma=1.5 m/s) to emulate spatial aggregation of a turbine fleet
         (Staffell & Pfenninger 2016 multi-turbine smoothing).

Weighting:
  within-country : site weight proportional to WRI station capacity (geographic
                   distribution of capacity inside the country)
  cross-country  : national magnitude from IRENA 2025 OnGrid installed capacity
                   (Solar PV / Wind), falling back to summed WRI capacity.

Inputs : data/global_renewable_sites/representative_sites.csv
         data/global_renewable_sites/weather_cache/*.npz
         data/global_renewable_sites/weather_time_index.npy
         paper_search_2026-04-27/.../country_solar_wind_capacity_irena_2025_on_grid.csv
Outputs: data/global_renewable_sites/country_hourly_cf.parquet  (iso2,tech,ts,cf)
         data/global_renewable_sites/country_capacity.csv
         data/global_renewable_sites/global_hourly_cf.csv       (ts,pv_cf,wind_cf)
         data/global_renewable_sites/site_cf_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites"
SITES = SITE_DIR / "representative_sites.csv"
CACHE = SITE_DIR / "weather_cache"
TIME_INDEX = SITE_DIR / "weather_time_index.npy"
IRENA = (ROOT / "paper_search_2026-04-27" / "data" / "installed_capacity"
         / "country_solar_wind_capacity_irena_2025_on_grid.csv")

# PV model params
G_STC = 1000.0
NOCT = 45.0
GAMMA = -0.0035
# Wind power curve params (m/s)
V_CI, V_R, V_CO = 3.5, 12.0, 25.0
WIND_SMOOTH_SIGMA = 1.5
# ERA5 underestimates hub-height wind at well-sited turbines (coarse grid smooths
# terrain). A mild multiplicative bias correction is standard practice
# (renewables.ninja applies country-specific factors ~1.2-1.3). Shape-preserving.
WIND_SPEED_BIAS = 1.2


def coord_id(lat: float, lon: float) -> str:
    return f"{lat:.3f}_{lon:.3f}".replace("-", "m")


def pv_cf(ghi: np.ndarray, t2m: np.ndarray) -> np.ndarray:
    tcell = t2m + (NOCT - 20.0) / 800.0 * ghi
    pr = 1.0 + GAMMA * (tcell - 25.0)
    return np.clip(ghi / G_STC * pr, 0.0, 1.0)


def _single_turbine_cf(v: np.ndarray) -> np.ndarray:
    cf = np.zeros_like(v, dtype=float)
    ramp = (v >= V_CI) & (v < V_R)
    cf[ramp] = (v[ramp] ** 3 - V_CI ** 3) / (V_R ** 3 - V_CI ** 3)
    cf[(v >= V_R) & (v < V_CO)] = 1.0
    return cf


def wind_cf(ws100: np.ndarray) -> np.ndarray:
    """Fleet CF: average the single-turbine curve over a Gaussian wind-speed
    kernel (multi-turbine smoothing), after a mild bias correction."""
    ws100 = ws100 * WIND_SPEED_BIAS
    offsets = np.arange(-3, 3.01, 0.5)
    wts = np.exp(-0.5 * (offsets / WIND_SMOOTH_SIGMA) ** 2)
    wts = wts / wts.sum()
    out = np.zeros_like(ws100, dtype=float)
    for off, w in zip(offsets, wts):
        out += w * _single_turbine_cf(np.maximum(ws100 + off, 0.0))
    return np.clip(out, 0.0, 1.0)


def irena_capacity_by_iso() -> dict[tuple[str, str], float]:
    """Map (iso2, tech) -> IRENA 2025 OnGrid MW via country-name matching."""
    import geonamescache

    gc = geonamescache.GeonamesCache()
    name_to_iso2 = {}
    for _, info in gc.get_countries().items():
        for key in (info.get("name"), ):
            if key:
                name_to_iso2[key.lower()] = info["iso"]
    patches = {
        "united states": "US", "usa": "US", "russian federation": "RU",
        "south korea": "KR", "korea, rep.": "KR", "korea rep": "KR",
        "iran (islamic republic of)": "IR", "iran": "IR", "viet nam": "VN",
        "bolivia (plurinational state of)": "BO", "venezuela (bolivarian republic of)": "VE",
        "tanzania, united rep. of": "TZ", "uk": "GB", "united kingdom": "GB",
        "czechia": "CZ", "czech republic": "CZ", "turkiye": "TR", "türkiye": "TR",
        "syrian arab republic": "SY", "moldova, rep. of": "MD", "lao pdr": "LA",
        "north macedonia": "MK", "cote d'ivoire": "CI", "congo, dem. rep.": "CD",
        "congo": "CG", "brunei darussalam": "BN", "cabo verde": "CV",
    }
    df = pd.read_csv(IRENA)
    out: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        nm = str(r["Country/area"]).strip().lower()
        iso2 = patches.get(nm) or name_to_iso2.get(nm)
        if not iso2:
            continue
        pv = pd.to_numeric(r.get("Solar photovoltaic (MW)"), errors="coerce")
        wd = pd.to_numeric(r.get("Wind energy (MW)"), errors="coerce")
        if pd.notna(pv):
            out[(iso2, "PV")] = float(pv)
        if pd.notna(wd):
            out[(iso2, "WIND")] = float(wd)
    return out


def main() -> None:
    # keep_default_na=False so ISO2 'NA' (Namibia) is not parsed as missing.
    sites = pd.read_csv(SITES, keep_default_na=False, na_values=[""])
    times = np.load(TIME_INDEX, allow_pickle=True)
    ts = pd.to_datetime([str(x) for x in times], utc=True)
    n = len(ts)
    print(f"sites={len(sites)}  hours={n}  range={ts.min()}..{ts.max()}")

    # per-site CF: compute BOTH pv and wind CF at every site coordinate (ERA5 has
    # irradiance AND wind everywhere) so every country gets a PV and a wind RESOURCE
    # shape regardless of which technology WRI happened to record there. This keeps
    # the equal-energy, resource-shape-based portfolio consistent with the original
    # (which had a Tong wind shape for every country, not only those with plants).
    site_pv: dict[str, np.ndarray] = {}
    site_wind: dict[str, np.ndarray] = {}
    summ = []
    missing = 0
    for _, s in sites.iterrows():
        f = CACHE / f"w_{coord_id(s['lat'], s['lon'])}.npz"
        if not f.exists():
            missing += 1
            continue
        w = np.load(f)
        ghi = np.nan_to_num(w["ghi"].astype(float))
        t2m = np.nan_to_num(w["t2m"].astype(float), nan=15.0)
        ws100 = np.nan_to_num(w["ws100"].astype(float))
        cpv, cwd = pv_cf(ghi, t2m), wind_cf(ws100)
        if len(cpv) != n:
            cpv, cwd = np.resize(cpv, n), np.resize(cwd, n)
        site_pv[s["site_id"]] = cpv
        site_wind[s["site_id"]] = cwd
        summ.append({"site_id": s["site_id"], "iso2": s["iso2"], "tech": s["tech"],
                     "lat": s["lat"], "lon": s["lon"], "wri_capacity_mw": s["wri_capacity_mw"],
                     "mean_cf_pv": float(cpv.mean()), "mean_cf_wind": float(cwd.mean())})
    if missing:
        print(f"WARNING: {missing} sites missing weather cache (run fetch stage first)")
    pd.DataFrame(summ).to_csv(SITE_DIR / "site_cf_summary.csv", index=False)
    site_ok = set(site_pv)

    # per-site hourly CF (each station's BUILT technology) for station-level pooling
    valid = [s for _, s in sites.iterrows() if s["site_id"] in site_ok]
    if valid:
        cf_mat = np.empty((len(valid), n), dtype="float32")
        sids, iso2s, techs = [], [], []
        for k, s in enumerate(valid):
            sid = s["site_id"]
            cf_mat[k] = site_pv[sid] if s["tech"] == "PV" else site_wind[sid]
            sids.append(sid); iso2s.append(s["iso2"]); techs.append(s["tech"])
        pd.DataFrame({
            "site_id": np.repeat(sids, n),
            "iso2": np.repeat(iso2s, n),
            "tech": np.repeat(techs, n),
            "timestamp_utc": np.tile(ts.values, len(valid)),
            "cf": cf_mat.reshape(-1),
        }).to_parquet(SITE_DIR / "site_hourly_cf.parquet", index=False)

    # national CF: PV from PV-tagged sites (fallback to the country's wind-site
    # coords), wind from WIND-tagged sites (fallback to PV-site coords). Within-
    # country weight = WRI station capacity.
    irena = irena_capacity_by_iso()

    def aggregate(grp_sites, cfdict):
        w = grp_sites["wri_capacity_mw"].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(grp_sites))
        w = w / w.sum()
        nat = np.zeros(n)
        for wi, sid in zip(w, grp_sites["site_id"]):
            nat += wi * cfdict[sid]
        return nat

    nat_rows, cap_rows = [], []
    for iso2, g in sites.groupby("iso2"):
        g = g[g["site_id"].isin(site_ok)]
        if g.empty or pd.isna(iso2):
            continue
        pv_g, wd_g = g[g["tech"] == "PV"], g[g["tech"] == "WIND"]
        nat_pv = aggregate(pv_g if not pv_g.empty else wd_g, site_pv)
        nat_wd = aggregate(wd_g if not wd_g.empty else pv_g, site_wind)
        for tech, nat, tg in (("PV", nat_pv, pv_g), ("WIND", nat_wd, wd_g)):
            wri_cap = float(tg["wri_capacity_mw"].sum())
            cap_rows.append({
                "iso2": iso2, "tech": tech, "country_long": g["country_long"].iloc[0],
                "n_sites": int(len(tg)), "has_native_sites": bool(len(tg)),
                "wri_capacity_mw": wri_cap,
                "irena_capacity_mw": irena.get((iso2, tech), np.nan),
                "capacity_mw_used": irena.get((iso2, tech), wri_cap),
                "mean_cf": float(nat.mean()),
            })
            for t_, cf_ in zip(ts, nat):
                nat_rows.append((iso2, tech, t_, cf_))
    nat_df = pd.DataFrame(nat_rows, columns=["iso2", "tech", "timestamp_utc", "cf"])
    nat_df.to_parquet(SITE_DIR / "country_hourly_cf.parquet", index=False)
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(SITE_DIR / "country_capacity.csv", index=False)

    # country metadata: capacity-weighted centroid + region/continent blocks so
    # the Result 2 power pool can generalize beyond the hand-curated 31 countries.
    import geonamescache
    cont_code = {k: v.get("continentcode") for k, v in geonamescache.GeonamesCache().get_countries().items()}
    block = {"NA": "Americas", "SA": "Americas", "EU": "EurAfrMENA", "AF": "EurAfrMENA",
             "AS": "Asia", "OC": "Oceania", "AN": "Other"}
    meta_rows = []
    for iso2, g in sites[sites["site_id"].isin(site_ok)].groupby("iso2"):
        if pd.isna(iso2):
            continue
        w = g["wri_capacity_mw"].to_numpy(dtype=float)
        cc = cont_code.get(iso2)
        meta_rows.append({
            "iso2": iso2, "country_long": g["country_long"].iloc[0],
            "lat": round(float(np.average(g["lat"], weights=w)), 3),
            "lon": round(float(np.average(g["lon"], weights=w)), 3),
            "region_code": cc, "continent_block": block.get(cc, "Other"),
        })
    pd.DataFrame(meta_rows).to_csv(SITE_DIR / "country_meta.csv", index=False)

    # global capacity-weighted CF (cross-country weight = IRENA, fallback WRI)
    glob = {}
    for tech in ["PV", "WIND"]:
        sub = cap_df[cap_df["tech"] == tech]
        wsum = sub["capacity_mw_used"].sum()
        g = np.zeros(n)
        nat_pivot = nat_df[nat_df["tech"] == tech].pivot_table(index="timestamp_utc", columns="iso2", values="cf")
        nat_pivot = nat_pivot.reindex(ts)
        for _, r in sub.iterrows():
            if r["iso2"] in nat_pivot.columns:
                g += (r["capacity_mw_used"] / wsum) * nat_pivot[r["iso2"]].to_numpy()
        glob[tech] = g
    pd.DataFrame({"timestamp_utc": ts, "pv_cf": glob["PV"], "wind_cf": glob["WIND"]}).to_csv(
        SITE_DIR / "global_hourly_cf.csv", index=False)

    print(f"national CF: {cap_df['iso2'].nunique()} countries "
          f"(PV={ (cap_df.tech=='PV').sum() }, WIND={ (cap_df.tech=='WIND').sum() })")
    print(f"global mean CF: PV={glob['PV'].mean():.3f}  WIND={glob['WIND'].mean():.3f}")
    print("outputs -> country_hourly_cf.parquet, country_capacity.csv, global_hourly_cf.csv, site_cf_summary.csv")


if __name__ == "__main__":
    main()
