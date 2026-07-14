#!/usr/bin/env python3
"""First-look probe: how much does allowing INTRA-COUNTRY dispatch lower the L0
(domestic) uncovered share, for large multi-timezone countries?

Three domestic supply bases, all built from the same station CF (UTC-aligned, so
each site keeps its true longitudinal solar phase), same demand, same NNLS:
  L0      current pipeline proxy = one capacity-weighted national PV + one national
          wind (2 columns; only PV/wind split is free, east-west mix frozen).
  L0_bin  route b = split the country's stations into 3 longitude bins, each bin a
          capacity-weighted PV + wind (<=6 columns; optimiser can favour the
          later-peaking western bin to cover the evening).
  L0_stn  upper bound = every station its own column (full domestic dispatch).

Reports the uncovered share for each and a rough implied domestic dispatch distance
(energy-weighted distance of the chosen bins from the national capacity centroid).
Demand = base human curve (the evening-peak case where follow-sun matters most; the
AI-mix curve would show a somewhat smaller but same-direction effect).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_geographic_portfolio_ai as base  # noqa: E402
import analyze_cfe_global_sites as G  # noqa: E402

SITE = G.SITE_DIR
BIG = ["CN", "US", "CA", "BR", "AU", "IN"]      # enough stations + real lon span
CTRL = ["DE", "GB", "JP"]                          # small-span controls
TARGETS = BIG + CTRL


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    site = pd.read_parquet(SITE / "site_hourly_cf.parquet")
    site["timestamp_utc"] = pd.to_datetime(site["timestamp_utc"])
    if site["timestamp_utc"].dt.tz is not None:
        site["timestamp_utc"] = site["timestamp_utc"].dt.tz_localize(None)
    idx = pd.DatetimeIndex(sorted(site["timestamp_utc"].unique()))
    cf = {sid: pd.Series(g["cf"].to_numpy(float),
                         index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
          for sid, g in site.groupby("site_id")}
    meta = pd.read_csv(SITE / "representative_sites.csv")
    meta = meta[meta.site_id.isin(cf)].copy()

    dem = pd.read_csv(G.DEMAND_GLOBAL)
    dem["timestamp_utc"] = pd.to_datetime(dem["timestamp_utc"])
    if dem["timestamp_utc"].dt.tz is not None:
        dem["timestamp_utc"] = dem["timestamp_utc"].dt.tz_localize(None)
    demand = {cc: pd.Series(g["human_volume_proxy"].to_numpy(float),
                            index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
              for cc, g in dem.groupby("cf_location")}

    def capw(rows):
        """capacity-weighted, mean-1 profile from [(site_id, cap), ...]"""
        num = sum(cf[s] * c for s, c in rows)
        den = sum(c for s, c in rows)
        if den <= 0:
            return None
        p = num / den
        m = float(p.mean())
        return None if m <= 0 else (p / m).to_numpy()

    def share(cols, d):
        cols = [c for c in cols if c is not None]
        if not cols:
            return np.nan, None
        S = np.stack(cols, axis=1)
        S_eq = base.equal_energy_align(S, float(np.mean(d.values)))
        w, sh, _ = base.portfolio_nnls(d.values, S_eq)
        return sh, w

    print(f"{'cty':>4} {'PV':>3} {'WD':>3} {'span°':>5} | {'L0':>6} {'L0_bin':>7} {'L0_stn':>7} | "
          f"{'Δbin':>6} {'Δstn':>6} | {'domkm':>6}")
    print("-" * 78)
    for c in TARGETS:
        g = meta[meta.iso2 == c]
        d = demand.get(c)
        if d is None or g.empty:
            print(f"{c:>4}  (no demand or no stations)")
            continue
        pv = [(r.site_id, r.wri_capacity_mw) for r in g[g.tech == "PV"].itertuples()]
        wd = [(r.site_id, r.wri_capacity_mw) for r in g[g.tech == "WIND"].itertuples()]
        # L0 national
        s_L0, _ = share([capw(pv), capw(wd)], d)
        # L0_bin: 3 longitude bins
        edges = np.quantile(g.lon.values, [1 / 3, 2 / 3])
        gb = g.assign(b=np.digitize(g.lon.values, edges))
        cols_bin, bin_info = [], []
        for b, sub in gb.groupby("b"):
            pvb = [(r.site_id, r.wri_capacity_mw) for r in sub[sub.tech == "PV"].itertuples()]
            wdb = [(r.site_id, r.wri_capacity_mw) for r in sub[sub.tech == "WIND"].itertuples()]
            cpv, cwd = capw(pvb), capw(wdb)
            for col, rows in ((cpv, pvb), (cwd, wdb)):
                if col is not None:
                    cols_bin.append(col)
                    w_cap = sum(x[1] for x in rows)
                    lon_c = sum(cf_lon(meta, x[0]) * x[1] for x in rows) / w_cap
                    lat_c = sum(cf_lat(meta, x[0]) * x[1] for x in rows) / w_cap
                    bin_info.append((lon_c, lat_c))
        s_bin, w_bin = share(cols_bin, d)
        # L0_stn upper bound
        cols_stn = [(cf[s] / cf[s].mean()).to_numpy() for s in g.site_id if cf[s].mean() > 0]
        s_stn, _ = share(cols_stn, d)
        # implied domestic distance: NNLS-weighted distance of active bins -> capacity centroid
        cen_lon = float((g.lon * g.wri_capacity_mw).sum() / g.wri_capacity_mw.sum())
        cen_lat = float((g.lat * g.wri_capacity_mw).sum() / g.wri_capacity_mw.sum())
        domkm = np.nan
        if w_bin is not None and len(bin_info) == len(w_bin) and w_bin.sum() > 0:
            dists = np.array([haversine_km(lo, la, cen_lon, cen_lat) for lo, la in bin_info])
            domkm = float((dists * w_bin).sum() / w_bin.sum())
        span = g.lon.max() - g.lon.min()
        print(f"{c:>4} {len(pv):>3} {len(wd):>3} {span:>5.0f} | "
              f"{s_L0*100:>5.1f}% {s_bin*100:>6.1f}% {s_stn*100:>6.1f}% | "
              f"{(s_L0-s_bin)*100:>5.1f} {(s_L0-s_stn)*100:>5.1f} | {domkm:>6.0f}")


def cf_lon(meta, sid):
    return float(meta.loc[meta.site_id == sid, "lon"].iloc[0])


def cf_lat(meta, sid):
    return float(meta.loc[meta.site_id == sid, "lat"].iloc[0])


if __name__ == "__main__":
    main()
