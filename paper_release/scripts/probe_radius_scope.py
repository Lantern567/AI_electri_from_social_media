#!/usr/bin/env python3
"""Re-classify dispatch scope by TRANSMISSION RADIUS instead of political buckets.

For a home country, sweep a dispatch radius R (km from the home load centroid). At
each R the supply pool = the home country's own STATIONS within R + foreign national
profiles within R, every unit derated by the distance-dependent HVDC efficiency. We
price the transmission of EVERY unit by its own distance (so intra-country lines are
included), size lines to the P95 of hourly flow exactly like the cost pipeline, and
read off the full-system cost as a function of R.

Question: does cost(R) have an INTERIOR minimum (an optimal radius cheaper than both
the national-average L0 and the global L3)? If so, the bucket scopes mis-classify and
a distance axis is the right framework.

Demand = base human curve; UTC-aligned station CF; central HVDC unit costs.
gen+ancillary held ~constant (the tx-vs-firming trade-off is what moves with R).
firming calibrated so the national-average baseline ~= 47 USD/MWh (the paper's L0).
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
import fullsystem_cost_params as P  # noqa: E402

SITE = G.SITE_DIR
GEN_LCOE = 100.0          # USD/MWh, held ~constant across R (mix shifts second-order)
ANC_LCOE = 4.0
E_ANN = 100.0 * 8760.0    # MWh/yr for a 100 MW-mean demand unit
RADII = [300, 600, 1000, 1500, 2500, 4000, 6000, 9000, 13000, 20000]


def hav(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
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
    smeta = pd.read_csv(SITE / "representative_sites.csv")
    smeta = smeta[smeta.site_id.isin(cf)].copy()
    cmeta = pd.read_csv(SITE / "country_meta.csv", keep_default_na=False, na_values=[""])
    cen = {r.iso2: (float(r.lon), float(r.lat)) for r in cmeta.itertuples()}

    dem = pd.read_csv(G.DEMAND_GLOBAL)
    dem["timestamp_utc"] = pd.to_datetime(dem["timestamp_utc"])
    if dem["timestamp_utc"].dt.tz is not None:
        dem["timestamp_utc"] = dem["timestamp_utc"].dt.tz_localize(None)
    demand = {cc: pd.Series(g["human_volume_proxy"].to_numpy(float),
                            index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
              for cc, g in dem.groupby("cf_location")}

    def capw(rows):
        num = sum(cf[s] * c for s, c in rows)
        den = sum(c for s, c in rows)
        if den <= 0:
            return None
        p = num / den
        m = float(p.mean())
        return None if m <= 0 else (p / m).to_numpy()

    # foreign national profiles (capacity-weighted), at country centroid
    nat = {}
    for c, g in smeta.groupby("iso2"):
        pv = [(r.site_id, r.wri_capacity_mw) for r in g[g.tech == "PV"].itertuples()]
        wd = [(r.site_id, r.wri_capacity_mw) for r in g[g.tech == "WIND"].itertuples()]
        cols = [x for x in (capw(pv), capw(wd)) if x is not None]
        if cols and c in cen:
            nat[c] = cols

    def nnls_cost(units, d):
        """units: list of (profile mean-1, distance_km). Returns dict of metrics."""
        cols, dists = [], []
        for prof, dist in units:
            eta = P.hvdc_efficiency(dist) if dist > 1.0 else 1.0
            cols.append(prof * eta)
            dists.append(dist)
        S = np.stack(cols, axis=1)
        S_eq = base.equal_energy_align(S, float(np.mean(d.values)))
        w, share, _ = base.portfolio_nnls(d.values, S_eq)
        # transmission sizing: P95 of each unit's hourly SENT flow (pre-loss), routed
        scale = S_eq / np.where(np.array([P.hvdc_efficiency(x) if x > 1 else 1.0 for x in dists]), 1, 1)
        mwkm = conv_mw = 0.0
        dmean_num = wsum = 0.0
        for j, dist in enumerate(dists):
            if dist <= 1.0:
                continue
            eta = P.hvdc_efficiency(dist)
            sent = (w[j] * (S_eq[:, j] / eta)) * 100.0          # MW sent on the line, 100 MW-mean demand
            p95 = float(np.percentile(sent, 95))
            route = P.TX_TORTUOSITY * dist
            mwkm += p95 * route
            conv_mw += p95
            ej = float(np.mean(w[j] * S_eq[:, j]))               # delivered energy weight
            dmean_num += ej * dist
            wsum += ej
        tx_annual = P.tx_annual_cost(mwkm, conv_mw, "central")
        tx_lcoe = tx_annual / E_ANN
        mean_km = dmean_num / wsum if wsum > 1e-9 else 0.0
        return {"share": share, "tx": tx_lcoe, "mean_km": mean_km, "n": len(cols)}

    for c in ["CN", "US"]:
        d = demand[c]
        hlon, hlat = cen[c]
        # build all candidate units with distance from home centroid
        home_units = []   # home stations
        for r in smeta[smeta.iso2 == c].itertuples():
            if cf[r.site_id].mean() > 0:
                dist = hav(r.lon, r.lat, hlon, hlat)
                home_units.append(((cf[r.site_id] / cf[r.site_id].mean()).to_numpy(), dist))
        for_units = []    # foreign nationals
        for fc, cols in nat.items():
            if fc == c:
                continue
            dist = hav(cen[fc][0], cen[fc][1], hlon, hlat)
            for col in cols:
                for_units.append((col, dist))
        all_units = home_units + for_units
        # reference: current national-average L0 (no tx) and bucket L3 (global, with tx)
        pvh = [(r.site_id, r.wri_capacity_mw) for r in smeta[(smeta.iso2 == c) & (smeta.tech == "PV")].itertuples()]
        wdh = [(r.site_id, r.wri_capacity_mw) for r in smeta[(smeta.iso2 == c) & (smeta.tech == "WIND")].itertuples()]
        L0nat = nnls_cost([(x, 0.0) for x in (capw(pvh), capw(wdh)) if x is not None], d)
        K_firm = 47.0 / L0nat["share"]
        print(f"\n===== {c}  (home stations={len(home_units)}, firming calib K={K_firm:.0f} $/MWh·share) =====")
        print(f"{'R(km)':>7} {'units':>5} {'uncov':>6} {'meanKm':>7} {'tx':>6} {'firm':>6} {'TOTAL':>7}")
        # L0 national reference
        f0 = K_firm * L0nat["share"]
        print(f"{'L0nat':>7} {L0nat['n']:>5} {L0nat['share']*100:>5.1f}% {0:>7.0f} "
              f"{0:>6.1f} {f0:>6.0f} {GEN_LCOE+0+f0+ANC_LCOE:>7.0f}")
        for R in RADII:
            units = [(p, dd) for (p, dd) in all_units if dd <= R]
            m = nnls_cost(units, d)
            firm = K_firm * m["share"]
            total = GEN_LCOE + m["tx"] + firm + ANC_LCOE
            print(f"{R:>7} {m['n']:>5} {m['share']*100:>5.1f}% {m['mean_km']:>7.0f} "
                  f"{m['tx']:>6.0f} {firm:>6.0f} {total:>7.0f}")


if __name__ == "__main__":
    main()
