#!/usr/bin/env python3
"""Full integrated computation of the new plan (NOTES_transmission_cost_and_scope_design.md):

  * STATION-LEVEL dispatch (individual WRI sites, UTC-aligned, NNLS).
  * SCOPE bounded by distance from the demand node: L0 (home only) -> T1 (<=D1)
    -> T2 (<=D2) -> T3 (same continent, no global). Nested.
  * GEOGRAPHICALLY DIFFERENTIATED transmission cost: per-country WACC r_c (into the
    CRF of gen/tx/storage), domestic T&D loss lambda_c, sunk-grid credit kappa_c by
    grid quality, plus the distance-priced HVDC. Domestic lines get kappa_c; cross-
    border lines pay full.

Generation = real (ATB capex x CRF(r_c) x nameplate from the chosen mix).
Transmission = real (HVDC line+converter, P95 flow, CRF(r_c), kappa/lambda).
Storage = firming proxy calibrated to ~47 USD/MWh at L0/r=0.07, x CRF(r_c) scaling.

Output: data_globalsites/r4_geodiff_scope.csv + console summary. Demand = base human
curve (structural result; AI-mix would be a touch smaller, same direction).
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
OUT = SITE.parent.parent / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"
YEAR = 2030
MEANMW = 100.0
E_ANN = MEANMW * 8760.0
D1, D2 = 1500.0, 3000.0          # T1, T2 distance caps (km); T3 = same continent
HVDC_MAX = 11700.0
F_T = 0.4                        # transmission share of WB T&D loss
ANC = 4.0
R_REF = 0.07
K_FIRM = 47.0 / 0.15            # firming USD/MWh per unit uncovered share, anchored

# ---- WACC: IRENA-2023-grounded regional proxy + high-income override (to be
# refined with the per-country Sci Data 2025 dataset). Documented, swept vs uniform.
REGION_WACC = {"EU": 0.052, "NA": 0.062, "AS": 0.078, "SA": 0.090,
               "OC": 0.050, "AF": 0.110, "AN": 0.070}
LOW_RISK = {"US", "CA", "DE", "FR", "GB", "NL", "SE", "NO", "DK", "FI", "CH",
            "AT", "BE", "IE", "IT", "ES", "PT", "JP", "KR", "AU", "NZ", "SG",
            "IL", "CN"}                              # ~0.045
GCC = {"AE", "SA", "QA", "KW", "BH", "OM"}           # ~0.060


def wacc(iso2, region, access):
    if iso2 in LOW_RISK:
        base_r = 0.045
    elif iso2 in GCC:
        base_r = 0.060
    else:
        base_r = REGION_WACC.get(region, 0.085)
    # light data nudge: poor electricity access -> higher country risk
    if access == access and access < 90:
        base_r += (90 - access) / 100 * 0.05
    return float(np.clip(base_r, 0.04, 0.15))


def kappa(access):
    if access != access:
        return 0.5
    if access >= 99:
        return 0.85
    if access >= 95:
        return 0.60
    if access >= 85:
        return 0.45
    return 0.25


def hav(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_all():
    site = pd.read_parquet(SITE / "site_hourly_cf.parquet")
    site["timestamp_utc"] = pd.to_datetime(site["timestamp_utc"])
    if site["timestamp_utc"].dt.tz is not None:
        site["timestamp_utc"] = site["timestamp_utc"].dt.tz_localize(None)
    idx = pd.DatetimeIndex(sorted(site["timestamp_utc"].unique()))
    cf, prof, cfm, smeta = {}, {}, {}, {}
    meta = pd.read_csv(SITE / "representative_sites.csv")
    meta = meta[meta.site_id.isin(site.site_id.unique())]
    raw = {sid: pd.Series(g["cf"].to_numpy(float),
                          index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
           for sid, g in site.groupby("site_id")}
    for r in meta.itertuples():
        s = raw.get(r.site_id)
        if s is None:
            continue
        m = float(s.mean())
        if m <= 0:
            continue
        cfm[r.site_id] = m
        prof[r.site_id] = (s / m).to_numpy()          # mean-1
        smeta[r.site_id] = {"iso2": r.iso2, "tech": r.tech, "lon": r.lon,
                            "lat": r.lat, "cap": r.wri_capacity_mw}
    cm = pd.read_csv(SITE / "country_meta.csv", keep_default_na=False, na_values=[""])
    cen = {r.iso2: (float(r.lon), float(r.lat), r.region_code, r.continent_block)
           for r in cm.itertuples()}
    dem = pd.read_csv(G.DEMAND_GLOBAL)
    dem["timestamp_utc"] = pd.to_datetime(dem["timestamp_utc"])
    if dem["timestamp_utc"].dt.tz is not None:
        dem["timestamp_utc"] = dem["timestamp_utc"].dt.tz_localize(None)
    demand = {cc: (pd.Series(g["human_volume_proxy"].to_numpy(float),
                             index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()).to_numpy()
              for cc, g in dem.groupby("cf_location")}
    gq = pd.read_csv(OUT / "country_grid_quality.csv")
    grid = {r.iso2: (r.td_loss_pct, r.access_pct) for r in gq.itertuples()}
    return prof, cfm, smeta, cen, demand, grid


def cost_for(scope_ids, smeta, prof, cfm, c, cen, r_c, kap, lam):
    """Run station NNLS over scope_ids and price the full system (USD/MWh of demand)."""
    clon, clat = cen[c][0], cen[c][1]
    cols, ids = [], []
    for sid in scope_ids:
        cols.append(prof[sid])
        ids.append(sid)
    S = np.stack(cols, axis=1)
    # distance only enters cost (energy-align cancels a scalar eta); fit on mean-1 cols
    dvec = (np.asarray(demand_g[c]) / np.mean(demand_g[c])) * MEANMW
    S_eq = base.equal_energy_align(S, float(np.mean(dvec)))
    w, share, fit = base.portfolio_nnls(dvec, S_eq)     # sum(w)=1 after rescale
    # generation nameplate from the mix (pre-loss): np_j = w_j*MEANMW/(eta_j*cf_j)
    np_pv = np_wd = 0.0
    tx_dom = tx_for = 0.0
    deliv = w * MEANMW                                   # delivered mean power per unit
    for j, sid in enumerate(ids):
        m = smeta[sid]
        dist = hav(m["lon"], m["lat"], clon, clat)
        eta = P.hvdc_efficiency(dist) if dist > 1.0 else 1.0
        npw = deliv[j] / (eta * cfm[sid])                # nameplate MW
        if m["tech"] == "PV":
            np_pv += npw
        else:
            np_wd += npw
        if dist > 1.0:
            sent = (w[j] * prof[sid] * MEANMW) / eta     # hourly MW on the line
            p95 = float(np.percentile(sent, 95))
            mwkm = p95 * P.TX_TORTUOSITY * dist
            ann = P.tx_annual_cost(mwkm, p95, "central", r_c)
            if m["iso2"] == c:
                tx_dom += ann * (1.0 - kap)
            else:
                tx_for += ann
    gen_ann = P.gen_annual_cost(np_pv, np_wd, YEAR, "central", r_c)
    gen_lcoe = gen_ann / E_ANN
    tx_lcoe = (tx_dom + tx_for) / E_ANN / (1.0 - F_T * lam)
    store_lcoe = K_FIRM * share * (P.crf(15.0, r_c) / P.crf(15.0, R_REF))
    total = gen_lcoe + tx_lcoe + store_lcoe + ANC
    return dict(share=share, gen=gen_lcoe, tx=tx_lcoe, store=store_lcoe,
                total=total, n=len(ids))


demand_g = {}


def main():
    global demand_g
    prof, cfm, smeta, cen, demand, grid = load_all()
    demand_g = demand
    by_country = {}
    for sid, m in smeta.items():
        by_country.setdefault(m["iso2"], []).append(sid)
    # precompute station country centroids list for distance scoping
    countries = [c for c in demand if c in cen and c in by_country]
    rows = []
    for c in sorted(countries):
        clon, clat = cen[c][0], cen[c][1]
        cont = cen[c][3]
        loss, access = grid.get(c, (np.nan, np.nan))
        lam = (loss / 100.0) if loss == loss else 0.09
        r_c = wacc(c, cen[c][2], access)
        kap = kappa(access)
        home = by_country[c]
        # foreign station distances
        fdist = {}
        for sid, m in smeta.items():
            if m["iso2"] == c:
                continue
            fdist[sid] = hav(m["lon"], m["lat"], clon, clat)
        scopes = {}
        scopes["L0"] = list(home)
        scopes["T1"] = home + [s for s, d in fdist.items() if d <= D1]
        scopes["T2"] = home + [s for s, d in fdist.items() if d <= D2]
        t3 = home + [s for s, d in fdist.items()
                     if d <= HVDC_MAX and cen.get(smeta[s]["iso2"], (0, 0, 0, ""))[3] == cont]
        scopes["T3"] = t3
        res = {}
        for sc, ids in scopes.items():
            try:
                res[sc] = cost_for(ids, smeta, prof, cfm, c, cen, r_c, kap, lam)
            except Exception as e:
                res[sc] = None
        if any(v is None for v in res.values()):
            continue
        opt = min(res, key=lambda k: res[k]["total"])
        rows.append({
            "iso2": c, "region": cen[c][2], "continent": cont,
            "wacc": round(r_c, 3), "access": access, "td_loss": loss,
            "kappa": round(kap, 2),
            **{f"unc_{k}": round(res[k]["share"] * 100, 1) for k in scopes},
            **{f"tot_{k}": round(res[k]["total"], 0) for k in scopes},
            **{f"tx_{k}": round(res[k]["tx"], 0) for k in scopes},
            "opt_scope": opt, "opt_total": round(res[opt]["total"], 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_geodiff_scope.csv", index=False, encoding="utf-8")
    print(f"computed {len(df)} countries -> r4_geodiff_scope.csv\n")
    print("=== optimal scope distribution ===")
    print(df.opt_scope.value_counts().reindex(["L0", "T1", "T2", "T3"]).fillna(0).astype(int).to_string())
    print("\n=== mean full-system LCOE by scope (USD/MWh) ===")
    for k in ["L0", "T1", "T2", "T3"]:
        print(f"  {k}: mean {df[f'tot_{k}'].mean():.0f}  median {df[f'tot_{k}'].median():.0f}")
    print("\n=== does grid quality predict optimal reach? (mean WACC / access by opt scope) ===")
    print(df.groupby("opt_scope")[["wacc", "access", "td_loss"]].mean().round(3).to_string())
    print("\n=== sample big countries ===")
    show = ["CN", "US", "DE", "IN", "BR", "NG", "RU", "AU", "JP", "ZA"]
    cols = ["iso2", "wacc", "access", "kappa", "unc_L0", "unc_T3",
            "tot_L0", "tot_T1", "tot_T2", "tot_T3", "opt_scope"]
    print(df[df.iso2.isin(show)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
