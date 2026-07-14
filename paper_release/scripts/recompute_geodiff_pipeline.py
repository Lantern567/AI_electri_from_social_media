#!/usr/bin/env python3
"""Full-chain recompute R1->R2->R3 under the new framework
(NOTES_transmission_cost_and_scope_design.md), mirroring decompose_fullsystem_portfolio
but with: STATION-LEVEL dispatch, distance-bounded scopes (L0/T1/T2/T3, no global,
continental ceiling), P_mix-2030 demand, the FULL duration-aware storage model
(reused gap_geometry + storage cost), overbuild co-optimisation, and GEOGRAPHICALLY
DIFFERENTIATED transmission cost (per-country WACC r_c, T&D loss lambda_c, sunk-grid
credit kappa_c).

R1 = the L0 (national) station-level match under P_mix -> per-country uncovered share.
R2 = the four scopes -> uncovered share + portfolio reach by scope.
R3 = full-system economics (gen + geo-diff tx + full storage + ancillary) -> optimal scope.

Outputs: data_globalsites/r4_geodiff_fullchain.csv  (+ console summary).
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
from decompose_fullsystem_portfolio import gap_geometry  # noqa: E402

SITE = G.SITE_DIR
OUT = SITE.parent.parent / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"
YEAR = 2030
MEANMW = 100.0
E_ANN = MEANMW * 8760.0
D1, D2 = 1500.0, 3000.0
HVDC_MAX = 11700.0
F_T = 0.4
OB_GRID = np.round(np.concatenate([np.arange(1.0, 1.81, 0.1), [2.0, 2.3, 2.6, 3.0]]), 2)
WORKLOADS = ["W0", "W1", "W2", "W3"]
RTE_EFF = 0.85
LAM_INF, LAM_TRAIN = 0.308, 0.132            # P_mix 2030
R_REF = 0.07

REGION_WACC = {"EU": 0.052, "NA": 0.062, "AS": 0.078, "SA": 0.090,
               "OC": 0.050, "AF": 0.110, "AN": 0.070}
LOW_RISK = {"US", "CA", "DE", "FR", "GB", "NL", "SE", "NO", "DK", "FI", "CH",
            "AT", "BE", "IE", "IT", "ES", "PT", "JP", "KR", "AU", "NZ", "SG", "IL", "CN"}
GCC = {"AE", "SA", "QA", "KW", "BH", "OM"}


STEFFEN_WACC = {}   # iso2 -> real WACC, loaded from Steffen et al. 2025 (figshare 28588943)


def wacc(iso2, region, access):
    if iso2 in STEFFEN_WACC:                       # per-country real WACC where available
        return float(STEFFEN_WACC[iso2])
    r = 0.045 if iso2 in LOW_RISK else (0.060 if iso2 in GCC else REGION_WACC.get(region, 0.085))
    if access == access and access < 90:           # proxy fallback for the ~40 uncovered
        r += (90 - access) / 100 * 0.05
    return float(np.clip(r, 0.04, 0.15))


def kappa(access):
    if access != access:
        return 0.5
    return 0.85 if access >= 99 else 0.60 if access >= 95 else 0.45 if access >= 85 else 0.25


def hav(lo1, la1, lo2, la2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi, dlam = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def storage_cost_r(p_peak, e_max, annual, year, r, case="central"):
    if annual <= 0 or p_peak <= 0:
        return 0.0
    eL, eD = P._e_li(case, year), P._e_ldes(case, year)
    cap_li = (P.P_LI * p_peak + eL * e_max) * 1000.0
    cap_ld = (P.P_LDES * p_peak + eD * e_max) * 1000.0
    lcos_li = (P.crf(P.LIFE_LI, r) + P.STORE_FOM) * cap_li / (annual * P.RTE_LI)
    lcos_ld = (P.crf(P.LIFE_LDES, r) + P.STORE_FOM) * cap_ld / (annual * P.RTE_LDES)
    return annual * min(lcos_li, lcos_ld)


def load():
    site = pd.read_parquet(SITE / "site_hourly_cf.parquet")
    site["timestamp_utc"] = pd.to_datetime(site["timestamp_utc"])
    if site["timestamp_utc"].dt.tz is not None:
        site["timestamp_utc"] = site["timestamp_utc"].dt.tz_localize(None)
    idx = pd.DatetimeIndex(sorted(site["timestamp_utc"].unique()))
    raw = {sid: pd.Series(g["cf"].to_numpy(float),
                          index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
           for sid, g in site.groupby("site_id")}
    meta = pd.read_csv(SITE / "representative_sites.csv")
    prof, cfm, smeta = {}, {}, {}
    for r in meta.itertuples():
        s = raw.get(r.site_id)
        if s is None or float(s.mean()) <= 0:
            continue
        cfm[r.site_id] = float(s.mean())
        prof[r.site_id] = (s / s.mean()).to_numpy()
        smeta[r.site_id] = {"iso2": r.iso2, "tech": r.tech, "lon": r.lon, "lat": r.lat}
    cm = pd.read_csv(SITE / "country_meta.csv", keep_default_na=False, na_values=[""])
    cen = {r.iso2: (float(r.lon), float(r.lat), r.region_code, r.continent_block) for r in cm.itertuples()}
    dem = pd.read_csv(G.DEMAND_GLOBAL)
    dem["timestamp_utc"] = pd.to_datetime(dem["timestamp_utc"])
    if dem["timestamp_utc"].dt.tz is not None:
        dem["timestamp_utc"] = dem["timestamp_utc"].dt.tz_localize(None)
    demand = {cc: pd.Series(g["human_volume_proxy"].to_numpy(float),
                            index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx).ffill().bfill()
              for cc, g in dem.groupby("cf_location")}
    gq = pd.read_csv(OUT / "country_grid_quality.csv")
    grid = {r.iso2: (r.td_loss_pct, r.access_pct) for r in gq.itertuples()}
    wf = pd.read_csv(OUT / "country_wacc.csv")
    STEFFEN_WACC.clear()
    STEFFEN_WACC.update({r.iso2: float(r.wacc) for r in wf.itertuples()})
    # production inputs for workload migration (receiver supply/demand + RTT)
    _, demand_prod, demand_meta, _, smeta_prod = G.load_inputs_expanded()
    rtt = G.build_rtt_distance(demand_meta, smeta_prod)
    utc_hour = idx.hour.to_numpy()
    return prof, cfm, smeta, cen, demand, grid, utc_hour, idx, demand_prod, demand_meta, rtt


def evaluate(scope_ids, smeta, prof, cfm, c, cen, dmw, r_c, kap, lam,
             workload, idx, demand_prod, demand_meta, rtt):
    """Mirror decompose(): station NNLS (+ workload migration) + OB co-opt + geo-diff
    cost. Returns metrics."""
    cl, ca = cen[c][0], cen[c][1]
    cols = np.stack([prof[s] for s in scope_ids], axis=1)
    S1 = base.equal_energy_align(cols, float(dmw.mean()))
    w, share, fit = base.portfolio_nnls(dmw, S1)
    if base.WORKLOAD_SCENARIOS.get(workload):           # W1-W3: migrate deferrable load
        resid = np.maximum(dmw - fit, 0.0)
        d_eff, _ = G.migrate_residual(c, workload, demand_prod, demand_meta, rtt,
                                      pd.Series(dmw, index=idx), resid)
        dmw = d_eff.to_numpy()
        S1 = base.equal_energy_align(cols, float(dmw.mean()))
        w, share, fit = base.portfolio_nnls(dmw, S1)
    scale = MEANMW / float(dmw.mean())
    d_mw = dmw * scale
    fit_mw = fit * scale
    # gen nameplate + tx corridors
    np_pv = np_wd = gen_busbar = 0.0
    dist = {s: hav(smeta[s]["lon"], smeta[s]["lat"], cl, ca) for s in scope_ids}
    by_src = {}
    for j, s in enumerate(scope_ids):
        wj = float(w[j])
        if wj <= 1e-6:
            continue
        m = smeta[s]
        eta = P.hvdc_efficiency(dist[s]) if dist[s] > 1 else 1.0
        nm = wj * 100.0 / (cfm[s] * eta)
        if m["tech"] == "PV":
            np_pv += nm
        else:
            np_wd += nm
        gen_busbar += nm * cfm[s] * 8760.0
        by_src.setdefault(m["iso2"], []).append(j)
    # transmission: per-source-country for foreign (de-coincidence); per-station for home
    tx_dom = tx_for = 0.0
    for src, cols_j in by_src.items():
        if src == c:                                  # home: each station its own short line
            for j in cols_j:
                s = scope_ids[j]
                if dist[s] <= 1:
                    continue
                eta = P.hvdc_efficiency(dist[s])
                sent = (w[j] * prof[s] * 100.0) / eta
                p95 = float(np.percentile(sent, 95))
                ann = P.tx_annual_cost(p95 * P.TX_TORTUOSITY * dist[s], p95, "central", r_c)
                tx_dom += ann * (1.0 - kap)
        else:                                         # foreign: aggregate corridor
            gc = np.mean([dist[scope_ids[j]] for j in cols_j])
            eta = P.hvdc_efficiency(gc)
            delivered = (S1[:, cols_j] @ np.array([w[j] for j in cols_j])) * scale
            p95 = float(np.percentile(delivered / eta, 95))
            tx_for += P.tx_annual_cost(p95 * P.TX_TORTUOSITY * gc, p95, "central", r_c)
    tx_lcoe = (tx_dom + tx_for) / E_ANN / (1.0 - F_T * lam)
    # overbuild co-optimisation (gen vs storage), energy-closure feasible
    best = None
    for ob in OB_GRID:
        net = d_mw - ob * fit_mw
        annual, e_max, p_peak = gap_geometry(net)
        excess = float(np.maximum(ob * fit_mw - d_mw, 0.0).sum())
        if annual > RTE_EFF * excess:
            continue
        gen_c = P.gen_annual_cost(np_pv * ob, np_wd * ob, YEAR, "central", r_c)
        sto_c = storage_cost_r(p_peak, e_max, annual, YEAR, r_c)
        anc_c = P.ancillary_annual_cost(gen_busbar * ob, "central")
        tot = gen_c + sto_c + anc_c
        if best is None or tot < best[0]:
            best = (tot, ob, gen_c, sto_c, anc_c)
    if best is None:
        ob = OB_GRID[-1]
        net = d_mw - ob * fit_mw
        annual, e_max, p_peak = gap_geometry(net)
        gen_c = P.gen_annual_cost(np_pv * ob, np_wd * ob, YEAR, "central", r_c)
        sto_c = storage_cost_r(p_peak, e_max, annual, YEAR, r_c)
        anc_c = P.ancillary_annual_cost(gen_busbar * ob, "central")
        best = (gen_c + sto_c + anc_c, ob, gen_c, sto_c, anc_c)
    _, ob_star, gen_c, sto_c, anc_c = best
    return dict(share=share, ob=ob_star, gen=gen_c / E_ANN, tx=tx_lcoe,
                store=sto_c / E_ANN, anc=anc_c / E_ANN,
                total=(gen_c + sto_c + anc_c) / E_ANN + tx_lcoe, n=len(scope_ids))


def main():
    prof, cfm, smeta, cen, demand, grid, utc_hour, idx, demand_prod, demand_meta, rtt = load()
    by_country = {}
    for s, m in smeta.items():
        by_country.setdefault(m["iso2"], []).append(s)
    countries = [c for c in demand if c in cen and c in by_country]
    SC = ["L0", "T1", "T2", "T3"]
    rows = []
    for c in sorted(countries):
        cl, ca, region, cont = cen[c]
        loss, access = grid.get(c, (np.nan, np.nan))
        lam = (loss / 100.0) if loss == loss else 0.09
        r_c, kap = wacc(c, region, access), kappa(access)
        tz = int(round(cl / 15.0))
        lh = (utc_hour + tz) % 24
        d_pmix = base.operator_mix(demand[c].to_numpy(), lh, LAM_INF, LAM_TRAIN)
        home = by_country[c]
        fdist = {s: hav(smeta[s]["lon"], smeta[s]["lat"], cl, ca)
                 for s, m in smeta.items() if m["iso2"] != c}
        scopes = {"L0": list(home),
                  "T1": home + [s for s, d in fdist.items() if d <= D1],
                  "T2": home + [s for s, d in fdist.items() if d <= D2],
                  "T3": home + [s for s, d in fdist.items()
                                if d <= HVDC_MAX and cen.get(smeta[s]["iso2"], (0, 0, 0, ""))[3] == cont]}
        res, ok = {}, True
        for sc in SC:
            for wl in WORKLOADS:
                try:
                    res[(sc, wl)] = evaluate(scopes[sc], smeta, prof, cfm, c, cen, d_pmix,
                                             r_c, kap, lam, wl, idx, demand_prod, demand_meta, rtt)
                except Exception:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        opt = min(res, key=lambda k: res[k]["total"])
        row = {"iso2": c, "region": region, "continent": cont, "wacc": round(r_c, 3),
               "access": access, "kappa": round(kap, 2),
               "opt_scope": opt[0], "opt_workload": opt[1], "opt_total": round(res[opt]["total"], 0)}
        for sc in SC:
            for wl in WORKLOADS:
                row[f"unc_{sc}_{wl}"] = round(res[(sc, wl)]["share"] * 100, 1)
                row[f"tot_{sc}_{wl}"] = round(res[(sc, wl)]["total"], 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_geodiff_16scen.csv", index=False, encoding="utf-8")
    # W0 slice for the existing figure (drop-in columns)
    w0 = df[["iso2", "region", "continent", "wacc", "access", "kappa"]].copy()
    for sc in SC:
        w0[f"unc_{sc}"] = df[f"unc_{sc}_W0"]
        w0[f"tot_{sc}"] = df[f"tot_{sc}_W0"]
    w0["opt_scope"] = df[[f"tot_{sc}_W0" for sc in SC]].idxmin(axis=1).str.replace("tot_", "").str.replace("_W0", "")
    w0.to_csv(OUT / "r4_geodiff_fullchain.csv", index=False, encoding="utf-8")

    print(f"FULL 16-scenario recompute (scope x workload): {len(df)} countries\n")
    print("=== R1: national L0_W0 uncovered (P_mix) ===")
    print(f"  mean {df.unc_L0_W0.mean():.1f}%  median {df.unc_L0_W0.median():.1f}%")
    print("\n=== jointly-optimal (scope, workload) distribution ===")
    print(df.groupby(["opt_scope", "opt_workload"]).size().to_string())
    print("\n=== optimal SCOPE distribution (over all 16) ===")
    print(df.opt_scope.value_counts().reindex(SC).fillna(0).astype(int).to_string())
    print("\n=== workload-migration lever at L0: mean cost / uncovered, W0->W3 ===")
    for wl in WORKLOADS:
        print(f"  {wl}: cost {df[f'tot_L0_{wl}'].mean():.0f}  uncovered {df[f'unc_L0_{wl}'].mean():.1f}%")
    print("\n=== mean full-system LCOE: scope (rows) x workload (cols) USD/MWh ===")
    print("        " + "".join(f"{wl:>8}" for wl in WORKLOADS))
    for sc in SC:
        print(f"   {sc}: " + "".join(f"{df[f'tot_{sc}_{wl}'].mean():8.0f}" for wl in WORKLOADS))
    print("\n=== sample ===")
    show = ["CN", "US", "JP", "IN", "NG", "BR", "DE", "RU"]
    cols = ["iso2", "unc_L0_W0", "unc_L0_W3", "tot_L0_W0", "tot_L0_W3", "opt_scope", "opt_workload"]
    print(df[df.iso2.isin(show)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
