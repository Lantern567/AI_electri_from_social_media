#!/usr/bin/env python3
"""STEP 1 of the full-system cost extension of Result 3 (corrected per the
fullsystem-cost-params-and-audit workflow).

Re-solves the 104-country expanded portfolio (analyze_cfe_global_sites) and, per
(country, power-scope L, workload-scope W, year, AI-operator), records a compact
PHYSICAL decomposition plus a CO-OPTIMISED VRE overbuild factor. The cost layer
(layer_fullsystem_cost.py) prices this cache, so cost-parameter sweeps never
re-run the NNLS.

Audit-driven corrections vs the first draft
-------------------------------------------
1. VRE OVERBUILD (blocker). Equal-energy (sum w = 1) builds generation = ~1.05x
   E_d, which is physically impossible for 24/7-CFE with storage round-trip < 1.
   We co-optimise an overbuild multiplier OB >= 1 on the chosen portfolio against
   the gen-vs-storage trade-off (at reference central prices for that year), and
   record OB* plus the resulting gap geometry. OB=1.0 geometry is also kept as the
   energy-balanced LOWER BOUND.
2. TRANSMISSION sized to P95 of the hourly delivered import flow (not nameplate),
   with the nameplate sizing kept as the conservative HIGH bound. Corridors are
   aggregated per SOURCE COUNTRY (PV+wind share one link / converter pair). Route
   length = 1.3 x great-circle (tortuosity). Distance-dependent one-way HVDC loss.
3. STORAGE computed on each scenario's post-pooling residual (scope-consistent),
   sized to the P95 gap (same as recompute_fig3c_storage.py).

Output: data_globalsites/fullsystem_portfolio_decomposition.csv
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("EXPANDED_DEMAND", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_global_sites as G          # noqa: E402
import fullsystem_cost_params as P            # noqa: E402
base = G.base

OUT = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"

YEARS = list(range(2025, 2051))   # every year 2025-2050 (dense full-system trajectory for panel d)
L_SCOPES = ["L0", "L1", "L2", "L3"]
W_SCOPES = ["W0", "W1", "W2", "W3"]
ALL_LW = [f"{p}_{w}" for p in L_SCOPES for w in W_SCOPES]
CORNERS = ["L0_W0", "L3_W0", "L0_W3", "L3_W3"]
MAIN_OP = "P_mix"
OTHER_OPS = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst"]
OB_GRID = np.round(np.arange(1.0, 3.001, 0.1), 2)   # overbuild search grid


def _ai_share(table, year):
    """AI inference/training share at an arbitrary year (linear interp over the
    literature anchor years, matching recompute_fig3c_storage.interp_dict)."""
    xs = sorted(table)
    return float(np.interp(year, xs, [table[x] for x in xs]))


def perturb(op, d_base, lh, lam_inf, lam_train):
    if op == "P0_baseline":
        return d_base
    if op == "P1_flatten":
        return base.operator_flatten(d_base, lam_inf, lam_train)
    if op == "P2_sharpen":
        return base.operator_sharpen(d_base, lh, lam_inf, lam_train)
    if op == "P2_emp":
        return base.operator_sharpen_emp(d_base, lh, lam_inf, lam_train)
    if op == "P3_burst":
        return base.operator_burst(d_base, lam_inf, lam_train)
    return base.operator_mix(d_base, lh, lam_inf, lam_train)


def cf_of(supply, src, tech):
    return float(supply[src]["pv" if tech == "PV" else "wind"].mean())


def gap_geometry(net_mw: np.ndarray):
    """Vectorised: from an hourly net-load (MW, positive = deficit) return
    (annual_unmet_mwh, e_max_p95_mwh, p_peak_p95_mw) using contiguous deficit
    runs sized to the P95 gap (matches recompute_fig3c_storage.py)."""
    pos = np.maximum(net_mw, 0.0)
    annual = float(pos.sum())
    if annual <= 0:
        return 0.0, 0.0, 0.0
    mask = pos > 1e-9
    padded = np.concatenate(([0], mask.view(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    csum = np.concatenate(([0.0], np.cumsum(pos)))
    run_e = csum[ends] - csum[starts]              # MWh per deficit run
    e_max = float(np.percentile(run_e, 95))
    p_peak = float(np.percentile(pos[mask], 95))
    return annual, e_max, p_peak


def decompose(cc, power, workload, demand, demand_meta, supply, smeta, rtt,
              dist_cache, d_pert, year):
    home_idx = demand[cc]["d"].index
    d_base = pd.Series(np.asarray(d_pert), index=home_idx)
    pool = G.global_pool(cc, power, supply, smeta)
    S, labels = base.supply_basis(supply, pool, home_idx, fixed=(power == "L0"))
    if S.size == 0:
        return None
    if power != "L0":
        for j, lab in enumerate(labels):
            if lab.split(":")[0] != cc:
                S[:, j] = S[:, j] * G.ETA
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    d_eff = d_base
    Suse = S1
    if base.WORKLOAD_SCENARIOS[workload]:
        resid = np.maximum(d_base.to_numpy() - fit, 0.0)
        d_eff, _ = G.migrate_residual(cc, workload, demand, demand_meta, rtt, d_base, resid)
        Suse = base.equal_energy_align(S, float(d_eff.mean()))
        w, share, fit = base.portfolio_nnls(d_eff.to_numpy(), Suse)

    scale = 100.0 / float(d_eff.mean())            # MW per curve-unit (mean = 100 MW)
    d_mw = d_eff.to_numpy() * scale
    fit_mw = fit * scale                            # portfolio output at home busbar (MW)

    # ---- generation nameplate by tech (OB=1 base) + busbar energy ----
    np_pv = np_wind = gen_busbar = 0.0
    for j, lab in enumerate(labels):
        wj = float(w[j])
        if wj <= 1e-6:
            continue
        src, tech = lab.split(":")
        cf = cf_of(supply, src, tech)
        if cf <= 1e-6:
            continue
        eta_j = P.hvdc_efficiency(dist_cache[(cc, src)]) if src != cc else 1.0
        nameplate = wj * 100.0 / (cf * eta_j)
        if tech == "PV":
            np_pv += nameplate
        else:
            np_wind += nameplate
        gen_busbar += nameplate * cf * 8760.0

    # ---- transmission corridors aggregated per source country (OB-independent) ----
    src_cols: dict[str, list[int]] = {}
    for j, lab in enumerate(labels):
        src = lab.split(":")[0]
        if src != cc and float(w[j]) > 1e-6:
            src_cols.setdefault(src, []).append(j)
    tx_p95_mwkm = tx_np_mwkm = tx_p95_mw = tx_np_mw = 0.0
    foreign_w = imp_km_wsum = max_km = 0.0
    for src, cols in src_cols.items():
        gc = dist_cache[(cc, src)]
        route = P.TX_TORTUOSITY * gc
        eta = P.hvdc_efficiency(gc)
        wsrc = np.array([w[j] for j in cols])
        delivered = (Suse[:, cols] @ wsrc) * scale          # MW delivered to home
        line_flow = delivered / eta                          # MW sent (pre-loss)
        p95 = float(np.percentile(line_flow, 95))
        nameplate_src = float(sum(w[j] * 100.0 / (cf_of(supply, src, lab_tech(labels[j])) * eta)
                                  for j in cols))
        tx_p95_mwkm += p95 * route
        tx_np_mwkm += nameplate_src * route
        tx_p95_mw += p95
        tx_np_mw += nameplate_src
        ws = float(wsrc.sum())
        foreign_w += ws
        imp_km_wsum += ws * gc
        max_km = max(max_km, gc)
    mean_km = (imp_km_wsum / foreign_w) if foreign_w > 1e-9 else 0.0

    # ---- co-optimise overbuild OB (gen vs storage, central prices for `year`) ----
    # ENERGY-CLOSURE CONSTRAINT (the audit blocker): storage can return at most
    # excess*RTE, so a 24/7 design must overbuild until uncovered(OB) <=
    # RTE_EFF*excess(OB). Among feasible OB, pick the least-cost gen-vs-storage
    # point. OB=1.0 (equal-energy) is kept as the (infeasible) lower-bound the
    # current paper implicitly assumes.
    RTE_EFF = 0.85          # Li-ion-dominant effective round-trip; documented assumption

    def geom_at(ob):
        net = d_mw - ob * fit_mw
        annual, e_max, p_peak = gap_geometry(net)
        excess = float(np.maximum(ob * fit_mw - d_mw, 0.0).sum())
        return annual, e_max, p_peak, excess

    best = None
    for ob in OB_GRID:
        annual, e_max, p_peak, excess = geom_at(ob)
        if annual > RTE_EFF * excess:          # infeasible: storage cannot close the gap
            continue
        gen_c = P.gen_annual_cost(np_pv * ob, np_wind * ob, year, "central")
        sto_c, _ = P.storage_annual_cost(p_peak, e_max, annual, "central", year)
        anc_c = P.ancillary_annual_cost(gen_busbar * ob, "central")
        tot = gen_c + sto_c + anc_c
        if best is None or tot < best[0]:
            best = (tot, ob, annual, e_max, p_peak)
    if best is None:                            # even OB=3 cannot close -> cap, flag
        ob_star = float(OB_GRID[-1])
        annual, e_max, p_peak, _ = geom_at(ob_star)
        unmet_ob, emax_ob, ppeak_ob = annual, e_max, p_peak
    else:
        _, ob_star, unmet_ob, emax_ob, ppeak_ob = best
    unmet_b, emax_b, ppeak_b, _ = geom_at(1.0)
    # literature 24/7-CFE overbuild band (Google/Princeton ZeroLab ~1.5-3x): the
    # P95-gap storage model is not a full hourly-closure optimisation, so these
    # fixed-overbuild points bracket the generation cost a strictly-hourly system
    # would carry.
    unmet_15, emax_15, ppeak_15, _ = geom_at(1.5)
    unmet_20, emax_20, ppeak_20, _ = geom_at(2.0)

    return {
        "country": cc, "power_scope": power, "workload_scenario": workload,
        "scenario": f"{power}_{workload}", "year": year,
        "np_pv_mw": np_pv, "np_wind_mw": np_wind, "gen_busbar_mwh": gen_busbar,
        "tx_p95_mwkm": tx_p95_mwkm, "tx_np_mwkm": tx_np_mwkm,
        "tx_p95_mw": tx_p95_mw, "tx_np_mw": tx_np_mw,
        "foreign_energy_share": foreign_w, "mean_import_km": mean_km,
        "max_import_km": max_km, "n_corridors": len(src_cols),
        "uncovered_share": float(share),
        # OB=1.0 energy-balanced lower bound
        "unmet_mwh_ob1": unmet_b, "emax_mwh_ob1": emax_b, "ppeak_mw_ob1": ppeak_b,
        # co-optimised overbuild
        "ob_star": ob_star,
        "unmet_mwh_ob": unmet_ob, "emax_mwh_ob": emax_ob, "ppeak_mw_ob": ppeak_ob,
        # fixed literature-overbuild band
        "unmet_mwh_15": unmet_15, "emax_mwh_15": emax_15, "ppeak_mw_15": ppeak_15,
        "unmet_mwh_20": unmet_20, "emax_mwh_20": emax_20, "ppeak_mw_20": ppeak_20,
    }


def lab_tech(label):
    return label.split(":")[1]


def main():
    t0 = time.time()
    print("loading expanded inputs (104 demand / 105 supply) ...")
    panel, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    rtt = G.build_rtt_distance(demand_meta, smeta)

    dist_cache = {}
    for cc in demand:
        lon1, lat1 = smeta[cc]["lon"], smeta[cc]["lat"]
        for src in supply:
            if src in smeta:
                dist_cache[(cc, src)] = base.great_circle_km(
                    lon1, lat1, smeta[src]["lon"], smeta[src]["lat"])

    cc_lh = {cc: base.as_arr(demand[cc]["local_hour"]).astype(int) for cc in demand}
    cc_d = {cc: base.as_arr(demand[cc]["d"]).astype(float) for cc in demand}

    jobs = [(yr, MAIN_OP, ALL_LW) for yr in YEARS]
    jobs += [(2030, op, CORNERS) for op in OTHER_OPS]

    rows = []
    for yr, op, scens in jobs:
        lam_inf = _ai_share(base.AI_SHARE_INFERENCE, yr)
        lam_train = _ai_share(base.AI_SHARE_TRAINING, yr)
        tjob = time.time()
        for cc in demand:
            d_pert = perturb(op, cc_d[cc], cc_lh[cc], lam_inf, lam_train)
            for sc in scens:
                power, workload = sc.split("_")
                rec = decompose(cc, power, workload, demand, demand_meta,
                                supply, smeta, rtt, dist_cache, d_pert, yr)
                if rec is None or np.isnan(rec["uncovered_share"]):
                    continue
                rec["perturbation"] = op
                rec["region"] = demand[cc]["region"]
                rec["continent"] = demand[cc]["continent"]
                rows.append(rec)
        print(f"  {yr} {op} ({len(scens)} scen): {time.time()-tjob:.1f}s")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fullsystem_portfolio_decomposition.csv", index=False)
    print(f"\nwrote {OUT / 'fullsystem_portfolio_decomposition.csv'}  "
          f"({len(df)} rows)  in {time.time()-t0:.1f}s")

    sub = df[(df.perturbation == "P_mix") & (df.year == 2030) & (df.workload_scenario == "W0")]
    g = sub.groupby("power_scope").agg(
        foreign_share=("foreign_energy_share", "mean"),
        mean_km=("mean_import_km", "mean"), max_km=("max_import_km", "max"),
        ob=("ob_star", "mean"), uncovered=("uncovered_share", "mean"),
    ).reindex(L_SCOPES)
    print("\nP_mix 2030 W0 — geography + overbuild by power scope:")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
