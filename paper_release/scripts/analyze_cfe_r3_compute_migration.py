#!/usr/bin/env python3
"""Result 3 RECOMPUTED in the compute-migration framing (no electricity transmission).

The redesigned R2 makes the geographic lever COMPUTE migration (move the workload),
not electricity transmission (move the power). R3 follows: full-system supply cost
is recomputed with electricity supply held NATIONAL (each country runs on its own
optimized wind+PV portfolio) and the geographic lever = how far compute may be
migrated (D0 in-country .. D3 global). Moving compute travels over the internet, so
there is NO cross-region HVDC transmission term: the four-component bill of the old
R3 (generation + transmission + firming storage + ancillary) becomes a THREE-
component bill (generation + firming storage + ancillary).

Because migrating compute to a surplus country consumes that country's otherwise-
curtailed clean surplus (the migration volume is capped by receiver surplus, exactly
as in R2), it carries no extra generation/storage cost. The residual mismatch a
country must firm therefore falls monotonically as the migration range widens, so
the full-system cost falls monotonically and is LOWEST at global range -- overturning
the old electricity-transmission result (cost 'U-shaped', optimal ~3000 km, global
most expensive). Moving compute is cheaper than moving power.

Cost model: reuses fullsystem_cost_params (NREL ATB generation, duration-aware
Li-ion/LDES firming, ancillary), drops transmission. Overbuild OB co-optimised
between generation and storage exactly as in decompose_fullsystem_portfolio.

Output: data_globalsites/r3_cm_cost_table.csv, r3_cm_cost_summary.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_global_sites as G
import analyze_cfe_r2_spatiotemporal as R2
import fullsystem_cost_params as P
base = G.base

OUT = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"

DIST_RINGS = R2.DIST_RINGS                 # D0/D1/D2/D3
GRACE_LEVELS = [0, 6, 24, 168]              # delay-grace levels (4 -> distance x grace 4x4 panels)
PHI = R2.PHI_DEFAULT                        # 0.40
MOVABLE = R2.MOVABLE_SHARE                  # 0.60
YEARS = list(range(2025, 2051))
MAIN_OP = "P_mix"
OTHER_OPS = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst"]
OB_GRID = np.round(np.arange(1.0, 3.001, 0.1), 2)
RTE_EFF = 0.85


def _ai_share(table, year):
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


def gap_geometry(net_mw):
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
    run_e = csum[ends] - csum[starts]
    return annual, float(np.percentile(run_e, 95)), float(np.percentile(pos[mask], 95))


def decompose_cm(cc, ring, dmax, grace, demand, supply, smeta, surplus_pools, idx, d_pert, year):
    d_base = pd.Series(np.asarray(d_pert), index=idx)
    # national supply: home country's PV+wind, NNLS-reweighted (no foreign, no HVDC)
    S, labels = base.supply_basis(supply, [cc], idx, fixed=False)
    if S.size == 0:
        return None
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)

    # temporal deferral (grace) then spatial migration (distance), exactly as R2
    d_tmp = R2.temporal_defer(d_base.to_numpy(), fit, grace, PHI) if grace > 0 else d_base.to_numpy()
    if grace > 0:
        S2 = base.equal_energy_align(S, float(d_tmp.mean()))
        w, share, fit = base.portfolio_nnls(d_tmp, S2)
    resid = np.maximum(d_tmp - fit, 0.0)
    if dmax > 0:
        mig = np.maximum(np.minimum(MOVABLE * d_tmp, np.minimum(resid, surplus_pools[cc][ring])), 0.0)
        d_eff = d_tmp - mig
    else:
        d_eff = d_tmp
    Suse = base.equal_energy_align(S, float(d_eff.mean()))
    w, share, fit = base.portfolio_nnls(d_eff, Suse)

    scale = 100.0 / float(np.mean(d_eff))
    d_mw = d_eff * scale
    fit_mw = fit * scale

    np_pv = np_wind = gen_busbar = 0.0
    for j, lab in enumerate(labels):
        wj = float(w[j])
        if wj <= 1e-6:
            continue
        src, tech = lab.split(":")
        cf = float(supply[src]["pv" if tech == "PV" else "wind"].mean())
        if cf <= 1e-6:
            continue
        nameplate = wj * 100.0 / cf                       # home, eta = 1 (no HVDC loss)
        if tech == "PV":
            np_pv += nameplate
        else:
            np_wind += nameplate
        gen_busbar += nameplate * cf * 8760.0

    def geom_at(ob):
        net = d_mw - ob * fit_mw
        annual, e_max, p_peak = gap_geometry(net)
        excess = float(np.maximum(ob * fit_mw - d_mw, 0.0).sum())
        return annual, e_max, p_peak, excess

    best = None
    for ob in OB_GRID:
        annual, e_max, p_peak, excess = geom_at(ob)
        if annual > RTE_EFF * excess:
            continue
        gen_c = P.gen_annual_cost(np_pv * ob, np_wind * ob, year, "central")
        sto_c, _ = P.storage_annual_cost(p_peak, e_max, annual, "central", year)
        anc_c = P.ancillary_annual_cost(gen_busbar * ob, "central")
        tot = gen_c + sto_c + anc_c
        if best is None or tot < best[0]:
            best = (tot, ob, annual, e_max, p_peak)
    if best is None:
        ob_star = float(OB_GRID[-1])
        annual, e_max, p_peak, _ = geom_at(ob_star)
    else:
        _, ob_star, annual, e_max, p_peak = best

    return {"country": cc, "dist_ring": ring, "grace_h": grace, "year": year,
            "np_pv_mw": np_pv, "np_wind_mw": np_wind, "gen_busbar_mwh": gen_busbar,
            "ob_star": ob_star, "unmet_mwh": annual, "emax_mwh": e_max, "ppeak_mw": p_peak,
            "uncovered_share": float(share), "region": demand[cc]["region"]}


def price(r, case="central"):
    ob = r["ob_star"]
    gen = P.gen_annual_cost(r["np_pv_mw"] * ob, r["np_wind_mw"] * ob, r["year"], case)
    store, _ = P.storage_annual_cost(r["ppeak_mw"], r["emax_mwh"], r["unmet_mwh"], case, r["year"])
    anc = P.ancillary_annual_cost(r["gen_busbar_mwh"] * ob, case)
    E_D = P.E_D_MWH
    return {"lcoe_gen": gen / E_D, "lcoe_store": store / E_D, "lcoe_anc": anc / E_D,
            "lcoe_elec": (gen + store + anc) / E_D}


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    surplus_pools, _ = R2.precompute_surplus_pools(demand, demand_meta, smeta)
    cc_lh = {cc: base.as_arr(demand[cc]["local_hour"]).astype(int) for cc in demand}
    cc_d = {cc: base.as_arr(demand[cc]["d"]).astype(float) for cc in demand}
    codes = [cc for cc in demand if cc in smeta]

    jobs = [(yr, MAIN_OP) for yr in YEARS] + [(2030, op) for op in OTHER_OPS]
    rows = []
    for yr, op in jobs:
        lam_inf = _ai_share(base.AI_SHARE_INFERENCE, yr)
        lam_train = _ai_share(base.AI_SHARE_TRAINING, yr)
        graces = GRACE_LEVELS
        for cc in codes:
            d_pert = perturb(op, cc_d[cc], cc_lh[cc], lam_inf, lam_train)
            for grace in graces:
                for ring, dmax in DIST_RINGS:
                    rec = decompose_cm(cc, ring, dmax, grace, demand, supply, smeta,
                                       surplus_pools, idx, d_pert, yr)
                    if rec is None or not np.isfinite(rec["uncovered_share"]):
                        continue
                    rec["perturbation"] = op
                    rec.update(price(rec))
                    rows.append(rec)
        print(f"  {yr} {op}: done")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r3_cm_cost_table.csv", index=False)

    # headline: P_mix 2030, g=0, by distance ring (mean over countries)
    h = df[(df.perturbation == "P_mix") & (df.year == 2030) & (df.grace_h == 0)]
    g = h.groupby("dist_ring").agg(
        gen=("lcoe_gen", "mean"), store=("lcoe_store", "mean"),
        anc=("lcoe_anc", "mean"), elec=("lcoe_elec", "mean"),
        ob=("ob_star", "mean"), unc=("uncovered_share", "mean")).reindex(["D0", "D1", "D2", "D3"])
    print("\n=== R3 compute-migration full-system LCOE (USD/MWh), P_mix 2030, g=0 ===")
    print((g * [1, 1, 1, 1, 1, 100]).round(1).to_string() if False else g.round(1).to_string())
    d0, d3 = g.loc["D0"], g.loc["D3"]
    print(f"\nD0 (in-country) = {d0.elec:.1f}  ->  D3 (global) = {d3.elec:.1f} USD/MWh  "
          f"(global is {d0.elec/max(d3.elec,1e-9):.2f}x cheaper; monotone? "
          f"{bool(np.all(np.diff(g.elec.values) < 0))})")

    # save summary pivot
    summ = (df[df.perturbation == "P_mix"].groupby(["year", "grace_h", "dist_ring"])
            ["lcoe_elec"].mean().reset_index())
    summ.to_csv(OUT / "r3_cm_cost_summary.csv", index=False)
    print(f"\nwrote {OUT/'r3_cm_cost_table.csv'} ({len(df)} rows) + r3_cm_cost_summary.csv")


if __name__ == "__main__":
    main()
