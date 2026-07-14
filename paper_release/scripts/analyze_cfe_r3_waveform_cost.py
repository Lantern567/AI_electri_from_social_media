#!/usr/bin/env python3
"""Result 3 full-system cost, WAVEFORM-matching routing (consistent with R2 waveform).

Identical cost model to analyze_cfe_r3_latency_routable.py (generation + firming
storage + ancillary, NO transmission; supply national; latency tolerance tau x
routable fraction phi), EXCEPT the routing step. The old model offloaded each
deficit hour by a QUANTITY cap min(phi*d, resid, SUM surplus_r) with the shared
surplus pool reused undepleted by every sender (double counting). Here the routable
shortfall is instead covered by WAVEFORM (shape) matching: a static non-negative mix
of reachable partners' equal-energy-normalised national clean shapes, magnitude
divided out, conservation-neutral. The load the LOCAL system must still firm is
d_eff = d - covered; everything downstream (overbuild optimisation, gen/firming/
ancillary layering) is unchanged.

Output: data_globalsites/r3_waveform_cost_table.csv  (same schema as r3_latency_cost_table.csv)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_global_sites as G
import fullsystem_cost_params as P
base = G.base

OUT = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"

TAU = [50, 100, 150, 200, 300, 500]
PHI = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
SPEED, SWITCH, WAN = base.LIGHT_SPEED_FIBER_KM_PER_MS, base.SWITCHING_OVERHEAD_MS, base.WAN_INFLATION_FACTOR
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
MAIN_OP = "P_mix"
OTHER_OPS = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst"]
OB_GRID = np.round(np.arange(1.0, 3.001, 0.1), 2)
RTE_EFF = 0.85


def rtt_ms(km):
    return 5.0 if km <= 0 else 2.0 * (km / SPEED + SWITCH) * WAN


def reach_km(tau):
    return max(SPEED * (tau / (2.0 * WAN) - SWITCH), 0.0)


def _ai_share(t, y):
    xs = sorted(t)
    return float(np.interp(y, xs, [t[x] for x in xs]))


def perturb(op, d, lh, li, lt):
    if op == "P0_baseline":
        return d
    if op == "P1_flatten":
        return base.operator_flatten(d, li, lt)
    if op == "P2_sharpen":
        return base.operator_sharpen(d, lh, li, lt)
    if op == "P2_emp":
        return base.operator_sharpen_emp(d, lh, li, lt)
    if op == "P3_burst":
        return base.operator_burst(d, li, lt)
    return base.operator_mix(d, lh, li, lt)


def gap_geometry(net):
    pos = np.maximum(net, 0.0)
    annual = float(pos.sum())
    if annual <= 0:
        return 0.0, 0.0, 0.0
    mask = pos > 1e-9
    padded = np.concatenate(([0], mask.view(np.int8), [0]))
    diff = np.diff(padded)
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    csum = np.concatenate(([0.0], np.cumsum(pos)))
    run_e = csum[ends] - csum[starts]
    return annual, float(np.percentile(run_e, 95)), float(np.percentile(pos[mask], 95))


def national_shape(supply, r, idx):
    """One mean-1 national clean-availability shape for country r (its VRE blend)."""
    S, _ = base.supply_basis(supply, [r], idx, fixed=False)
    if S.size == 0:
        return None
    v = S.mean(axis=1)
    m = float(v.mean())
    return v / m if m > 0 else v


def precompute_reach(codes, smeta):
    receivers = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    reach = {c: {t: [] for t in TAU} for c in codes}
    for c in codes:
        hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
        for r in receivers:
            if r == c:
                continue
            rtt = rtt_ms(base.great_circle_km(hlon, hlat, smeta[r]["lon"], smeta[r]["lat"]))
            for t in TAU:
                if rtt <= t:
                    reach[c][t].append(r)
    return reach, receivers


def waveform_cover(r_route, P_mat):
    """Load covered by matching the routable shortfall SHAPE to a static non-negative
    mix of partner clean shapes (equal-energy-normalised); magnitude divided out."""
    if P_mat is None or P_mat.shape[1] == 0 or r_route.sum() <= 0:
        return np.zeros_like(r_route)
    P1 = base.equal_energy_align(P_mat, float(r_route.mean()))
    _, _, fit_route = base.portfolio_nnls(r_route, P1)
    return np.minimum(r_route, np.maximum(fit_route, 0.0))


def decompose(cc, tau, phi, demand, supply, smeta, reach, nat_shape, idx, d_pert, year):
    d_base = pd.Series(np.asarray(d_pert), index=idx)
    S, labels = base.supply_basis(supply, [cc], idx, fixed=False)
    if S.size == 0:
        return None
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share0, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    d = d_base.to_numpy()
    resid0 = np.maximum(d - fit, 0.0)                       # local shortfall (first fit)
    if phi > 0:
        r_route = np.minimum(phi * d, resid0)
        recv = [r for r in reach[cc][tau] if nat_shape.get(r) is not None]
        P_mat = np.stack([nat_shape[r] for r in recv], axis=1) if recv else None
        covered = waveform_cover(r_route, P_mat)
        d_eff = d - covered
        # reported residual: fraction of TOTAL demand still uncovered by local+routed
        # timing (approach A, identical metric to the R2 waveform grid).
        unc_share = float((resid0 - covered).sum() / max(d.sum(), 1e-12))
    else:
        d_eff = d
        unc_share = float(share0)
    # local system re-optimises for the load it must still firm (d_eff) -> cost/firming
    Suse = base.equal_energy_align(S, float(np.mean(d_eff)))
    w, share, fit = base.portfolio_nnls(d_eff, Suse)
    scale = 100.0 / float(np.mean(d_eff))
    d_mw, fit_mw = d_eff * scale, fit * scale

    np_pv = np_wind = gen_busbar = 0.0
    for j, lab in enumerate(labels):
        wj = float(w[j])
        if wj <= 1e-6:
            continue
        src, tech = lab.split(":")
        cf = float(supply[src]["pv" if tech == "PV" else "wind"].mean())
        if cf <= 1e-6:
            continue
        nameplate = wj * 100.0 / cf
        if tech == "PV":
            np_pv += nameplate
        else:
            np_wind += nameplate
        gen_busbar += nameplate * cf * 8760.0

    def geom_at(ob):
        net = d_mw - ob * fit_mw
        a, e, p = gap_geometry(net)
        ex = float(np.maximum(ob * fit_mw - d_mw, 0.0).sum())
        return a, e, p, ex
    best = None
    for ob in OB_GRID:
        a, e, p, ex = geom_at(ob)
        if a > RTE_EFF * ex:
            continue
        tot = (P.gen_annual_cost(np_pv * ob, np_wind * ob, year, "central")
               + P.storage_annual_cost(p, e, a, "central", year)[0]
               + P.ancillary_annual_cost(gen_busbar * ob, "central"))
        if best is None or tot < best[0]:
            best = (tot, ob, a, e, p)
    if best is None:
        ob_star = float(OB_GRID[-1]); a, e, p, _ = geom_at(ob_star)
    else:
        _, ob_star, a, e, p = best
    gen = P.gen_annual_cost(np_pv * ob_star, np_wind * ob_star, year, "central")
    sto = P.storage_annual_cost(p, e, a, "central", year)[0]
    anc = P.ancillary_annual_cost(gen_busbar * ob_star, "central")
    E_D = P.E_D_MWH
    return {"country": cc, "tau_ms": tau, "phi": phi, "reach_km": round(reach_km(tau)),
            "year": year, "ob_star": ob_star, "uncovered_share": unc_share,
            "region": demand[cc]["region"], "lcoe_gen": gen / E_D, "lcoe_store": sto / E_D,
            "lcoe_anc": anc / E_D, "lcoe_elec": (gen + sto + anc) / E_D}


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    codes = [c for c in demand if c in smeta]
    reach, receivers = precompute_reach(codes, smeta)
    nat_shape = {r: national_shape(supply, r, idx) for r in receivers}
    print(f"  receivers with a national clean shape = {sum(v is not None for v in nat_shape.values())}/{len(receivers)}")
    cc_lh = {c: base.as_arr(demand[c]["local_hour"]).astype(int) for c in demand}
    cc_d = {c: base.as_arr(demand[c]["d"]).astype(float) for c in demand}

    jobs = [(y, MAIN_OP) for y in YEARS] + [(2030, op) for op in OTHER_OPS]
    rows = []
    for yr, op in jobs:
        li, lt = _ai_share(base.AI_SHARE_INFERENCE, yr), _ai_share(base.AI_SHARE_TRAINING, yr)
        for cc in codes:
            dp = perturb(op, cc_d[cc], cc_lh[cc], li, lt)
            for tau in TAU:
                for phi in PHI:
                    rec = decompose(cc, tau, phi, demand, supply, smeta, reach, nat_shape, idx, dp, yr)
                    if rec is None or not np.isfinite(rec["uncovered_share"]):
                        continue
                    rec["perturbation"] = op
                    rows.append(rec)
        print(f"  {yr} {op}: done")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r3_waveform_cost_table.csv", index=False)

    h = df[(df.perturbation == "P_mix") & (df.year == 2030)]
    piv = (h.groupby(["phi", "tau_ms"]).lcoe_elec.median().unstack()).round(1).reindex(index=PHI, columns=TAU)
    print("\n=== R3 WAVEFORM LCOE (USD/MWh) median, P_mix 2030 : rows=phi, cols=tau(ms) ===")
    print("reach(km):", {t: round(reach_km(t)) for t in TAU})
    print(piv.to_string())
    unc = (h.groupby(["phi", "tau_ms"]).uncovered_share.median().unstack() * 100).round(1).reindex(index=PHI, columns=TAU)
    print("\n=== R3 WAVEFORM uncovered share (%) median, P_mix 2030 ===")
    print(unc.to_string())
    hg = h[(h.phi == 1.0)]
    for lab, comp in [("gen", "lcoe_gen"), ("firming", "lcoe_store"), ("anc", "lcoe_anc"), ("total", "lcoe_elec")]:
        loc = float(hg[hg.tau_ms == 50][comp].median())
        glo = float(hg[hg.tau_ms == 500][comp].median())
        print(f"  {lab:8s} phi=1: local(50ms)={loc:6.1f} -> global(500ms)={glo:6.1f} USD/MWh")
    print(f"wrote r3_waveform_cost_table.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
