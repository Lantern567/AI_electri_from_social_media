#!/usr/bin/env python3
"""Result 3 full-system cost in the LATENCY x ROUTABLE-FRACTION framing.

Same compute-migration cost model as analyze_cfe_r3_compute_migration (generation +
firming storage + ancillary, NO transmission; supply national), but the geographic
lever is now LATENCY TOLERANCE tau (ms, RTT-gated routing) instead of a distance ring,
and the flexibility lever is the ROUTABLE FRACTION phi instead of delay grace. No
temporal deferral (the Cloudflare demand proxy is interactive web traffic).

Per (country, tau, phi, year, AI-operator): route the phi share of each deficit hour
to latency-feasible (RTT<=tau) hyperscaler receivers with clean surplus, then price the
residual. Cost falls as tau loosens (more reach) and phi grows (more routable);
tight real-time latency (tau=50 ms) admits no cross-border routing -> baseline cost.

Output: data_globalsites/r3_latency_cost_table.csv
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

TAU = [50, 100, 150, 200, 300, 500]             # latency tolerance levels (ms), aligned with R2
PHI = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]        # routable fraction, aligned with R2
SPEED, SWITCH, WAN = base.LIGHT_SPEED_FIBER_KM_PER_MS, base.SWITCHING_OVERHEAD_MS, base.WAN_INFLATION_FACTOR
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]    # sampled years (sufficient for panels d/e)
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


def precompute_pools(demand, smeta, codes, idx):
    surplus = {c: np.maximum(demand[c]["s"].reindex(idx).ffill().bfill().to_numpy()
                             - demand[c]["d"].reindex(idx).ffill().bfill().to_numpy(), 0.0) for c in codes}
    pools = {}
    for c in codes:
        hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
        pools[c] = {t: np.zeros(len(idx)) for t in TAU}
        for r in codes:
            if r == c or r not in G.HYPERSCALER_EXTENDED:
                continue
            rtt = rtt_ms(base.great_circle_km(hlon, hlat, smeta[r]["lon"], smeta[r]["lat"]))
            for t in TAU:
                if rtt <= t:
                    pools[c][t] = pools[c][t] + surplus[r]
    return pools


def decompose(cc, tau, phi, demand, supply, smeta, pools, idx, d_pert, year):
    d_base = pd.Series(np.asarray(d_pert), index=idx)
    S, labels = base.supply_basis(supply, [cc], idx, fixed=False)
    if S.size == 0:
        return None
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    d = d_base.to_numpy()
    if phi > 0:
        resid = np.maximum(d - fit, 0.0)
        mig = np.maximum(np.minimum(phi * d, np.minimum(resid, pools[cc][tau])), 0.0)
        d_eff = d - mig
    else:
        d_eff = d
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
            "year": year, "ob_star": ob_star, "uncovered_share": float(share),
            "region": demand[cc]["region"], "lcoe_gen": gen / E_D, "lcoe_store": sto / E_D,
            "lcoe_anc": anc / E_D, "lcoe_elec": (gen + sto + anc) / E_D}


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    codes = [c for c in demand if c in smeta]
    pools = precompute_pools(demand, smeta, codes, idx)
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
                    rec = decompose(cc, tau, phi, demand, supply, smeta, pools, idx, dp, yr)
                    if rec is None or not np.isfinite(rec["uncovered_share"]):
                        continue
                    rec["perturbation"] = op
                    rows.append(rec)
        print(f"  {yr} {op}: done")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r3_latency_cost_table.csv", index=False)

    h = df[(df.perturbation == "P_mix") & (df.year == 2030)]
    piv = (h.groupby(["phi", "tau_ms"]).lcoe_elec.median().unstack()).round(1).reindex(index=PHI, columns=TAU)
    print("\n=== R3 latency LCOE (USD/MWh) median, P_mix 2030 : rows=phi, cols=tau(ms) ===")
    print("reach(km):", {t: round(reach_km(t)) for t in TAU})
    print(piv.to_string())
    print(f"\n(phi=1) tau 50ms={piv.loc[1.0,50]:.1f} -> 500ms={piv.loc[1.0,500]:.1f}  "
          f"(monotone down in tau: {bool(np.all(np.diff(piv.loc[1.0].values)<=0.01))})")
    print(f"wrote r3_latency_cost_table.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
