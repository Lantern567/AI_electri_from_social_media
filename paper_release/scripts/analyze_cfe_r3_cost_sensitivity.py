#!/usr/bin/env python3
"""SI sensitivity: the Result-3 full-system electricity LCOE vs the two firming-
storage assumptions the reviewer flagged -- (a) the percentile used to SIZE storage
to the gap (base P95), and (b) the Li-ion / LDES technology crossover (base: cheaper
of the two, which crosses near ~10 h duration).

Built on the SAME cost model as analyze_cfe_r3_latency_routable.py (national
generation + firming storage on the routed residual + ancillary, NO transmission),
so the base case reproduces the headline: ~125 USD/MWh local (tau=50 ms, phi=0) and
~95 USD/MWh global (tau=500 ms, phi=1) at P_mix 2030. Only the storage sizing
percentile and tech treatment are varied; the overbuild is re-co-optimised under
each assumption so each column is internally self-consistent.

Output: data_globalsites/r3_cost_sensitivity.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_global_sites as G                 # noqa: E402
import fullsystem_cost_params as P                   # noqa: E402
import analyze_cfe_r3_waveform_cost as R3L           # noqa: E402  (waveform routing)
base = G.base

DATA = R3L.OUT
E_D = P.E_D_MWH
YEAR = 2030
OP = "P_mix"
CELLS = {"local (tau=50, phi=0)": (50, 0.0), "global (tau=500, phi=1)": (500, 1.0)}
PCTS = [90, 95, 99, 100]
XVARS = [("cheaper", "cheaper-of (base)"), ("li", "Li-ion only"), ("ldes", "LDES only"),
         ("ldes_lo", "LDES cost x0.5"), ("ldes_hi", "LDES cost x1.5")]


def gap_geom_pct(net, pct):
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
    return annual, float(np.percentile(run_e, pct)), float(np.percentile(pos[mask], pct))


def store_cost_variant(p_peak, e_max, unmet, year, variant):
    if unmet <= 0 or p_peak <= 0:
        return 0.0
    case = "central"
    eL = P._e_li(case, year)
    eD = P._e_ldes(case, year)
    if variant == "ldes_lo":
        eD *= 0.5
    elif variant == "ldes_hi":
        eD *= 1.5
    cap_li = (P.P_LI * p_peak + eL * e_max) * 1000.0
    cap_ld = (P.P_LDES * p_peak + eD * e_max) * 1000.0
    lcos_li = (P._CRF_LI + P.STORE_FOM) * cap_li / (unmet * P.RTE_LI)
    lcos_ld = (P._CRF_LDES + P.STORE_FOM) * cap_ld / (unmet * P.RTE_LDES)
    if variant == "li":
        lcos = lcos_li
    elif variant == "ldes":
        lcos = lcos_ld
    else:
        lcos = min(lcos_li, lcos_ld)
    return unmet * lcos


def cell_cost(cc, tau, phi, supply, reach, nat_shape, idx, dp, year, pct, variant):
    """Reproduce analyze_cfe_r3_waveform_cost.decompose with a chosen storage
    sizing percentile and tech variant; return (elec_lcoe, store_lcoe)."""
    d_base = pd.Series(np.asarray(dp), index=idx)
    S, labels = base.supply_basis(supply, [cc], idx, fixed=False)
    if S.size == 0:
        return None
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    d = d_base.to_numpy()
    if phi > 0:
        resid = np.maximum(d - fit, 0.0)
        r_route = np.minimum(phi * d, resid)
        recv = [r for r in reach[cc][tau] if nat_shape.get(r) is not None]
        P_mat = np.stack([nat_shape[r] for r in recv], axis=1) if recv else None
        covered = R3L.waveform_cover(r_route, P_mat)
        d_eff = d - covered
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

    def geom(ob):
        net = d_mw - ob * fit_mw
        a, e, p = gap_geom_pct(net, pct)
        ex = float(np.maximum(ob * fit_mw - d_mw, 0.0).sum())
        return a, e, p, ex

    best = None
    for ob in R3L.OB_GRID:
        a, e, p, ex = geom(ob)
        if a > R3L.RTE_EFF * ex:
            continue
        tot = (P.gen_annual_cost(np_pv * ob, np_wind * ob, year, "central")
               + store_cost_variant(p, e, a, year, variant)
               + P.ancillary_annual_cost(gen_busbar * ob, "central"))
        if best is None or tot < best[0]:
            best = (tot, ob, a, e, p)
    if best is None:
        ob = float(R3L.OB_GRID[-1]); a, e, p, _ = geom(ob)
    else:
        _, ob, a, e, p = best
    gen = P.gen_annual_cost(np_pv * ob, np_wind * ob, year, "central")
    sto = store_cost_variant(p, e, a, year, variant)
    anc = P.ancillary_annual_cost(gen_busbar * ob, "central")
    return (gen + sto + anc) / E_D, sto / E_D


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    codes = [c for c in demand if c in smeta]
    reach, receivers = R3L.precompute_reach(codes, smeta)
    nat_shape = {r: R3L.national_shape(supply, r, idx) for r in receivers}
    cc_lh = {c: base.as_arr(demand[c]["local_hour"]).astype(int) for c in demand}
    cc_d = {c: base.as_arr(demand[c]["d"]).astype(float) for c in demand}
    li, lt = R3L._ai_share(base.AI_SHARE_INFERENCE, YEAR), R3L._ai_share(base.AI_SHARE_TRAINING, YEAR)
    dp = {c: R3L.perturb(OP, cc_d[c], cc_lh[c], li, lt) for c in codes}

    rows = []
    for cname, (tau, phi) in CELLS.items():
        # (a) percentile sweep, cheaper-of storage
        for pct in PCTS:
            vals = [cell_cost(c, tau, phi, supply, reach, nat_shape, idx, dp[c], YEAR, pct, "cheaper") for c in codes]
            vals = [v for v in vals if v is not None]
            elec = [v[0] for v in vals]; sto = [v[1] for v in vals]
            rows.append({"cell": cname, "axis": "sizing percentile", "variant": f"P{pct}",
                         "elec_median": round(float(np.median(elec)), 1),
                         "store_median": round(float(np.median(sto)), 1)})
        # (b) crossover sweep, base P95
        for variant, lbl in XVARS:
            vals = [cell_cost(c, tau, phi, supply, reach, nat_shape, idx, dp[c], YEAR, 95, variant) for c in codes]
            vals = [v for v in vals if v is not None]
            elec = [v[0] for v in vals]; sto = [v[1] for v in vals]
            rows.append({"cell": cname, "axis": "Li/LDES crossover", "variant": lbl,
                         "elec_median": round(float(np.median(elec)), 1),
                         "store_median": round(float(np.median(sto)), 1)})

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "r3_cost_sensitivity.csv", index=False)
    print("\n=== 2030 P_mix full-system LCOE sensitivity (USD/MWh, 104-country median) ===")
    for cname in CELLS:
        print(f"\n[{cname}]")
        print(out[out.cell == cname][["axis", "variant", "elec_median", "store_median"]].to_string(index=False))
    print(f"\nwrote {DATA / 'r3_cost_sensitivity.csv'}")


if __name__ == "__main__":
    main()
