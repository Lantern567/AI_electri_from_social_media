# -*- coding: utf-8 -*-
"""Full LP-dispatch firm-cost run replacing the P95 heuristic for the Fig-5 headline cells.
Scenarios: P_mix demand shape, years 2025-2050, LOCAL tier (tau=50, phi=0) and GLOBAL tier
(tau=500, phi=1). Only the cost changes; geometric uncovered_share is unchanged.

Phase 1 (single proc): heavy loading + routing -> per-(country,year,tier) residual d_mw/fit_mw
+ gen1/anc1 + year energy costs.  Phase 2 (parallel): solve the dispatch LP per residual.
Output: data_globalsites/r3_dispatch_cost_table.csv
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullsystem_cost_params as P
import storage_dispatch_lp as LP
import analyze_cfe_r3_waveform_cost as A
G = A.G; base = G.base

YEARS = list(A.YEARS)
TIERS = [(50, 0.0, "local"), (500, 1.0, "global")]
OUT = A.OUT

def e_costs(case, year):
    eli = getattr(P, "e_li", getattr(P, "_e_li"))
    eld = getattr(P, "e_ldes", getattr(P, "_e_ldes"))
    return float(eli(case, year)), float(eld(case, year))

def front_half(cc, tau, phi, demand, supply, smeta, reach, nat_shape, idx, d_pert):
    """Replicate decompose() lines 117-158 -> residual + gen/anc(ob=1) + uncovered_share."""
    d_base = pd.Series(np.asarray(d_pert), index=idx)
    S, labels = base.supply_basis(supply, [cc], idx, fixed=False)
    if S.size == 0:
        return None
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share0, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    d = d_base.to_numpy()
    resid0 = np.maximum(d - fit, 0.0)
    if phi > 0:
        r_route = np.minimum(phi * d, resid0)
        recv = [r for r in reach[cc][tau] if nat_shape.get(r) is not None]
        P_mat = np.stack([nat_shape[r] for r in recv], axis=1) if recv else None
        covered = A.waveform_cover(r_route, P_mat)
        d_eff = d - covered
        unc = float((resid0 - covered).sum() / max(d.sum(), 1e-12))
    else:
        d_eff = d
        unc = float(share0)
    Suse = base.equal_energy_align(S, float(np.mean(d_eff)))
    w, share, fit = base.portfolio_nnls(d_eff, Suse)
    scale = 100.0 / float(np.mean(d_eff))
    d_mw, fit_mw = d_eff * scale, fit * scale
    np_pv = np_wind = gbb = 0.0
    for j, lab in enumerate(labels):
        wj = float(w[j])
        if wj <= 1e-6:
            continue
        src, tech = lab.split(":")
        cf = float(supply[src]["pv" if tech == "PV" else "wind"].mean())
        if cf <= 1e-6:
            continue
        nm = wj * 100.0 / cf
        np_pv += nm if tech == "PV" else 0.0
        np_wind += nm if tech != "PV" else 0.0
        gbb += nm * cf * 8760.0
    return {"d_mw": d_mw.astype(np.float32), "fit_mw": fit_mw.astype(np.float32),
            "np_pv": np_pv, "np_wind": np_wind, "gbb": gbb, "unc": unc,
            "region": demand[cc]["region"]}

def solve_job(job):
    r = LP.solve_dispatch(job["d_mw"].astype(float), job["fit_mw"].astype(float),
                          job["gen1"], job["anc1"], job["ecl"], job["ecd"],
                          rel_target=1.0, cyclic=True)
    if not r["ok"]:
        return None
    return {"country": job["cc"], "tau_ms": job["tau"], "phi": job["phi"], "tier": job["tier"],
            "year": job["year"], "perturbation": "P_mix", "region": job["region"],
            "ob_star": round(r["ob"], 3), "uncovered_share": job["unc"],
            "lcoe_gen": r["lcoe_gen"], "lcoe_store": r["lcoe_store"], "lcoe_anc": r["lcoe_anc"],
            "lcoe_elec": r["lcoe_elec"], "lcoe_gen_floor": job["gen1"] / P.E_D_MWH,
            "ldes_energy_share": round(r["ldes_energy_share"], 3),
            "P_li": round(r["P_li"], 1), "E_li": round(r["E_li"], 1),
            "P_ld": round(r["P_ld"], 1), "E_ld": round(r["E_ld"], 1)}

def main():
    t0 = time.time()
    print("Phase 1: load + routing + residuals ...", flush=True)
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    codes = [c for c in demand if c in smeta]
    reach, receivers = A.precompute_reach(codes, smeta)
    nat_shape = {r: A.national_shape(supply, r, idx) for r in receivers}
    cc_lh = {c: base.as_arr(demand[c]["local_hour"]).astype(int) for c in demand}
    cc_d = {c: base.as_arr(demand[c]["d"]).astype(float) for c in demand}

    grid = bool(os.environ.get("GRID"))
    yearly = bool(os.environ.get("YEARLY"))
    if grid:                                          # 2030 full 6x6 (tau,phi) grid for Fig5 a/b
        yrs = [2030]
        TAU_G = [50, 100, 150, 200, 300, 500]; PHI_G = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cells = [(t, p, f"t{t}p{int(round(p*100))}") for t in TAU_G for p in PHI_G]
    elif yearly:                                      # phi=1 x 6 tau x all years for Fig5 c/d
        yrs = YEARS
        cells = [(t, 1.0, f"t{t}p100") for t in [50, 100, 150, 200, 300, 500]]
    else:
        yrs = [2030] if os.environ.get("SMOKE") else YEARS
        cells = TIERS
    if os.environ.get("SMOKE"):
        codes = codes[:3]
    jobs = []
    for yr in yrs:
        li, lt = A._ai_share(base.AI_SHARE_INFERENCE, yr), A._ai_share(base.AI_SHARE_TRAINING, yr)
        ecl, ecd = e_costs("central", yr)
        for cc in codes:
            dp = A.perturb("P_mix", cc_d[cc], cc_lh[cc], li, lt)
            for tau, phi, tier in cells:
                fh = front_half(cc, tau, phi, demand, supply, smeta, reach, nat_shape, idx, dp)
                if fh is None:
                    continue
                gen1 = P.gen_annual_cost(fh["np_pv"], fh["np_wind"], yr, "central")
                anc1 = P.ancillary_annual_cost(fh["gbb"], "central")
                jobs.append({"cc": cc, "year": yr, "tau": tau, "phi": phi, "tier": tier,
                             "d_mw": fh["d_mw"], "fit_mw": fh["fit_mw"], "gen1": gen1, "anc1": anc1,
                             "ecl": ecl, "ecd": ecd, "unc": fh["unc"], "region": fh["region"]})
    print(f"  {len(jobs)} residual jobs built in {time.time()-t0:.0f}s", flush=True)

    print("Phase 2: parallel LP dispatch ...", flush=True)
    from multiprocessing import Pool
    nproc = min(24, max(1, (os.cpu_count() or 4) - 2))
    t1 = time.time()
    with Pool(nproc) as pool:
        rows = [r for r in pool.map(solve_job, jobs, chunksize=4) if r is not None]
    print(f"  {len(rows)} solved in {time.time()-t1:.0f}s on {nproc} procs", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    outname = ("r3_dispatch_grid_2030.csv" if grid else
               "r3_dispatch_yearly_phi1.csv" if yearly else "r3_dispatch_cost_table.csv")
    df.to_csv(OUT / outname, index=False)
    # headline summary: median across countries, per year x tier
    print("\n=== LP-dispatch median LCOE (USD/MWh), P_mix, by year x tier ===", flush=True)
    for comp in ["lcoe_gen", "lcoe_store", "lcoe_anc", "lcoe_elec"]:
        piv = df.groupby(["year", "tier"])[comp].median().unstack().round(1)
        print(f"\n[{comp}]\n{piv.to_string()}", flush=True)
    # premium Pi = (firm - gen)/gen, per country then median
    df["pi"] = (df["lcoe_elec"] - df["lcoe_gen"]) / df["lcoe_gen"]
    pi = (df.groupby(["year", "tier"])["pi"].median() * 100).unstack().round(0)
    print(f"\n[premium Pi %, vs overbuilt gen denominator (paper convention)]\n{pi.to_string()}", flush=True)
    df["pi_floor"] = (df["lcoe_elec"] - df["lcoe_gen_floor"]) / df["lcoe_gen_floor"]
    pif = (df.groupby(["year", "tier"])["pi_floor"].median() * 100).unstack().round(0)
    print(f"\n[premium Pi %, vs ob=1 pure-generation floor]\n{pif.to_string()}", flush=True)
    print(f"\nwrote r3_dispatch_cost_table.csv ({len(df)} rows) | total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
