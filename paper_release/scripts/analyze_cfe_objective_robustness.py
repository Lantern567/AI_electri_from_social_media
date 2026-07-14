#!/usr/bin/env python3
"""Objective-robustness check for the uncovered-share floor (reviewer Major-2).

The published mismatch/routing residuals use NNLS, i.e.

    min_w || Phi w - d ||_2   s.t. w >= 0      (then rescaled to equal energy)

which minimises SQUARED error, not the reported metric -- the uncovered energy
sum_t max(d_t - (Phi w)_t, 0).  A squared-error optimum need not minimise the
uncovered share, and equal energy is imposed only by a post-hoc scalar rescale,
so the NNLS residual is only an UPPER bound on the geometric floor.

This script recomputes the same problem with the DIRECT objective via a linear
program, on the IDENTICAL basis and with equal energy as a HARD constraint, and
reports the two side by side per country, for BOTH calibers the paper headlines:

  - domestic baseline  (phi = 0)            -> paper median ~41.8 %
  - global routed floor (phi = 1, tau=500)  -> paper median ~12.7 %

  step-1 local LP:  min_{w>=0,u>=0} sum u_t  s.t. u_t >= d_t-(Phi w)_t,
                    sum_t (Phi w)_t = sum_t d_t
  step-2 route  LP: max_{w>=0,cov>=0} sum cov_t  s.t. cov_t <= r_route_t,
                    cov_t <= (P w)_t,  sum_t (P w)_t = sum_t r_route_t

Feasible set = NNLS-after-rescaling's set, so  share_LP <= share_NNLS  always.
Small gap  => 41.8 / 12.7 are validated as ~the floor and, if anything,
conservative (an upper bound on the residual).

Env OBJROB_LIMIT=N restricts to the first N usable countries (smoke run).
Output: data_globalsites_stations_expanded/r2_objective_robustness.csv
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST
import analyze_cfe_r2_waveform_routing as WF   # reach/shape helpers + constants

OUT = WF.OUT
LPM = "highs"


def lp_min_uncovered(d: np.ndarray, S: np.ndarray):
    """min sum_t u_t  s.t.  u_t >= d_t-(S w)_t, u>=0, w>=0, sum(S w)=sum(d)."""
    T, K = S.shape
    c = np.concatenate([np.zeros(K), np.ones(T)])
    A_ub = sp.hstack([sp.csr_matrix(-S), -sp.identity(T, format="csr")], format="csr")
    b_ub = -d
    colsum = np.asarray(S.sum(axis=0)).ravel()
    A_eq = sp.csr_matrix(np.concatenate([colsum, np.zeros(T)]).reshape(1, -1))
    b_eq = np.array([float(d.sum())])
    bounds = [(0.0, None)] * (K + T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=LPM)
    if not res.success:
        return None, None
    w = res.x[:K]
    fit = S @ w
    return float(np.maximum(d - fit, 0.0).sum() / max(d.sum(), 1e-12)), fit


def lp_max_coverage(r_route: np.ndarray, P: np.ndarray) -> float:
    """max sum_t cov_t  s.t. cov_t<=r_route_t, cov_t<=(P w)_t, w>=0,
    sum(P w)=sum(r_route).  Returns covered energy sum."""
    T, K = P.shape
    obj = np.concatenate([np.zeros(K), -np.ones(T)])                 # maximize sum cov
    A_ub = sp.hstack([sp.csr_matrix(-P), sp.identity(T, format="csr")], format="csr")  # cov-Pw<=0
    b_ub = np.zeros(T)
    colsum = np.asarray(P.sum(axis=0)).ravel()
    A_eq = sp.csr_matrix(np.concatenate([colsum, np.zeros(T)]).reshape(1, -1))
    b_eq = np.array([float(r_route.sum())])
    bounds = [(0.0, None)] * K + [(0.0, float(x)) for x in np.maximum(r_route, 0.0)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=LPM)
    if not res.success:
        return np.nan
    return float(res.x[K:].sum())


def main():
    t0 = time.time()
    print("loading expanded inputs (104 demand + 461 stations) ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)

    S_home, shape_home = {}, {}
    for c in demand:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        if getattr(S, "size", 0):
            S_home[c], shape_home[c] = S, WF.national_shape(S)
        else:
            S_home[c], shape_home[c] = None, None

    countries = [c for c in demand if S_home.get(c) is not None and c in smeta]
    limit = int(os.environ.get("OBJROB_LIMIT", "0"))
    if limit > 0:
        countries = countries[:limit]
    print(f"  usable countries = {len(countries)}" + (f"  (LIMIT={limit})" if limit else ""))
    reach = WF.precompute_reach(countries, smeta)

    rows = []
    for ci, c in enumerate(countries):
        d = demand[c]["d"].to_numpy(float)
        dsum = max(d.sum(), 1e-12)
        S1 = base.equal_energy_align(S_home[c], float(d.mean()))

        # domestic baseline (phi = 0)
        _, sh_nnls_dom, fit_nnls = base.portfolio_nnls(d, S1)
        sh_lp_dom, fit_lp = lp_min_uncovered(d, S1)
        if sh_lp_dom is None:
            print(f"  domestic LP failed for {c}; skipping")
            continue
        resid_nnls = np.maximum(d - fit_nnls, 0.0)
        resid_lp = np.maximum(d - fit_lp, 0.0)

        # global routed floor (phi = 1, tau = 500)
        partners = [r for r in reach[c][500] if shape_home.get(r) is not None]
        P = np.stack([shape_home[r] for r in partners], axis=1) if partners else None
        if P is None:
            sh_nnls_rt, sh_lp_rt = sh_nnls_dom, sh_lp_dom
        else:
            r_n = np.minimum(d, resid_nnls)                     # NNLS route (reproduces 12.7)
            if r_n.sum() > 0:
                P1n = base.equal_energy_align(P, float(r_n.mean()))
                _, _, fit_rt = base.portfolio_nnls(r_n, P1n)
                cov_n = np.minimum(r_n, np.maximum(fit_rt, 0.0))
                sh_nnls_rt = float((resid_nnls - cov_n).sum() / dsum)
            else:
                sh_nnls_rt = sh_nnls_dom
            r_l = np.minimum(d, resid_lp)                       # LP route (min-uncovered analog)
            if r_l.sum() > 0:
                P1l = base.equal_energy_align(P, float(r_l.mean()))
                cov_l = lp_max_coverage(r_l, P1l)
                sh_lp_rt = float((resid_lp.sum() - cov_l) / dsum) if np.isfinite(cov_l) else np.nan
            else:
                sh_lp_rt = sh_lp_dom

        rows.append({"country": c, "region": demand[c]["region"],
                     "n_home_stations": int(S_home[c].shape[1]),
                     "n_partners": len(partners) if P is not None else 0,
                     "dom_nnls": sh_nnls_dom, "dom_lp": sh_lp_dom,
                     "rt_nnls": sh_nnls_rt, "rt_lp": sh_lp_rt})
        if (ci + 1) % 10 == 0:
            print(f"  ... {ci + 1}/{len(countries)}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r2_objective_robustness.csv", index=False)

    def med(col):
        return float(np.median(df[col].dropna()) * 100)

    print("\n=== median uncovered share (%), {} countries ===".format(len(df)))
    print(f"  DOMESTIC  (phi=0):        NNLS = {med('dom_nnls'):5.1f}   L1-LP = {med('dom_lp'):5.1f}"
          f"   (gap {med('dom_nnls')-med('dom_lp'):+.1f} pp)   [paper ~41.8]")
    print(f"  ROUTED (phi=1, tau=500):  NNLS = {med('rt_nnls'):5.1f}   L1-LP = {med('rt_lp'):5.1f}"
          f"   (gap {med('rt_nnls')-med('rt_lp'):+.1f} pp)   [paper ~12.7]")
    g_dom = (df["dom_nnls"] - df["dom_lp"]) * 100
    g_rt = (df["rt_nnls"] - df["rt_lp"]) * 100
    print(f"\n  per-country NNLS-minus-LP gap (pp):"
          f"  domestic median={g_dom.median():.2f} p90={g_dom.quantile(.9):.2f} max={g_dom.max():.2f}")
    print(f"                                      routed   median={g_rt.median():.2f} "
          f"p90={g_rt.quantile(.9):.2f} max={g_rt.max():.2f}")
    (OUT / "r2_objective_robustness_provenance.json").write_text(json.dumps({
        "compare": "NNLS L2 projection + post-hoc equal-energy rescale (paper) vs "
                   "direct L1 min-uncovered LP with equal-energy hard constraint (floor)",
        "same_basis_as": "analyze_cfe_r2_waveform_routing.py",
        "n_countries": int(len(df)),
        "median_pct": {k: med(k) for k in ["dom_nnls", "dom_lp", "rt_nnls", "rt_lp"]},
    }, indent=2))
    print(f"\noutput -> {OUT / 'r2_objective_robustness.csv'}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
