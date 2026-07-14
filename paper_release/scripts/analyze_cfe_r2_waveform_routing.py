#!/usr/bin/env python3
"""Result 2, WAVEFORM-MATCHING reformulation of cross-longitude routing.

Motivation
----------
The earlier latency x routable-fraction grid (analyze_cfe_r2_latency_routable.py)
capped each hour's migration by SUM_{i in reach} surplus_i(t), an additive MW
budget that was reused undepleted by every sending country -- i.e. it was a
QUANTITY match that did not conserve the shared surplus. This script replaces
that with a pure WAVEFORM (shape/timing) match, in which absolute magnitude is
divided out at every step by equal-energy normalisation, so there is no shared
energy budget to conserve and no double counting is possible.

Model (two-step, per country c, latency tau, routable fraction phi)
-------------------------------------------------------------------
1. LOCAL FIT (identical to the baseline, so phi=0 reproduces ~41.8%):
     S1 = equal_energy_align(S_home[c], mean(d_c))
     _, share0, fit = portfolio_nnls(d_c, S1)
     resid(t) = max(d_c(t) - fit(t), 0)           # hours local clean supply falls short
2. WAVEFORM ROUTING of the shortfall:
     r_route(t) = min(phi * d_c(t), resid(t))      # the part of the shortfall we may route
     P = [ national clean-availability SHAPE of each reachable partner r ]   # one column per r
     P1 = equal_energy_align(P, mean(r_route))      # magnitude divided out
     _, _, fit_route = portfolio_nnls(r_route, P1)  # best STATIC longitude-mix of partner shapes
     covered(t) = min(r_route(t), fit_route(t))
     uncovered(t) = resid(t) - covered(t)
     U_c(tau,phi) = sum_t uncovered(t) / sum_t d_c(t)

Why this is conservation-neutral: a partner r's clean SHAPE appears in the basis
of every country that can reach it, but each country normalises that shape to its
OWN routable shortfall energy and asks a pure temporal-correlation question. Two
countries both finding r's daytime shape useful just means both are in r's night;
no MW is transferred, nothing is depleted. The residual floor is therefore the
phase-structure limit: shortfall hours that NO static non-negative mix of
reachable partner shapes can reproduce (weights are time-invariant).

Reachability and axes are unchanged from the baseline grid:
  X = latency tolerance tau (ms), RTT = 2*(km/200 + 20 ms)*1.4, gates reach.
  Y = routable fraction phi (0..1).
  Partners must be hyperscaler-capable (G.HYPERSCALER_EXTENDED) and within tau.
  Supply stays national; routing moves only the load's timing match, not power.

Env: R2WF_LIMIT=N restricts to the first N usable countries for a quick smoke run.

Output: data_globalsites_stations_expanded/
    r2_waveform_grid.csv, r2_waveform_median.csv
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST

OUT = (G.ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "data_globalsites_stations_expanded")

TAU_MS = [50, 100, 150, 200, 300, 500]          # latency tolerance levels
PHI = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]        # routable fraction levels
SPEED = base.LIGHT_SPEED_FIBER_KM_PER_MS        # 200 km/ms
SWITCH = base.SWITCHING_OVERHEAD_MS             # 20 ms
WAN = base.WAN_INFLATION_FACTOR                 # 1.4


def rtt_ms(km):
    if km <= 0:
        return 5.0
    return 2.0 * (km / SPEED + SWITCH) * WAN


def reach_km(tau):
    km = SPEED * (tau / (2.0 * WAN) - SWITCH)
    return max(km, 0.0)


def precompute_reach(codes, smeta):
    """reach[c][tau] = list of partner countries r (hyperscaler, within tau, r != c)."""
    reach = {c: {tau: [] for tau in TAU_MS} for c in codes}
    for c in codes:
        hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
        for r in codes:
            if r == c or r not in G.HYPERSCALER_EXTENDED:
                continue
            rtt = rtt_ms(base.great_circle_km(hlon, hlat, smeta[r]["lon"], smeta[r]["lat"]))
            for tau in TAU_MS:
                if rtt <= tau:
                    reach[c][tau].append(r)
    return reach


def national_shape(S):
    """Collapse a T x K station-basis matrix (each column mean 1) to one mean-1
    national clean-availability shape (the country's blended renewable timing)."""
    v = S.mean(axis=1)
    m = float(v.mean())
    return v / m if m > 0 else v


def main():
    print("loading expanded inputs (104 demand + 461 stations) ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)

    S_home, shape_home = {}, {}
    for c in demand:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        if S.size:
            S_home[c] = S
            shape_home[c] = national_shape(S)
        else:
            S_home[c] = None
            shape_home[c] = None

    countries = [c for c in demand if S_home.get(c) is not None and c in smeta]
    limit = int(os.environ.get("R2WF_LIMIT", "0"))
    if limit > 0:
        countries = countries[:limit]
    print(f"  usable countries = {len(countries)}"
          + (f"  (LIMIT={limit})" if limit else ""))
    reach = precompute_reach(countries, smeta)

    rows = []
    for ci, c in enumerate(countries):
        d = demand[c]["d"].to_numpy(float)
        dsum = max(d.sum(), 1e-12)
        # step 1: local fit (baseline), identical to original evaluate()
        S1 = base.equal_energy_align(S_home[c], float(d.mean()))
        _, share0, fit = base.portfolio_nnls(d, S1)
        resid = np.maximum(d - fit, 0.0)
        for tau in TAU_MS:
            partners = [r for r in reach[c][tau] if shape_home.get(r) is not None]
            P = (np.stack([shape_home[r] for r in partners], axis=1)
                 if partners else None)
            for phi in PHI:
                if phi <= 0 or P is None:
                    share_f, mig = share0, 0.0
                else:
                    r_route = np.minimum(phi * d, resid)
                    if r_route.sum() <= 0:
                        share_f, mig = share0, 0.0
                    else:
                        P1 = base.equal_energy_align(P, float(r_route.mean()))
                        _, _, fit_route = base.portfolio_nnls(r_route, P1)
                        covered = np.minimum(r_route, np.maximum(fit_route, 0.0))
                        uncovered = resid - covered
                        share_f = float(uncovered.sum() / dsum)
                        mig = float(covered.sum() / dsum)
                if not np.isfinite(share_f):
                    continue
                rows.append({"country": c, "region": demand[c]["region"],
                             "continent": demand[c]["continent"], "tau_ms": tau,
                             "phi": phi, "reach_km": round(reach_km(tau)),
                             "n_partners": len(partners),
                             "uncovered_share": share_f,
                             "uncovered_share_baseline": share0,
                             "migrated_share": mig})
        if (ci + 1) % 20 == 0:
            print(f"  ... {ci + 1}/{len(countries)} countries")

    grid = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "r2_waveform_grid.csv", index=False)

    med = (grid.groupby(["phi", "tau_ms"]).uncovered_share.median().unstack() * 100).round(1)
    med = med.reindex(index=PHI, columns=TAU_MS)
    med.to_csv(OUT / "r2_waveform_median.csv")

    print("\n=== WAVEFORM median uncovered share (%) : rows = routable phi, cols = latency tau (ms) ===")
    print("reach(km) per tau:", {t: round(reach_km(t)) for t in TAU_MS})
    print(med.to_string())
    base_corner = med.loc[0.0, 50]
    floor = med.loc[1.0, 500]
    print(f"\nbaseline (phi=0) median = {base_corner:.1f}%   |   floor (phi=1, tau=500 global) = {floor:.1f}%")
    # side-by-side with the old quantity-match grid, if present
    old = OUT / "r2_latency_median.csv"
    if old.exists():
        om = pd.read_csv(old, index_col=0)
        try:
            old_floor = float(om.loc[1.0, "500"]) if "500" in om.columns else float(om.iloc[-1, -1])
            print(f"old quantity-match floor (phi=1, tau=500) = {old_floor:.1f}%  ->  "
                  f"waveform floor = {floor:.1f}%  (delta {floor - old_floor:+.1f} pp)")
        except Exception as e:
            print("  (could not read old floor:", e, ")")
    (OUT / "r2_waveform_provenance.json").write_text(json.dumps({
        "design": "R2 waveform: local NNLS shortfall, routable shortfall matched to STATIC "
                  "non-negative mix of reachable partners' equal-energy-normalised national "
                  "clean shapes; no surplus-MW pool, conservation-neutral by construction",
        "tau_ms": TAU_MS, "phi": PHI, "reach_km": {t: round(reach_km(t)) for t in TAU_MS},
        "partner_basis": "one mean-1 national clean-availability shape per reachable hyperscaler partner",
        "metric": "uncovered share U on demand after covering routable shortfall by partner timing",
    }, indent=2))
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
