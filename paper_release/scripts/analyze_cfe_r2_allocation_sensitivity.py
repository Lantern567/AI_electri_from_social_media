#!/usr/bin/env python3
"""SI robustness / due-diligence: sensitivity of the Result 2 routing grid to the
surplus ALLOCATION RULE and to surplus DEFINITION (global reach, tau = 500 ms).

The headline grid (analyze_cfe_r2_latency_routable.py) lets each deficit country draw
against the UNION of reachable clean surplus WITHOUT depleting it -- every receiver's
hourly surplus is a non-rival shape resource shared by all senders. That measures the
geographic-complementarity POTENTIAL (an upper bound). Here we re-clear the same
migration under a SHARED, FINITE surplus budget (each receiver's hourly surplus is
consumed at most once) and report a range of allocation rules / surplus definitions.

At global reach every sender reaches every hyperscaler receiver, so the bipartite graph
is complete and order-free clearing is exact. Variants (all vs the published upper bound A):
  A  independent pool (headline, double-counted UPPER BOUND)            -- reproduced as a check
  B  shared budget, PROPORTIONAL rationing (neutral)                    -- mig_c = desired_c * ratio(t)
  C  shared budget, PRIORITY smallest-first (coverage-maximising)       -- optimistic-median bound
  D  shared budget, proportional, receivers = ALL countries (not just 39 hyperscalers)
  E  shared budget, proportional, CAPACITY-weighted (absolute installed VRE MW)
       -> tests whether large absolute-surplus countries recover benefit; also yields the
          capacity(energy)-weighted uncovered share (reviewer's equal- vs energy-weight point)
where ratio(t) = min(1, sum surplus(t) / sum desired(t)), desired_c=min(phi*d, home residual).

Global reach maximises surplus sharing, so these corrections upper-bound the correction at
any narrower reach. Total clean-matched migration per hour = min(sum desired, sum surplus) is
fixed across non-wasteful rules, so the MEAN uncovered share is ~allocation-invariant; only
the cross-country MEDIAN moves with the distribution (B vs C).

Output: data_globalsites_stations_expanded/r2_allocation_sensitivity.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST
import analyze_cfe_r2_latency_routable as R2

OUT = R2.OUT
PHI = R2.PHI


def priority_fill(D, avail):
    """Smallest-first (coverage-maximising) allocation. D=(C,T) desired, avail=(T,)."""
    order = np.argsort(D, axis=0)
    Dsort = np.take_along_axis(D, order, axis=0)
    csum = np.cumsum(Dsort, axis=0)
    prev = csum - Dsort
    alloc_sorted = np.clip(avail[None, :] - prev, 0.0, Dsort)
    out = np.empty_like(D)
    np.put_along_axis(out, order, alloc_sorted, axis=0)
    return out


def main():
    print("loading expanded inputs (104 demand + 461 stations) ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)

    # absolute installed VRE capacity (MW) per country, for the capacity-weighted variant
    cap = pd.read_csv(G.SITE_DIR / "country_capacity.csv", keep_default_na=False, na_values=[""])
    cap_tot = {}
    for r in cap.itertuples():
        cap_tot[r.iso2] = cap_tot.get(r.iso2, 0.0) + (float(r.capacity_mw_used) if r.capacity_mw_used not in ("", None) else 0.0)

    S_home, resid, dvec, surplus, wcap = {}, {}, {}, {}, {}
    for c in demand:
        if c not in smeta:
            continue
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        if S is None or not S.size:
            continue
        d = demand[c]["d"].to_numpy(float)
        S1 = base.equal_energy_align(S, float(d.mean()))
        _, _, fit = base.portfolio_nnls(d, S1)
        S_home[c] = S
        dvec[c] = d
        resid[c] = np.maximum(d - fit, 0.0)
        s = demand[c]["s"].reindex(idx).ffill().bfill().to_numpy(float)
        surplus[c] = np.maximum(s - d, 0.0)
        # capacity-size scale: absolute annual-energy weight proportional to installed VRE MW
        wcap[c] = cap_tot.get(c, 0.0) / max(float(d.mean()), 1e-12)
    codes = list(S_home.keys())
    receivers_h = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    print(f"  usable countries = {len(codes)} | hyperscaler receivers = {len(receivers_h)}")

    Q_h = np.sum([surplus[r] for r in receivers_h], axis=0)        # hyperscaler surplus (T,)
    Q_all = np.sum([surplus[r] for r in codes], axis=0)            # all-country surplus (T,)
    Qw_h = np.sum([wcap[r] * surplus[r] for r in receivers_h], axis=0)  # capacity-weighted surplus

    def uncovered(c, mig):
        d_eff = dvec[c] - mig
        S2 = base.equal_energy_align(S_home[c], float(d_eff.mean()))
        _, share_f, _ = base.portfolio_nnls(d_eff, S2)
        return float(share_f)

    rows = []
    for phi in PHI:
        desired = {c: np.minimum(phi * dvec[c], resid[c]) for c in codes}
        D = np.vstack([desired[c] for c in codes])                  # (C,T)
        total_desire = D.sum(axis=0)
        total_desire_w = np.sum([wcap[c] * desired[c] for c in codes], axis=0)
        eps = 1e-12
        ratio_h = np.where(total_desire > eps, np.minimum(1.0, Q_h / np.maximum(total_desire, eps)), 0.0)
        ratio_all = np.where(total_desire > eps, np.minimum(1.0, Q_all / np.maximum(total_desire, eps)), 0.0)
        ratio_w = np.where(total_desire_w > eps, np.minimum(1.0, Qw_h / np.maximum(total_desire_w, eps)), 0.0)
        avail_h = np.minimum(total_desire, Q_h)
        migC = priority_fill(D, avail_h)

        shares = {k: [] for k in ("A", "B", "C", "D", "E")}
        wsum = 0.0
        wE = 0.0
        for i, c in enumerate(codes):
            own = surplus[c] if c in G.HYPERSCALER_EXTENDED else 0.0
            migA = np.minimum(desired[c], np.maximum(Q_h - own, 0.0))   # independent pool (upper bound)
            migB = desired[c] * ratio_h
            migCi = migC[i]
            migD = desired[c] * ratio_all
            migE = desired[c] * ratio_w
            uA, uB, uC, uD, uE = (uncovered(c, m) for m in (migA, migB, migCi, migD, migE))
            for k, v in zip("ABCDE", (uA, uB, uC, uD, uE)):
                shares[k].append(v)
            wsum += wcap[c]
            wE += wcap[c] * uE
        def med(k):
            return round(float(np.median(shares[k])) * 100, 2)
        def mean(k):
            return round(float(np.mean(shares[k])) * 100, 2)
        rec = {"phi": phi,
               "A_indep_pool_median": med("A"),
               "B_shared_prop_median": med("B"), "B_shared_prop_mean": mean("B"),
               "C_shared_priority_median": med("C"),
               "D_all_receivers_median": med("D"),
               "E_capwt_median": med("E"),
               "E_capwt_weighted_mean": round(wE / max(wsum, 1e-12) * 100, 2)}
        rows.append(rec)
        print(f"  phi={phi:.1f} | A={rec['A_indep_pool_median']:5.1f} "
              f"B={rec['B_shared_prop_median']:5.1f}(mean {rec['B_shared_prop_mean']:5.1f}) "
              f"C={rec['C_shared_priority_median']:5.1f} D={rec['D_all_receivers_median']:5.1f} "
              f"E={rec['E_capwt_median']:5.1f}(wmean {rec['E_capwt_weighted_mean']:5.1f})")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "r2_allocation_sensitivity.csv", index=False)
    print("\n=== allocation/surplus sensitivity (global reach tau=500 ms), median uncovered % ===")
    print(out.to_string(index=False))
    (OUT / "r2_allocation_sensitivity_provenance.json").write_text(json.dumps({
        "purpose": "due-diligence range for routing benefit under a physically finite shared surplus budget",
        "variants": {"A": "independent non-rival pool (headline upper bound)",
                     "B": "shared finite budget, proportional rationing (neutral)",
                     "C": "shared finite budget, priority smallest-first (optimistic median)",
                     "D": "shared finite budget, proportional, receivers=all 104 countries",
                     "E": "shared finite budget, capacity(MW)-weighted; also reports capacity-weighted mean"},
        "reach": "global, tau=500 ms (worst case for double-counting)",
        "phi": PHI}, indent=2))
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
