#!/usr/bin/env python3
"""Result 2, REDESIGNED as a spatiotemporal compute-flexibility grid.

The old R2 matrix was (electricity reach L0-L3) x (workload-migration share W0-W3).
This redesign holds the ELECTRICITY supply fixed at the national level (L0: each
country matches against its own station portfolio) and resolves the WORKLOAD-
MIGRATION lever into its two physical degrees of freedom:

  Y axis = maximum compute MIGRATION DISTANCE D (spatial flexibility)
           D0 = in-country only (0 km)        -> no migration
           D1 = metro/national  (<= 500 km)
           D2 = continental     (<= 3000 km)
           D3 = intercontinental/global (no cap)
      A job's deferrable+latency-tolerant share (classes B/C/D ~= 60%) may be
      relocated SAME-HOUR to a hyperscaler country with clean surplus that lies
      within D, gated by per-class fiber latency L(km)=km/200+20 ms (Luo & Yang).
      Class A (interactive ~300 ms) always stays local.

  X axis = DELAY GRACE TIME g (temporal flexibility / deadline slack), in hours
           g in {0, 1, 3, 6, 12, 24, 168}, 24 h = production anchor (Google CICS,
           Meta Carbon Explorer). The temporally-flexible share phi (~40%, Google
           Borg 24h-SLO) is redistributed within non-overlapping windows of length
           g toward locally cleaner hours, conserving energy within each window
           (virtual-storage / Demand-Side-Management form, Riepin-Brown-Zavala
           2024; Google Virtual Capacity Curves). Headline uses lossless (eta=1).

Cell metric (unchanged from current R2, so cells stay comparable):
    uncovered share U = sum_t max(d~(t) - s~(t), 0) / sum_t d~(t)
where d~ is the flexibility-adjusted demand and s~ is the home L0 NNLS portfolio.

Solve order per cell (temporal-first, then spatial, then re-portfolio), to avoid
double-counting surplus:
    (1) pass-1 NNLS home portfolio on d -> clean target fit
    (2) temporal DSM: redistribute phi*d within g-windows under the clean target
    (3) re-fit, take residual deficit
    (4) spatial routing: move movable residual to within-D surplus (same hour)
    (5) pass-2 NNLS on the effective demand -> U

Regression guarantee: the (D0, g=0) corner reproduces today's L0_W0 uncovered
share (no migration, no deferral).

Outputs: reports/cfe_geographic_portfolio_ai/data_globalsites_stations_expanded/
    r2_spatiotemporal_grid.csv        (per-country, full D x g grid at phi=0.40)
    r2_spatiotemporal_phisweep.csv    (per-country D x g grid for phi in sweep)
    r2_spatiotemporal_median.csv      (cross-country median pivot, headline)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST

OUT = (G.ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "data_globalsites_stations_expanded")

# ----------------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------------
DIST_RINGS = [("D0", 0.0), ("D1", 500.0), ("D2", 3000.0), ("D3", float("inf"))]
GRACE_HOURS = [0, 1, 3, 6, 12, 24, 168]
PHI_DEFAULT = 0.40
PHI_SWEEP = [0.20, 0.30, 0.40, 0.50]

# Temporally-flexible share is redistributed; movable (spatial) share = B+C+D.
MOVABLE_SHARE = base.WORKLOAD_SHARE["B"] + base.WORKLOAD_SHARE["C"] + base.WORKLOAD_SHARE["D"]  # 0.60
ETA_DSM = 1.0  # lossless headline (within-window net-zero); eta<1 is an SI sensitivity


# ----------------------------------------------------------------------------
# Temporal deferral (deadline-constrained virtual-storage dispatch)
# ----------------------------------------------------------------------------
def temporal_defer(d, s_clean, grace_h, phi, eta=1.0):
    """Defer the temporally-flexible share phi of demand to follow the clean
    supply curve, subject to a per-job deadline slack of grace_h hours.

    Causal virtual-storage / deadline-slack model (CarbonScaler slack
    T-(t+l) + Google-VCC daily conservation + Riepin-Brown-Zavala storage):
    a job arriving at hour t may run anywhere in [t, t+grace_h]. Each hour the
    deferrable cohort phi*d[t] enters a FIFO reservoir; it is served opportunist-
    ically up to the clean headroom (s_clean - non-deferrable floor) and the
    per-hour depth cap, and is FORCE-served when its deadline (age = grace_h) is
    reached. Deferral is forward-only (a job cannot run before it arrives) and
    every deferred MWh is served within its window, so energy is conserved.
    Headline uses lossless eta=1; eta<1 charges a round-trip loss on the deferred
    share (an SI sensitivity). Returns the reshaped demand array d~."""
    T = len(d)
    if grace_h <= 0 or phi <= 0:
        return d.copy()
    g = int(grace_h)
    floor = (1.0 - phi) * d                     # non-deferrable load stays in place
    head = np.maximum(s_clean - floor, 0.0)     # clean headroom under supply
    cap_extra = 2.0 * phi * d                   # depth cap: d~ <= (1+phi) d
    loss = (1.0 - eta)                          # round-trip loss applied on discharge
    d_tilde = floor.copy()
    # FIFO reservoir of [arrival_hour, remaining_energy]
    pend = []                                   # list used as a queue (head = pend[0])
    qi = 0                                       # index of queue head (avoid pop(0) cost)
    for t in range(T):
        e_in = phi * d[t]
        if e_in > 0:
            pend.append([t, e_in])
        served = 0.0
        # force-serve any cohort that has reached its deadline (age == grace_h)
        while qi < len(pend) and pend[qi][0] <= t - g:
            served += pend[qi][1]
            pend[qi][1] = 0.0
            qi += 1
        # opportunistic serve, oldest-first, up to clean headroom and depth cap
        room = min(head[t], cap_extra[t]) - served
        while room > 1e-12 and qi < len(pend):
            take = pend[qi][1] if pend[qi][1] <= room else room
            pend[qi][1] -= take
            served += take
            room -= take
            if pend[qi][1] <= 1e-12:
                qi += 1
        d_tilde[t] = floor[t] + served * (1.0 - loss)
    # flush any unserved remainder into the final hours (end-of-series deadline)
    rem = sum(pend[i][1] for i in range(qi, len(pend)))
    if rem > 1e-9:
        d_tilde[T - 1] += rem * (1.0 - loss)
    return d_tilde


# ----------------------------------------------------------------------------
# Spatial surplus pools (precomputed once per country x distance ring)
# ----------------------------------------------------------------------------
def precompute_surplus_pools(demand, demand_meta, smeta):
    """For each demand country and each distance ring, sum the same-hour clean
    surplus of all hyperscaler receivers within that ring. B/C/D share the same
    receiver set within a ring (fiber latency only excludes interactive A), so a
    single surplus series per (country, ring) suffices."""
    codes = [c for c in demand_meta["cf_location"] if c in demand and c in smeta]
    idx = demand[codes[0]]["d"].index
    surplus = {c: np.maximum(demand[c]["s"].reindex(idx).ffill().bfill().to_numpy()
                             - demand[c]["d"].reindex(idx).ffill().bfill().to_numpy(), 0.0)
               for c in codes}
    pools = {}
    for c in codes:
        hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
        pools[c] = {}
        for ring, dmax in DIST_RINGS:
            if dmax <= 0:
                continue
            acc = np.zeros(len(idx))
            for r in codes:
                if r == c or r not in G.HYPERSCALER_EXTENDED:
                    continue
                dkm = base.great_circle_km(hlon, hlat, smeta[r]["lon"], smeta[r]["lat"])
                if dkm > dmax:
                    continue
                # latency feasibility: movable classes B/C/D all pass within these
                # rings; only interactive A (always local) would be excluded.
                acc = acc + surplus[r]
            pools[c][ring] = acc
    return pools, idx


# ----------------------------------------------------------------------------
# Grid evaluation (supply fixed at L0 home stations)
# ----------------------------------------------------------------------------
def run_grid(phi, demand, countries, station_curves, station_meta, smeta,
             surplus_pools, S_home_cache):
    """Full D x g grid for one deferrable fraction phi. Temporal deferral is
    computed once per (country, grace) and reused across distance rings."""
    rows = []
    for c in countries:
        d_base = demand[c]["d"].to_numpy()
        S = S_home_cache[c]
        # pass 1: home portfolio on the original demand -> clean target
        S1 = base.equal_energy_align(S, float(d_base.mean()))
        _, share0, fit0 = base.portfolio_nnls(d_base, S1)
        for g in GRACE_HOURS:
            # (2) temporal deferral against the clean target (once per country x grace)
            d_tmp = temporal_defer(d_base, fit0, g, phi, eta=ETA_DSM)
            # (3) re-fit -> residual deficit after deferral
            S2 = base.equal_energy_align(S, float(d_tmp.mean()))
            _, share_t, fit_t = base.portfolio_nnls(d_tmp, S2)
            resid = np.maximum(d_tmp - fit_t, 0.0)
            deferred = float(0.5 * np.abs(d_tmp - d_base).sum() / max(d_base.sum(), 1e-12))
            movable_demand = MOVABLE_SHARE * d_tmp
            for ring, dmax in DIST_RINGS:
                # (4) spatial routing within distance ring D (same-hour foreign surplus)
                if dmax > 0:
                    mig = np.maximum(np.minimum(movable_demand,
                                                np.minimum(resid, surplus_pools[c][ring])), 0.0)
                    d_eff = d_tmp - mig
                    migrated = float(mig.sum() / max(d_tmp.sum(), 1e-12))
                else:
                    d_eff = d_tmp
                    migrated = 0.0
                # (5) pass-2 NNLS on the effective demand -> cell metric U
                S3 = base.equal_energy_align(S, float(d_eff.mean()))
                _, share_f, _ = base.portfolio_nnls(d_eff, S3)
                if not np.isfinite(share_f):
                    continue
                rows.append({"country": c, "region": demand[c]["region"],
                             "continent": demand[c]["continent"],
                             "dist_ring": ring, "dist_km": (None if np.isinf(dmax) else dmax),
                             "grace_h": g, "phi": phi,
                             "uncovered_share": share_f,
                             "uncovered_share_baseline": share0,
                             "uncovered_share_temporal_only": share_t,
                             "migrated_share": migrated, "deferred_share": deferred})
    return pd.DataFrame(rows)


def main():
    print("loading expanded inputs (104-country demand + 461 stations) ...")
    panel, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)
    print(f"  demand={len(demand)}  stations={len(station_curves)}")

    print("precomputing home (L0) station bases + surplus pools ...")
    S_home_cache = {}
    for c in demand:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, labels = ST.station_basis(station_curves, pool, idx)
        S_home_cache[c] = S if S.size else None
    surplus_pools, _ = precompute_surplus_pools(demand, demand_meta, smeta)

    countries = [c for c in demand if S_home_cache[c] is not None]
    print(f"  usable countries (have home stations) = {len(countries)}")

    # ---- main grid: phi = 0.40 over the full D x g matrix ----
    print(f"main grid: {len(countries)} countries x {len(DIST_RINGS)} dist x {len(GRACE_HOURS)} grace ...")
    grid = run_grid(PHI_DEFAULT, demand, countries, station_curves, station_meta,
                    smeta, surplus_pools, S_home_cache)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "r2_spatiotemporal_grid.csv", index=False)

    # cross-country median pivot (headline heatmap values, %)
    med = (grid.groupby(["dist_ring", "grace_h"]).uncovered_share.median().unstack() * 100).round(1)
    med = med.reindex(index=[r for r, _ in DIST_RINGS], columns=GRACE_HOURS)
    med.to_csv(OUT / "r2_spatiotemporal_median.csv")

    # ---- phi sweep (for SI) over the full grid ----
    print(f"phi sweep: {PHI_SWEEP} ...")
    srows = []
    for phi in PHI_SWEEP:
        sub = grid if phi == PHI_DEFAULT else run_grid(
            phi, demand, countries, station_curves, station_meta, smeta,
            surplus_pools, S_home_cache)
        srows.append(sub)
    sweep = pd.concat(srows, ignore_index=True)
    sweep.to_csv(OUT / "r2_spatiotemporal_phisweep.csv", index=False)

    # ---- report + regression check ----
    print("\n=== headline median uncovered share (%) : rows = migration distance, cols = grace hours ===")
    print(med.to_string())

    base_corner = med.loc["D0", 0]
    print(f"\nregression check: (D0, g=0) median = {base_corner:.1f}%  (should match existing L0_W0)")
    existing = OUT / "r2_portfolio_scenarios.csv"
    if existing.exists():
        ex = pd.read_csv(existing)
        l0w0 = (ex[(ex.power_scope == "L0") & (ex.workload_scenario == "W0")]
                .uncovered_share.median() * 100)
        print(f"  existing L0_W0 median = {l0w0:.1f}%   delta = {base_corner - l0w0:+.2f} pp")

    print("\nmin-U corner (D3, g=168) median = {:.1f}%".format(med.loc["D3", 168]))
    print("provenance -> writing")
    (OUT / "r2_spatiotemporal_provenance.json").write_text(json.dumps({
        "design": "R2 redesigned: y=migration distance (D0-D3), x=delay grace (h); supply fixed national L0",
        "dist_rings_km": {r: (None if np.isinf(d) else d) for r, d in DIST_RINGS},
        "grace_hours": GRACE_HOURS, "phi_default": PHI_DEFAULT, "phi_sweep": PHI_SWEEP,
        "movable_share_BCD": MOVABLE_SHARE, "eta_dsm": ETA_DSM,
        "deferral": "within-window water-filling DSM, net-zero energy per window (Riepin-Brown-Zavala; Google VCC)",
        "metric": "uncovered share U on flexibility-adjusted demand vs home L0 NNLS portfolio",
        "solve_order": "pass1 NNLS -> temporal DSM -> refit/residual -> spatial route within D -> pass2 NNLS",
    }, indent=2))
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
