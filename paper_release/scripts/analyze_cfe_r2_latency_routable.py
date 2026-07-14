#!/usr/bin/env python3
"""Result 2, REDESIGNED as a latency-tolerance x routable-fraction grid.

The demand proxy is Cloudflare HUMAN web traffic, i.e. interactive load that is
served in real time. The realistic flexibility is therefore NOT hour-scale job
deferral but (a) how much network LATENCY a request can tolerate -- which gates how
far it may be routed to a datacentre that has clean renewable surplus right now --
and (b) what FRACTION of the load is geographically routable at all (the rest is
pinned local by data residency / stateful sessions / sub-50 ms real-time SLAs).

  X axis = latency tolerance tau (ms): a request may be served at any site whose
           round-trip time RTT(home, site) <= tau. RTT uses the measured-calibrated
           fibre model RTT = 2*(km/200 + 20 ms)*1.4 (Luo & Yang; cross-validated
           against the Azure region-to-region P50 latency matrix). At the country
           scale RTT spans ~5 ms (in-country) to ~340 ms (antipodal), so tau maps to
           reach: 50 ms in-country, ~100 ms ~3000 km, ~300 ms near-global, >=500 ms global.
  Y axis = routable fraction phi (0..1): the share of demand that may be served
           remotely; (1-phi) stays on the home grid.

Supply is held NATIONAL (each country runs on its own station portfolio). In every
deficit hour the routable share is migrated SAME-HOUR to latency-feasible receiver
countries (hyperscaler-capable) that have clean surplus. Cell metric is unchanged:
uncovered share U = sum_t max(d~(t)-s(t),0)/sum_t d~(t). The (phi=0) row and the
(tau=50 ms, no cross-border reach) column both reproduce the home baseline.

Output: data_globalsites_stations_expanded/
    r2_latency_grid.csv, r2_latency_median.csv, r2_latency_reach.csv
"""
from __future__ import annotations
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
    """Max great-circle reach (km) for a latency budget tau (inverse of rtt_ms)."""
    km = SPEED * (tau / (2.0 * WAN) - SWITCH)
    return max(km, 0.0)


def precompute_pools(demand, demand_meta, smeta):
    codes = [c for c in demand_meta["cf_location"] if c in demand and c in smeta]
    idx = demand[codes[0]]["d"].index
    surplus = {c: np.maximum(demand[c]["s"].reindex(idx).ffill().bfill().to_numpy()
                             - demand[c]["d"].reindex(idx).ffill().bfill().to_numpy(), 0.0)
               for c in codes}
    pools = {}
    for c in codes:
        hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
        pools[c] = {tau: np.zeros(len(idx)) for tau in TAU_MS}
        for r in codes:
            if r == c or r not in G.HYPERSCALER_EXTENDED:
                continue
            rtt = rtt_ms(base.great_circle_km(hlon, hlat, smeta[r]["lon"], smeta[r]["lat"]))
            for tau in TAU_MS:
                if rtt <= tau:
                    pools[c][tau] = pools[c][tau] + surplus[r]
    return pools, idx


def evaluate(c, tau, phi, demand, S_home, pools):
    d = demand[c]["d"].to_numpy()
    S = S_home[c]
    S1 = base.equal_energy_align(S, float(d.mean()))
    _, share0, fit = base.portfolio_nnls(d, S1)
    if phi <= 0:
        return share0, share0, 0.0
    resid = np.maximum(d - fit, 0.0)
    pool = pools[c][tau]
    mig = np.maximum(np.minimum(phi * d, np.minimum(resid, pool)), 0.0)
    d_eff = d - mig
    S2 = base.equal_energy_align(S, float(d_eff.mean()))
    _, share_f, _ = base.portfolio_nnls(d_eff, S2)
    return share_f, share0, float(mig.sum() / max(d.sum(), 1e-12))


def main():
    print("loading expanded inputs (104 demand + 461 stations) ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)
    S_home = {}
    for c in demand:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        S_home[c] = S if S.size else None
    pools, _ = precompute_pools(demand, demand_meta, smeta)
    countries = [c for c in demand if S_home[c] is not None]
    print(f"  usable countries = {len(countries)}")

    rows = []
    for c in countries:
        for tau in TAU_MS:
            for phi in PHI:
                share_f, share0, mig = evaluate(c, tau, phi, demand, S_home, pools)
                if not np.isfinite(share_f):
                    continue
                rows.append({"country": c, "region": demand[c]["region"],
                             "continent": demand[c]["continent"], "tau_ms": tau,
                             "phi": phi, "reach_km": round(reach_km(tau)),
                             "uncovered_share": share_f, "uncovered_share_baseline": share0,
                             "migrated_share": mig})
    grid = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "r2_latency_grid.csv", index=False)

    med = (grid.groupby(["phi", "tau_ms"]).uncovered_share.median().unstack() * 100).round(1)
    med = med.reindex(index=PHI, columns=TAU_MS)
    med.to_csv(OUT / "r2_latency_median.csv")
    reach = pd.DataFrame({"tau_ms": TAU_MS, "reach_km": [round(reach_km(t)) for t in TAU_MS]})
    reach.to_csv(OUT / "r2_latency_reach.csv", index=False)

    print("\n=== median uncovered share (%) : rows = routable fraction phi, cols = latency tolerance tau (ms) ===")
    print("reach(km) per tau:", {t: round(reach_km(t)) for t in TAU_MS})
    print(med.to_string())
    base_corner = med.loc[0.0, 50]
    print(f"\nregression: (phi=0) median = {base_corner:.1f}%  | (phi=1.0, tau=500ms) = {med.loc[1.0,500]:.1f}%")
    existing = OUT / "r2_portfolio_scenarios.csv"
    if existing.exists():
        ex = pd.read_csv(existing)
        l0 = (ex[(ex.power_scope == "L0") & (ex.workload_scenario == "W0")].uncovered_share.median() * 100)
        print(f"existing L0_W0 median = {l0:.1f}%  delta = {base_corner - l0:+.2f} pp")
    (OUT / "r2_latency_provenance.json").write_text(json.dumps({
        "design": "R2 v2: x=latency tolerance tau(ms, RTT-gated), y=routable fraction phi; supply national; same-hour routing, no temporal deferral",
        "tau_ms": TAU_MS, "phi": PHI, "reach_km": {t: round(reach_km(t)) for t in TAU_MS},
        "rtt_model": "2*(km/200 + 20ms)*1.4 (Luo&Yang; Azure-measured P50 cross-validation)",
        "metric": "uncovered share U on routed demand vs home national portfolio",
    }, indent=2))
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
