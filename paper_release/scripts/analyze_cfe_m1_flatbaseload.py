#!/usr/bin/env python3
"""M1 robustness: how the headline mismatch and routing potential move if the real
data-centre load is FLATTER than the human-traffic demand proxy.

The Cloudflare human-traffic proxy captures the human-facing, latency-bearing
INTERACTIVE / inference-serving load (the routable fraction). Flat training/batch
baseload (caching, CDN cache hits, model training) is non-diurnal and is modelled
separately (the lambda_train flat share). To bound how much a flatter true load
would change the picture, we blend each country's demand with a mean-preserving flat
component, d_beta = (1-beta)*d_human + beta*<d_human>, for beta in {0, 0.25, 0.5}
(beta = the flat-baseload share), and recompute:
  R1  national equal-energy mismatch (the 40% headline, capacity-weighted VRE);
  R2  global routing potential (phi=1, tau=500 ms independent-pool, the 16.7% upper bound).

A flatter load (higher beta) is LESS anti-correlated with midday solar, so both the
mismatch and the routing gap fall monotonically -- i.e. the human-traffic numbers are
an UPPER BOUND for the routable interactive load and conservative w.r.t. flat baseload.

Output: data_globalsites_stations_expanded/m1_flatbaseload_sensitivity.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST
import analyze_cfe_r2_latency_routable as R2L

OUT = R2L.OUT
BETAS = [0.0, 0.25, 0.50]


def main():
    print("loading expanded inputs (104 demand + stations) ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)

    S_home, dvec, svec = {}, {}, {}
    for c in demand:
        if c not in smeta:
            continue
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        if S is None or not S.size:
            continue
        S_home[c] = S
        dvec[c] = demand[c]["d"].to_numpy(float)
        svec[c] = demand[c]["s"].reindex(idx).ffill().bfill().to_numpy(float)  # capacity-weighted VRE, equal-energy
    codes = list(S_home.keys())
    receivers = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    print(f"  usable countries = {len(codes)} | hyperscaler receivers = {len(receivers)}")

    # partner national clean shapes (station-based, consistent with the R2 waveform grid)
    nat_shape = {}
    for r in receivers:
        v = S_home[r].mean(axis=1)
        m = float(v.mean())
        nat_shape[r] = v / m if m > 0 else v

    rows = []
    for beta in BETAS:
        d_b = {c: (1 - beta) * dvec[c] + beta * float(dvec[c].mean()) for c in codes}
        r1_shares, routed_shares = [], []
        for c in codes:
            d = d_b[c]
            # --- R1: national equal-energy mismatch (capacity-weighted VRE) ---
            s = svec[c]
            s_al = s * (d.mean() / s.mean()) if s.mean() > 0 else s
            r1 = float(np.maximum(d - s_al, 0.0).sum() / max(d.sum(), 1e-12))
            r1_shares.append(r1)
            # --- R2: global routing FLOOR (phi=1, tau=500), WAVEFORM (timing) match ---
            S = S_home[c]
            S1 = base.equal_energy_align(S, float(d.mean()))
            _, _, fit = base.portfolio_nnls(d, S1)
            resid = np.maximum(d - fit, 0.0)                      # phi=1 -> routable shortfall = resid
            cols = [nat_shape[r] for r in receivers if r != c]
            if cols and resid.sum() > 0:
                P1 = base.equal_energy_align(np.stack(cols, axis=1), float(resid.mean()))
                _, _, fit_route = base.portfolio_nnls(resid, P1)
                covered = np.minimum(resid, np.maximum(fit_route, 0.0))
                share_f = float((resid - covered).sum() / max(d.sum(), 1e-12))
            else:
                share_f = float(resid.sum() / max(d.sum(), 1e-12))
            routed_shares.append(share_f)

        rows.append({"beta_flat_share": beta,
                     "R1_national_mismatch_median": round(float(np.median(r1_shares)) * 100, 1),
                     "R2_global_routing_potential_median": round(float(np.median(routed_shares)) * 100, 1)})
        print(f"  beta={beta:.2f}: R1 national mismatch={rows[-1]['R1_national_mismatch_median']:.1f}%  "
              f"R2 global routing potential={rows[-1]['R2_global_routing_potential_median']:.1f}%")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "m1_flatbaseload_sensitivity.csv", index=False)
    print("\n=== M1 flat-baseload sensitivity (104-country median, %) ===")
    print(out.to_string(index=False))
    print(f"\nwrote {OUT / 'm1_flatbaseload_sensitivity.csv'}")


if __name__ == "__main__":
    main()
