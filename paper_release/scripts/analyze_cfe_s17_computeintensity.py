#!/usr/bin/env python3
"""S17.4 robustness: how the headline mismatch and routing residual move if the demand
shape is reweighted by COMPUTE INTENSITY rather than raw request count.

The Cloudflare proxy models the interactive electricity-use shape as
    d_c(t) proportional to sum_k n_k(t) e_k
(request rate of class k times mean energy per request). The headline uses the raw
aggregate request rate R_c(t) = sum_k n_k(t), i.e. it assumes the hour-to-hour class
mix (hence mean energy per request) is stationary. Reviewer Major-5 asks what happens
if high-compute-intensity requests (LLM conversation, code, image/video generation) are
MORE concentrated in the interactive evening peak than low-compute requests (search,
static pages, CDN hits) -- then per-hour mean energy e(t) rises with local interactive
intensity and the energy-weighted shape sharpens; the opposite composition flattens it.

We parameterise this by a single compute-intensity concentration exponent kappa: model
per-request energy as e(t) ~ (R(t)/<R>)^kappa, so the energy-weighted demand shape is
    d_kappa(t) proportional to R(t) * (R(t)/<R>)^kappa  = R(t)^(1+kappa),
renormalised to preserve each country's annual energy total.
  kappa = 0    -> raw request count (the headline).
  kappa > 0    -> high-compute requests concentrated at the interactive peak (sharper).
  kappa < 0    -> low-compute/CDN-dominated (flatter; complements the flat-baseload sweep).

Recompute, exactly as in the flat-baseload sensitivity:
  R1  national equal-energy mismatch (capacity-weighted VRE), 104-country median;
  R2  global routing residual (phi=1, tau=500 ms, waveform/timing match), 104-country median.

Output: <OUT>/s17_computeintensity_sensitivity.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST
import analyze_cfe_r2_latency_routable as R2L

OUT = R2L.OUT
KAPPAS = [-0.5, -0.25, 0.0, 0.5, 1.0]


def reweight(d: np.ndarray, kappa: float) -> np.ndarray:
    """Energy-weighted shape d * (d/<d>)^kappa, renormalised to the same total."""
    if kappa == 0.0:
        return d
    m = float(d.mean())
    if m <= 0:
        return d
    w = np.power(np.clip(d / m, 1e-9, None), kappa)
    dk = d * w
    tot = dk.sum()
    return dk * (d.sum() / tot) if tot > 0 else d


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
        svec[c] = demand[c]["s"].reindex(idx).ffill().bfill().to_numpy(float)
    codes = list(S_home.keys())
    receivers = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    print(f"  usable countries = {len(codes)} | hyperscaler receivers = {len(receivers)}")

    nat_shape = {}
    for r in receivers:
        v = S_home[r].mean(axis=1)
        m = float(v.mean())
        nat_shape[r] = v / m if m > 0 else v

    rows = []
    for kappa in KAPPAS:
        d_k = {c: reweight(dvec[c], kappa) for c in codes}
        r1_shares, routed_shares = [], []
        for c in codes:
            d = d_k[c]
            # R1: national equal-energy mismatch
            s = svec[c]
            s_al = s * (d.mean() / s.mean()) if s.mean() > 0 else s
            r1 = float(np.maximum(d - s_al, 0.0).sum() / max(d.sum(), 1e-12))
            r1_shares.append(r1)
            # R2: global routing residual (phi=1, tau=500), waveform match
            S = S_home[c]
            S1 = base.equal_energy_align(S, float(d.mean()))
            _, _, fit = base.portfolio_nnls(d, S1)
            resid = np.maximum(d - fit, 0.0)
            cols = [nat_shape[r] for r in receivers if r != c]
            if cols and resid.sum() > 0:
                P1 = base.equal_energy_align(np.stack(cols, axis=1), float(resid.mean()))
                _, _, fit_route = base.portfolio_nnls(resid, P1)
                covered = np.minimum(resid, np.maximum(fit_route, 0.0))
                share_f = float((resid - covered).sum() / max(d.sum(), 1e-12))
            else:
                share_f = float(resid.sum() / max(d.sum(), 1e-12))
            routed_shares.append(share_f)

        rows.append({"kappa_compute_intensity": kappa,
                     "R1_national_mismatch_median": round(float(np.median(r1_shares)) * 100, 1),
                     "R2_global_routing_residual_median": round(float(np.median(routed_shares)) * 100, 1)})
        print(f"  kappa={kappa:+.2f}: R1 mismatch={rows[-1]['R1_national_mismatch_median']:.1f}%  "
              f"R2 routing residual={rows[-1]['R2_global_routing_residual_median']:.1f}%")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "s17_computeintensity_sensitivity.csv", index=False)
    print("\n=== S17.4 compute-intensity sensitivity (104-country median, %) ===")
    print(out.to_string(index=False))
    print(f"\nwrote {OUT / 's17_computeintensity_sensitivity.csv'}")


if __name__ == "__main__":
    main()
