#!/usr/bin/env python3
"""WAVEFORM-matching compute-migration FLOW + follow-the-sun phase data (Fig 2c, Fig 3).

This is the waveform analogue of analyze_cfe_r2_st_flows.py. The old pipeline
allocated each sender's migrated volume to receivers IN PROPORTION TO EACH
RECEIVER'S HOURLY SURPLUS (mig * surplus_r / pool), a quantity-based, double-counted
assignment. Here the sender's routable shortfall is instead matched to a STATIC
non-negative mix of reachable partners' clean-availability SHAPES (one mean-1
national shape per partner), and the per-receiver flow / landing solar hour are
read off the NNLS weights of that shape fit. Magnitude is divided out at every
step by equal-energy normalisation, so nothing is depleted and no surplus is
reused -- the assignment is conservation-neutral, consistent with the waveform
grid in analyze_cfe_r2_waveform_routing.py.

Computed at full routability (phi = 1), matching the "phi = 100% routable" panels.

Outputs (data_globalsites_stations_expanded/):
    r2_wf_compute_flows.csv          sender_country, receiver_country, weight, recv_lon, recv_lat, send_lon, send_lat
    r2_wf_compute_flows_tau{100,150,200}.csv
    r2_wf_followsun.csv              country, x_peak, y_solar, reach, phase_lag, migrated_share
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST

OUT = (G.ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "data_globalsites_stations_expanded")
FLOW_MIN = 0.0015   # drop negligible sender->receiver flows (fraction of sender demand)
TAU_MAP_LEVELS = [100, 150, 200]
SPEED = base.LIGHT_SPEED_FIBER_KM_PER_MS     # 200 km/ms
SWITCH = base.SWITCHING_OVERHEAD_MS          # 20 ms
WAN = base.WAN_INFLATION_FACTOR              # 1.4


def rtt_ms(km):
    if km <= 0:
        return 5.0
    return 2.0 * (km / SPEED + SWITCH) * WAN


def national_shape(S):
    """Collapse a T x K station basis (columns mean 1) to one mean-1 national shape."""
    v = S.mean(axis=1)
    m = float(v.mean())
    return v / m if m > 0 else v


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)
    utc_hour = idx.hour.to_numpy()

    codes = [c for c in demand if c in smeta]
    S_home, shape_home = {}, {}
    for c in codes:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        S_home[c] = S if S.size else None
        shape_home[c] = national_shape(S) if (S is not None and S.size) else None
    codes = [c for c in codes if S_home[c] is not None]

    receivers = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    phasor = {}
    for r in receivers:
        solar = (utc_hour + smeta[r]["lon"] / 15.0) % 24.0       # local solar hour at r
        phasor[r] = np.exp(1j * 2 * np.pi * solar / 24.0)

    def build_flows(tau=None):
        """Sender->receiver waveform flows (+ follow-sun phase) for latency budget tau.
        tau=None reaches every hyperscaler (global); a finite tau restricts each sender
        to receivers within its RTT-gated reach."""
        flow_rows, fs_rows = [], []
        for c in codes:
            d = demand[c]["d"].to_numpy(float)
            dsum = max(d.sum(), 1e-9)
            S1 = base.equal_energy_align(S_home[c], float(d.mean()))
            _, _, fit0 = base.portfolio_nnls(d, S1)
            resid = np.maximum(d - fit0, 0.0)         # phi = 1 -> r_route = resid
            if resid.sum() <= 0:
                continue

            recv = [r for r in receivers if r != c and shape_home[r] is not None]
            if tau is not None:
                hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
                recv = [r for r in recv if rtt_ms(base.great_circle_km(
                    hlon, hlat, smeta[r]["lon"], smeta[r]["lat"])) <= tau]
            if not recv:
                continue

            P = np.stack([shape_home[r] for r in recv], axis=1)          # T x R, cols mean 1
            P1 = base.equal_energy_align(P, float(resid.mean()))         # cols mean = resid.mean()
            w, _, fit_route = base.portfolio_nnls(resid, P1)            # NNLS shape fit
            covered = np.minimum(resid, np.maximum(fit_route, 0.0))
            tot_cov = float(covered.sum())
            if tot_cov <= 0:
                continue

            contr = P1 * w                                              # T x R, contr[t,r]=w_r*P1_r(t)
            fit_pos = np.where(fit_route > 0, fit_route, 1.0)
            frac = contr / fit_pos[:, None]                            # partner share of the fit at t
            cov_r = (covered[:, None] * frac).sum(axis=0)              # covered energy per partner

            for r, val in zip(recv, cov_r):
                weight = float(val) / dsum
                if weight >= FLOW_MIN:
                    flow_rows.append({"sender_country": c, "receiver_country": r,
                                      "weight": weight,
                                      "recv_lon": smeta[r]["lon"], "recv_lat": smeta[r]["lat"],
                                      "send_lon": smeta[c]["lon"], "send_lat": smeta[c]["lat"]})

            # energy-weighted receiver solar hour at which covered compute lands
            Phas = np.stack([phasor[r] for r in recv], axis=1)         # T x R
            wphasor = (contr * Phas).sum(axis=1)                       # per-hour weighted phasor
            sender_phasor = (covered * wphasor / fit_pos).sum()
            y_solar = (np.angle(sender_phasor) / (2 * np.pi) * 24.0) % 24.0
            lh = demand[c]["local_hour"].to_numpy().astype(int)
            x_peak = base.peak_local_hour(d, lh)
            phase_lag = base.diurnal_phase_lag(d, demand[c]["s"].to_numpy(), lh)
            reach = ((x_peak - y_solar + 12) % 24) - 12                # signed; + = runs west (earlier)
            fs_rows.append({"country": c, "x_peak": float(x_peak), "y_solar": float(y_solar),
                            "reach": float(reach), "phase_lag": float(phase_lag),
                            "migrated_share": tot_cov / dsum})
        return pd.DataFrame(flow_rows), pd.DataFrame(fs_rows)

    flows, fs = build_flows(tau=None)               # global reach -> Fig 3 global panel, Fig 2 (c)
    OUT.mkdir(parents=True, exist_ok=True)
    flows.to_csv(OUT / "r2_wf_compute_flows.csv", index=False)
    fs.to_csv(OUT / "r2_wf_followsun.csv", index=False)
    for tau in TAU_MAP_LEVELS:
        ft, _ = build_flows(tau=tau)
        ft.to_csv(OUT / f"r2_wf_compute_flows_tau{tau}.csv", index=False)
        print(f"pairing flows (tau={tau} ms): {len(ft)} edges, "
              f"{ft.sender_country.nunique()} senders -> {ft.receiver_country.nunique()} receivers")

    print(f"\nglobal flows: {len(flows)} edges, {flows.sender_country.nunique()} senders -> "
          f"{flows.receiver_country.nunique()} receivers")
    top = flows.groupby("receiver_country")["weight"].sum().sort_values(ascending=False).head(10)
    print("top compute-receiver hubs (sum of inbound weight):")
    print(top.round(3).to_string())
    b, a = np.polyfit(fs["x_peak"], fs["y_solar"], 1)
    print(f"\nfollow-sun fit: y_solar = {a:.2f} + {b:+.3f}*x_peak  (slope ~0 = pins at noon)")
    print(f"median receiver solar hour = {fs.y_solar.median():.1f}  "
          f"(evening-peak senders' median west reach = {fs[fs.x_peak>=17].reach.median():.1f} h)")
    print(f"median migrated (covered) share at phi=1 global = {fs.migrated_share.median()*100:.1f}%  "
          f"mean = {fs.migrated_share.mean()*100:.1f}%")
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
