#!/usr/bin/env python3
"""Compute-migration FLOW + follow-the-sun phase data for the redesigned Fig 2
panels (b) chord, (d) map, (e) scatter.

At the spatial upper bound (D3 global, grace g=0 so the spatial lever is isolated
from temporal deferral), each demand country routes its movable compute (classes
B/C/D ~60%) same-hour to hyperscaler RECEIVER countries that have clean renewable
surplus at that hour. We disaggregate the migrated volume per receiver (in
proportion to each receiver's hourly surplus) to get a sender->receiver compute
flow matrix, and the energy-weighted RECEIVER solar hour at which the migrated
compute runs (the compute analogue of follow-the-sun: jobs land where it is
currently sunny).

Outputs (data_globalsites_stations_expanded/):
    r2_st_compute_flows.csv   sender_country, receiver_country, weight, recv_lon, recv_lat, send_lon, send_lat
    r2_st_followsun.csv       country, x_peak, y_solar, reach, phase_lag, migrated_share
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G
import analyze_cfe_global_sites_stations as ST

OUT = (G.ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "data_globalsites_stations_expanded")
MOVABLE_SHARE = base.WORKLOAD_SHARE["B"] + base.WORKLOAD_SHARE["C"] + base.WORKLOAD_SHARE["D"]
FLOW_MIN = 0.0015   # drop negligible sender->receiver flows (fraction of sender demand)

# Sender->receiver pairing is also written at several INTERMEDIATE latency-tolerance
# levels for the dedicated pairing figure (Fig 3): a sender may only reach receivers whose
# RTT(home, site) <= tau. tau=100 ms -> ~3 Mm (regional), 150 ms -> ~6.7 Mm (cross-
# continental, not antipodal), 200 ms -> ~10 Mm (half the globe). The global-reach flows
# (tau=None) and follow-the-sun phase are written unchanged for Fig 2 panel (c).
TAU_MAP_LEVELS = [100, 150, 200]
SPEED = base.LIGHT_SPEED_FIBER_KM_PER_MS     # 200 km/ms
SWITCH = base.SWITCHING_OVERHEAD_MS          # 20 ms
WAN = base.WAN_INFLATION_FACTOR              # 1.4


def rtt_ms(km):
    if km <= 0:
        return 5.0
    return 2.0 * (km / SPEED + SWITCH) * WAN


def main():
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    station_curves, station_meta = ST.load_stations(idx, smeta)
    utc_hour = idx.hour.to_numpy()

    codes = [c for c in demand if c in smeta]
    # per-country home station basis (L0 supply)
    S_home = {}
    for c in codes:
        pool = ST.station_pool(c, "L0", station_meta, smeta)
        S, _ = ST.station_basis(station_curves, pool, idx)
        S_home[c] = S if S.size else None
    codes = [c for c in codes if S_home[c] is not None]

    # receiver pool: hyperscaler countries, with hourly surplus and a solar-hour phasor
    receivers = [r for r in codes if r in G.HYPERSCALER_EXTENDED]
    surplus = {}
    phasor = {}
    for r in receivers:
        s_r = demand[r]["s"].reindex(idx).ffill().bfill().to_numpy()
        d_r = demand[r]["d"].reindex(idx).ffill().bfill().to_numpy()
        surplus[r] = np.maximum(s_r - d_r, 0.0)
        solar = (utc_hour + smeta[r]["lon"] / 15.0) % 24.0           # local solar hour at r
        phasor[r] = np.exp(1j * 2 * np.pi * solar / 24.0)

    def build_flows(tau=None):
        """Sender->receiver flows (and follow-sun phase) for a latency budget tau.
        tau=None pools every hyperscaler globally (the spatial upper bound); a finite
        tau restricts each sender to receivers within its RTT-gated reach."""
        flow_rows, fs_rows = [], []
        for c in codes:
            d_base = demand[c]["d"].to_numpy()
            S = S_home[c]
            S1 = base.equal_energy_align(S, float(d_base.mean()))
            _, _, fit0 = base.portfolio_nnls(d_base, S1)
            resid0 = np.maximum(d_base - fit0, 0.0)

            recv = [r for r in receivers if r != c]
            if tau is not None:
                hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
                recv = [r for r in recv if rtt_ms(base.great_circle_km(
                    hlon, hlat, smeta[r]["lon"], smeta[r]["lat"])) <= tau]
            pool = np.zeros(len(idx))
            wphasor = np.zeros(len(idx), dtype=complex)     # surplus-weighted solar phasor per hour
            for r in recv:
                pool += surplus[r]
                wphasor += surplus[r] * phasor[r]
            mig = np.maximum(np.minimum(MOVABLE_SHARE * d_base, np.minimum(resid0, pool)), 0.0)
            tot_mig = float(mig.sum())
            if tot_mig <= 0:
                continue
            denom = np.where(pool > 0, pool, 1.0)

            # per-receiver flow = migrated volume allocated by receiver surplus share
            for r in recv:
                f = float((mig * surplus[r] / denom).sum())
                if f / max(d_base.sum(), 1e-9) >= FLOW_MIN:
                    flow_rows.append({"sender_country": c, "receiver_country": r,
                                      "weight": f / max(d_base.sum(), 1e-9),
                                      "recv_lon": smeta[r]["lon"], "recv_lat": smeta[r]["lat"],
                                      "send_lon": smeta[c]["lon"], "send_lat": smeta[c]["lat"]})

            # energy-weighted receiver solar hour at which migrated compute runs
            sender_phasor = (mig * wphasor / denom).sum()
            y_solar = (np.angle(sender_phasor) / (2 * np.pi) * 24.0) % 24.0
            lh = demand[c]["local_hour"].to_numpy().astype(int)
            x_peak = base.peak_local_hour(d_base, lh)
            s_home = demand[c]["s"].to_numpy()
            phase_lag = base.diurnal_phase_lag(d_base, s_home, lh)
            reach = ((x_peak - y_solar + 12) % 24) - 12     # signed; positive = runs west (earlier)
            fs_rows.append({"country": c, "x_peak": float(x_peak), "y_solar": float(y_solar),
                            "reach": float(reach), "phase_lag": float(phase_lag),
                            "migrated_share": tot_mig / max(d_base.sum(), 1e-9)})
        return pd.DataFrame(flow_rows), pd.DataFrame(fs_rows)

    flows, fs = build_flows(tau=None)               # global reach -> Fig 3 global panel, Fig 2 (c)
    OUT.mkdir(parents=True, exist_ok=True)
    flows.to_csv(OUT / "r2_st_compute_flows.csv", index=False)
    fs.to_csv(OUT / "r2_st_followsun.csv", index=False)
    for tau in TAU_MAP_LEVELS:                       # intermediate reach -> Fig 3 pairing panels
        ft, _ = build_flows(tau=tau)
        ft.to_csv(OUT / f"r2_st_compute_flows_tau{tau}.csv", index=False)
        print(f"pairing flows (tau={tau} ms): {len(ft)} edges, "
              f"{ft.sender_country.nunique()} senders -> {ft.receiver_country.nunique()} receivers")

    print(f"flows: {len(flows)} edges, {flows.sender_country.nunique()} senders -> "
          f"{flows.receiver_country.nunique()} receivers")
    top = flows.groupby("receiver_country")["weight"].sum().sort_values(ascending=False).head(10)
    print("top compute-receiver hubs (sum of inbound weight):")
    print(top.round(3).to_string())
    b, a = np.polyfit(fs["x_peak"], fs["y_solar"], 1)
    print(f"\nfollow-sun fit: y_solar = {a:.2f} + {b:+.3f}*x_peak  (slope ~0 = pins at noon)")
    print(f"median receiver solar hour = {fs.y_solar.median():.1f}  "
          f"(evening-peak senders' median west reach = "
          f"{fs[fs.x_peak>=17].reach.median():.1f} h)")
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
