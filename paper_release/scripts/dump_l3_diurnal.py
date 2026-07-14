#!/usr/bin/env python3
"""Dump per-country diurnal curves at L3 power scope for fig 2(d) upgrade.

For each demand country, runs the same two-pass scope-consistent procedure used
in the canonical Result 2:
  1. NNLS-fit the L3 station portfolio to the country's original demand
     (pass-1, equivalent to L3 x W0).
  2. Migrate W3-eligible deferrable workload (B+C+D classes) out of the
     residual deficit hours to RTT-feasible hyperscaler receivers with clean
     surplus, producing the post-migration demand d_eff.
  3. Re-fit the L3 portfolio to d_eff (pass-2, equivalent to L3 x W3).

Then folds the four hourly series to local-hour means and writes one row per
(country, local_hour) with columns:
    country, local_hour,
    demand_orig, demand_migrated,
    portfolio_w0_fit, portfolio_w3_fit

These let fig 2(d) show the NNLS-weighted L3 portfolio shape (rather than the
misleading unweighted global mean) AND the post-W3-migration demand shape.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_cfe_geographic_portfolio_ai as base  # noqa: E402
import analyze_cfe_global_sites as G  # noqa: E402
from analyze_cfe_global_sites_stations import (  # noqa: E402
    ETA, load_stations, station_basis, station_pool,
)

ROOT = Path(__file__).resolve().parents[2]
_REPORT = ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai"
# EXPANDED (104-country) output feeds fig2(d) via data_globalsites/; the legacy
# 31-country run writes to data/.
OUT = (_REPORT / "data_globalsites" / "r2_l3_diurnal.csv"
       if os.environ.get("EXPANDED_DEMAND", "1") != "0"
       else _REPORT / "data" / "r2_l3_diurnal.csv")


def main() -> int:
    expanded = bool(os.environ.get("EXPANDED_DEMAND", "1") != "0")
    print(f"loading inputs ({'EXPANDED 104' if expanded else 'original 31'}) ...", flush=True)
    if expanded:
        _, demand, demand_meta, _, smeta = G.load_inputs_expanded()
        idx = demand[next(iter(demand))]["d"].index
        station_curves, station_meta = load_stations(idx, smeta)
        rtt = G.build_rtt_distance(demand_meta, smeta)
    else:
        _, demand, demand_meta, _, smeta = G.load_inputs()
        idx = demand[next(iter(demand))]["d"].index
        station_curves, station_meta = load_stations(idx, smeta)
        rtt = base.build_rtt_matrix_azure(list(demand_meta["cf_location"]))

    rows = []
    for k, cc in enumerate(demand, 1):
        home_idx = demand[cc]["d"].index
        d_base = demand[cc]["d"]
        lh = demand[cc]["local_hour"].reindex(home_idx).astype(int).to_numpy()
        pool = station_pool(cc, "L3", station_meta, smeta)
        S, _labels = station_basis(station_curves, pool, home_idx)
        if S.size == 0:
            continue
        for j, sid in enumerate(_labels):
            if station_meta[sid]["iso2"] != cc:
                S[:, j] = S[:, j] * ETA
        S1 = base.equal_energy_align(S, float(d_base.mean()))
        w1, _share1, fit1 = base.portfolio_nnls(d_base.to_numpy(), S1)
        resid = np.maximum(d_base.to_numpy() - fit1, 0.0)
        d_eff, _mig = G.migrate_residual(cc, "W3", demand, demand_meta, rtt, d_base, resid)
        S2 = base.equal_energy_align(S, float(d_eff.mean()))
        w2, _share2, fit2 = base.portfolio_nnls(d_eff.to_numpy(), S2)
        tmp = pd.DataFrame({
            "local_hour": lh,
            "demand_orig": d_base.to_numpy(),
            "demand_migrated": d_eff.to_numpy(),
            "portfolio_w0_fit": fit1,
            "portfolio_w3_fit": fit2,
        })
        g = tmp.groupby("local_hour", as_index=False).mean()
        g.insert(0, "country", cc)
        rows.append(g)
        if k % 20 == 0 or k == len(demand):
            print(f"  [{k}/{len(demand)}]", flush=True)

    if not rows:
        raise SystemExit("no countries processed")
    df = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df['country'].nunique()} countries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
