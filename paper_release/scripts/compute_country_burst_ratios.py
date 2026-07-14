#!/usr/bin/env python3
"""Per-country d_burst/mean ratios from Cloudflare hourly demand.

The R3 P3 operator amplifies the top-1% of demand hours by gamma*lam, with the
empirical gamma derived from BurstGPT via:
    gamma_op = (gamma_emp - 1) / (d_burst / mean)

The denominator was previously hard-coded to 1.75 as a generic guess. Here we
compute it from the actual Cloudflare hourly human-volume proxy used as the R3
base demand series, on a per-country basis.

We report:
- median d_burst/mean across 104 countries (replaces the 1.75 constant)
- per-country distribution (P25, P50, P75) for SI Table S3
- the resulting per-country gamma_op when paired with BurstGPT's POOLED token-
  weighted hourly P99/mean (gamma_emp = 9.31), giving the operative gamma_op =
  4.84 used by the P3 operator; the per-slice high-burst value (gamma_emp ~=
  10.6, slices 1 & 3) is reported separately as the upper end of the within-
  trace band (see SI S2.5/S2.7).

Outputs
-------
reports/ai_inference_growth_curve_fit/country_burst_ratios.csv
reports/ai_inference_growth_curve_fit/country_burst_ratio_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEMAND = ROOT / "collective_attention_research_plan" / "data" / "cloudflare_global_demand" / "country_hourly_demand.csv.gz"
OUT_DIR = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"


def main() -> int:
    print("Loading Cloudflare hourly demand ...", flush=True)
    df = pd.read_csv(DEMAND, parse_dates=["timestamp_utc"])
    print(f"  rows={len(df):,}  countries={df.cf_location.nunique()}", flush=True)

    # Compute per-country: mean, median, P99, max, and the burst ratio
    rows = []
    for cc, g in df.groupby("cf_location"):
        d = g["human_volume_proxy"].to_numpy()
        d = d[d > 0]  # drop zero/null hours
        if len(d) < 100:
            continue
        mean = float(d.mean())
        if mean <= 0:
            continue
        rows.append({
            "country": cc,
            "n_hours": int(len(d)),
            "mean": mean,
            "median": float(np.median(d)),
            "p95": float(np.quantile(d, 0.95)),
            "p99": float(np.quantile(d, 0.99)),
            "max": float(d.max()),
            "d_burst_over_mean": float(np.quantile(d, 0.99) / mean),
            "d_p95_over_mean": float(np.quantile(d, 0.95) / mean),
            "d_max_over_mean": float(d.max() / mean),
        })
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "country_burst_ratios.csv", index=False)

    # Summary stats across countries
    ratios = out["d_burst_over_mean"].to_numpy()
    summary = {
        "n_countries": int(len(out)),
        "d_burst_over_mean": {
            "p25": float(np.percentile(ratios, 25)),
            "p50": float(np.percentile(ratios, 50)),
            "mean": float(np.mean(ratios)),
            "p75": float(np.percentile(ratios, 75)),
            "min": float(ratios.min()),
            "max": float(ratios.max()),
        },
        "description": (
            "P99 / mean of hourly human-traffic demand per country, from "
            "Cloudflare Radar 2025-05-02 to 2026-05-01. Replaces the prior "
            "hard-coded constant d_burst/mean = 1.75 used in the gamma_op "
            "derivation. The cross-country median is the recommended single "
            "value; the distribution is reported for SI Table S3."
        ),
    }
    # Pair the per-country d_burst/mean with BurstGPT's POOLED token-weighted
    # hourly P99/mean (gamma_emp). This is the operative value used by the P3
    # operator (gamma_op = 4.84); load it from the v2 fit rather than hard-
    # coding so this stays the single source of truth and cannot drift from the
    # operator default. The per-slice high-burst value (gamma_emp ~= 10.6,
    # slices 1 & 3) is reported separately as the upper end of the within-trace
    # band (bracketed by the gamma sensitivity in SI S7).
    v2 = json.loads((OUT_DIR / "burstgpt_workload_shape_params_v2.json").read_text())
    gamma_emp = float(v2["burst_stats_hourly"]["tokens"]["p99_over_mean"])  # pooled ~= 9.31
    gamma_emp_high_slice = 10.6  # per-slice (1 & 3) upper end, see SI S2.5/S2.7
    summary["gamma_op_from_burstgpt"] = {
        "gamma_emp_pooled_tokens": gamma_emp,
        "p25": float((gamma_emp - 1.0) / np.percentile(ratios, 75)),  # high ratio -> low gamma
        "p50": float((gamma_emp - 1.0) / np.percentile(ratios, 50)),
        "p75": float((gamma_emp - 1.0) / np.percentile(ratios, 25)),
        "recommended_default": float((gamma_emp - 1.0) / np.percentile(ratios, 50)),
        "gamma_emp_high_burst_slice": gamma_emp_high_slice,
        "gamma_op_high_burst_slice": float((gamma_emp_high_slice - 1.0) / np.percentile(ratios, 50)),
    }
    (OUT_DIR / "country_burst_ratio_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nCross-country d_burst/mean distribution:")
    print(out["d_burst_over_mean"].describe().round(3).to_string())
    print("\nRecommended gamma_op (using median d_burst/mean):",
          summary["gamma_op_from_burstgpt"]["recommended_default"])
    print(f"\nWrote {OUT_DIR / 'country_burst_ratios.csv'}")
    print(f"Wrote {OUT_DIR / 'country_burst_ratio_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
