#!/usr/bin/env python3
"""Empirical fit of AI-inference workload shape from BurstGPT v2.0 traces.

Replaces the stylized P2 (Gaussian @ 13h, sigma=3h) and P3 (top-1% x gamma=3)
operators with parameters fitted to a real Azure OpenAI GPT serving trace.

Inputs
------
collective_attention_research_plan/data/burstgpt_trace/BurstGPT_without_fails_*.csv
  Columns: Timestamp (s, local-tz, t=0 -> 00:00 day 1), Model, Request tokens,
  Response tokens, Total tokens, Log Type.

What we fit
-----------
1. Diurnal profile of inference load. Two definitions:
     (a) request count per hour-of-day (request rate)
     (b) total tokens per hour-of-day (energy proxy; output token generation
         dominates LLM serving compute)
   Per the BurstGPT README the timestamp is "calibrated to the local time
   zone", so hour-of-day directly indexes user-local hour.

2. Diurnal Gaussian fit -> empirical (peak_hour, sigma_h, FWHM_h).
   Replaces P2's stylized (13h, sigma=3h).

3. Burst amplification gamma. For each 5-min bin we compute total tokens; we
   define burst hours as bins whose hour-of-day belongs to the top-q quantile
   of hourly mean rate, and gamma_eff = (P99 / median) of tokens in those
   high-traffic hours - i.e. how much the top-1% spikes exceed the typical
   high-traffic-hour level. We report multiple definitions for transparency.

Outputs
-------
collective_attention_research_plan/reports/ai_inference_growth_curve_fit/
  burstgpt_diurnal_fit.csv          -- 24 rows: hour, share_count, share_tokens
  burstgpt_workload_shape_params.json -- empirical P2 / P3 parameters
  burstgpt_burst_distribution.csv   -- per-quantile burst stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "collective_attention_research_plan" / "data" / "burstgpt_trace"
OUT_DIR = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trace() -> pd.DataFrame:
    files = sorted(TRACE_DIR.glob("BurstGPT_without_fails_*.csv"))
    if not files:
        raise FileNotFoundError(f"no BurstGPT files in {TRACE_DIR}")
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=["Timestamp", "Total tokens", "Response tokens"])
        df["file"] = f.name
        parts.append(df)
        print(f"  loaded {f.name}: {len(df):,} rows", flush=True)
    out = pd.concat(parts, ignore_index=True)
    return out


def gaussian(h, peak, sigma, base, amp):
    return base + amp * np.exp(-((h - peak) ** 2) / (2 * sigma ** 2))


def fit_diurnal(df: pd.DataFrame) -> dict:
    """Return per-hour shares + Gaussian fit params."""
    df = df.copy()
    df["hour"] = (df["Timestamp"] // 3600).astype(int) % 24
    g = df.groupby("hour").agg(
        n=("Timestamp", "size"),
        tokens=("Total tokens", "sum"),
        resp_tokens=("Response tokens", "sum"),
    ).reindex(range(24)).fillna(0)
    g["share_count"] = g["n"] / g["n"].sum()
    g["share_tokens"] = g["tokens"] / g["tokens"].sum()
    g["share_resp_tokens"] = g["resp_tokens"] / g["resp_tokens"].sum()

    out = {"per_hour": g.reset_index().to_dict("list")}
    for col in ("share_count", "share_tokens", "share_resp_tokens"):
        y = g[col].to_numpy()
        h = np.arange(24)
        # initial guesses: peak at argmax, sigma=3, baseline=min, amp=max-min
        p0 = (float(h[np.argmax(y)]), 3.0, float(y.min()), float(y.max() - y.min()))
        try:
            popt, _ = curve_fit(gaussian, h, y, p0=p0, maxfev=20000)
            peak, sigma, base, amp = popt
            # wrap peak into [0, 24)
            peak = peak % 24
            sigma = abs(sigma)
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
            out[f"fit_{col}"] = {
                "peak_hour": float(peak),
                "sigma_h": float(sigma),
                "fwhm_h": float(fwhm),
                "baseline_share": float(base),
                "amplitude": float(amp),
            }
        except Exception as e:
            out[f"fit_{col}"] = {"error": str(e)}
    return out


def fit_burst(df: pd.DataFrame) -> dict:
    """5-min token-volume bins -> burst quantile structure.

    The R3 P3 operator amplifies the top-1% of hourly bins by (gamma * lambda)
    on top of their baseline (1 + gamma*lambda). To map an empirical burst ratio
    onto the gamma parameter, the relevant reference is the MEAN (not the
    median) of the trace - because the operator's lambda-share of mean already
    accounts for the typical level, and gamma quantifies how much the AI-added
    mass spikes ABOVE its own mean at top-1% hours.
    """
    df = df.copy()
    df["bin5m"] = (df["Timestamp"] // 300).astype(np.int64)
    df["hour"] = (df["Timestamp"] // 3600).astype(int) % 24
    by_bin = df.groupby("bin5m").agg(
        tokens=("Total tokens", "sum"),
        hour=("hour", "first"),
    )
    hour_mean = df.groupby("hour")["Total tokens"].sum() / df["bin5m"].nunique()
    busy_hours = hour_mean.sort_values(ascending=False).head(6).index.tolist()
    busy_bins = by_bin[by_bin["hour"].isin(busy_hours)]["tokens"]
    all_bins = by_bin["tokens"]

    # Hourly aggregation - the natural unit for the P3 operator (which works on
    # an hourly demand series).
    df["bin1h"] = (df["Timestamp"] // 3600).astype(np.int64)
    by_hr = df.groupby("bin1h").agg(tokens=("Total tokens", "sum"))
    hr_vals = by_hr["tokens"].to_numpy()

    quants = {
        # 5-min bin stats (instantaneous burst)
        "bin5m_p50": float(busy_bins.quantile(0.50)),
        "bin5m_mean": float(busy_bins.mean()),
        "bin5m_p99": float(busy_bins.quantile(0.99)),
        "bin5m_p999": float(busy_bins.quantile(0.999)),
        "bin5m_max": float(busy_bins.max()),
        "bin5m_p99_over_p50": float(busy_bins.quantile(0.99) / max(busy_bins.quantile(0.50), 1.0)),
        "bin5m_p99_over_mean": float(busy_bins.quantile(0.99) / max(busy_bins.mean(), 1.0)),
        # Whole-trace 5-min bin stats
        "whole_5m_p99_over_mean": float(all_bins.quantile(0.99) / max(all_bins.mean(), 1.0)),
        # 1-hour bin stats - this is the natural granularity of P3
        "bin1h_mean": float(hr_vals.mean()),
        "bin1h_median": float(np.median(hr_vals)),
        "bin1h_p99": float(np.quantile(hr_vals, 0.99)),
        "bin1h_p999": float(np.quantile(hr_vals, 0.999)),
        "bin1h_max": float(hr_vals.max()),
        # *** Empirical gamma for the P3 operator: top-1% hour excess over mean
        # P3 adds gamma*lambda*d at burst hour. If burst hour AI load is
        # gamma_emp * (lambda*mean_d), excess = (gamma_emp - 1) * lambda*mean_d.
        # Equating with operator excess: d_burst * gamma_op * lambda ~ (gamma_emp - 1) * lambda * mean_d
        # gamma_op = (gamma_emp - 1) * mean_d / d_burst.
        # For human-traffic d, d_burst (P99 of hourly) ~ 1.5-2.0 x mean_d, so
        # gamma_op ~ (gamma_emp - 1) / 1.75. We report the empirical gamma_emp
        # and the converted operator gamma_op (using factor 1.75 for human d).
        "burst_hours_local": busy_hours,
    }
    gemp = quants["bin1h_p99"] / max(quants["bin1h_mean"], 1.0)
    quants["gamma_emp_p99_over_mean_hourly"] = float(gemp)
    quants["gamma_op_equivalent"] = float((gemp - 1.0) / 1.75)  # see derivation above
    quants["d_burst_to_mean_assumption"] = 1.75

    dist = pd.DataFrame({
        "quantile": [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0],
        "hourly_tokens": [
            float(np.quantile(hr_vals, q)) if q < 1.0 else float(hr_vals.max())
            for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        ],
        "hourly_over_mean": [
            float(np.quantile(hr_vals, q) / hr_vals.mean()) if q < 1.0
            else float(hr_vals.max() / hr_vals.mean())
            for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        ],
    })

    # Distribution for write-out
    dist = pd.DataFrame({
        "quantile": [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0],
        "busy_hour_bin_tokens": [
            float(busy_bins.quantile(q)) if q < 1.0 else float(busy_bins.max())
            for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        ],
        "whole_trace_bin_tokens": [
            float(all_bins.quantile(q)) if q < 1.0 else float(all_bins.max())
            for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        ],
    })
    return {"summary": quants, "distribution": dist}


def main() -> int:
    print("Loading BurstGPT trace ...", flush=True)
    df = load_trace()
    print(f"Total: {len(df):,} rows", flush=True)

    print("Fitting diurnal profile ...", flush=True)
    d = fit_diurnal(df)
    # write per-hour shares
    per_hour = pd.DataFrame(d.pop("per_hour"))
    per_hour.to_csv(OUT_DIR / "burstgpt_diurnal_fit.csv", index=False)
    print(per_hour[["hour", "share_count", "share_tokens"]].round(4).to_string(index=False), flush=True)

    print("\nFitting burst distribution ...", flush=True)
    b = fit_burst(df)
    b["distribution"].to_csv(OUT_DIR / "burstgpt_burst_distribution.csv", index=False)
    print(json.dumps(b["summary"], indent=2), flush=True)

    out_params = {
        "source": "BurstGPT v2.0 (Azure OpenAI GPT serving trace, ~10M req)",
        "trace_files": [p.name for p in sorted(TRACE_DIR.glob("BurstGPT_without_fails_*.csv"))],
        "diurnal_fit": {k: v for k, v in d.items() if k.startswith("fit_")},
        "burst": b["summary"],
    }
    (OUT_DIR / "burstgpt_workload_shape_params.json").write_text(
        json.dumps(out_params, indent=2)
    )
    print(f"\nWrote {OUT_DIR / 'burstgpt_workload_shape_params.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
