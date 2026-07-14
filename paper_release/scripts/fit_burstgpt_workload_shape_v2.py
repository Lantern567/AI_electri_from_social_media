#!/usr/bin/env python3
"""BurstGPT v2.0 workload-shape fit, v2: pooled across all 3 trace files,
bootstrap CIs, temporal stationarity, and total-tokens vs response-tokens
weighting comparison.

Inputs
------
collective_attention_research_plan/data/burstgpt_trace/BurstGPT_without_fails_{1,2,3}.csv
  ~10.1M cleaned Azure OpenAI GPT serving requests across ~4 months total.
  Timestamps are seconds from 00:00 day 1, calibrated to user-local timezone
  (per README).

Fitting protocol
----------------
1. Diurnal aggregation:
   - hour-of-day = (timestamp // 3600) % 24
   - For weighting in [count, total_tokens, response_tokens]:
     - normalize hourly volume to share (sum = 1)
     - fit Gaussian g(h) = base + amp * exp(-(h - mu)^2 / (2 sigma^2))
   - Report (mu, sigma, FWHM) for each weighting.

2. Bootstrap CI:
   - Block-resample 5-min bins with replacement, 500 iterations.
   - Within each iteration, recompute hourly share + Gaussian fit.
   - Report 2.5 / 50 / 97.5 percentiles for (mu, sigma).

3. Burst quantile structure:
   - 1-hour bins (the natural granularity of the R3 operator).
   - Compute P50, P90, P95, P99, P999 / mean.
   - Empirical gamma_emp = bin1h P99 / mean (operator-mapped via separate
     per-country d_burst/mean from compute_country_burst_ratios.py).

4. Temporal stationarity:
   - Slice by file (each is a separate ~2-4 month window).
   - Refit per slice, report drift in (mu, sigma, gamma_emp).

Outputs
-------
reports/ai_inference_growth_curve_fit/
  burstgpt_diurnal_fit_v2.csv        24 rows x N weightings
  burstgpt_workload_shape_params_v2.json
  burstgpt_bootstrap_ci.csv          500 bootstrap samples for (mu, sigma)
  burstgpt_stationarity.csv          per-file fit comparison
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

RNG = np.random.default_rng(20260601)
N_BOOTSTRAP = 500


def gaussian(h, peak, sigma, base, amp):
    return base + amp * np.exp(-((h - peak) ** 2) / (2 * sigma ** 2))


def load_all() -> pd.DataFrame:
    files = sorted(TRACE_DIR.glob("BurstGPT_without_fails_*.csv"))
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=["Timestamp", "Total tokens", "Response tokens"])
        df["src"] = f.stem.split("_")[-1]  # 1 / 2 / 3
        parts.append(df)
        print(f"  loaded {f.name}: {len(df):,} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def hourly_shares(df: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    """24-vector of normalized shares (sum = 1)."""
    if weight_col is None:
        s = df.groupby("hour").size()
    else:
        s = df.groupby("hour")[weight_col].sum()
    s = s.reindex(range(24)).fillna(0).to_numpy()
    tot = s.sum()
    return s / tot if tot > 0 else s


def fit_gaussian(y: np.ndarray) -> dict:
    h = np.arange(24)
    p0 = (float(h[np.argmax(y)]), 3.0, float(y.min()), float(y.max() - y.min()))
    try:
        popt, _ = curve_fit(gaussian, h, y, p0=p0, maxfev=20000)
        peak, sigma, base, amp = popt
        peak = float(peak) % 24
        sigma = abs(float(sigma))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        return {"peak_hour": peak, "sigma_h": sigma, "fwhm_h": float(fwhm),
                "baseline_share": float(base), "amplitude": float(amp),
                "ok": True}
    except Exception as e:
        return {"error": str(e), "ok": False}


def burst_stats(df: pd.DataFrame, weight_col: str) -> dict:
    df = df.copy()
    df["bin1h"] = (df["Timestamp"] // 3600).astype(np.int64)
    by_hr = df.groupby("bin1h")[weight_col].sum().to_numpy()
    mean = float(by_hr.mean())
    return {
        "n_bins": int(len(by_hr)),
        "mean": mean,
        "median": float(np.median(by_hr)),
        "p90": float(np.quantile(by_hr, 0.90)),
        "p95": float(np.quantile(by_hr, 0.95)),
        "p99": float(np.quantile(by_hr, 0.99)),
        "p999": float(np.quantile(by_hr, 0.999)),
        "max": float(by_hr.max()),
        "p99_over_mean": float(np.quantile(by_hr, 0.99) / mean) if mean > 0 else float("nan"),
        "p999_over_mean": float(np.quantile(by_hr, 0.999) / mean) if mean > 0 else float("nan"),
    }


def bootstrap_ci(df: pd.DataFrame, weight_col: str | None, n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Block-bootstrap by 5-min bin to get CI on (peak_hour, sigma_h).

    Resamples 5-min token-volume bins with replacement and refits the Gaussian
    to the per-hour share derived from each resample.
    """
    df = df.copy()
    df["bin5m"] = (df["Timestamp"] // 300).astype(np.int64)
    df["hour"] = (df["Timestamp"] // 3600).astype(int) % 24
    # Pre-aggregate to (bin5m, hour, weight) — much faster to bootstrap.
    if weight_col is None:
        agg = df.groupby(["bin5m", "hour"]).size().reset_index(name="w")
    else:
        agg = df.groupby(["bin5m", "hour"])[weight_col].sum().reset_index(name="w")
    bins = agg["bin5m"].unique()
    n_bins = len(bins)
    rows = []
    bin_to_idx = pd.Series(np.arange(n_bins), index=bins)
    agg["bin_idx"] = bin_to_idx[agg["bin5m"]].to_numpy()
    # Build sparse matrix-like representation: list per bin of (hour, w)
    per_bin = agg.groupby("bin_idx").agg(list)  # hour list + w list per bin
    hour_lists = per_bin["hour"].to_list()
    w_lists = per_bin["w"].to_list()

    for it in range(n_boot):
        sample_idx = RNG.integers(0, n_bins, size=n_bins)
        hourly = np.zeros(24)
        for i in sample_idx:
            for h, w in zip(hour_lists[i], w_lists[i]):
                hourly[h] += w
        tot = hourly.sum()
        if tot <= 0:
            continue
        y = hourly / tot
        f = fit_gaussian(y)
        if f.get("ok"):
            rows.append({"iter": it, "peak_hour": f["peak_hour"],
                         "sigma_h": f["sigma_h"], "fwhm_h": f["fwhm_h"]})
        if (it + 1) % 50 == 0:
            print(f"    bootstrap {it+1}/{n_boot}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    print("Loading all 3 BurstGPT trace files ...", flush=True)
    df = load_all()
    df["hour"] = (df["Timestamp"] // 3600).astype(int) % 24
    print(f"Total: {len(df):,} requests, {df.Timestamp.max()/86400:.1f} day span", flush=True)

    # --- Diurnal share + Gaussian fit by 3 weight definitions ---
    out_per_hour = pd.DataFrame({"hour": np.arange(24)})
    fits = {}
    for label, col in [("count", None), ("tokens", "Total tokens"),
                       ("resp_tokens", "Response tokens")]:
        y = hourly_shares(df, col)
        out_per_hour[f"share_{label}"] = y
        fits[label] = fit_gaussian(y)
    out_per_hour.to_csv(OUT_DIR / "burstgpt_diurnal_fit_v2.csv", index=False)
    print("\nGaussian fits (full trace):")
    for label, f in fits.items():
        if f.get("ok"):
            print(f"  {label:12s}: peak={f['peak_hour']:.2f}h  sigma={f['sigma_h']:.2f}h  "
                  f"FWHM={f['fwhm_h']:.2f}h")

    # --- Burst quantile structure ---
    bursts = {}
    for label, col in [("tokens", "Total tokens"), ("resp_tokens", "Response tokens")]:
        bursts[label] = burst_stats(df, col)
    print("\nBurst stats (1-h bins):")
    for label, b in bursts.items():
        print(f"  {label:12s}: mean={b['mean']:.0f}  P99={b['p99']:.0f}  "
              f"P99/mean={b['p99_over_mean']:.2f}  P999/mean={b['p999_over_mean']:.2f}")

    # --- Bootstrap CI for tokens weighting ---
    print(f"\nBootstrap CI ({N_BOOTSTRAP} iters, tokens weighting) ...", flush=True)
    boot = bootstrap_ci(df, "Total tokens", n_boot=N_BOOTSTRAP)
    boot.to_csv(OUT_DIR / "burstgpt_bootstrap_ci.csv", index=False)
    ci = {
        "peak_hour_p2_5": float(np.percentile(boot["peak_hour"], 2.5)),
        "peak_hour_p50": float(np.percentile(boot["peak_hour"], 50)),
        "peak_hour_p97_5": float(np.percentile(boot["peak_hour"], 97.5)),
        "sigma_h_p2_5": float(np.percentile(boot["sigma_h"], 2.5)),
        "sigma_h_p50": float(np.percentile(boot["sigma_h"], 50)),
        "sigma_h_p97_5": float(np.percentile(boot["sigma_h"], 97.5)),
        "n_successful_iters": int(len(boot)),
    }
    print(f"  peak_hour: 95% CI [{ci['peak_hour_p2_5']:.2f}, {ci['peak_hour_p97_5']:.2f}]")
    print(f"  sigma_h:   95% CI [{ci['sigma_h_p2_5']:.2f}, {ci['sigma_h_p97_5']:.2f}]")

    # --- Temporal stationarity: fit per file slice ---
    print("\nStationarity (per-file Gaussian fit, tokens):")
    rows = []
    for src, sub in df.groupby("src"):
        y = hourly_shares(sub, "Total tokens")
        f = fit_gaussian(y)
        b = burst_stats(sub, "Total tokens")
        if f.get("ok"):
            rows.append({
                "trace_slice": src,
                "n_requests": int(len(sub)),
                "duration_days": float((sub.Timestamp.max() - sub.Timestamp.min()) / 86400),
                "peak_hour": f["peak_hour"],
                "sigma_h": f["sigma_h"],
                "fwhm_h": f["fwhm_h"],
                "p99_over_mean": b["p99_over_mean"],
            })
            print(f"  slice {src}: peak={f['peak_hour']:.2f}h sigma={f['sigma_h']:.2f}h "
                  f"P99/mean={b['p99_over_mean']:.2f}")
    stationarity = pd.DataFrame(rows)
    stationarity.to_csv(OUT_DIR / "burstgpt_stationarity.csv", index=False)

    # --- Apply per-country d_burst/mean for gamma_op recommendation ---
    burst_summary_path = OUT_DIR / "country_burst_ratio_summary.json"
    if burst_summary_path.exists():
        cb = json.loads(burst_summary_path.read_text())
        d_med = cb["d_burst_over_mean"]["p50"]
        d_q25 = cb["d_burst_over_mean"]["p25"]
        d_q75 = cb["d_burst_over_mean"]["p75"]
    else:
        d_med = d_q25 = d_q75 = 1.75

    gamma_emp_tokens = bursts["tokens"]["p99_over_mean"]
    gamma_emp_resp = bursts["resp_tokens"]["p99_over_mean"]
    gamma_op_band = {
        "gamma_emp_tokens": float(gamma_emp_tokens),
        "gamma_emp_resp_tokens": float(gamma_emp_resp),
        "d_burst_over_mean_country_p50": float(d_med),
        "d_burst_over_mean_country_p25": float(d_q25),
        "d_burst_over_mean_country_p75": float(d_q75),
        "gamma_op_recommended": float((gamma_emp_tokens - 1.0) / d_med),
        "gamma_op_low": float((gamma_emp_tokens - 1.0) / d_q75),
        "gamma_op_high": float((gamma_emp_tokens - 1.0) / d_q25),
    }
    print("\nOperator gamma (with per-country d_burst/mean band):")
    print(f"  recommended: {gamma_op_band['gamma_op_recommended']:.2f}")
    print(f"  IQR-derived band: [{gamma_op_band['gamma_op_low']:.2f}, "
          f"{gamma_op_band['gamma_op_high']:.2f}]")

    # --- Persist all parameters ---
    out_params = {
        "source": "BurstGPT v2.0 (Azure OpenAI GPT serving trace)",
        "trace_files": sorted([f.name for f in TRACE_DIR.glob("BurstGPT_without_fails_*.csv")]),
        "n_requests_total": int(len(df)),
        "trace_span_days": float(df.Timestamp.max() / 86400),
        "gaussian_fit": fits,
        "bootstrap_ci_95pct_tokens": ci,
        "burst_stats_hourly": bursts,
        "operator_gamma_with_country_band": gamma_op_band,
        "stationarity_per_slice": stationarity.to_dict("records"),
    }
    (OUT_DIR / "burstgpt_workload_shape_params_v2.json").write_text(
        json.dumps(out_params, indent=2)
    )
    print(f"\nWrote {OUT_DIR / 'burstgpt_workload_shape_params_v2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
