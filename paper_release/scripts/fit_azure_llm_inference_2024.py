#!/usr/bin/env python3
"""Cross-validation fit on Azure LLM Inference Trace 2024 (separate from BurstGPT).

Inputs
------
collective_attention_research_plan/data/azure_llm_inference_2024/
  AzureLLMInferenceTrace_conv_1week.csv  (~27M LLM serving conversation requests)
  AzureLLMInferenceTrace_code_1week.csv  (~17M LLM serving code-assist requests)

Both: TIMESTAMP (datetime in UTC), ContextTokens (input), GeneratedTokens (output).
Collected May 10-19 2024 from two Azure-hosted LLM inference services.

Difference from BurstGPT
------------------------
- Timestamps here are UTC absolute datetimes (BurstGPT was relative seconds in
  user-local time). So peak hour in UTC reflects the dominant user population's
  local timezone. For the WORKLOAD SHAPE (sigma, burst structure) the
  comparison is still valid.
- Two service types: 'conv' (conversational) and 'code' (code assistant).

Outputs
-------
reports/ai_inference_growth_curve_fit/azure_llm_inference_fit.csv
reports/ai_inference_growth_curve_fit/azure_llm_inference_params.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "collective_attention_research_plan" / "data" / "azure_llm_inference_2024"
OUT_DIR = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"


def gaussian(h, peak, sigma, base, amp):
    return base + amp * np.exp(-((h - peak) ** 2) / (2 * sigma ** 2))


def fit_gaussian(y: np.ndarray) -> dict:
    h = np.arange(24)
    p0 = (float(h[np.argmax(y)]), 3.0, float(y.min()), float(y.max() - y.min()))
    try:
        popt, _ = curve_fit(gaussian, h, y, p0=p0, maxfev=20000)
        peak, sigma, base, amp = popt
        peak = float(peak) % 24
        sigma = abs(float(sigma))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        return {"peak_hour_utc": peak, "sigma_h": sigma, "fwhm_h": float(fwhm),
                "baseline_share": float(base), "amplitude": float(amp), "ok": True}
    except Exception as e:
        return {"error": str(e), "ok": False}


def fit_service(name: str, csv_path: Path) -> dict:
    print(f"\nLoading {csv_path.name} (this may take a minute) ...", flush=True)
    # Use efficient parsing: read TIMESTAMP, parse only hour, drop original
    df = pd.read_csv(csv_path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
    df["ts"] = pd.to_datetime(df["TIMESTAMP"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["hour"] = df["ts"].dt.hour.astype(int)
    df["TotalTokens"] = df["ContextTokens"] + df["GeneratedTokens"]
    span_days = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
    print(f"  rows={len(df):,} span_days={span_days:.2f}", flush=True)

    # Hourly aggregation per 1-hour bin (UTC absolute)
    df["bin1h"] = df["ts"].dt.floor("h")
    by_hr = df.groupby("bin1h").agg(
        n=("ts", "size"),
        tot=("TotalTokens", "sum"),
        resp=("GeneratedTokens", "sum"),
    )
    # Hourly shares (24 hours of day, normalized)
    per_hr = df.groupby("hour").agg(
        n=("ts", "size"),
        tot=("TotalTokens", "sum"),
        resp=("GeneratedTokens", "sum"),
    ).reindex(range(24)).fillna(0)
    per_hr["share_count"] = per_hr["n"] / per_hr["n"].sum()
    per_hr["share_tokens"] = per_hr["tot"] / per_hr["tot"].sum()
    per_hr["share_resp_tokens"] = per_hr["resp"] / per_hr["resp"].sum()

    # Gaussian fits
    fits = {}
    for label, col in [("count", "share_count"), ("tokens", "share_tokens"),
                       ("resp_tokens", "share_resp_tokens")]:
        fits[label] = fit_gaussian(per_hr[col].to_numpy())

    # Burst stats
    bursts = {}
    for label, col in [("tokens", "tot"), ("resp_tokens", "resp")]:
        v = by_hr[col].to_numpy()
        mean = float(v.mean())
        bursts[label] = {
            "n_bins": int(len(v)),
            "mean": mean,
            "p99": float(np.quantile(v, 0.99)),
            "p999": float(np.quantile(v, 0.999)),
            "p99_over_mean": float(np.quantile(v, 0.99) / mean) if mean > 0 else float("nan"),
            "p999_over_mean": float(np.quantile(v, 0.999) / mean) if mean > 0 else float("nan"),
        }
    return {"name": name, "n_requests": int(len(df)),
            "span_days": float(span_days), "diurnal_per_hour": per_hr.reset_index().to_dict("list"),
            "gaussian_fits_utc": fits, "burst_stats_hourly": bursts}


def main() -> int:
    services = {}
    for name, fname in [("conv", "AzureLLMInferenceTrace_conv_1week.csv"),
                         ("code", "AzureLLMInferenceTrace_code_1week.csv")]:
        services[name] = fit_service(name, TRACE_DIR / fname)

    print("\n=== Azure LLM Inference 2024 fit summary ===")
    for sname, s in services.items():
        print(f"\n{sname}: {s['n_requests']:,} req over {s['span_days']:.1f} days")
        for label, f in s["gaussian_fits_utc"].items():
            if f.get("ok"):
                print(f"  {label:12s}: peak_utc={f['peak_hour_utc']:.2f}h  "
                      f"sigma={f['sigma_h']:.2f}h  FWHM={f['fwhm_h']:.2f}h")
        for label, b in s["burst_stats_hourly"].items():
            print(f"  burst {label:8s}: P99/mean={b['p99_over_mean']:.2f}  "
                  f"P999/mean={b['p999_over_mean']:.2f}")

    # Combined fit (conv + code) on share_tokens
    print("\n=== Combined (conv + code) Gaussian fit on token share ===")
    combined_per_hour = np.zeros(24)
    for s in services.values():
        per = pd.DataFrame(s["diurnal_per_hour"])
        combined_per_hour += per["tot"].to_numpy()
    combined_share = combined_per_hour / combined_per_hour.sum()
    fit_combined = fit_gaussian(combined_share)
    if fit_combined.get("ok"):
        print(f"  Combined: peak_utc={fit_combined['peak_hour_utc']:.2f}h  "
              f"sigma={fit_combined['sigma_h']:.2f}h  FWHM={fit_combined['fwhm_h']:.2f}h")

    # Cross-validation comparison with BurstGPT
    bg_params_path = OUT_DIR / "burstgpt_workload_shape_params_v2.json"
    if bg_params_path.exists():
        bg = json.loads(bg_params_path.read_text())
        bg_tokens = bg["gaussian_fit"]["tokens"]
        bg_burst = bg["burst_stats_hourly"]["tokens"]
        print("\n=== Cross-validation vs BurstGPT v2 (tokens-weighted) ===")
        print(f"  BurstGPT (local-tz): peak={bg_tokens['peak_hour']:.2f}h "
              f"sigma={bg_tokens['sigma_h']:.2f}h  P99/mean={bg_burst['p99_over_mean']:.2f}")
        if fit_combined.get("ok"):
            print(f"  Azure LLM 2024 (UTC, may shift):  peak={fit_combined['peak_hour_utc']:.2f}h "
                  f"sigma={fit_combined['sigma_h']:.2f}h")
        for sname, s in services.items():
            b = s["burst_stats_hourly"]["tokens"]
            print(f"  Azure LLM '{sname}':  P99/mean={b['p99_over_mean']:.2f}  "
                  f"P999/mean={b['p999_over_mean']:.2f}")

    # Persist
    out = {
        "source": "Azure LLM Inference Dataset 2024 (Patel et al., Azure Public Dataset)",
        "url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
        "trace_files": ["AzureLLMInferenceTrace_conv_1week.csv", "AzureLLMInferenceTrace_code_1week.csv"],
        "note": "Timestamps are UTC absolute; peak_hour_utc differs from BurstGPT's local-tz peak. Sigma and burst stats are directly comparable.",
        "services": services,
        "combined_token_gaussian_utc": fit_combined,
    }
    (OUT_DIR / "azure_llm_inference_params.json").write_text(json.dumps(out, indent=2))
    # Per-hour CSV
    rows = []
    for sname, s in services.items():
        per = pd.DataFrame(s["diurnal_per_hour"])
        per["service"] = sname
        rows.append(per)
    pd.concat(rows, ignore_index=True).to_csv(
        OUT_DIR / "azure_llm_inference_fit.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'azure_llm_inference_params.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
