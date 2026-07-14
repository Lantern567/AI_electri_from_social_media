"""Compute the data-backed Supplementary Information validation tables.

Outputs (printed as Markdown-ready blocks):
  1. RTT model validation: modeled distance-based RTT (rtt_matrix_ms.csv) vs
     measured Azure region-to-region P50 latency (azure_rtt_raw.csv).
  2. Political-feasibility robustness: median residual uncovered share by
     political-exclusion level x corner scenario (from r2_political_sensitivity.csv).

These back SI Sections S11 (RTT) and S13 (Result 1-2 robustness). Run from repo:
  python collective_attention_research_plan/scripts/build_si_validation_tables.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "cfe_geographic_portfolio_ai" / "data"

# Representative Azure region for each modeled country code that hosts a
# public Azure region (countries without a region are excluded from the RTT
# anchor comparison; they have no measured latency).
CC_TO_AZURE = {
    "AU": "Australia East", "BR": "Brazil South", "CA": "Canada Central",
    "DE": "Germany West Central", "FR": "France Central", "GB": "UK South",
    "IN": "Central India", "IT": "Italy North", "JP": "Japan East",
    "KR": "Korea Central", "MX": "Mexico Central", "MY": "Malaysia West",
    "NZ": "New Zealand North", "PL": "Poland Central", "SE": "Sweden Central",
    "US": "East US", "ZA": "South Africa North",
}


def rtt_validation():
    # Modeled RTT from the distance model used for the 104-country extension:
    # one-way L = dist_km / 200 + 20 ms; RTT = L * 2 (round trip) * 1.4 (WAN inflation).
    dist = pd.read_csv(DATA / "distance_matrix_km.csv", index_col=0)
    modeled = (dist / 200.0 + 20.0) * 2.0 * 1.4
    azure = pd.read_csv(DATA / "azure_rtt_raw.csv", index_col=0)
    azure = azure.apply(pd.to_numeric, errors="coerce")

    def azure_rtt(a, b):
        ra, rb = CC_TO_AZURE[a], CC_TO_AZURE[b]
        vals = []
        for r, c in [(ra, rb), (rb, ra)]:
            if r in azure.index and c in azure.columns:
                v = azure.at[r, c]
                if pd.notna(v):
                    vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    ccs = [c for c in CC_TO_AZURE if c in modeled.index and c in modeled.columns]
    rows = []
    for i, a in enumerate(ccs):
        for b in ccs[i + 1:]:
            m = modeled.at[a, b]
            z = azure_rtt(a, b)
            if pd.notna(m) and pd.notna(z):
                rows.append((a, b, float(m), z, float(m) - z))
    df = pd.DataFrame(rows, columns=["a", "b", "modeled_ms", "azure_ms", "delta_ms"])

    r = df["modeled_ms"].corr(df["azure_ms"])
    mae = df["delta_ms"].abs().median()
    bias = df["delta_ms"].median()
    print("## RTT validation (modeled distance-RTT vs measured Azure P50)")
    print(f"n country pairs with a measured Azure anchor: {len(df)}")
    print(f"Pearson r(modeled, measured) = {r:.3f}")
    print(f"median |error| = {mae:.0f} ms ; median signed error (modeled - measured) = {bias:+.0f} ms")
    print(f"modeled range = [{df.modeled_ms.min():.0f}, {df.modeled_ms.max():.0f}] ms ; "
          f"measured range = [{df.azure_ms.min():.0f}, {df.azure_ms.max():.0f}] ms")
    show = ["US-GB", "US-DE", "US-JP", "US-BR", "DE-FR", "GB-DE", "JP-AU",
            "JP-KR", "IN-GB", "BR-DE", "US-IN", "AU-US"]
    print("\nRepresentative pairs:")
    print("| Pair | Modeled RTT (ms) | Measured Azure RTT (ms) | Δ (ms) |")
    print("|---|---:|---:|---:|")
    for lbl in show:
        a, b = lbl.split("-")
        sub = df[((df.a == a) & (df.b == b)) | ((df.a == b) & (df.b == a))]
        if len(sub):
            row = sub.iloc[0]
            print(f"| {a}-{b} | {row.modeled_ms:.0f} | {row.azure_ms:.0f} | {row.delta_ms:+.0f} |")
    print()


def political_robustness():
    df = pd.read_csv(DATA / "r2_political_sensitivity.csv")
    print("## Political-feasibility robustness (median residual uncovered share, %)")
    piv = (df.groupby(["political_level", "scenario"])["uncovered_share"]
             .median().unstack("scenario") * 100).round(2)
    cols = [c for c in ["L0_W0", "L3_W0", "L0_W3", "L3_W3"] if c in piv.columns]
    print(piv[cols].to_string())
    mig = (df.groupby("scenario")["migrated_share"].median() * 100).round(2)
    print("\nmedian migrated share by scenario (%):")
    print(mig.to_string())
    print(f"\nn countries = {df['country'].nunique()} ; political levels = {sorted(df['political_level'].unique())}")
    print()


if __name__ == "__main__":
    rtt_validation()
    political_robustness()
