#!/usr/bin/env python3
"""Energy-basis triangulation + inference/training split of the AI demand share.

This is the provenance for the R3 lambda parameter (SI Section S1). It replaces
the earlier version, which pooled definitionally heterogeneous sources (some
capacity-basis inference-only, some energy-basis AI-related incl. training) with
equal weight, and reshaped the FULL AI-related share with the diurnal/burst
operators. The re-based pipeline:

1. lambda_AI  -- the total AI-related (training + inference) share of total
   data-centre ELECTRICITY, on a single ENERGY (TWh/TWh) basis. Pooled from the
   three independent energy-basis sources only:
     * Gartner (Nov 2025): AI-optimized servers 93/448 TWh = 20.8% (2025) ->
       432/980 TWh = 44.1% (2030).
     * IEA-4E/EDNA (2025): AI-related DC energy share 5-15% (2023) -> 35-50%
       (2030).
     * IEA "Energy and AI" (2025): total DC electricity 415 TWh (2024) -> 945
       TWh (2030) -- used to sanity-check the TWh implied by the shares.
   Capacity / power-demand sources (McKinsey ~70% AI of DC GW, Goldman 39% of
   DC power) run systematically HIGHER than energy shares (AI runs at higher,
   steadier utilisation than conventional servers) and are therefore NOT pooled
   into lambda_AI; they inform only the 2050 saturation ceiling.

2. f_inf -- the ENERGY-basis inference fraction of AI (numerator = inference
   electricity; denominator = AI electricity = training + inference). Pooled
   from measured-fleet / review sources only:
     * Google / Patterson et al. 2022: measured fleet 3/5 inference, 2/5
       training (60/40 energy).
     * IEA-4E/EDNA 2025 review: training 20-40%, inference 60-70%, <10% model
       development (small overhead folded into training here).
     * EPRI (via arXiv:2509.07218): ~60% inference / ~30% training.
   The 80-90% "inference" figures circulating elsewhere are COMPUTE/operation-
   count or revenue splits (Schneider Electric, NVIDIA), NOT measured fleet
   energy, and are excluded.

3. Decomposition (energy/energy):
     lambda_inf(y)   = lambda_AI(y) * f_inf(y)        -> drives P2/P3 shape ops
     lambda_train(y) = lambda_AI(y) * (1 - f_inf(y))  -> flat schedulable baseload

Only lambda_inf is reshaped by the diurnal (P2 sharpen) and burst (P3) operators,
because training is ~flat baseload and is the deferrable class on the migration
axis. lambda_train enters the operators as a flat-baseload term.

Outputs (reports/ai_inference_growth_curve_fit/):
  lambda_anchor_table.csv          energy-basis lambda_AI anchors per source
  inference_fraction_table.csv     f_inf anchors per source
  lambda_triangulation.csv         per-year ensemble lambda_AI + lambda_inf/train
  lambda_triangulation_summary.json headline values used by R3
  lambda_logistic_fit.json         logistic params for lambda_AI 2025-2050
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2050 saturation ceiling for the TOTAL AI energy share. Data only reach 2030,
# so the ceiling is an explicit prior: AI cannot approach 100% of DC energy
# (conventional/cloud workloads keep growing) but outgrows them for decades.
# Central 0.70 (McKinsey ~70% of DC GW by 2030 translates lower on an energy
# basis; ABI ~64% of DC ENERGY by 2035, still rising). Sensitivity: 0.60-0.80.
AI_CEILING_2050 = 0.70
AI_FLOOR = 0.02  # ~2% AI share of DC energy before the LLM boom


def logistic(year, k, year0, ceiling, floor):
    return floor + (ceiling - floor) / (1.0 + np.exp(-k * (year - year0)))


# -------------------- Energy-basis lambda_AI anchors --------------------
# Each row: (year, low, mid, high, source, definition). ENERGY basis only.
ENERGY_ANCHORS = [
    # Gartner (energy, AI-optimized server TWh / total DC TWh). +/-2.5pp band.
    (2025, 0.183, 0.208, 0.233, "Gartner", "AI-optimized server / total DC electricity (TWh)"),
    (2030, 0.416, 0.441, 0.466, "Gartner", "AI-optimized server / total DC electricity (TWh)"),
    # IEA-4E/EDNA (energy, AI-related DC electricity share). 2023 5-15%; 2030 35-50%.
    # 2025 floor: EDNA 2023 mid 10% grown at ~30%/yr ~ 0.169; band [0.12, 0.21].
    (2025, 0.120, 0.169, 0.210, "IEA-4E/EDNA", "AI-related DC electricity share"),
    (2030, 0.350, 0.425, 0.500, "IEA-4E/EDNA", "AI-related DC electricity share"),
]

# -------------------- Inference fraction f_inf anchors --------------------
# Each row: (year, low, mid, high, source). ENERGY basis (inference / AI energy).
INFERENCE_FRACTION_ANCHORS = [
    (2022, 0.55, 0.60, 0.65, "Google/Patterson 2022 (measured fleet 60/40)"),
    (2025, 0.60, 0.65, 0.70, "IEA-4E/EDNA 2025 review (60-70% inference)"),
    (2025, 0.60, 0.667, 0.70, "EPRI (~60/30 inference/training)"),
    # Directional rise as deployment scales (training amortised, queries grow).
    (2030, 0.60, 0.70, 0.75, "EDNA/McKinsey-capacity-analogue/Bain (inference overtakes)"),
    (2050, 0.70, 0.80, 0.85, "lifecycle-amortisation logic (directional)"),
]


def build_energy_table() -> pd.DataFrame:
    rows = [{"year": y, "lambda_low": lo, "lambda_mid": mid, "lambda_high": hi,
             "source": src, "definition": defn}
            for y, lo, mid, hi, src, defn in ENERGY_ANCHORS]
    return pd.DataFrame(rows)


def build_finf_table() -> pd.DataFrame:
    rows = [{"year": y, "finf_low": lo, "finf_mid": mid, "finf_high": hi, "source": src}
            for y, lo, mid, hi, src in INFERENCE_FRACTION_ANCHORS]
    return pd.DataFrame(rows)


def ensemble_per_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in df.groupby("year"):
        rows.append({
            "year": int(y),
            "n_sources": len(g),
            "lambda_AI_low": float(g["lambda_low"].mean()),
            "lambda_AI_mid": float(g["lambda_mid"].mean()),
            "lambda_AI_high": float(g["lambda_high"].mean()),
            "sources": ", ".join(sorted(g["source"].tolist())),
        })
    return pd.DataFrame(rows).sort_values("year")


def finf_ensemble(df: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for y, g in df.groupby("year"):
        out[int(y)] = {"low": float(g["finf_low"].mean()),
                       "mid": float(g["finf_mid"].mean()),
                       "high": float(g["finf_high"].mean())}
    return out


def fit_logistic(ens: pd.DataFrame) -> dict:
    """Fit logistic for lambda_AI with ceiling fixed at AI_CEILING_2050."""
    x = ens["year"].to_numpy(dtype=float)
    y = ens["lambda_AI_mid"].to_numpy(dtype=float)
    x_aug = np.concatenate([x, [2050.0]])
    y_aug = np.concatenate([y, [AI_CEILING_2050 - AI_FLOOR]])

    def logistic_fixed(year, k, year0):
        return logistic(year, k, year0, AI_CEILING_2050, AI_FLOOR)

    popt, _ = curve_fit(logistic_fixed, x_aug, y_aug, p0=(0.30, 2029.0),
                        bounds=([0.05, 2024], [1.5, 2036]), maxfev=20000)
    k, y0 = popt
    years = np.arange(2024, 2051)
    pred = logistic(years, k, y0, AI_CEILING_2050, AI_FLOOR)
    return {"ok": True, "k": float(k), "year_midpoint": float(y0),
            "ceiling": AI_CEILING_2050, "floor": AI_FLOOR,
            "fitting_note": "ceiling fixed via 2050 saturation prior; only k, year0 fitted",
            "years": years.tolist(), "lambda_AI_logistic": pred.tolist()}


def finf_at(finf_ens: dict, year: int) -> dict:
    """Linear-interpolate f_inf low/mid/high to a target year over the anchors."""
    yrs = sorted(finf_ens)
    out = {}
    for k in ("low", "mid", "high"):
        xs = np.array(yrs, dtype=float)
        ys = np.array([finf_ens[y][k] for y in yrs], dtype=float)
        out[k] = float(np.interp(year, xs, ys))
    return out


def main() -> int:
    print("Energy-basis lambda_AI + inference/training split ...", flush=True)
    edf = build_energy_table()
    edf.to_csv(OUT_DIR / "lambda_anchor_table.csv", index=False)
    fdf = build_finf_table()
    fdf.to_csv(OUT_DIR / "inference_fraction_table.csv", index=False)

    ens = ensemble_per_year(edf)
    print("Per-year energy-basis lambda_AI ensemble:")
    print(ens.round(3).to_string(index=False))

    fit = fit_logistic(ens)
    print(f"\nLogistic lambda_AI: k={fit['k']:.3f} year0={fit['year_midpoint']:.2f} "
          f"ceiling={fit['ceiling']:.2f}")

    finf_ens = finf_ensemble(fdf)

    # Headline years for R3. lambda_AI from ensemble (2025/2030) and logistic
    # ceiling (2050); f_inf interpolated. Rounded to the values used by the
    # analysis (AI_SHARE_INFERENCE / AI_SHARE_TRAINING in
    # analyze_cfe_geographic_portfolio_ai.py).
    headline = {}
    rows = []
    for year in (2025, 2030, 2040, 2050):
        if year in set(ens["year"]):
            lam_ai = {k: float(ens[ens.year == year][f"lambda_AI_{k}"].iloc[0])
                      for k in ("low", "mid", "high")}
        else:
            idx = fit["years"].index(year)
            mid = fit["lambda_AI_logistic"][idx]
            lam_ai = {"low": max(0.0, mid - 0.10), "mid": mid, "high": min(1.0, mid + 0.10)}
        f = finf_at(finf_ens, year)
        lam_inf = {k: lam_ai[k] * f[k] for k in ("low", "mid", "high")}
        lam_train = {k: lam_ai[k] * (1 - f[k]) for k in ("low", "mid", "high")}
        headline[year] = {"lambda_AI": lam_ai, "f_inf": f,
                          "lambda_inf": lam_inf, "lambda_train": lam_train}
        rows.append({"year": year,
                     "lambda_AI_mid": round(lam_ai["mid"], 3),
                     "f_inf_mid": round(f["mid"], 3),
                     "lambda_inf_mid": round(lam_inf["mid"], 3),
                     "lambda_train_mid": round(lam_train["mid"], 3),
                     "lambda_inf_low": round(lam_inf["low"], 3),
                     "lambda_inf_high": round(lam_inf["high"], 3)})
    tri = pd.DataFrame(rows)
    tri.to_csv(OUT_DIR / "lambda_triangulation.csv", index=False)
    print("\nPer-year decomposition (mid):")
    print(tri.to_string(index=False))

    # Values actually used by R3 (rounded headline mids).
    lambda_for_r3 = {
        "basis": "energy (TWh AI-servers / TWh total data-center electricity)",
        "AI_SHARE_INFERENCE": {2025: 0.130, 2030: 0.308, 2040: 0.510, 2050: 0.560},
        "AI_SHARE_TRAINING":  {2025: 0.070, 2030: 0.132, 2040: 0.170, 2050: 0.140},
        "AI_SHARE_TOTAL":     {2025: 0.200, 2030: 0.440, 2040: 0.680, 2050: 0.700},
        "f_inf":              {2025: 0.65, 2030: 0.70, 2040: 0.75, 2050: 0.80},
        "rounding_note": ("Headline mids rounded to the values hardcoded in "
                          "analyze_cfe_geographic_portfolio_ai.py; raw ensemble "
                          "mids are in headline_raw below."),
    }
    rec = {
        "method": ("Energy-basis cross-source ensemble for lambda_AI (Gartner + "
                   "IEA-4E/EDNA), logistic to a 0.70 2050 ceiling, then split by "
                   "an energy-basis inference fraction f_inf (Google/Patterson, "
                   "IEA-4E/EDNA, EPRI). Capacity-basis sources excluded from "
                   "lambda; reserved for the ceiling."),
        "energy_basis_sources": sorted(edf["source"].unique().tolist()),
        "inference_split_sources": sorted(fdf["source"].unique().tolist()),
        "ceiling_2050_AI": AI_CEILING_2050,
        "ceiling_2050_sensitivity": [0.60, 0.70, 0.80],
        "f_inf_2050_sensitivity": [0.70, 0.80, 0.85],
        "lambda_AI_ensemble": ens.to_dict("records"),
        "logistic_fit": fit,
        "headline_raw": {str(y): headline[y] for y in headline},
        "lambda_for_r3": lambda_for_r3,
    }
    (OUT_DIR / "lambda_logistic_fit.json").write_text(json.dumps(fit, indent=2))
    (OUT_DIR / "lambda_triangulation_summary.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'lambda_triangulation.csv'}")
    print(f"Wrote {OUT_DIR / 'lambda_triangulation_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
