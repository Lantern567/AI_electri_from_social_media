#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity sweep for R3 (SI Section S7).

Evaluates how the headline 2030 global storage account changes when each AI-
workload parameter is moved within its empirical uncertainty band, holding
all others at their central value:

  λ_inf_{2030}:  energy-basis inference-share low / mid / high (0.21/0.308/0.375)
  P2 (μ, σ):     bootstrap 95% CI bounds (low-low, mid, high-high)
  P3 γ:          {3.0 stylized, 4.84 default, 6.0 upper}
  P_mix weights: {25/65/10 default, 35/55/10 more-batch, 15/70/15 more-burst}

For each parameter setting we:
1. Apply the perturbed operator to baseline demand (no re-run of full R3 — we
   re-use the analyze_cfe_global_sites_stations evaluate_station mechanism for
   only the (year, op, sensitivity_variant) combinations not already in
   r3_perturbation_cost.csv).
2. Re-evaluate the L0_W0 and L3_W3 corner scenarios for those variants.
3. Aggregate to global accounts (M USD/yr) and report a (low, mid, high)
   bracket per parameter.

Output
------
reports/cfe_geographic_portfolio_ai/data_globalsites_stations_expanded/
  r3_sensitivity_sweep.csv     country-level, (parameter, variant, scenario)
  r3_sensitivity_summary.csv   aggregated global accounts (M USD/yr)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_cfe_geographic_portfolio_ai as base  # noqa: E402
import analyze_cfe_global_sites as G  # noqa: E402


YEAR = 2030  # focus on 2030 (the modal year of the headline result)
SCENARIOS = ["L0_W0", "L3_W3"]
# Country-level (data_globalsites) evaluation, to stay consistent with the R3
# cost headline; output kept alongside the expanded artifacts.
OUT_DIR = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites_stations_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Re-basis: operators take (lam_inf, lam_train). The sweep varies one AI-shape
# factor at a time while holding the 2030 inference/training split at central.
LAM_INF = base.AI_SHARE_INFERENCE[YEAR]
LAM_TRAIN = base.AI_SHARE_TRAINING[YEAR]


def operator_burst_g(d, lam_inf, gamma, lam_train=LAM_TRAIN):
    return base.operator_burst(d, lam_inf, lam_train, gamma)


def operator_sharpen_param(d, lh, lam_inf, peak, sigma, lam_train=LAM_TRAIN):
    return base.operator_sharpen(d, lh, lam_inf, lam_train, peak_hour=peak, sigma_h=sigma)


def operator_mix_param(d, lh, lam_inf, ab, ai, ae, lam_train=LAM_TRAIN):
    return base.operator_mix(d, lh, lam_inf, lam_train, alpha_batch=ab,
                             alpha_interactive=ai, alpha_burst=ae)


def sensitivity_variants():
    """Yield (parameter_name, variant_label, operator_function)."""
    # P3 gamma
    for g, lab in [(3.0, "stylized_g3"), (4.84, "default_g4.84"), (6.0, "upper_g6")]:
        yield ("gamma", lab, lambda d, lh, lam, g=g: operator_burst_g(d, lam, g))
    # P2 (peak, sigma) at 95% CI bounds (joint corners)
    for peak, sigma, lab in [(17.75, 3.53, "ci_low"),
                              (18.0,  3.79, "ci_mid"),
                              (18.39, 4.05, "ci_high")]:
        yield ("P2_peak_sigma", lab,
               lambda d, lh, lam, p=peak, s=sigma: operator_sharpen_param(d, lh, lam, p, s))
    # P_mix weights
    for ab, ai, ae, lab in [(0.25, 0.65, 0.10, "default_25_65_10"),
                              (0.35, 0.55, 0.10, "more_batch_35_55_10"),
                              (0.15, 0.70, 0.15, "more_burst_15_70_15")]:
        yield ("Pmix_weights", lab,
               lambda d, lh, lam, b=ab, i=ai, e=ae: operator_mix_param(d, lh, lam, b, i, e))
    # Inference-share band (energy-basis lambda_inf low/mid/high for 2030),
    # paired with the default P3_burst operator since that is the most
    # lambda_inf-sensitive. Training share held at central LAM_TRAIN.
    for lam_val, lab in [(0.210, "lambda_inf_low"), (0.308, "lambda_inf_mid"), (0.375, "lambda_inf_high")]:
        yield ("lambda_inf_band", lab,
               lambda d, lh, _lam, lv=lam_val: operator_burst_g(d, lv, 4.84))


def main() -> int:
    print("Loading expanded 104-country inputs ...", flush=True)
    panel, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    rtt = G.build_rtt_distance(demand_meta, smeta)
    print(f"  {len(demand)} countries (country-level evaluation)", flush=True)

    capex_yr = YEAR if YEAR in base.NREL_BATTERY_CAPEX else 2030
    lam_default = base.AI_SHARE_INFERENCE[YEAR]  # re-basis: shape operators take lambda_inf
    rows = []

    for param_name, variant, op_fn in sensitivity_variants():
        print(f"  variant: {param_name} = {variant}", flush=True)
        for cc in demand:
            d_base = base.as_arr(demand[cc]["d"])
            lh = base.as_arr(demand[cc]["local_hour"]).astype(int)
            d_pert = op_fn(d_base, lh, lam_default)
            for sc in SCENARIOS:
                power, workload = sc.split("_")
                rec = G.evaluate(cc, power, workload, demand, demand_meta,
                                 supply, smeta, rtt, d_perturbed=d_pert)
                share = rec.get("uncovered_share", float("nan"))
                if np.isnan(share):
                    continue
                rows.append({
                    "year": YEAR, "parameter": param_name, "variant": variant,
                    "country": cc, "scenario": sc,
                    "uncovered_share": share,
                    "storage_cost_usd_yr_per_100mw_mid": base.gap_to_storage_cost(share, capex_yr, "mid"),
                })

    pc = pd.DataFrame(rows)
    pc.to_csv(OUT_DIR / "r3_sensitivity_sweep.csv", index=False)
    print(f"  wrote {len(pc):,} rows", flush=True)

    # Aggregate to global account
    fleet = base.GLOBAL_FLEET_GW[YEAR]
    n_units = fleet * 1000.0 / 100.0
    agg = []
    for (param, variant, sc), g in pc.groupby(["parameter", "variant", "scenario"]):
        ms = g["uncovered_share"].mean()
        annual_mwh = ms * 8760 * 100 * n_units
        cmid = base.battery_annualized_cost_per_mwh(base.NREL_BATTERY_CAPEX[capex_yr]["mid"]) * annual_mwh / base.STORAGE_ROUND_TRIP
        agg.append({
            "year": YEAR, "parameter": param, "variant": variant, "scenario": sc,
            "mean_uncovered_share": ms,
            "global_storage_cost_musd_yr_mid": cmid / 1e6,
        })
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(OUT_DIR / "r3_sensitivity_summary.csv", index=False)
    print("\nSensitivity summary (2030 global storage cost, M USD/yr):")
    piv = aggdf.pivot_table(values="global_storage_cost_musd_yr_mid",
                              index=["parameter", "variant"],
                              columns="scenario").round(1)
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
