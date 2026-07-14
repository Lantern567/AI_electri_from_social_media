#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity sweep for R3 in the LATENCY x ROUTABLE
framing (SI Section S7, new caliber).

Replaces run_r3_sensitivity_sweep.py (old L x W storage-bill version). For each
AI-workload parameter moved within its empirical band (others central), we
re-price the full-system cost at the two corners of the latency x routable grid:

  national  = (tau = 50 ms, phi = 0)    no cross-border routing -> baseline
  global    = (tau = 500 ms, phi = 1.0) full-routing, hemisphere reach

Reported per corner: cross-country median full-system electricity cost
(USD/MWh) and median uncovered demand share (%). The point is that no AI-shape
parameter moves these corners much — the latency x routable lever (national
~125 -> global ~95 USD/MWh, gap 41.8% -> ~17%) dominates.

Parameters swept (same bands as the old sweep, operator pairing unchanged):
  lambda_inf_{2030}: 0.21 / 0.308 / 0.375  (on P3_burst, the most lambda-sensitive)
  P2 (mu, sigma):    bootstrap 95% CI corners (on P2_sharpen)
  P3 gamma:          3.0 / 4.84 / 6.0       (on P3_burst)
  P_mix weights:     25/65/10 / 35/55/10 / 15/70/15  (on P_mix)

Output: data_globalsites/r3_latency_sensitivity.csv  (country-level)
        data_globalsites/r3_latency_sensitivity_summary.csv (median corners)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_cfe_geographic_portfolio_ai as base   # noqa: E402
import analyze_cfe_global_sites as G                 # noqa: E402
import analyze_cfe_r3_waveform_cost as R3            # noqa: E402  (waveform routing)

YEAR = 2030
CORNERS = [("national", 50, 0.0), ("global", 500, 1.0)]
OUT = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"

LAM_INF = base.AI_SHARE_INFERENCE[YEAR]
LAM_TRAIN = base.AI_SHARE_TRAINING[YEAR]


def op_burst_g(d, lh, lam, gamma):
    return base.operator_burst(d, lam, LAM_TRAIN, gamma)


def op_sharpen_p(d, lh, lam, peak, sigma):
    return base.operator_sharpen(d, lh, lam, LAM_TRAIN, peak_hour=peak, sigma_h=sigma)


def op_mix_p(d, lh, lam, ab, ai, ae):
    return base.operator_mix(d, lh, lam, LAM_TRAIN, alpha_batch=ab,
                             alpha_interactive=ai, alpha_burst=ae)


def variants():
    """(parameter, variant_label, operator(d, lh, lam_inf))."""
    for lam, lab in [(0.210, "low_0.21"), (0.308, "mid_0.308"), (0.375, "high_0.375")]:
        yield ("lambda_inf", lab, lambda d, lh, _l, lv=lam: op_burst_g(d, lh, lv, base.P3_GAMMA))
    for peak, sigma, lab in [(17.75, 3.53, "ci_low"), (18.0, 3.79, "mid"), (18.39, 4.05, "ci_high")]:
        yield ("P2_mu_sigma", lab, lambda d, lh, l, p=peak, s=sigma: op_sharpen_p(d, lh, l, p, s))
    for g, lab in [(3.0, "stylized_3.0"), (4.84, "default_4.84"), (6.0, "upper_6.0")]:
        yield ("P3_gamma", lab, lambda d, lh, l, gv=g: op_burst_g(d, lh, l, gv))
    for ab, ai, ae, lab in [(0.25, 0.65, 0.10, "default_0.25"),
                            (0.35, 0.55, 0.10, "more_batch_0.35"),
                            (0.15, 0.70, 0.15, "more_shaped_0.15")]:
        yield ("Pmix_batch", lab, lambda d, lh, l, b=ab, i=ai, e=ae: op_mix_p(d, lh, l, b, i, e))


def main():
    print("loading expanded inputs ...", flush=True)
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    idx = demand[next(iter(demand))]["d"].index
    codes = [c for c in demand if c in smeta]
    print(f"  {len(codes)} countries; precomputing waveform routing shapes ...", flush=True)
    reach, receivers = R3.precompute_reach(codes, smeta)
    nat_shape = {r: R3.national_shape(supply, r, idx) for r in receivers}
    cc_lh = {c: base.as_arr(demand[c]["local_hour"]).astype(int) for c in demand}
    cc_d = {c: base.as_arr(demand[c]["d"]).astype(float) for c in demand}

    rows = []
    for param, variant, op_fn in variants():
        print(f"  {param} = {variant}", flush=True)
        for cc in codes:
            dp = op_fn(cc_d[cc], cc_lh[cc], LAM_INF)
            for corner, tau, phi in CORNERS:
                rec = R3.decompose(cc, tau, phi, demand, supply, smeta, reach, nat_shape, idx, dp, YEAR)
                if rec is None or not np.isfinite(rec["uncovered_share"]):
                    continue
                rows.append({"parameter": param, "variant": variant, "country": cc,
                             "corner": corner, "uncovered_pct": rec["uncovered_share"] * 100.0,
                             "lcoe_elec": rec["lcoe_elec"], "lcoe_gen": rec["lcoe_gen"],
                             "lcoe_store": rec["lcoe_store"], "lcoe_anc": rec["lcoe_anc"]})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r3_latency_sensitivity.csv", index=False)

    summ = (df.groupby(["parameter", "variant", "corner"])
              .agg(median_uncovered_pct=("uncovered_pct", "median"),
                   median_lcoe=("lcoe_elec", "median"),
                   median_store=("lcoe_store", "median")).reset_index())
    summ.to_csv(OUT / "r3_latency_sensitivity_summary.csv", index=False)

    print("\n=== R3 latency sensitivity (2030) — median across 104 countries ===")
    for metric in ["median_lcoe", "median_uncovered_pct"]:
        piv = summ.pivot_table(values=metric, index=["parameter", "variant"], columns="corner")
        piv = piv[["national", "global"]].round(1)
        unit = "USD/MWh" if metric == "median_lcoe" else "% uncovered"
        print(f"\n-- {unit} --")
        print(piv.to_string())
    print(f"\nwrote r3_latency_sensitivity.csv ({len(df)} rows) + summary")


if __name__ == "__main__":
    main()
