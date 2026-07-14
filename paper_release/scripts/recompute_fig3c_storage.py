#!/usr/bin/env python3
"""Complete (duration-aware) recompute of the Figure-3c forward-looking firming
bill, with annual storage-cost data and storage-technology scenarios.

What changes vs the old Fig-3c model
------------------------------------
Old: every unmet MWh is firmed by a 4-hour Li-ion battery at a single NREL capex
point per year (daily-cycling proxy). That over-prices the long (multi-hour to
multi-day) clean-energy gaps that dominate 24/7-CFE shortfalls (Result 1: P95
run-length up to 19 h), and ignores long-duration storage (LDES).

New (this script):
  1. ANNUAL storage cost, not 3 hand anchors:
     - Li-ion 4h: NREL ATB 2024 utility-scale battery capex (public), the three
       ATB cost cases (Conservative/Moderate/Advanced) interpolated to every year.
     - LDES: public anchors (Sepulveda 2021 Nature Energy 10.1038/s41560-021-00796-8
       "design space" $1-20/kWh; DOE Long-Duration Storage Shot $0.05/kWh-cycle by
       2030; Form Energy / LDES-Council public targets), interpolated to every year.
       (BNEF's proprietary annual series is not used — it is paywalled.)
  2. DURATION-AWARE sizing: for each demand country the home-supply (L0xW0) deficit
     time series is decomposed into gaps; the firming asset is sized to the worst
     gap (energy E_max, power P_peak) and the CHEAPEST technology is chosen:
        LCOS = (CRF+FOM) * (p_power*P_peak + e_energy*E_max)*1000 / (annual_unmet*RTE)
     Li-ion (low p, high e) wins for short gaps; LDES (high p, low e) wins for long
     gaps — the crossover is ~10 h, matching the DOE "10h+ = LDES" definition.
  3. DENSE YEARS: the hourly model is re-run for every year 2025-2050 (the global
     geographic reachability is year-independent — only AI share lambda, fleet GW
     and storage cost move with year — so densifying is cheap).
  4. THREE storage scenarios (Conservative / Central / Optimistic) => a bill band.

Output: data_globalsites/r3c_storage_scenarios.csv (year x operator x scenario).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("EXPANDED_DEMAND", "1")
import analyze_cfe_global_sites as G          # noqa: E402
base = G.base

OUT = base.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"
YEARS = list(range(2025, 2051))
OPERATORS = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
SCENARIOS = ["Conservative", "Central", "Optimistic"]

# ---------------------------------------------------------------- fleet models
# Global data-centre nameplate capacity (GW). The bill scales linearly with this,
# so it is applied as a post-multiplier on the (fleet-independent) per-100 MW geometry.
#
#   "frozen"  = the paper's original assumption: McKinsey 82.3 GW(2025) -> 219 GW(2030),
#               then FROZEN at 219 GW. Reproduces the current Fig.3d (post-2030 decline
#               is an artefact of holding the fleet flat while unit cost keeps falling).
#
#   "growing_*" = de-frozen trajectories. The 2025/2030 GW anchors are McKinsey
#               (published); the 2035/2040/2050 values follow published TOTAL-DC
#               electricity (TWh) growth, pinned to the IEA 2030=945 TWh anchor and
#               scaled by published TWh ratios (IEA Energy&AI 2035 Base ~1,200 TWh;
#               Deloitte 2040 midpoint ~1,900 TWh; BNEF/Deloitte/Shift 2050 cluster
#               ~1,680-3,750 TWh). Exact values + citations:
#               reports/datacenter_fleet_projection_2050/datacenter_fleet_gw_projection_2050.md (§5).
#               growing_central coincides with frozen through 2030 and only de-freezes after.
FLEET_FROZEN = {2025: 82.3, 2030: 219.0, 2040: 219.0, 2050: 219.0}
FLEET_GROW_CENTRAL = {2025: 82.3, 2030: 219.0, 2035: 285.0, 2040: 415.0, 2050: 700.0}
FLEET_GROW_LOW = {2025: 82.3, 2030: 171.0, 2035: 210.0, 2040: 280.0, 2050: 500.0}
FLEET_GROW_HIGH = {2025: 96.8, 2030: 298.0, 2035: 360.0, 2040: 560.0, 2050: 850.0}
FLEET_MODELS = {
    "frozen": FLEET_FROZEN,
    "growing_low": FLEET_GROW_LOW,
    "growing_central": FLEET_GROW_CENTRAL,
    "growing_high": FLEET_GROW_HIGH,
}

# ---------------------------------------------------------------- storage cost
# Li-ion 4h system capex (USD/kWh): NREL ATB 2024, three cost cases at the ATB
# milestone years (these are the values already used by the project's base model).
# REAL annual NREL ATB 2024 (v3.0.0) 4h utility-scale battery overnight capital
# cost, USD/kWh (= OCC USD/kW ÷ 4h), pulled year-by-year 2022-2050 from the OEDI
# data lake (oedi-data-lake.s3 .../ATB/electricity/csv/2024/v3.0.0/ATBe.csv) and
# saved to data_globalsites/nrel_atb_2024_battery_4h.csv. ATB Conservative/Moderate/
# Advanced map to our Conservative/Central/Optimistic.
_ATB = pd.read_csv(OUT / "nrel_atb_2024_battery_4h.csv")
_ATB_YR = _ATB["year"].to_numpy(dtype=float)
_ATB_COL = {"Conservative": "Conservative", "Central": "Moderate", "Optimistic": "Advanced"}
_ATB_VAL = {scen: _ATB[col].to_numpy(dtype=float) for scen, col in _ATB_COL.items()}
P_LI = 280.0       # Li-ion power-capacity cost (USD/kW), ~flat
RTE_LI = 0.86
LIFE_LI = 15

# LDES energy-capacity cost (USD/kWh): public anchors interpolated to every year.
LDES_ANCHOR_YR = [2025, 2030, 2050]
E_LDES = {
    "Conservative": [120.0, 90.0, 60.0],   # LDES scales slowly, stays pricey
    "Central":      [80.0,  45.0, 25.0],    # moderate scale-up (Sepulveda mid)
    "Optimistic":   [60.0,  30.0, 12.0],    # DOE Storage-Shot / iron-air at scale
}
P_LDES = 1000.0    # LDES power-capacity cost (USD/kW), higher than Li-ion
RTE_LDES = 0.60
LIFE_LDES = 20

DISCOUNT = 0.07
FOM_FRAC = 0.025


def crf(life):
    r = DISCOUNT
    return r * (1 + r) ** life / ((1 + r) ** life - 1)


CRF_LI = crf(LIFE_LI)
CRF_LDES = crf(LIFE_LDES)


def e_li(scen, year):
    # real ATB annual 4h-system USD/kWh, minus the power-cost share, gives the
    # Li-ion energy-capacity cost (USD/kWh). 2022-2050 are exact points (no anchor
    # interpolation); np.interp only guards the range ends.
    return float(np.interp(year, _ATB_YR, _ATB_VAL[scen])) - P_LI / 4.0


def e_ldes(scen, year):
    return float(np.interp(year, LDES_ANCHOR_YR, E_LDES[scen]))


def cheapest_lcos(p_peak, e_max, annual_unmet, scen, year):
    """Levelised firming cost (USD/MWh delivered) under the cheaper of Li-ion vs
    LDES, for a single-asset sized to the worst gap (P_peak MW, E_max MWh)."""
    if annual_unmet <= 0 or p_peak <= 0:
        return 0.0, "none"
    eL, eD = e_li(scen, year), e_ldes(scen, year)
    capex_li = (P_LI * p_peak + eL * e_max) * 1000.0          # USD
    capex_ld = (P_LDES * p_peak + eD * e_max) * 1000.0
    lcos_li = (CRF_LI + FOM_FRAC) * capex_li / (annual_unmet * RTE_LI)
    lcos_ld = (CRF_LDES + FOM_FRAC) * capex_ld / (annual_unmet * RTE_LDES)
    return (lcos_li, "Li-ion") if lcos_li <= lcos_ld else (lcos_ld, "LDES")


# ---------------------------------------------------------------- year inputs
def interp_dict(dct, year):
    xs = sorted(dct)
    return float(np.interp(year, xs, [dct[x] for x in xs]))


def gaps(g):
    """Return (duration_hours, energy) for each contiguous deficit run in g."""
    out = []
    n = len(g)
    i = 0
    while i < n:
        if g[i] > 1e-9:
            j = i
            e = 0.0
            while j < n and g[j] > 1e-9:
                e += g[j]
                j += 1
            out.append((j - i, e))
            i = j
        else:
            i += 1
    return out


def operator_curve(op, d_base, lh, li, lt):
    if op == "P0_baseline":
        return d_base
    if op == "P1_flatten":
        return base.operator_flatten(d_base, li, lt)
    if op == "P2_sharpen":
        return base.operator_sharpen(d_base, lh, li, lt)
    if op == "P2_emp":
        return base.operator_sharpen_emp(d_base, lh, li, lt)
    if op == "P3_burst":
        return base.operator_burst(d_base, li, lt)
    return base.operator_mix(d_base, lh, li, lt)


def main():
    print("loading expanded demand (104 countries) ...")
    panel, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    print(f"  demand countries: {len(demand)}")

    # pre-extract per-country base arrays
    cc_d = {cc: base.as_arr(demand[cc]["d"]).astype(float) for cc in demand}
    cc_s = {cc: base.as_arr(demand[cc]["s"]).astype(float) for cc in demand}
    cc_lh = {cc: base.as_arr(demand[cc]["local_hour"]).astype(int) for cc in demand}
    N = len(demand)

    rows = []
    country_rows = []
    for year in YEARS:
        li = interp_dict(base.AI_SHARE_INFERENCE, year)
        lt = interp_dict(base.AI_SHARE_TRAINING, year)
        # number of 100 MW units per fleet model; the geometry below is per-100 MW
        # and fleet-independent, so we bill it once and scale by each model.
        n_units_by_model = {m: interp_dict(d, year) * 1000.0 / 100.0
                            for m, d in FLEET_MODELS.items()}
        n_units_frozen = n_units_by_model["frozen"]   # Fig-4 country maps stay on frozen
        for op in OPERATORS:
            # per-country firming geometry at the status-quo corner (L0 x W0)
            geom = []  # (p_peak, e_max, annual_unmet) per country, per 100 MW
            for cc in demand:
                d = cc_d[cc]
                s = cc_s[cc]
                d_pert = operator_curve(op, d, cc_lh[cc], li, lt)
                g = np.maximum(0.0, d_pert - s)
                tot = float(g.sum())
                if tot <= 0:
                    continue
                scale = 100.0 / float(d_pert.mean())          # MW per curve-unit
                gg = gaps(g)
                # size to the P95 gap (standard capacity planning — do not over-
                # build for the single worst 1-in-N event)
                gap_e = np.array([e for _, e in gg])
                e_max = float(np.percentile(gap_e, 95)) * scale            # MWh
                pos = g[g > 1e-9]
                p_peak = float(np.percentile(pos, 95)) * scale             # MW
                annual_unmet = tot * scale                     # MWh/yr per 100 MW
                geom.append((cc, p_peak, e_max, annual_unmet))
            for scen in SCENARIOS:
                bill = 0.0
                unmet_sum = 0.0
                tech_ldes_share = 0.0
                for cc_, p_peak, e_max, annual_unmet in geom:
                    lcos, tech = cheapest_lcos(p_peak, e_max, annual_unmet, scen, year)
                    bill += annual_unmet * lcos
                    unmet_sum += annual_unmet
                    if tech == "LDES":
                        tech_ldes_share += annual_unmet
                    # per-country bill (fleet share x unmet x cost) — kept for the
                    # realistic blend, used by the Figure-4 maps (frozen fleet, unchanged)
                    if op == "P_mix":
                        country_rows.append({
                            "year": year, "storage_scenario": scen, "country": cc_,
                            "country_bill_musd": (n_units_frozen / N) * annual_unmet * lcos / 1e6,
                            "tech": tech,
                        })
                # distribute the fleet evenly across demand countries (matches the
                # base model's mean-country x fleet aggregation). The per-100 MW bill
                # above is fleet-independent; scale it by each fleet model so a single
                # run yields both the frozen baseline and the de-frozen growth band.
                ldes_share = (tech_ldes_share / unmet_sum) if unmet_sum else 0.0
                for fmodel, n_units in n_units_by_model.items():
                    global_bill = (n_units / N) * bill          # USD/yr
                    rows.append({
                        "year": year, "perturbation": op, "storage_scenario": scen,
                        "fleet_model": fmodel,
                        "global_storage_bill_busd": global_bill / 1e9,
                        "ldes_energy_share": ldes_share,
                        "fleet_gw": interp_dict(FLEET_MODELS[fmodel], year),
                        "lambda_inf": li,
                    })
        print(f"  {year} done")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r3c_storage_scenarios.csv", index=False)
    print(f"wrote {OUT / 'r3c_storage_scenarios.csv'}  ({len(df)} rows)")

    cdf = pd.DataFrame(country_rows)
    cdf.to_csv(OUT / "r3c_country_bill.csv", index=False)
    print(f"wrote {OUT / 'r3c_country_bill.csv'}  ({len(cdf)} rows)")

    # quick sanity print: P_mix Central-storage bill at 2025/2030/2035/2040/2050,
    # frozen vs growing_central — should show the post-2030 shape reverse from
    # decline (frozen) to a continued rise (growing).
    for fm in ("frozen", "growing_central"):
        vals = {}
        for y in (2025, 2030, 2035, 2040, 2050):
            sub = df[(df.year == y) & (df.perturbation == "P_mix") &
                     (df.storage_scenario == "Central") & (df.fleet_model == fm)]
            if len(sub):
                vals[y] = round(float(sub["global_storage_bill_busd"].iloc[0]), 1)
        print(f"  P_mix Central [{fm}]: {vals}")


if __name__ == "__main__":
    main()
