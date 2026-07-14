#!/usr/bin/env python3
"""STEP 2 of the full-system cost extension of Result 3.

Prices the cached physical decomposition (fullsystem_portfolio_decomposition.csv)
with the verified parameter set (fullsystem_cost_params) and produces the
headline comparison the user asked for: what the FOUR previously-omitted cost
components (generation, transmission, data-centre facility, ancillary) add on top
of the existing firming-storage bill, expressed BOTH as a system LCOE (USD/MWh)
and as an absolute annual bill (B USD/yr), and whether the "L3 is ~10x cheaper"
result survives once generation and transmission are priced.

Per 100 MW-mean unit, per (scenario, year, operator), for a chosen cost `case`
(central/low/high) and overbuild mode (star = co-optimised; ob1 = equal-energy
lower bound; ob15/ob20 = literature 24/7 overbuild band):

  gen   = generation capex+FOM at the (overbuild-scaled) nameplate
  tx    = HVDC line+converters, sized to P95 hourly flow (np = nameplate, HIGH bound)
  store = duration-aware Li-ion/LDES firming on the post-pooling residual
  anc   = residual VRE balancing/reserve
  dc    = data-centre FACILITY capex+non-energy opex (scope-invariant; reported apart)
  it    = IT/GPU capex (context only)

Electricity-system LCOE = (gen+tx+store+anc)/E_d.  Outputs printed + saved to
data_globalsites/fullsystem_cost_table.csv and fullsystem_cost_summary.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import fullsystem_cost_params as P            # noqa: E402

DATA = P.DATA
E_D = P.E_D_MWH                               # 876,000 MWh/yr per 100 MW unit

# fleet trajectory for the absolute bill (growing-central; recompute_fig3c)
FLEET_GROW_CENTRAL = {2025: 82.3, 2030: 219.0, 2035: 285.0, 2040: 415.0, 2050: 700.0}


def fleet_units(year):
    xs = sorted(FLEET_GROW_CENTRAL)
    gw = float(np.interp(year, xs, [FLEET_GROW_CENTRAL[x] for x in xs]))
    return gw * 1000.0 / 100.0               # number of 100 MW units


OB_GEOM = {
    "star": ("ob_star", "unmet_mwh_ob", "emax_mwh_ob", "ppeak_mw_ob"),
    "ob1":  (None, "unmet_mwh_ob1", "emax_mwh_ob1", "ppeak_mw_ob1"),
    "ob15": (1.5, "unmet_mwh_15", "emax_mwh_15", "ppeak_mw_15"),
    "ob20": (2.0, "unmet_mwh_20", "emax_mwh_20", "ppeak_mw_20"),
}


def row_costs(r, case="central", ob_mode="star", tx_sizing="p95"):
    """Per-100 MW-unit annualised costs (USD/yr) for one decomposition row."""
    obcol, unmetc, emaxc, ppeakc = OB_GEOM[ob_mode]
    ob = float(r["ob_star"]) if obcol == "ob_star" else (1.0 if obcol is None else obcol)
    gen = P.gen_annual_cost(r["np_pv_mw"] * ob, r["np_wind_mw"] * ob, r["year"], case)
    gen_busbar = r["gen_busbar_mwh"] * ob
    store, _ = P.storage_annual_cost(r[ppeakc], r[emaxc], r[unmetc], case, r["year"])
    if tx_sizing == "p95":
        tx = P.tx_annual_cost(r["tx_p95_mwkm"], r["tx_p95_mw"], case)
    else:
        tx = P.tx_annual_cost(r["tx_np_mwkm"], r["tx_np_mw"], case)
    anc = P.ancillary_annual_cost(gen_busbar, case)
    dc = P.dc_facility_annual_cost(case)
    it = P.it_annual_cost(case)
    return {"gen": gen, "tx": tx, "store": store, "anc": anc, "dc": dc, "it": it,
            "ob": ob, "elec": gen + tx + store + anc}


def add_cost_columns(df, case="central", ob_mode="star", tx_sizing="p95"):
    recs = [row_costs(r, case, ob_mode, tx_sizing) for _, r in df.iterrows()]
    c = pd.DataFrame(recs, index=df.index)
    for col in ["gen", "tx", "store", "anc", "dc", "it", "elec", "ob"]:
        df[col + "_usd_yr"] = c[col]
    # LCOE (USD/MWh delivered demand) per component
    for col in ["gen", "tx", "store", "anc", "elec", "dc", "it"]:
        df["lcoe_" + col] = df[col + "_usd_yr"] / E_D
    return df


def fmt(x, n=1):
    return f"{x:.{n}f}"


def main():
    df = pd.read_csv(DATA / "fullsystem_portfolio_decomposition.csv")
    print(f"loaded {len(df)} decomposition rows")

    # ---------- headline: P_mix 2030, central, co-optimised OB, by power scope ----------
    base = df[(df.perturbation == "P_mix") & (df.year == 2030)].copy()
    add_cost_columns(base, "central", "star", "p95")

    print("\n" + "=" * 78)
    print("HEADLINE — P_mix, 2030, central prices, co-optimised overbuild, tx=P95")
    print("Electricity-system LCOE (USD/MWh of delivered demand), mean over 104 countries")
    print("=" * 78)
    w0 = base[base.workload_scenario == "W0"]
    g = w0.groupby("power_scope").agg(
        ob=("ob_usd_yr", "mean"),
        gen=("lcoe_gen", "mean"), tx=("lcoe_tx", "mean"),
        store=("lcoe_store", "mean"), anc=("lcoe_anc", "mean"),
        elec=("lcoe_elec", "mean"),
        fgn=("foreign_energy_share", "mean"), km=("mean_import_km", "mean"),
    ).reindex(["L0", "L1", "L2", "L3"])
    print(g.round(1).to_string())

    l0, l3 = g.loc["L0"], g.loc["L3"]
    print("\n--- L0 vs L3 ratios (the reframing) ---")
    print(f"  firming/storage-only LCOE:  L0={l0.store:.1f}  L3={l3.store:.1f}  "
          f"ratio L0/L3 = {l0.store / max(l3.store,1e-9):.1f}x")
    print(f"  TOTAL electricity LCOE:     L0={l0.elec:.1f}  L3={l3.elec:.1f}  "
          f"ratio L0/L3 = {l0.elec / max(l3.elec,1e-9):.2f}x")
    print(f"  transmission added at L3:   {l3.tx:.1f} USD/MWh ({100*l3.tx/l3.elec:.0f}% of L3 total)")
    print(f"  generation share:           L0={100*l0.gen/l0.elec:.0f}%   L3={100*l3.gen/l3.elec:.0f}%")

    # component-share table at the four reference scopes
    print("\n--- component share of electricity LCOE (%) ---")
    for sc in ["L0", "L1", "L2", "L3"]:
        rr = g.loc[sc]
        tot = rr.elec
        print(f"  {sc}: gen {100*rr.gen/tot:4.0f}  tx {100*rr.tx/tot:4.0f}  "
              f"store {100*rr.store/tot:4.0f}  anc {100*rr.anc/tot:4.0f}   "
              f"(total {tot:5.1f} USD/MWh)")

    # ---------- DC facility / IT context ----------
    dc_lcoe = base["lcoe_dc"].iloc[0]
    it_lcoe = base["lcoe_it"].iloc[0]
    print("\n--- scope-invariant layers (reported separately) ---")
    print(f"  DC facility capex+opex:  {dc_lcoe:.0f} USD/MWh   (vs L3 electricity {l3.elec:.0f})")
    print(f"  IT/GPU capex (4-yr):     {it_lcoe:.0f} USD/MWh")

    # ---------- overbuild sensitivity (P_mix 2030 L3, central) ----------
    print("\n--- overbuild sensitivity (P_mix 2030, L3_W0, central) ---")
    r_l3 = df[(df.perturbation == "P_mix") & (df.year == 2030) &
              (df.scenario == "L3_W0")]
    for mode, lbl in [("ob1", "equal-energy (OB=1, lower bound)"),
                      ("star", "co-optimised OB*"),
                      ("ob15", "overbuild 1.5x"),
                      ("ob20", "overbuild 2.0x")]:
        tmp = r_l3.copy(); add_cost_columns(tmp, "central", mode, "p95")
        print(f"  {lbl:32}: OB={tmp.ob_usd_yr.mean():.2f}  gen={tmp.lcoe_gen.mean():.1f}  "
              f"store={tmp.lcoe_store.mean():.1f}  elec={tmp.lcoe_elec.mean():.1f} USD/MWh")

    # ---------- tx sizing + case band (P_mix 2030 L3) ----------
    print("\n--- transmission sizing & price-case band (P_mix 2030 L3_W0) ---")
    for case in ["low", "central", "high"]:
        for siz in ["p95", "nameplate"]:
            tmp = r_l3.copy(); add_cost_columns(tmp, case, "star", siz)
            print(f"  case={case:7} tx={siz:9}: tx={tmp.lcoe_tx.mean():5.1f}  "
                  f"elec={tmp.lcoe_elec.mean():5.1f} USD/MWh")

    # ---------- absolute global bill (B USD/yr), growing-central fleet ----------
    print("\n" + "=" * 78)
    print("ABSOLUTE global bill (B USD/yr) — P_mix, growing-central fleet, central, OB*")
    print("=" * 78)
    for yr in [2025, 2030, 2050]:
        sub = df[(df.perturbation == "P_mix") & (df.year == yr) &
                 (df.workload_scenario == "W0")].copy()
        if sub.empty:
            continue
        add_cost_columns(sub, "central", "star", "p95")
        n = fleet_units(yr)
        for sc in ["L0", "L3"]:
            s = sub[sub.power_scope == sc]
            per_unit = s[["gen_usd_yr", "tx_usd_yr", "store_usd_yr", "anc_usd_yr",
                          "elec_usd_yr", "dc_usd_yr"]].mean()
            b = per_unit * n / 1e9
            print(f"  {yr} {sc}: gen {b.gen_usd_yr:6.0f}  tx {b.tx_usd_yr:6.0f}  "
                  f"store {b.store_usd_yr:6.0f}  anc {b.anc_usd_yr:5.0f}  "
                  f"| elec {b.elec_usd_yr:6.0f}  + DC-fac {b.dc_usd_yr:6.0f}  B USD/yr")

    # ---------- save tidy tables ----------
    full = df[(df.perturbation == "P_mix")].copy()
    add_cost_columns(full, "central", "star", "p95")
    keep = ["country", "year", "scenario", "power_scope", "workload_scenario",
            "ob_usd_yr", "foreign_energy_share", "mean_import_km",
            "lcoe_gen", "lcoe_tx", "lcoe_store", "lcoe_anc", "lcoe_elec",
            "lcoe_dc", "lcoe_it"]
    full[keep].to_csv(DATA / "fullsystem_cost_table.csv", index=False)
    print(f"\nwrote {DATA / 'fullsystem_cost_table.csv'} ({len(full)} rows)")


if __name__ == "__main__":
    main()
