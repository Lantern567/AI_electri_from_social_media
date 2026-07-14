#!/usr/bin/env python3
"""Station-level variant of Result 2: pool over the 461 individual renewable
STATIONS directly (not country aggregates). Scopes by station location:
  L0 = home country's own stations
  L1 = all stations in the home country's world region
  L2 = all stations on the home continent block
  L3 = all 461 global stations (theoretical upper bound)

The NNLS picks individual plants worldwide, and can pick the best station WITHIN
a country. Workload axis reuses the 2-pass scope-consistent migration from the
country-level analysis. R1 (home mismatch) is unchanged (its number already
equals the capacity-weighted aggregate of a country's stations).

Output: reports/cfe_geographic_portfolio_ai/data_globalsites_stations/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base
import analyze_cfe_global_sites as G

SITE_DIR = G.SITE_DIR
OUT = G.ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites_stations"
OUT.mkdir(parents=True, exist_ok=True)
ETA = G.ETA
POWER_SCOPES = G.POWER_SCOPES
WORKLOAD_CODES = G.WORKLOAD_CODES


def load_stations(idx_naive, smeta):
    site = pd.read_parquet(SITE_DIR / "site_hourly_cf.parquet")
    site["timestamp_utc"] = pd.to_datetime(site["timestamp_utc"])
    if site["timestamp_utc"].dt.tz is not None:
        site["timestamp_utc"] = site["timestamp_utc"].dt.tz_localize(None)
    curves, meta = {}, {}
    for sid, g in site.groupby("site_id"):
        s = pd.Series(g["cf"].to_numpy(float),
                      index=pd.DatetimeIndex(g["timestamp_utc"])).reindex(idx_naive).ffill().bfill()
        iso2 = g["iso2"].iloc[0]
        curves[sid] = s
        meta[sid] = {"iso2": iso2, "tech": g["tech"].iloc[0],
                     "region_code": smeta.get(iso2, {}).get("region_code"),
                     "continent_block": smeta.get(iso2, {}).get("continent_block")}
    return curves, meta


def station_pool(c, scope, station_meta, smeta):
    # Distance-bounded scopes, consistent with G.global_pool: L0 home stations,
    # L1 <=1500 km, L2 <=3000 km, L3 same continent (NO global). Distance is from
    # the home country centroid to each station's country centroid.
    if scope == "L0":
        return [s for s, m in station_meta.items() if m["iso2"] == c]
    hc = smeta.get(c, {})
    hlon, hlat = hc.get("lon"), hc.get("lat")
    cont = hc.get("continent_block")

    def dkm(m):
        sm = smeta.get(m["iso2"])
        if not sm or hlon is None:
            return 1e9
        return base.great_circle_km(hlon, hlat, sm["lon"], sm["lat"])

    if scope == "L1":
        return [s for s, m in station_meta.items() if dkm(m) <= G.SCOPE_D1_KM]
    if scope == "L2":
        return [s for s, m in station_meta.items() if dkm(m) <= G.SCOPE_D2_KM]
    if scope == "L3":                                    # global: all stations (no cap)
        return list(station_meta.keys())
    raise KeyError(scope)


def station_basis(station_curves, site_ids, target_index):
    cols, labels = [], []
    for sid in site_ids:
        cf = station_curves[sid]
        m = float(cf.mean())
        if m > 0:
            cols.append((cf / m).to_numpy())
            labels.append(sid)
    if not cols:
        return np.zeros((len(target_index), 0)), []
    return np.stack(cols, axis=1), labels


def evaluate_station(c, power, workload, demand, demand_meta, station_curves, station_meta, smeta, rtt, d_perturbed=None):
    home_idx = demand[c]["d"].index
    d_base = demand[c]["d"] if d_perturbed is None else pd.Series(np.asarray(d_perturbed), index=home_idx)
    pool = station_pool(c, power, station_meta, smeta)
    S, labels = station_basis(station_curves, pool, home_idx)
    if S.size == 0:
        return {"country": c, "power_scope": power, "workload_scenario": workload,
                "scenario": f"{power}_{workload}", "uncovered_share": float("nan"),
                "n_supply_basis": 0, "n_active_basis": 0, "migrated_share": 0.0}
    if power != "L0":
        for j, sid in enumerate(labels):
            if station_meta[sid]["iso2"] != c:
                S[:, j] = S[:, j] * ETA
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    migrated = 0.0
    if base.WORKLOAD_SCENARIOS[workload]:
        resid = np.maximum(d_base.to_numpy() - fit, 0.0)
        d_eff, migrated = G.migrate_residual(c, workload, demand, demand_meta, rtt, d_base, resid)
        S2 = base.equal_energy_align(S, float(d_eff.mean()))
        w, share, fit = base.portfolio_nnls(d_eff.to_numpy(), S2)
    out = {"country": c, "power_scope": power, "workload_scenario": workload,
           "scenario": f"{power}_{workload}", "uncovered_share": share,
           "n_supply_basis": int(S.shape[1]), "n_active_basis": int((w > 1e-4).sum()),
           "migrated_share": migrated}
    # top hubs as station -> (iso2:tech)
    order = np.argsort(w)[::-1]
    for rank in range(min(3, len(order))):
        sid = labels[order[rank]]
        out[f"top{rank+1}_label"] = f"{station_meta[sid]['iso2']}:{station_meta[sid]['tech']}"
        out[f"top{rank+1}_weight"] = float(w[order[rank]])
    return out


def run_r3_stations(demand, demand_meta, station_curves, station_meta, smeta, rtt):
    perturbations = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
    years = [2025, 2030, 2050]
    scenarios = base.R3_SCENARIO_SUBSET
    rows = []
    for year in years:
        lam_inf = base.AI_SHARE_INFERENCE[year]
        lam_train = base.AI_SHARE_TRAINING[year]
        lam = base.AI_SHARE[year]  # total AI-related share, for the ai_share column
        capex_yr = year if year in base.NREL_BATTERY_CAPEX else 2030
        for cc in demand:
            d_base = base.as_arr(demand[cc]["d"])
            lh = base.as_arr(demand[cc]["local_hour"]).astype(int)
            for op in perturbations:
                if op == "P0_baseline":
                    d_pert = d_base
                elif op == "P1_flatten":
                    d_pert = base.operator_flatten(d_base, lam_inf, lam_train)
                elif op == "P2_sharpen":
                    d_pert = base.operator_sharpen(d_base, lh, lam_inf, lam_train)
                elif op == "P2_emp":
                    d_pert = base.operator_sharpen_emp(d_base, lh, lam_inf, lam_train)
                elif op == "P_mix":
                    d_pert = base.operator_mix(d_base, lh, lam_inf, lam_train)
                else:
                    # P3_burst — empirical gamma from BurstGPT v2 fit (base default).
                    d_pert = base.operator_burst(d_base, lam_inf, lam_train)
                for sc in scenarios:
                    power, workload = sc.split("_")
                    rec = evaluate_station(cc, power, workload, demand, demand_meta,
                                           station_curves, station_meta, smeta, rtt, d_perturbed=d_pert)
                    share = rec.get("uncovered_share", float("nan"))
                    if np.isnan(share):
                        continue
                    rows.append({"year": year, "lambda": lam, "lambda_inf": lam_inf,
                                 "lambda_train": lam_train, "country": cc,
                                 "region": demand[cc]["region"], "continent": demand[cc]["continent"],
                                 "perturbation": op, "scenario": sc, "uncovered_share": share,
                                 "annual_unmet_mwh_per_100mw": share * 8760 * 100,
                                 "storage_cost_usd_yr_per_100mw_mid": base.gap_to_storage_cost(share, capex_yr, "mid"),
                                 "storage_cost_usd_yr_per_100mw_low": base.gap_to_storage_cost(share, capex_yr, "low")})
    pc = pd.DataFrame(rows)
    agg = []
    for year in years:
        fleet = base.GLOBAL_FLEET_GW[year]; n_units = fleet * 1000.0 / 100.0
        capex_yr = year if year in base.NREL_BATTERY_CAPEX else 2030
        for op in pc["perturbation"].unique():
            for sc in pc["scenario"].unique():
                sub = pc[(pc.year == year) & (pc.perturbation == op) & (pc.scenario == sc)]
                if sub.empty:
                    continue
                ms = sub["uncovered_share"].mean(); annual_mwh = ms * 8760 * 100 * n_units
                cmid = base.battery_annualized_cost_per_mwh(base.NREL_BATTERY_CAPEX[capex_yr]["mid"]) * annual_mwh / base.STORAGE_ROUND_TRIP
                clow = base.battery_annualized_cost_per_mwh(base.NREL_BATTERY_CAPEX[capex_yr]["low"]) * annual_mwh / base.STORAGE_ROUND_TRIP
                agg.append({"year": year, "perturbation": op, "scenario": sc,
                            "mean_uncovered_share": ms, "global_annual_unmet_gwh": annual_mwh / 1000.0,
                            "global_storage_cost_musd_yr_mid": cmid / 1e6,
                            "global_storage_cost_usd_yr_low": clow / 1e6,
                            "global_fleet_gw": fleet, "ai_share": base.AI_SHARE[year], "capex_year_used": capex_yr})
    return pc, pd.DataFrame(agg)


def main():
    import os
    expanded = bool(os.environ.get("EXPANDED_DEMAND"))
    out = OUT.parent / "data_globalsites_stations_expanded" if expanded else OUT
    out.mkdir(parents=True, exist_ok=True)
    print(f"loading demand ({'EXPANDED' if expanded else '31'}) + country meta + station curves ...")
    if expanded:
        panel, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
        idx = demand[next(iter(demand))]["d"].index
        station_curves, station_meta = load_stations(idx, smeta)
        rtt = G.build_rtt_distance(demand_meta, smeta)
    else:
        panel, demand, demand_meta, supply, smeta = G.load_inputs()
        idx = demand[next(iter(demand))]["d"].index
        station_curves, station_meta = load_stations(idx, smeta)
        rtt = base.build_rtt_matrix_azure(list(demand_meta["cf_location"]))
    print(f"  demand={len(demand)}  stations={len(station_curves)}")

    print("R1 (station-derived home aggregate; identical to country-level) ...")
    r1, diur = G.run_r1(demand)
    r1.to_csv(out / "r1_country_mismatch.csv", index=False)
    diur.to_csv(out / "r1_diurnal_profiles.csv", index=False)

    rows = []
    for cc in demand:
        for power in POWER_SCOPES:
            for workload in WORKLOAD_CODES:
                rec = evaluate_station(cc, power, workload, demand, demand_meta,
                                       station_curves, station_meta, smeta, rtt)
                rec["region"] = demand[cc]["region"]
                rows.append(rec)
    r2 = pd.DataFrame(rows)
    r2.to_csv(out / "r2_portfolio_scenarios_stations.csv", index=False)

    # station-level L3 weight matrix (which stations serve whom), aggregated to iso2:tech
    wrows = []
    for cc in demand:
        home_idx = demand[cc]["d"].index
        d = demand[cc]["d"].to_numpy()
        pool = station_pool(cc, "L3", station_meta, smeta)
        S, labels = station_basis(station_curves, pool, home_idx)
        for j, sid in enumerate(labels):
            if station_meta[sid]["iso2"] != cc:
                S[:, j] = S[:, j] * ETA
        S_eq = base.equal_energy_align(S, float(d.mean()))
        w, share, fit = base.portfolio_nnls(d, S_eq)
        for j, sid in enumerate(labels):
            wrows.append({"demand_country": cc, "site_id": sid,
                          "supply_country": station_meta[sid]["iso2"],
                          "tech": station_meta[sid]["tech"], "weight": float(w[j])})
    w_df = pd.DataFrame(wrows)
    w_df.to_csv(out / "r2_l3_station_weights.csv", index=False)
    # aggregate station weights to (demand_country, supply_country, tech) for fig2 schema
    s4 = (w_df.groupby(["demand_country", "supply_country", "tech"], as_index=False)["weight"].sum())
    s4["is_home"] = s4["demand_country"] == s4["supply_country"]
    s4.to_csv(out / "r2_s4_weight_matrix.csv", index=False)

    print("R3 (station-level AI perturbation -> storage cost) ...")
    r3c, r3g = run_r3_stations(demand, demand_meta, station_curves, station_meta, smeta, rtt)
    r3c.to_csv(out / "r3_perturbation_cost.csv", index=False)
    r3g.to_csv(out / "r3_global_aggregate.csv", index=False)

    # canonical-named R2 scenarios file (same schema as country-level)
    r2.to_csv(out / "r2_portfolio_scenarios.csv", index=False)

    piv = (r2.groupby(["power_scope", "workload_scenario"]).uncovered_share.median().unstack() * 100).round(1)
    piv = piv.reindex(index=["L0", "L1", "L2", "L3"], columns=WORKLOAD_CODES)
    print("\nSTATION-LEVEL R2 median uncovered share (%):")
    print(piv.to_string())
    print("\nbasis sizes (W0): " + ", ".join(
        f"{s}={int(r2[(r2.power_scope==s)&(r2.workload_scenario=='W0')].n_supply_basis.median())}"
        for s in ["L0", "L1", "L2", "L3"]))
    cross = w_df[(w_df.supply_country != w_df.demand_country) & (w_df.weight > 1e-3)]
    topc = cross.groupby(["supply_country", "tech"]).weight.sum().sort_values(ascending=False).head(8)
    print("\ntop cross-border station hubs (aggregated to country:tech):")
    print(topc.round(2).to_string())
    print(f"distinct supply countries used: {cross.supply_country.nunique()}, distinct stations: {cross.site_id.nunique()}")
    (out / "provenance.json").write_text(json.dumps({
        "mode": "station-level pooling (461 real stations as NNLS basis)",
        "scopes": "L0 home-country stations / L1 region stations / L2 continent stations / L3 all 461",
    }, indent=2))
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
