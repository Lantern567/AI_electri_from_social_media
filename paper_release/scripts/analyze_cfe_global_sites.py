#!/usr/bin/env python3
"""Result 2 (and R1) recomputed with GLOBAL station-based renewable supply.

Replaces the 31-country Tong national CF supply with capacity-weighted,
station-based hourly CF for all ~102 renewable countries (WRI Global Power Plant
Database locations + capacity, Open-Meteo ERA5 hourly weather, standard PV/wind
CF models; see build_global_supply_cf.py). Demand stays at the 31 Cloudflare
countries. The power-transmission scope becomes a nested geographic reachability
ladder over the global supply pool:

  L0 Home      : the demand country's own station-based VRE
  L1 Region    : all supply countries in the same world region (AF/AS/EU/NA/SA/OC)
  L2 Continent : Americas / EurAfrMENA / Asia / Oceania block
  L3 Global    : all 102 supply countries (theoretical upper bound)

The workload-class compute-migration axis (W0..W3) is unchanged from the
31-country analysis (Azure RTT + hyperscaler receiver mask, Luo & Yang 2026).

Outputs (same schema as the original, in a parallel dir so the 31-country
results are preserved):
  reports/cfe_geographic_portfolio_ai/data_globalsites/
      r1_country_mismatch.csv, r1_diurnal_profiles.csv,
      r2_portfolio_scenarios.csv, r2_s4_weight_matrix.csv,
      global_supply_provenance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cfe_geographic_portfolio_ai as base

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites"
PANEL = base.PANEL_PATH
OUT = ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "data_globalsites"
OUT.mkdir(parents=True, exist_ok=True)

ETA = base.TRANSMISSION_EFFICIENCY
POWER_SCOPES = base.POWER_SCOPES
WORKLOAD_CODES = base.WORKLOAD_SCENARIO_CODES

# expanded-demand inputs (Cloudflare global demand fetch) + mappings
DEMAND_GLOBAL = (ROOT / "collective_attention_research_plan" / "data"
                 / "cloudflare_global_demand" / "country_hourly_demand.csv.gz")
REGION_FROM_CODE = {"AF": "Africa/MENA", "AS": "Asia-Pacific", "EU": "Europe",
                    "NA": "North America", "SA": "Latin America", "OC": "Asia-Pacific",
                    "AN": "Other"}
# Countries with at least one major hyperscaler (AWS/Azure/GCP/OCI) region as of 2025,
# used to gate which demand countries can RECEIVE migrated compute (workload axis).
HYPERSCALER_EXTENDED = {
    "US", "CA", "BR", "CL", "MX", "GB", "IE", "FR", "DE", "NL", "BE", "CH", "IT",
    "ES", "SE", "NO", "FI", "PL", "AT", "GR", "DK", "JP", "KR", "IN", "ID", "MY",
    "TH", "AU", "NZ", "ZA", "IL", "AE", "SA", "BH", "KW", "TR", "EG", "CN", "TW", "QA",
}


# -------------------- load demand (31) + supply (102) --------------------

def load_inputs():
    panel = pd.read_csv(PANEL, parse_dates=["timestamp_utc"])
    panel = panel.sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)

    cf = pd.read_parquet(SITE_DIR / "country_hourly_cf.parquet")
    cf["timestamp_utc"] = pd.to_datetime(cf["timestamp_utc"], utc=True)
    # keep_default_na=False so ISO2 'NA' (Namibia) and region code 'NA' (North
    # America) survive as strings instead of being parsed as missing.
    cap = pd.read_csv(SITE_DIR / "country_capacity.csv", keep_default_na=False, na_values=[""])
    meta = pd.read_csv(SITE_DIR / "country_meta.csv", keep_default_na=False, na_values=[""])

    # common hourly index = the panel hours (tz-aware UTC)
    idx = pd.DatetimeIndex(sorted(panel["timestamp_utc"].unique()))
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx

    # supply curves per iso2: pv, wind (unit aligned to idx), and capacity-weighted VRE
    cap_lu = {(r.iso2, r.tech): r.capacity_mw_used for r in cap.itertuples()}
    supply: dict[str, dict] = {}
    for iso2, g in cf.groupby("iso2"):
        rec = {}
        for tech in ("PV", "WIND"):
            sub = g[g["tech"] == tech]
            if sub.empty:
                continue
            s = pd.Series(sub["cf"].to_numpy(), index=pd.DatetimeIndex(sub["timestamp_utc"]).tz_localize(None))
            rec[tech.lower() if tech == "PV" else "wind"] = s.reindex(idx_naive).ffill().bfill()
        if not rec:
            continue
        cpv = cap_lu.get((iso2, "PV"), 0.0) or 0.0
        cwd = cap_lu.get((iso2, "WIND"), 0.0) or 0.0
        pv = rec.get("pv"); wd = rec.get("wind")
        if pv is None:
            vre = wd
        elif wd is None:
            vre = pv
        else:
            tot = cpv + cwd
            vre = (cpv * pv + cwd * wd) / tot if tot > 0 else 0.5 * (pv + wd)
        rec["pv"] = pv if pv is not None else pd.Series(0.0, index=idx_naive)
        rec["wind"] = wd if wd is not None else pd.Series(0.0, index=idx_naive)
        rec["s"] = vre
        supply[iso2] = rec

    smeta = {r.iso2: {"lat": r.lat, "lon": r.lon, "region_code": r.region_code,
                      "continent_block": r.continent_block} for r in meta.itertuples()}

    # demand curves (31) from panel; home supply from station CF where available
    demand: dict[str, dict] = {}
    drows = []
    for cc, sub in panel.groupby("cf_location"):
        sub = sub.dropna(subset=["demand_scaled"]).sort_values("timestamp_utc")
        if len(sub) < 24 * 30 or cc not in supply:
            continue
        ts = pd.DatetimeIndex(sub["timestamp_utc"]).tz_localize(None)
        d = pd.Series(sub["demand_scaled"].to_numpy(float), index=ts).reindex(idx_naive).ffill().bfill()
        lh = pd.Series(sub["local_hour"].to_numpy(), index=ts).reindex(idx_naive).ffill().bfill()
        s = supply[cc]["s"].copy()
        if s.mean() > 0:
            s = s * (d.mean() / s.mean())  # equal-energy
        demand[cc] = {"d": d, "s": s, "pv": supply[cc]["pv"], "wind": supply[cc]["wind"],
                      "local_hour": lh, "region": str(sub["region"].iloc[0]),
                      "continent": base.CONTINENT.get(cc, "Other"),
                      "tong_country": sub["tong_country"].iloc[0] if "tong_country" in sub else cc}
        drows.append({"cf_location": cc, "region": str(sub["region"].iloc[0]),
                      "continent": base.CONTINENT.get(cc, "Other")})
    demand_meta = pd.DataFrame(drows)
    return panel, demand, demand_meta, supply, smeta


def _build_supply_from_cf():
    """Build supply curves + meta from station CF (idx derived from CF, not the panel)."""
    cf = pd.read_parquet(SITE_DIR / "country_hourly_cf.parquet")
    cf["timestamp_utc"] = pd.to_datetime(cf["timestamp_utc"], utc=True)
    cap = pd.read_csv(SITE_DIR / "country_capacity.csv", keep_default_na=False, na_values=[""])
    meta = pd.read_csv(SITE_DIR / "country_meta.csv", keep_default_na=False, na_values=[""])
    idx = pd.DatetimeIndex(sorted(cf["timestamp_utc"].unique()))
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    cap_lu = {(r.iso2, r.tech): r.capacity_mw_used for r in cap.itertuples()}
    supply = {}
    for iso2, g in cf.groupby("iso2"):
        rec = {}
        for tech in ("PV", "WIND"):
            sub = g[g["tech"] == tech]
            if sub.empty:
                continue
            s = pd.Series(sub["cf"].to_numpy(), index=pd.DatetimeIndex(sub["timestamp_utc"]).tz_localize(None))
            rec["pv" if tech == "PV" else "wind"] = s.reindex(idx_naive).ffill().bfill()
        cpv = cap_lu.get((iso2, "PV"), 0.0) or 0.0
        cwd = cap_lu.get((iso2, "WIND"), 0.0) or 0.0
        pv, wd = rec.get("pv"), rec.get("wind")
        if pv is None:
            vre = wd
        elif wd is None:
            vre = pv
        else:
            tot = cpv + cwd
            vre = (cpv * pv + cwd * wd) / tot if tot > 0 else 0.5 * (pv + wd)
        rec["pv"] = pv if pv is not None else pd.Series(0.0, index=idx_naive)
        rec["wind"] = wd if wd is not None else pd.Series(0.0, index=idx_naive)
        rec["s"] = vre
        supply[iso2] = rec
    smeta = {r.iso2: {"lat": r.lat, "lon": r.lon, "region_code": r.region_code,
                      "continent_block": r.continent_block} for r in meta.itertuples()}
    return idx_naive, supply, smeta


def load_inputs_expanded(min_cov: float = 0.9):
    """Demand = Cloudflare global fetch (all fetched countries with >=min_cov hourly
    coverage and a home station supply); supply = station CF. Same return signature
    as load_inputs (panel is None)."""
    idx_naive, supply, smeta = _build_supply_from_cf()
    dem = pd.read_csv(DEMAND_GLOBAL)
    dem["timestamp_utc"] = pd.to_datetime(dem["timestamp_utc"], utc=True).dt.tz_localize(None)
    demand, drows = {}, []
    for cc, g in dem.groupby("cf_location"):
        if cc not in supply or cc not in smeta:
            continue
        ser = pd.Series(g["human_volume_proxy"].to_numpy(float), index=pd.DatetimeIndex(g["timestamp_utc"]))
        ser = ser[~ser.index.duplicated(keep="last")]
        if ser.reindex(idx_naive).notna().mean() < min_cov:
            continue
        d = ser.reindex(idx_naive).ffill().bfill()
        if not np.isfinite(d.to_numpy()).all() or d.mean() <= 0 or d.std() == 0:
            continue
        tz = int(round(float(smeta[cc]["lon"]) / 15.0))
        local_hour = pd.Series((idx_naive.hour.to_numpy() + tz) % 24, index=idx_naive)
        s = supply[cc]["s"].copy()
        if s.mean() > 0:
            s = s * (d.mean() / s.mean())
        region = REGION_FROM_CODE.get(smeta[cc]["region_code"], "Other")
        demand[cc] = {"d": d, "s": s, "pv": supply[cc]["pv"], "wind": supply[cc]["wind"],
                      "local_hour": local_hour, "region": region,
                      "continent": smeta[cc]["continent_block"], "tong_country": cc}
        drows.append({"cf_location": cc, "region": region, "continent": smeta[cc]["continent_block"]})
    return None, demand, pd.DataFrame(drows), supply, smeta


def build_rtt_distance(demand_meta, smeta):
    """Distance-based RTT (ms) among demand countries (Luo & Yang 2026 model),
    used when the demand set extends beyond the 31 Azure-mapped countries."""
    codes = list(demand_meta["cf_location"])
    rtt = pd.DataFrame(np.full((len(codes), len(codes)), 5.0), index=codes, columns=codes)
    for a in codes:
        for b in codes:
            if a == b:
                continue
            dkm = base.great_circle_km(smeta[a]["lon"], smeta[a]["lat"], smeta[b]["lon"], smeta[b]["lat"])
            one_way = dkm / base.LIGHT_SPEED_FIBER_KM_PER_MS + base.SWITCHING_OVERHEAD_MS
            rtt.loc[a, b] = 2.0 * one_way * base.WAN_INFLATION_FACTOR
    return rtt


# -------------------- global nested power pool --------------------

# Distance-bounded dispatch scopes (replace the old political buckets). Codes kept
# as L0-L3 so schemas/figures are unchanged; meaning is now: L0 national, L1 <=1500 km,
# L2 <=3000 km, L3 same continent (NO global / cross-continent pooling).
SCOPE_D1_KM = 1500.0
SCOPE_D2_KM = 3000.0


def global_pool(c: str, scope: str, supply: dict, smeta: dict) -> list[str]:
    if scope == "L0":
        return [c] if c in supply else []
    if c not in smeta:
        return [c] if c in supply else []
    cont = smeta[c]["continent_block"]
    hlon, hlat = smeta[c]["lon"], smeta[c]["lat"]
    allc = [i for i in supply if i in smeta]

    def dkm(i):
        return base.great_circle_km(hlon, hlat, smeta[i]["lon"], smeta[i]["lat"])

    if scope == "L1":
        cand = [i for i in allc if dkm(i) <= SCOPE_D1_KM]
    elif scope == "L2":
        cand = [i for i in allc if dkm(i) <= SCOPE_D2_KM]
    elif scope == "L3":                                   # global: all reachable supply (no cap)
        cand = list(allc)
    else:
        raise KeyError(scope)
    if c not in cand and c in supply:
        cand.append(c)
    return cand


def migrate_residual(c, workload, demand, demand_meta, rtt, d_base, resid_deficit):
    """Migrate the deferrable workload classes out of the hours that are STILL in
    deficit after power pooling (resid_deficit), to RTT-feasible hyperscaler
    receivers with local renewable surplus. Scope-consistent so more migration
    can never increase the residual gap. Returns (d_eff, migrated_share)."""
    enabled = base.WORKLOAD_SCENARIOS[workload]
    idx = d_base.index
    if not enabled:
        return d_base.copy(), 0.0
    resid = pd.Series(np.asarray(resid_deficit), index=idx)
    denom = sum(base.WORKLOAD_SHARE[k] for k in enabled)
    total = pd.Series(0.0, index=idx)
    for cls in ("B", "C", "D"):
        if cls not in enabled:
            continue
        tau = base.WORKLOAD_LATENCY_MS[cls]
        receivers = [r for r in demand_meta["cf_location"]
                     if r != c and r in rtt.columns and rtt.at[c, r] <= tau
                     and (r in HYPERSCALER_EXTENDED) and r in demand]
        if not receivers:
            continue
        pool_surplus = pd.Series(0.0, index=idx)
        for r in receivers:
            s_r = demand[r]["s"].reindex(idx).ffill().bfill()
            d_r = demand[r]["d"].reindex(idx).ffill().bfill()
            pool_surplus = pool_surplus + np.maximum(s_r - d_r, 0.0)
        class_demand = d_base * base.WORKLOAD_SHARE[cls]
        class_budget = resid * (base.WORKLOAD_SHARE[cls] / denom)
        mig = np.minimum(class_demand, np.minimum(class_budget, np.maximum(pool_surplus - total, 0.0)))
        total = total + np.maximum(mig, 0.0)
    d_eff = d_base - total
    return d_eff, float(total.sum() / max(d_base.sum(), 1e-12))


def evaluate(c, power, workload, demand, demand_meta, supply, smeta, rtt, d_perturbed=None):
    home_idx = demand[c]["d"].index
    d_base = demand[c]["d"] if d_perturbed is None else pd.Series(np.asarray(d_perturbed), index=home_idx)
    pool = global_pool(c, power, supply, smeta)
    S, labels = base.supply_basis(supply, pool, home_idx, fixed=(power == "L0"))
    if S.size == 0:
        return {"country": c, "power_scope": power, "workload_scenario": workload,
                "scenario": f"{power}_{workload}", "uncovered_share": float("nan"),
                "n_supply_basis": 0, "n_active_basis": 0, "migrated_share": 0.0}
    if power != "L0":
        for j, lab in enumerate(labels):
            if lab.split(":")[0] != c:
                S[:, j] = S[:, j] * ETA
    # pass 1: portfolio without migration -> residual deficit after pooling
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    migrated = 0.0
    if base.WORKLOAD_SCENARIOS[workload]:
        resid = np.maximum(d_base.to_numpy() - fit, 0.0)
        d_eff, migrated = migrate_residual(c, workload, demand, demand_meta, rtt, d_base, resid)
        # pass 2: re-solve the same pool against the migrated demand
        S2 = base.equal_energy_align(S, float(d_eff.mean()))
        w, share, fit = base.portfolio_nnls(d_eff.to_numpy(), S2)
    out = {"country": c, "power_scope": power, "workload_scenario": workload,
           "scenario": f"{power}_{workload}", "uncovered_share": share,
           "n_supply_basis": int(S.shape[1]), "n_active_basis": int((w > 1e-4).sum()),
           "migrated_share": migrated}
    order = np.argsort(w)[::-1]
    for rank in range(min(3, len(order))):
        out[f"top{rank+1}_label"] = labels[order[rank]]
        out[f"top{rank+1}_weight"] = float(w[order[rank]])
    return out


# -------------------- R1 + R2 + weight matrix --------------------

def run_r1(demand: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diur = [], []
    for cc, c in demand.items():
        d, s, lh = base.as_arr(c["d"]), base.as_arr(c["s"]), base.as_arr(c["local_hour"]).astype(int)
        pv, wd = base.as_arr(c["pv"]), base.as_arr(c["wind"])
        wm = base.waveform_mismatch(d, s)
        rl = base.run_length_distribution(d, s)
        rows.append({"country": cc, "region": c["region"], "continent": c["continent"],
                     "uncovered_share_local": wm["uncovered_share"],
                     "excess_share_local": wm["excess_share"],
                     "max_uncovered_x_mean": wm["max_uncovered"],
                     "n_runs": rl["n_runs"], "median_run_h": rl["median_run_hours"],
                     "p95_run_h": rl["p95_run_hours"], "max_run_h": rl["max_run_hours"],
                     "peak_demand_local_h": base.peak_local_hour(d, lh),
                     "peak_supply_local_h": base.peak_local_hour(s, lh),
                     "phase_lag_h": base.diurnal_phase_lag(d, s, lh),
                     "spectral_diurnal_d": base.spectral_share(d, 22, 26),
                     "spectral_diurnal_s": base.spectral_share(s, 22, 26)})
        for h in range(24):
            m = lh == h
            if m.any():
                diur.append({"country": cc, "region": c["region"], "local_hour": h,
                             "demand": float(d[m].mean()), "supply_vre": float(s[m].mean()),
                             "supply_pv": float(pv[m].mean()), "supply_wind": float(wd[m].mean())})
    r1 = pd.DataFrame(rows).sort_values("uncovered_share_local", ascending=False)
    return r1, pd.DataFrame(diur)


def run_r2(demand, demand_meta, supply, smeta, rtt) -> pd.DataFrame:
    rows = []
    for cc in demand:
        for power in POWER_SCOPES:
            for workload in WORKLOAD_CODES:
                rec = evaluate(cc, power, workload, demand, demand_meta, supply, smeta, rtt)
                rec["region"] = demand[cc]["region"]
                rec["continent"] = demand[cc]["continent"]
                rows.append(rec)
    return pd.DataFrame(rows)


def run_weight_matrix(demand, supply, smeta) -> pd.DataFrame:
    rows = []
    for cc in demand:
        home_idx = demand[cc]["d"].index
        d = demand[cc]["d"].to_numpy()
        pool = global_pool(cc, "L3", supply, smeta)
        S, labels = base.supply_basis(supply, pool, home_idx)
        if S.size == 0:
            continue
        for j, lab in enumerate(labels):
            if lab.split(":")[0] != cc:
                S[:, j] = S[:, j] * ETA
        S_eq = base.equal_energy_align(S, float(d.mean()))
        w, share, fit = base.portfolio_nnls(d, S_eq)
        for j, lab in enumerate(labels):
            src, tech = lab.split(":")
            rows.append({"demand_country": cc, "supply_country": src, "tech": tech,
                         "weight": float(w[j]), "is_home": (src == cc)})
    return pd.DataFrame(rows)


def run_r3(demand, demand_meta, supply, smeta, rtt):
    """AI-perturbation -> residual gap -> NREL storage cost, over the global pool.
    Power scopes are year-independent here (global geographic reachability),
    so year effects enter only via AI share lambda, fleet GW, and battery capex."""
    perturbations = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
    years = [2025, 2030, 2050]
    scenarios = base.R3_SCENARIO_SUBSET  # L0_W0, L3_W0, L0_W3, L3_W3
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
                    rec = evaluate(cc, power, workload, demand, demand_meta, supply, smeta,
                                   rtt, d_perturbed=d_pert)
                    share = rec.get("uncovered_share", float("nan"))
                    if np.isnan(share):
                        continue
                    rows.append({
                        "year": year, "lambda": lam, "lambda_inf": lam_inf,
                        "lambda_train": lam_train, "country": cc,
                        "region": demand[cc]["region"], "continent": demand[cc]["continent"],
                        "perturbation": op, "scenario": sc, "uncovered_share": share,
                        "annual_unmet_mwh_per_100mw": share * 8760 * 100,
                        "storage_cost_usd_yr_per_100mw_mid": base.gap_to_storage_cost(share, capex_yr, "mid"),
                        "storage_cost_usd_yr_per_100mw_low": base.gap_to_storage_cost(share, capex_yr, "low"),
                    })
    pc = pd.DataFrame(rows)
    agg = []
    for year in years:
        fleet = base.GLOBAL_FLEET_GW[year]
        n_units = fleet * 1000.0 / 100.0
        capex_yr = year if year in base.NREL_BATTERY_CAPEX else 2030
        for op in pc["perturbation"].unique():
            for sc in pc["scenario"].unique():
                sub = pc[(pc.year == year) & (pc.perturbation == op) & (pc.scenario == sc)]
                if sub.empty:
                    continue
                ms = sub["uncovered_share"].mean()
                annual_mwh = ms * 8760 * 100 * n_units
                cmid = base.battery_annualized_cost_per_mwh(base.NREL_BATTERY_CAPEX[capex_yr]["mid"]) * annual_mwh / base.STORAGE_ROUND_TRIP
                clow = base.battery_annualized_cost_per_mwh(base.NREL_BATTERY_CAPEX[capex_yr]["low"]) * annual_mwh / base.STORAGE_ROUND_TRIP
                agg.append({"year": year, "perturbation": op, "scenario": sc,
                            "mean_uncovered_share": ms, "global_annual_unmet_gwh": annual_mwh / 1000.0,
                            "global_storage_cost_musd_yr_mid": cmid / 1e6,
                            "global_storage_cost_usd_yr_low": clow / 1e6,
                            "global_fleet_gw": fleet, "ai_share": base.AI_SHARE[year],
                            "capex_year_used": capex_yr})
    return pc, pd.DataFrame(agg)


def main():
    import os
    # The 104-country headline uses the expanded Cloudflare-global demand set;
    # set EXPANDED_DEMAND=1 to reproduce it. The default (31 Azure-mapped
    # countries) keeps the measured Azure RTT matrix.
    expanded = bool(os.environ.get("EXPANDED_DEMAND"))
    if expanded:
        print("loading EXPANDED demand (Cloudflare global) + global station supply ...")
        panel, demand, demand_meta, supply, smeta = load_inputs_expanded()
        rtt = build_rtt_distance(demand_meta, smeta)
    else:
        print("loading demand (31) + global station supply (102) ...")
        panel, demand, demand_meta, supply, smeta = load_inputs()
        rtt = base.build_rtt_matrix_azure(list(demand_meta["cf_location"]))
    print(f"  demand countries: {len(demand)}  |  supply countries: {len(supply)}")

    print("R1 (station-based home supply) ...")
    r1, diur = run_r1(demand)
    r1.to_csv(OUT / "r1_country_mismatch.csv", index=False)
    diur.to_csv(OUT / "r1_diurnal_profiles.csv", index=False)

    print("R2 (4x4 scenarios over global pool) ...")
    r2 = run_r2(demand, demand_meta, supply, smeta, rtt)
    r2.to_csv(OUT / "r2_portfolio_scenarios.csv", index=False)

    print("R2 L3 weight matrix (global hubs) ...")
    s4 = run_weight_matrix(demand, supply, smeta)
    s4.to_csv(OUT / "r2_s4_weight_matrix.csv", index=False)

    print("R3 (AI perturbation -> storage cost over global pool) ...")
    r3c, r3g = run_r3(demand, demand_meta, supply, smeta, rtt)
    r3c.to_csv(OUT / "r3_perturbation_cost.csv", index=False)
    r3g.to_csv(OUT / "r3_global_aggregate.csv", index=False)

    (OUT / "global_supply_provenance.json").write_text(json.dumps({
        "supply_source": "WRI Global Power Plant Database (locations+capacity) + Open-Meteo ERA5 hourly weather",
        "cf_models": {"pv": "GHI/1000 * (1+gamma*(Tcell-25)); PVWatts/GSEE-style",
                      "wind": "smoothed IEC power curve on wind_speed_100m, x1.2 bias correction"},
        "n_supply_countries": len(supply), "n_demand_countries": len(demand),
        "power_scopes": "nested geographic reachability L0 home / L1 region / L2 continent / L3 global",
        "transmission_efficiency": ETA,
        "note": "31-country HVDC corridor list dropped (incompatible with global pool); "
                "scopes are nested geographic upper bounds.",
    }, indent=2))

    print("\nR2 median uncovered share by scenario (%):")
    piv = (r2.groupby(["power_scope", "workload_scenario"])["uncovered_share"].median()
           .unstack("workload_scenario") * 100).round(1)
    print(piv.reindex(index=["L0", "L1", "L2", "L3"], columns=WORKLOAD_CODES).to_string())
    print("\nR1 uncovered share (station supply), top/bottom 5:")
    print(r1[["country", "uncovered_share_local", "phase_lag_h"]].head(5).to_string(index=False))
    print(r1[["country", "uncovered_share_local", "phase_lag_h"]].tail(5).to_string(index=False))
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
