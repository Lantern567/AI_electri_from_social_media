#!/usr/bin/env python3
"""Cross-country renewable portfolio + AI-perturbed demand analysis.

This script produces the data tables and figures for the new academic report
"cfe_geographic_portfolio_ai". It is organized around three results:

  R1  Per-country waveform + spectral mismatch between Cloudflare human-traffic
      demand and the country's own VRE supply (PV+wind, capacity-weighted).
  R2  Cross-country supply portfolio optimization under five physical-constraint
      scenarios (S0 local, S1 compute-follow-sun, S2 regional grid, S3
      continental grid, S4 unconstrained upper bound).
  R3  AI-inference demand perturbations (Flatten / Sharpen / Burst-amplify)
      applied at McKinsey-anchored AI shares; rerun R1 + R2 and convert
      residual gap to NREL battery storage cost (per-100MW and global-fleet).

Inputs
------
  collective_attention_research_plan/reports/prime_time_cfe_penalty_ptp_r2_r3/
      ptp_country_hourly_cfe_panel.csv.gz
      (Country x hour panel of equal-energy normalized demand, hourly PV/wind/VRE
       capacity factors, region label and local hour.)

Outputs
-------
  collective_attention_research_plan/reports/cfe_geographic_portfolio_ai/
      data/r1_country_mismatch.csv
      data/r1_spectral.csv
      data/r2_portfolio_scenarios.csv
      data/r2_portfolio_weights_sample.csv
      data/r3_perturbation_cost.csv
      data/r3_global_aggregate.csv
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import nnls

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = (
    ROOT
    / "collective_attention_research_plan"
    / "reports"
    / "prime_time_cfe_penalty_ptp_r2_r3"
    / "ptp_country_hourly_cfe_panel.csv.gz"
)
OUT_ROOT = (
    ROOT
    / "collective_attention_research_plan"
    / "reports"
    / "cfe_geographic_portfolio_ai"
)
OUT_DATA = OUT_ROOT / "data"
OUT_DATA.mkdir(parents=True, exist_ok=True)

# -------------------- assumptions --------------------
# AI demand-share trajectory, re-based onto a consistent ENERGY (TWh/TWh) basis
# and split into inference vs training. See fit_lambda_triangulation.py ->
# reports/ai_inference_growth_curve_fit/lambda_triangulation_summary.json and
# SI Section S1.
#
# lambda_AI (total AI-related share of data-center ELECTRICITY) is the energy-
# basis cross-source ensemble of Gartner (AI-optimized-server TWh: 21% 2025 ->
# 44% 2030), IEA-4E/EDNA (AI-related DC energy share: 5-15% -> 35-50%) and IEA
# "Energy and AI" 2025 (total DC TWh: 415 -> 945). Capacity-basis figures
# (McKinsey/Goldman GW shares, which run higher than energy shares) are reserved
# for the 2050 ceiling only. lambda_AI is then split by an ENERGY-basis inference
# fraction f_inf (Google/Patterson 2022 measured 60/40; IEA-4E/EDNA 60-70%
# inference; EPRI ~60/30):
#     lambda_inf   = lambda_AI * f_inf       -> drives the diurnal/burst operators
#     lambda_train = lambda_AI * (1 - f_inf) -> flat, schedulable baseload
# Only the INFERENCE share is reshaped by the P2 (sharpen) / P3 (burst) shape
# operators, because training is ~flat baseload and is the deferrable class (D)
# on the migration axis. This replaces the prior single-lambda treatment, which
# reshaped the FULL AI-related share and so conflated flat training load into the
# diurnal/burst peak (inflating the peaky-scenario bills).
#
# f_inf:        2025 = 0.65, 2030 = 0.70, 2050 = 0.80
# lambda_AI mid: 2025 = 0.20, 2030 = 0.44, 2050 = 0.70 (ceiling)
AI_SHARE_INFERENCE = {2025: 0.130, 2030: 0.308, 2040: 0.510, 2050: 0.560}
AI_SHARE_TRAINING = {2025: 0.070, 2030: 0.132, 2040: 0.170, 2050: 0.140}
# Total AI-related share (= inference + training); used for reporting and the
# cost-aggregation `ai_share` column. The shape operators take the two
# components separately (lam_inf, lam_train); see operator_* below.
AI_SHARE = {y: round(AI_SHARE_INFERENCE[y] + AI_SHARE_TRAINING[y], 4)
            for y in AI_SHARE_INFERENCE}

# Global data-center fleet (GW) - McKinsey scenario, frozen after 2030
GLOBAL_FLEET_GW = {2025: 82.3, 2030: 219.0, 2040: 219.0, 2050: 219.0}

# NREL utility-scale battery (4h) capex (USD/kWh), 2025 update; 2030 linear interp
NREL_BATTERY_CAPEX = {
    2025: {"low": 236, "mid": 332, "high": 415},
    2030: {"low": 194, "mid": 289.5, "high": 382},
    2035: {"low": 152, "mid": 247, "high": 349},
    2050: {"low": 111, "mid": 184, "high": 333},
}
DISCOUNT_RATE = 0.07
BATTERY_LIFE_YR = 15
FIXED_OM_FRAC = 0.025

# Continental / regional adjacency groups for R2 portfolio scenarios.
# Continental groups are large geographic blocks (HVDC plausibility).
CONTINENT = {
    "AR": "Americas", "BR": "Americas", "CA": "Americas", "CL": "Americas",
    "CO": "Americas", "MX": "Americas", "NZ": "Oceania",     "PE": "Americas",
    "PY": "Americas", "US": "Americas", "AU": "Oceania",
    "DE": "EurAfrMENA", "DZ": "EurAfrMENA", "FR": "EurAfrMENA",
    "GB": "EurAfrMENA", "GH": "EurAfrMENA", "IT": "EurAfrMENA",
    "LY": "EurAfrMENA", "MA": "EurAfrMENA", "NG": "EurAfrMENA",
    "PL": "EurAfrMENA", "SE": "EurAfrMENA", "TN": "EurAfrMENA",
    "UA": "EurAfrMENA", "ZA": "EurAfrMENA",
    "ID": "Asia",     "IN": "Asia",     "JP": "Asia",
    "KR": "Asia",     "MY": "Asia",     "TH": "Asia",
}

# Cross-country power transmission round-trip efficiency
TRANSMISSION_EFFICIENCY = 0.92

# -----------------------------------------------------------------------------
# Workload-class migration design (anchored to published values).
# Latency budgets τ_k from Luo & Yang (2026) "AI Inference as Relocatable
# Electricity Demand", Table 4. Workload class shares from Meta Carbon Responder
# (Xing et al. 2023) Fig 1 (real-time services ~60-70%, AI training + data
# pipeline ~30-40%) cross-referenced with Luo & Yang's task taxonomy.
# -----------------------------------------------------------------------------
WORKLOAD_CLASSES = ["A", "B", "C", "D"]
WORKLOAD_LATENCY_MS = {
    "A": 300,          # interactive (real-time chat, voice copilot)
    "B": 1500,         # standard online (chatbot, enterprise Q&A)
    "C": 60_000,       # weakly real-time (background agents, async workflows)
    "D": 3_600_000,    # offline batch (1h+; training, batch summarisation)
}
WORKLOAD_SHARE = {
    "A": 0.40,
    "B": 0.30,
    "C": 0.20,
    "D": 0.10,
}

# Workload migration scenarios: which classes are ALLOWED to migrate at all.
# Class A always stays local; W3 enables maximum migration (B+C+D ≈ 60% of d_c).
WORKLOAD_SCENARIOS = {
    "W0": set(),
    "W1": {"D"},
    "W2": {"C", "D"},
    "W3": {"B", "C", "D"},
}

# Power transmission scope codes
POWER_SCOPES = ["L0", "L1", "L2", "L3"]
WORKLOAD_SCENARIO_CODES = ["W0", "W1", "W2", "W3"]

POWER_LABEL = {
    "L0": "Local",
    "L1": "Regional grid",
    "L2": "Continental grid",
    "L3": "Global pool",
}
WORKLOAD_LABEL = {
    "W0": "W0 None",
    "W1": "W1 Batch only (D, 10%)",
    "W2": "W2 + Background (C+D, 30%)",
    "W3": "W3 + Standard (B+C+D, 60%)",
}

# Latency model parameters from Luo & Yang (2026) Section 5.4
LIGHT_SPEED_FIBER_KM_PER_MS = 200.0   # ~2/3 c in fiber
SWITCHING_OVERHEAD_MS = 20.0
WAN_INFLATION_FACTOR = 1.4

# -----------------------------------------------------------------------------
# Real-world feasibility masks (3 independent dimensions).
# -----------------------------------------------------------------------------
# (1) HVDC maximum practical distance, scaled by year.
# 2025 = 3,000 km matches today's commercial maximum (Belo Monte Brazil = 2,500 km;
# Xlinks Morocco-UK = 3,800 km in construction).
# 2030 = 4,000 km reflects projects under construction or financial close.
# 2050 = 5,000 km is the theoretical UHVDC upper bound.
HVDC_MAX_KM_BY_YEAR = {
    2025: 3000.0,
    2030: 4000.0,
    2050: 5000.0,
}
DEFAULT_HVDC_MAX_KM = HVDC_MAX_KM_BY_YEAR[2025]

# (2) Political / sanctions exclusion mask with 4 relaxation levels. For the
# 31-country sample, the most well-known hostile pairs (CN-IN, RU-EU, IL-arab,
# IR-SA, TW-CN) are not in scope, so the framework supports populating with
# realistic but lower-stakes exclusions. Levels: NONE, LOOSE (active 2025
# disputes only), MODERATE (+ historical regional tensions), STRICT (+ trade
# bloc enforcement).
POLITICAL_MASK_LEVELS = {
    "NONE": set(),
    "LOOSE": {
        # Active 2025 disputes within the 31-country sample
        ("UA", "DZ"), ("UA", "LY"),  # post-2022 alignment shifts
    },
    "MODERATE": {
        ("UA", "DZ"), ("UA", "LY"),
        ("MA", "DZ"), ("DZ", "MA"),    # Western Sahara dispute
        ("AR", "GB"), ("GB", "AR"),    # Falklands legacy
        ("KR", "JP"),                  # historical East Asia tensions (one-way)
        ("IN", "GB"),                  # historic energy strategic independence
    },
    "STRICT": {
        # Adds trade-bloc enforcement: only intra-bloc + adjacency partners
        ("UA", "DZ"), ("UA", "LY"),
        ("MA", "DZ"), ("DZ", "MA"),
        ("AR", "GB"), ("GB", "AR"),
        ("KR", "JP"),
        ("IN", "GB"),
        # Strict view: EU-MENA only via Spain (none in set), so MENA cut from EU
        ("DE", "DZ"), ("DE", "LY"), ("FR", "LY"), ("IT", "LY"),
        ("PL", "DZ"), ("PL", "LY"), ("SE", "DZ"), ("SE", "LY"),
        # Strict view: US-South-China-Sea-aligned: limit India-China-aligned
        ("IN", "MY"), ("IN", "TH"),
        # Some Latin America historic tensions
        ("CL", "AR"), ("AR", "CL"),
    },
}
DEFAULT_POLITICAL_LEVEL = "LOOSE"
POLITICAL_EXCLUSIONS = POLITICAL_MASK_LEVELS[DEFAULT_POLITICAL_LEVEL]

# (3) Cloud-region availability for compute receivers. A country can RECEIVE
# migrated compute only if it has at least one major hyperscaler (AWS / Azure /
# GCP) region as of 2025. Cross-referenced against public cloud region maps.
HYPERSCALER_AVAILABLE = {
    "AU", "BR", "CA", "CL", "DE", "FR", "GB", "ID", "IN", "IT",
    "JP", "KR", "MX", "MY", "NZ", "PL", "SE", "TH", "US", "ZA",
}
# Excluded (no major public hyperscaler region as of 2025):
# AR, CO, PE, PY, DZ, GH, LY, MA, NG, TN, UA

# (4) Concrete HVDC corridor list (replacing pure distance cap).
# Source: ENTSO-E grid map, IEA Global Energy and CO2 Status Report, Wikipedia
# "List of HVDC projects". For each year, set of country pairs (undirected)
# corresponds to commissioned (2025), + under construction (2030), + aspirational
# investable corridors (2050). Within-region AC mesh handled by region-level
# scope L1 plus the corridor list; corridors here are explicit cross-region
# HVDC links.
HVDC_CORRIDORS_BY_YEAR = {
    2025: {
        # Within ENTSO-E continental Europe (HVDC + AC mesh treated together)
        ("DE", "FR"), ("DE", "PL"), ("DE", "SE"), ("DE", "IT"),
        ("FR", "GB"), ("FR", "IT"), ("FR", "DE"),
        ("IT", "FR"), ("IT", "DE"),
        ("PL", "SE"), ("PL", "DE"), ("PL", "UA"),
        ("SE", "DE"), ("SE", "PL"),
        ("GB", "FR"),
        ("UA", "PL"),  # post-2022 synchronization with ENTSO-E
        # Within North America
        ("US", "CA"), ("CA", "US"), ("US", "MX"), ("MX", "US"),
        # Within Latin America (Itaipu, mesh)
        ("BR", "PY"), ("PY", "BR"), ("BR", "AR"), ("AR", "BR"),
        ("BR", "CO"), ("CO", "BR"),
        # Within Maghreb (limited)
        ("MA", "DZ"), ("DZ", "TN"),
        # Within Asia-Pacific (limited)
        ("MY", "TH"), ("TH", "MY"),
    },
    2030: {
        # 2025 corridors + under construction
        # (will be union-ed with 2025 in code; here we list ADDITIONAL pairs)
        ("MA", "GB"), ("GB", "MA"),    # Xlinks Morocco-UK 3,800 km submarine
        ("DE", "GB"), ("GB", "DE"),    # NeuConnect under construction
        ("MA", "FR"), ("FR", "MA"),    # Mediterranean Sea proposed projects
        ("DZ", "IT"), ("IT", "DZ"),    # Mediterranean energy ring
        ("UA", "DE"), ("UA", "SE"),    # Post-2022 EU integration extensions
        ("BR", "PE"), ("PE", "BR"),    # Andean grid integration
        ("AR", "CL"), ("CL", "AR"),    # Andean integration
        ("ID", "MY"), ("MY", "ID"),    # ASEAN power grid
        ("TH", "ID"), ("ID", "TH"),    # ASEAN power grid
    },
    2050: {
        # 2030 + aspirational with financial pre-conditions
        ("JP", "KR"), ("KR", "JP"),    # Korea-Japan submarine cable (long-proposed)
        ("ZA", "MA"), ("MA", "ZA"),    # Pan-African transmission backbone
        ("IN", "MY"), ("MY", "IN"),    # Andaman Sea cable
        ("ZA", "BR"), ("BR", "ZA"),    # Trans-Atlantic (Sun Cable scale, aspirational)
        ("ZA", "IT"), ("IT", "ZA"),    # Africa-Europe integration
        ("AU", "ID"), ("ID", "AU"),    # Sun-Cable-like Australia-Indonesia
    },
}

# 31-country centroids (approximate, used for great-circle latency estimation)
COUNTRY_CENTROIDS = {
    "AR": (-64, -34), "AU": (134, -25), "BR": (-55, -10), "CA": (-100, 56),
    "CL": (-71, -35), "CO": (-74, 4),   "DE": (10, 51),   "DZ": (3, 28),
    "FR": (2, 47),    "GB": (-2, 54),   "GH": (-1, 8),    "ID": (118, -2),
    "IN": (78, 22),   "IT": (12, 42),   "JP": (138, 36),  "KR": (128, 36),
    "LY": (17, 27),   "MA": (-7, 31),   "MX": (-102, 23), "MY": (102, 4),
    "NG": (8, 10),    "NZ": (172, -41), "PE": (-76, -10), "PL": (19, 52),
    "PY": (-58, -23), "SE": (15, 62),   "TH": (101, 15),  "TN": (9, 34),
    "UA": (32, 49),   "US": (-100, 38), "ZA": (24, -29),
}

# R3 (AI perturbation) is evaluated over the FULL 4x4 power x workload matrix
# (same 16 scenarios as Result 2), not just the four corners, so the L x W
# satisfaction-rate heatmap in Figure 3 can be drawn. The four corners
# (L0_W0 / L3_W0 / L0_W3 / L3_W3) remain the labelled reference points.
R3_SCENARIO_SUBSET = [f"{p}_{w}" for p in ("L0", "L1", "L2", "L3")
                      for w in ("W0", "W1", "W2", "W3")]
R3_CORNERS = ["L0_W0", "L3_W0", "L0_W3", "L3_W3"]

# Kept for backward compatibility
MIGRATION_FRACTION = 0.30

# Storage round-trip efficiency for cost translation
STORAGE_ROUND_TRIP = 0.85


def battery_annualized_cost_per_mwh(capex_usd_per_kwh: float) -> float:
    """Return annualized storage cost per MWh of yearly supply (4h battery,
    daily-cycling proxy). Includes capital recovery + fixed O&M.
    """
    r = DISCOUNT_RATE
    n = BATTERY_LIFE_YR
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    return capex_usd_per_kwh * 1000.0 * (crf + FIXED_OM_FRAC) / 365.0


# -------------------- data loading --------------------

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH, parse_dates=["timestamp_utc"])
    df = df.sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)
    df["region"] = df["region"].astype(str)
    df["continent"] = df["cf_location"].map(CONTINENT).fillna("Other")
    return df


def country_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-country dataframe of meta info."""
    cols = ["cf_location", "tong_country", "region", "continent"]
    meta = panel[cols].drop_duplicates().reset_index(drop=True)
    return meta


# -------------------- R1 helpers --------------------

def per_country_curves(panel: pd.DataFrame) -> dict[str, dict]:
    """For each country, return demand and supply arrays plus timestamp index.

    Demand is `demand_scaled` (equal-energy normalized so mean(d) = mean(s_vre)).
    Supply is the capacity-weighted VRE curve already in the panel.
    PV / wind separately are kept as basis functions for R2.
    """
    out: dict[str, dict] = {}
    for cc, sub in panel.groupby("cf_location"):
        sub = sub.dropna(subset=["demand_scaled", "supply_vre_cf"]).sort_values("timestamp_utc")
        if len(sub) < 24 * 30:
            continue
        d = sub["demand_scaled"].to_numpy(dtype=float)
        s = sub["supply_vre_cf"].to_numpy(dtype=float)
        pv = sub["supply_pv_cf"].to_numpy(dtype=float)
        wd = sub["supply_wind_cf"].to_numpy(dtype=float)
        lh = sub["local_hour"].to_numpy(dtype=int)
        ts = pd.DatetimeIndex(sub["timestamp_utc"].values)
        # Force equal-energy alignment per country (mean(s) = mean(d))
        if s.mean() > 0:
            s = s * (d.mean() / s.mean())
        out[cc] = {
            "ts": ts,
            "d": pd.Series(d, index=ts),
            "s": pd.Series(s, index=ts),
            "pv": pd.Series(pv, index=ts),
            "wind": pd.Series(wd, index=ts),
            "local_hour": pd.Series(lh, index=ts),
            "region": sub["region"].iloc[0],
            "continent": sub["continent"].iloc[0],
            "tong_country": sub["tong_country"].iloc[0],
        }
    return out


def as_arr(x):
    """Return numpy array of x (works for pandas Series or ndarray)."""
    return x.values if isinstance(x, pd.Series) else np.asarray(x)


def waveform_mismatch(d: np.ndarray, s: np.ndarray) -> dict:
    """Hour-by-hour mismatch metrics under equal-energy supply."""
    uncovered = np.maximum(d - s, 0.0)
    excess = np.maximum(s - d, 0.0)
    return {
        "mean_demand": float(d.mean()),
        "uncovered_share": float(uncovered.sum() / d.sum()),
        "excess_share": float(excess.sum() / d.sum()),
        "hours_uncovered": int((d > s).sum()),
        "max_uncovered": float(uncovered.max() / max(d.mean(), 1e-12)),
    }


def diurnal_profile(values: np.ndarray, local_hour: np.ndarray) -> np.ndarray:
    """Return mean profile by local hour 0-23."""
    out = np.zeros(24, dtype=float)
    for h in range(24):
        mask = local_hour == h
        if mask.any():
            out[h] = values[mask].mean()
    return out


def peak_local_hour(values: np.ndarray, local_hour: np.ndarray) -> int:
    prof = diurnal_profile(values, local_hour)
    return int(np.argmax(prof))


def spectral_share(values: np.ndarray, period_low: float, period_high: float,
                   dt_hours: float = 1.0) -> float:
    """Fraction of total spectral power in [period_low, period_high] hours."""
    x = values - values.mean()
    n = len(x)
    if n < 48:
        return float("nan")
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=dt_hours)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    band = (periods >= period_low) & (periods <= period_high)
    total = power[1:].sum()  # exclude DC
    return float(power[band].sum() / total) if total > 0 else float("nan")


def diurnal_phase_lag(d: np.ndarray, s: np.ndarray, local_hour: np.ndarray) -> float:
    """Return signed lag (hours) of demand peak after supply peak (local hours)."""
    pd_ = peak_local_hour(d, local_hour)
    ps = peak_local_hour(s, local_hour)
    lag = (pd_ - ps) % 24
    if lag > 12:
        lag -= 24
    return float(lag)


def run_length_distribution(d: np.ndarray, s: np.ndarray) -> dict:
    deficit = d > s
    runs = []
    cur = 0
    for v in deficit:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    arr = np.array(runs) if runs else np.array([0])
    return {
        "n_runs": int(len(runs)),
        "median_run_hours": float(np.median(arr)) if runs else 0.0,
        "p95_run_hours": float(np.percentile(arr, 95)) if runs else 0.0,
        "max_run_hours": int(arr.max()),
    }


# -------------------- R2 portfolio optimization --------------------

def equal_energy_align(matrix: np.ndarray, target_mean: float) -> np.ndarray:
    """Scale each column to have a given mean. Columns that are all-zero stay 0."""
    out = matrix.copy()
    means = out.mean(axis=0)
    for j in range(out.shape[1]):
        if means[j] > 0:
            out[:, j] = out[:, j] * (target_mean / means[j])
    return out


def portfolio_nnls(d: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve min ||S w - d||_2 s.t. w >= 0 (NNLS), then enforce equal-energy
    by rescaling w so that mean(S w) = mean(d). Returns (w, uncovered_share, fit)."""
    w, _ = nnls(S, d)
    fit = S @ w
    if fit.mean() > 0:
        w = w * (d.mean() / fit.mean())
        fit = S @ w
    uncovered = np.maximum(d - fit, 0.0)
    share = float(uncovered.sum() / d.sum())
    return w, share, fit


def supply_basis(curves: dict, countries: list[str], target_index: pd.DatetimeIndex, fixed: bool = False) -> tuple[np.ndarray, list[str]]:
    """Build a T x K matrix of PV and wind basis functions, aligned to target_index.

    Each basis column has mean = 1 (so that NNLS weights have an interpretable
    'fraction of total capacity' meaning under the equal-energy constraint).
    Missing timestamps are forward-filled then back-filled within the country
    series before reindexing; columns with no valid mean are dropped.
    """
    cols = []
    labels = []
    if fixed and countries:
        c0 = countries[0]
        if c0 in curves and "s" in curves[c0]:
            sblend = curves[c0]["s"].reindex(target_index).ffill().bfill()
            if sblend.notna().all() and sblend.mean() > 0:
                return (sblend / sblend.mean()).values.reshape(-1, 1), [f"{c0}:VRE"]
    for c in countries:
        if c not in curves:
            continue
        pv = curves[c]["pv"].reindex(target_index).ffill().bfill()
        wd = curves[c]["wind"].reindex(target_index).ffill().bfill()
        if pv.notna().all() and pv.mean() > 0:
            cols.append((pv / pv.mean()).values)
            labels.append(f"{c}:PV")
        if wd.notna().all() and wd.mean() > 0:
            cols.append((wd / wd.mean()).values)
            labels.append(f"{c}:WIND")
    if not cols:
        return np.zeros((len(target_index), 0)), []
    return np.stack(cols, axis=1), labels


# Map each of the 31 countries to its nearest Azure region for measured-RTT lookup.
# Source: Azure region map https://azure.microsoft.com/en-us/explore/global-infrastructure/geographies/
COUNTRY_TO_AZURE_REGION = {
    "AR": "Brazil South",        # nearest Azure region in South America
    "AU": "Australia East",
    "BR": "Brazil South",
    "CA": "Canada Central",
    "CL": "Brazil South",        # no Azure in Chile yet
    "CO": "Mexico Central",
    "DE": "Germany West Central",
    "DZ": "Italy North",          # Mediterranean proxy
    "FR": "France Central",
    "GB": "UK South",
    "GH": "South Africa North",   # nearest in Africa
    "ID": "Southeast Asia",       # close to Indonesia Central but use SE Asia for stability
    "IN": "Central India",
    "IT": "Italy North",
    "JP": "Japan East",
    "KR": "Korea Central",
    "LY": "UAE Central",          # closer than EU regions for Eastern Libya
    "MA": "France South",         # nearest EU region (Morocco-Iberia border-ish)
    "MX": "Mexico Central",
    "MY": "Malaysia West",
    "NG": "South Africa North",
    "NZ": "New Zealand North",
    "PE": "Brazil South",
    "PL": "Poland Central",
    "PY": "Brazil South",
    "SE": "Sweden Central",
    "TH": "Southeast Asia",
    "TN": "Italy North",
    "UA": "Poland Central",
    "US": "East US",
    "ZA": "South Africa North",
}


_AZURE_RTT_CACHE: pd.DataFrame | None = None


def load_azure_rtt() -> pd.DataFrame:
    """Load the published Azure region-to-region P50 latency table (ms).
    Cached after first load."""
    global _AZURE_RTT_CACHE
    if _AZURE_RTT_CACHE is None:
        path = OUT_DATA / "azure_rtt_raw.csv"
        _AZURE_RTT_CACHE = pd.read_csv(path, index_col="Source")
    return _AZURE_RTT_CACHE


def build_rtt_matrix_azure(countries: list[str]) -> pd.DataFrame:
    """Build a 31×31 RTT matrix using measured Azure region-to-region latency,
    mapped via COUNTRY_TO_AZURE_REGION."""
    azure = load_azure_rtt()
    n = len(countries)
    out = pd.DataFrame(np.full((n, n), np.nan), index=countries, columns=countries)
    for ci in countries:
        for cj in countries:
            if ci == cj:
                out.loc[ci, cj] = 5.0
                continue
            ai = COUNTRY_TO_AZURE_REGION.get(ci)
            aj = COUNTRY_TO_AZURE_REGION.get(cj)
            if ai is None or aj is None:
                continue
            if ai not in azure.index or aj not in azure.columns:
                continue
            v = azure.at[ai, aj]
            if pd.isna(v) and aj in azure.index and ai in azure.columns:
                v = azure.at[aj, ai]   # use reverse direction as fallback
            out.loc[ci, cj] = float(v) if pd.notna(v) else np.nan
    # If still NaN (Azure region missing data), fall back to distance estimate
    centroids_avail = COUNTRY_CENTROIDS
    for ci in countries:
        for cj in countries:
            if pd.notna(out.loc[ci, cj]):
                continue
            if ci in centroids_avail and cj in centroids_avail:
                lon1, lat1 = centroids_avail[ci]
                lon2, lat2 = centroids_avail[cj]
                d = great_circle_km(lon1, lat1, lon2, lat2)
                out.loc[ci, cj] = 2 * (d / LIGHT_SPEED_FIBER_KM_PER_MS + SWITCHING_OVERHEAD_MS) * WAN_INFLATION_FACTOR
    return out


def great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two (lon, lat) points in kilometres."""
    R = 6371.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_distance_matrix(countries: list[str]) -> pd.DataFrame:
    """Pairwise great-circle distance (km) between country centroids."""
    n = len(countries)
    mat = pd.DataFrame(np.zeros((n, n)), index=countries, columns=countries)
    for ci in countries:
        for cj in countries:
            if ci == cj:
                mat.loc[ci, cj] = 0.0
                continue
            if ci not in COUNTRY_CENTROIDS or cj not in COUNTRY_CENTROIDS:
                mat.loc[ci, cj] = np.inf
                continue
            lon1, lat1 = COUNTRY_CENTROIDS[ci]
            lon2, lat2 = COUNTRY_CENTROIDS[cj]
            mat.loc[ci, cj] = great_circle_km(lon1, lat1, lon2, lat2)
    return mat


def hvdc_corridor_set(year: int) -> set[tuple[str, str]]:
    """Return the cumulative set of HVDC corridor pairs available by `year`."""
    corridors: set[tuple[str, str]] = set()
    for y, pairs in HVDC_CORRIDORS_BY_YEAR.items():
        if y <= year:
            for c, i in pairs:
                corridors.add((c, i))
                corridors.add((i, c))
    return corridors


def hvdc_feasible(distance_matrix: pd.DataFrame, c: str, i: str,
                  max_km: float = DEFAULT_HVDC_MAX_KM,
                  corridor_set: set[tuple[str, str]] | None = None,
                  political_exclusions: set[tuple[str, str]] | None = None,
                  use_corridor_list: bool = False) -> bool:
    """Power feasibility between c and i. Three filters applied in order:
      1) political exclusion mask (hard veto)
      2) if use_corridor_list: must be in the year-cumulative HVDC corridor set
      3) else: great-circle distance ≤ max_km
    Same country is always feasible.
    """
    if c == i:
        return True
    pol = political_exclusions if political_exclusions is not None else POLITICAL_EXCLUSIONS
    if (c, i) in pol or (i, c) in pol:
        return False
    if use_corridor_list:
        if corridor_set is None:
            return False
        return (c, i) in corridor_set
    if c not in distance_matrix.index or i not in distance_matrix.columns:
        return False
    return distance_matrix.at[c, i] <= max_km


def can_receive_compute(country: str) -> bool:
    """Return True if `country` has a hyperscaler region able to host migrated compute."""
    return country in HYPERSCALER_AVAILABLE


def build_rtt_matrix(countries: list[str]) -> pd.DataFrame:
    """Estimated round-trip time (ms) between every pair of countries.

    Uses the Luo & Yang (2026) latency model:
        L_sc one-way ≈ d / (200 km/ms) + 20 ms overhead
        RTT = 2 * L_sc * 1.4 (WAN inflation)
    Intra-country RTT is set to 5 ms (a representative metropolitan value).
    """
    n = len(countries)
    rtt = pd.DataFrame(np.zeros((n, n)), index=countries, columns=countries)
    for ci in countries:
        for cj in countries:
            if ci == cj:
                rtt.loc[ci, cj] = 5.0
                continue
            if ci not in COUNTRY_CENTROIDS or cj not in COUNTRY_CENTROIDS:
                rtt.loc[ci, cj] = np.inf
                continue
            lon1, lat1 = COUNTRY_CENTROIDS[ci]
            lon2, lat2 = COUNTRY_CENTROIDS[cj]
            d = great_circle_km(lon1, lat1, lon2, lat2)
            one_way = d / LIGHT_SPEED_FIBER_KM_PER_MS + SWITCHING_OVERHEAD_MS
            rtt.loc[ci, cj] = 2.0 * one_way * WAN_INFLATION_FACTOR
    return rtt


def power_pool(panel_meta: pd.DataFrame, demand_country: str, scope: str,
               distance_matrix: pd.DataFrame | None = None,
               hvdc_max_km: float = DEFAULT_HVDC_MAX_KM,
               corridor_set: set[tuple[str, str]] | None = None,
               political_exclusions: set[tuple[str, str]] | None = None,
               use_corridor_list: bool = False) -> list[str]:
    """Return supply-country codes whose power can reach `demand_country` under
    the given scope AND real-world feasibility (HVDC + political + cloud)."""
    region = panel_meta.set_index("cf_location").at[demand_country, "region"]
    cont = panel_meta.set_index("cf_location").at[demand_country, "continent"]
    if scope == "L0":
        return [demand_country]
    if scope == "L1":
        candidates = [c for c, r in zip(panel_meta["cf_location"], panel_meta["region"]) if r == region]
    elif scope == "L2":
        candidates = [c for c, k in zip(panel_meta["cf_location"], panel_meta["continent"]) if k == cont]
    elif scope == "L3":
        candidates = list(panel_meta["cf_location"])
    else:
        raise KeyError(scope)
    if distance_matrix is None:
        distance_matrix = build_distance_matrix(list(panel_meta["cf_location"]))
    return [i for i in candidates
            if hvdc_feasible(distance_matrix, demand_country, i,
                              max_km=hvdc_max_km,
                              corridor_set=corridor_set,
                              political_exclusions=political_exclusions,
                              use_corridor_list=use_corridor_list)]


def compute_pool(panel_meta: pd.DataFrame, demand_country: str, scope: str) -> list[str]:
    """Return countries that can RECEIVE compute migrated from `demand_country`
    under the given compute-migration scope. Excludes the home country itself.
    M0 returns empty (no migration); M1 regional, M2 continental, M3 global."""
    if scope == "M0":
        return []
    region = panel_meta.set_index("cf_location").at[demand_country, "region"]
    cont = panel_meta.set_index("cf_location").at[demand_country, "continent"]
    if scope == "M1":
        return [c for c, r in zip(panel_meta["cf_location"], panel_meta["region"])
                if r == region and c != demand_country]
    if scope == "M2":
        return [c for c, k in zip(panel_meta["cf_location"], panel_meta["continent"])
                if k == cont and c != demand_country]
    if scope == "M3":
        return [c for c in panel_meta["cf_location"] if c != demand_country]
    raise KeyError(scope)


def reshape_demand_with_compute_pool(curves: dict, demand_country: str,
                                     compute_receivers: list[str],
                                     alpha: float,
                                     d_baseline: pd.Series | None = None) -> pd.Series:
    """Migrate up to fraction `alpha` of `d_baseline` (defaults to home demand)
    out of `demand_country` to the receiver pool, in hours where the home has a
    deficit and at least one receiver has surplus. Receivers are constrained by
    compute-migration scope; surplus is summed across the receiver pool.

    Returns the effective demand at the home country after migration.
    """
    d_home = curves[demand_country]["d"] if d_baseline is None else d_baseline
    s_home = curves[demand_country]["s"]
    idx = d_home.index
    if alpha <= 0 or not compute_receivers:
        return d_home.copy()
    pool_surplus = pd.Series(0.0, index=idx)
    for c in compute_receivers:
        if c == demand_country or c not in curves:
            continue
        s_c = curves[c]["s"].reindex(idx).ffill().bfill()
        d_c = curves[c]["d"].reindex(idx).ffill().bfill()
        pool_surplus = pool_surplus + np.maximum(s_c - d_c, 0)
    home_deficit = np.maximum(d_home - s_home, 0)
    migrate = np.minimum(alpha * d_home, np.minimum(home_deficit, pool_surplus))
    return d_home - migrate


# Backward-compatible alias
def reshape_demand_follow_sun(curves, demand_country, pool, alpha):
    return reshape_demand_with_compute_pool(curves, demand_country,
                                            [c for c in pool if c != demand_country],
                                            alpha)


def reshape_demand_workload_class(curves: dict, demand_country: str,
                                  workload_scenario: str,
                                  rtt_matrix: pd.DataFrame,
                                  panel_meta: pd.DataFrame,
                                  d_baseline: pd.Series | None = None
                                  ) -> tuple[pd.Series, float]:
    """Workload-class-aware compute migration.

    Decomposes home demand into the four workload classes (A/B/C/D) using
    WORKLOAD_SHARE. For each class that is enabled in the workload scenario,
    determine which countries are RTT-feasible (RTT <= τ_k); migrate the class
    fraction up to (home_deficit, pool_surplus) constraints. Class A always
    stays local. Returns (effective home demand series, total migrated share).
    """
    d_home = curves[demand_country]["d"] if d_baseline is None else d_baseline
    s_home = curves[demand_country]["s"]
    idx = d_home.index
    enabled = WORKLOAD_SCENARIOS[workload_scenario]
    if not enabled:
        return d_home.copy(), 0.0
    # Class A always stays local
    d_eff = d_home * WORKLOAD_SHARE["A"]
    total_migrated = pd.Series(0.0, index=idx)
    home_deficit = np.maximum(d_home - s_home, 0)
    for cls in ("B", "C", "D"):
        d_class = d_home * WORKLOAD_SHARE[cls]
        if cls not in enabled:
            d_eff = d_eff + d_class
            continue
        # RTT-feasible receivers for this class, intersected with the cloud
        # availability mask (only hyperscaler-capable countries can host compute).
        tau = WORKLOAD_LATENCY_MS[cls]
        receivers = [c for c in panel_meta["cf_location"]
                     if c != demand_country and c in rtt_matrix.columns
                     and rtt_matrix.at[demand_country, c] <= tau
                     and can_receive_compute(c)]
        if not receivers:
            d_eff = d_eff + d_class
            continue
        # Pool surplus from RTT-feasible receivers
        pool_surplus = pd.Series(0.0, index=idx)
        for r in receivers:
            if r not in curves:
                continue
            s_r = curves[r]["s"].reindex(idx).ffill().bfill()
            d_r = curves[r]["d"].reindex(idx).ffill().bfill()
            pool_surplus = pool_surplus + np.maximum(s_r - d_r, 0)
        # Migrate class share, capped by home deficit and pool surplus
        # Distribute home deficit across enabled classes proportional to their share
        class_fraction_of_total = (WORKLOAD_SHARE[cls]
                                   / sum(WORKLOAD_SHARE[c] for c in enabled))
        class_deficit_budget = home_deficit * class_fraction_of_total
        migrate_class = np.minimum(d_class,
                                   np.minimum(class_deficit_budget,
                                              pool_surplus - total_migrated))
        migrate_class = np.maximum(migrate_class, 0)
        d_eff = d_eff + (d_class - migrate_class)
        total_migrated = total_migrated + migrate_class
    migrated_share = float(total_migrated.sum() / max(d_home.sum(), 1e-12))
    return d_eff, migrated_share


def evaluate_scenario_2d(demand_country: str, power_scope: str, workload_scenario: str,
                         curves: dict, panel_meta: pd.DataFrame,
                         rtt_matrix: pd.DataFrame,
                         distance_matrix: pd.DataFrame | None = None,
                         hvdc_max_km: float = DEFAULT_HVDC_MAX_KM,
                         corridor_set: set[tuple[str, str]] | None = None,
                         political_exclusions: set[tuple[str, str]] | None = None,
                         use_corridor_list: bool = False,
                         eta: float = TRANSMISSION_EFFICIENCY,
                         perturbed_demand: pd.Series | np.ndarray | None = None) -> dict:
    """Evaluate a 2D scenario (power_scope, workload_scenario) for a demand country.

    Two independent accessibility axes anchored to literature:
      - power_scope (L0/L1/L2/L3): which countries' supply can flow to home
      - workload_scenario (W0/W1/W2/W3): which workload classes are enabled
        for migration; class-specific RTT feasibility (τ_k from Luo & Yang 2026)
        and pool surplus determine actual migrated amount.
    """
    home_idx = curves[demand_country]["d"].index
    if perturbed_demand is None:
        d_base = curves[demand_country]["d"]
    elif isinstance(perturbed_demand, pd.Series):
        d_base = perturbed_demand.reindex(home_idx).ffill().bfill()
    else:
        d_base = pd.Series(np.asarray(perturbed_demand), index=home_idx)

    d_eff, migrated_share = reshape_demand_workload_class(
        curves, demand_country, workload_scenario, rtt_matrix, panel_meta,
        d_baseline=d_base,
    )

    if distance_matrix is None:
        distance_matrix = build_distance_matrix(list(panel_meta["cf_location"]))
    pool = power_pool(panel_meta, demand_country, power_scope,
                      distance_matrix=distance_matrix, hvdc_max_km=hvdc_max_km,
                      corridor_set=corridor_set,
                      political_exclusions=political_exclusions,
                      use_corridor_list=use_corridor_list)
    S, labels = supply_basis(curves, pool, home_idx)
    if S.size == 0:
        return {"country": demand_country, "power_scope": power_scope,
                "workload_scenario": workload_scenario,
                "scenario": f"{power_scope}_{workload_scenario}",
                "uncovered_share": float("nan"),
                "n_supply_basis": 0, "n_active_basis": 0,
                "migrated_share": migrated_share}

    if power_scope != "L0":
        for j, lab in enumerate(labels):
            src = lab.split(":")[0]
            if src != demand_country:
                S[:, j] = S[:, j] * eta

    target_d = d_eff.values
    S_eq = equal_energy_align(S, float(np.mean(target_d)))
    w, share, fit = portfolio_nnls(target_d, S_eq)
    active = int((w > 1e-4).sum())
    out = {
        "country": demand_country,
        "power_scope": power_scope,
        "workload_scenario": workload_scenario,
        "scenario": f"{power_scope}_{workload_scenario}",
        "uncovered_share": share,
        "n_supply_basis": int(S_eq.shape[1]) if S_eq.size else 0,
        "n_active_basis": active,
        "migrated_share": migrated_share,
    }
    if w.size and S_eq.size:
        order = np.argsort(w)[::-1]
        for rank in range(min(3, len(order))):
            idx = order[rank]
            out[f"top{rank+1}_label"] = labels[idx]
            out[f"top{rank+1}_weight"] = float(w[idx])
    return out


# Backward-compatible wrapper
_OLD_TO_2D = {
    "S0_local": ("L0", "W0"),
    "S1_compute_followsun": ("L0", "W2"),
    "S2_regional": ("L1", "W0"),
    "S3_continental": ("L2", "W0"),
    "S4_global": ("L3", "W0"),
}


def evaluate_scenario(demand_country, scenario, curves, panel_meta,
                      rtt_matrix=None,
                      alpha=MIGRATION_FRACTION, eta=TRANSMISSION_EFFICIENCY,
                      perturbed_demand=None):
    """Backward-compatible wrapper around evaluate_scenario_2d."""
    if rtt_matrix is None:
        rtt_matrix = build_rtt_matrix(list(panel_meta["cf_location"]))
    if scenario in _OLD_TO_2D:
        power, workload = _OLD_TO_2D[scenario]
    elif "_" in scenario and scenario.startswith("L"):
        parts = scenario.split("_")
        power, workload = parts[0], parts[1]
    else:
        raise KeyError(f"unknown scenario {scenario}")
    rec = evaluate_scenario_2d(demand_country, power, workload, curves, panel_meta,
                               rtt_matrix=rtt_matrix, eta=eta,
                               perturbed_demand=perturbed_demand)
    rec["scenario"] = scenario
    return rec


# -------------------- R3 demand perturbation operators --------------------

def operator_flatten(d: np.ndarray, lam_inf: float, lam_train: float = 0.0) -> np.ndarray:
    """P1: 'AI is flat' scenario. Both AI inference (batch) and training flatten
    the curve, so the full AI-related share (lam_inf + lam_train) is averaged
    toward the annual mean."""
    lam = lam_inf + lam_train
    return (1 - lam) * d + lam * d.mean()


# P2 / P3 parameters: empirical fits from BurstGPT v2.0 trace (Azure OpenAI
# GPT serving, ~10.1M cleaned requests across 3 trace files / 335 days, hour-
# of-day calibrated to local user timezone). 95% bootstrap CIs are reported in
# burstgpt_workload_shape_params_v2.json. The gamma derivation uses the
# cross-country median d_burst/mean = 1.72 from Cloudflare hourly demand
# (compute_country_burst_ratios.py), replacing the prior 1.75 constant.
#
# v2 pooled fit (token-weighted, 10.1M requests):
#   peak_hour: 18.05 [17.75, 18.39] (95% CI)
#   sigma_h:    3.79 [3.53, 4.05]   (95% CI)
#   gamma_emp = P99/mean(hourly) = 9.31; gamma_op = (9.31 - 1) / 1.72 = 4.84
P2_PEAK_HOUR = 18.0           # v2 fit; was 17.7 from single-file v1 fit
P2_SIGMA_H = 3.8              # v2 fit; was 4.2 from single-file v1 fit
P2_PEAK_HOUR_STYLIZED = 13.0  # legacy stylized; kept for §4.6 sensitivity
P2_SIGMA_H_STYLIZED = 3.0     # legacy stylized; kept for §4.6 sensitivity

P3_GAMMA = 4.84               # v2 fit + per-country d_burst/mean median
P3_GAMMA_STYLIZED = 3.0       # legacy stylized; kept for §4.6 sensitivity

# Composite workload mix (P_mix). Real AI inference is a mix of latency
# regimes (batch / interactive / event-bursty). Mix weights are an AI-specific
# adaptation of Carbon Responder (Xing et al. 2023 Meta) workload classes,
# adjusted for production LLM serving where chat/copilot dominate:
#   alpha_batch:       0.25  scheduled / batchable inference (embeddings,
#                            content moderation, eval, fine-tune jobs)
#   alpha_interactive: 0.65  real-time chat / copilot (BurstGPT-shaped diurnal)
#   alpha_burst:       0.10  event-triggered viral / news spikes
# Sensitivity bands reported in SI Section S5.
PMIX_ALPHA_BATCH = 0.25
PMIX_ALPHA_INTERACTIVE = 0.65
PMIX_ALPHA_BURST = 0.10

# BurstGPT v2 empirical 24-hour diurnal shape (token-share weighted), used as a
# Gaussian-free alternative to the P2 sharpen operator. Loaded lazily from
# reports/ai_inference_growth_curve_fit/burstgpt_diurnal_fit_v2.csv to avoid
# import-time IO; see _load_burstgpt_empirical_shape() below. Falls back to the
# Gaussian default if the fit file is missing.
_BURSTGPT_EMP_SHAPE_24H = None


def _load_burstgpt_empirical_shape() -> np.ndarray:
    """Return BurstGPT-empirical 24-vector (token-share, sum=1)."""
    global _BURSTGPT_EMP_SHAPE_24H
    if _BURSTGPT_EMP_SHAPE_24H is not None:
        return _BURSTGPT_EMP_SHAPE_24H
    p = (ROOT / "collective_attention_research_plan" / "reports"
         / "ai_inference_growth_curve_fit" / "burstgpt_diurnal_fit_v2.csv")
    if p.exists():
        df = pd.read_csv(p)
        if {"hour", "share_tokens"}.issubset(df.columns) and len(df) == 24:
            shape = df.sort_values("hour")["share_tokens"].to_numpy()
            _BURSTGPT_EMP_SHAPE_24H = shape
            return shape
    # Fallback: Gaussian with P2 defaults (silent — caller will use Gaussian)
    h = np.arange(24)
    g = np.exp(-((h - P2_PEAK_HOUR) ** 2) / (2 * P2_SIGMA_H ** 2))
    g = g / g.sum()
    _BURSTGPT_EMP_SHAPE_24H = g
    return g


def operator_sharpen(d: np.ndarray, local_hour: np.ndarray, lam_inf: float,
                     lam_train: float = 0.0,
                     peak_hour: float = P2_PEAK_HOUR,
                     sigma_h: float = P2_SIGMA_H) -> np.ndarray:
    """AI-inference working-hours profile.

    Only the INFERENCE share (lam_inf) is reshaped to the BurstGPT v2 evening
    Gaussian (peak_hour, sigma_h) = (18.0, 3.8); the TRAINING share (lam_train)
    enters as flat baseload (it is ~constant, schedulable load and does not
    sharpen the evening peak). The legacy stylized (13.0, 3.0) is exposed via
    P2_*_STYLIZED. Energy is preserved: mean stays at d.mean().
    """
    h = np.arange(24)
    work = np.exp(-((h - peak_hour) ** 2) / (2 * sigma_h ** 2))
    work = work / work.mean()
    work_hourly = work[local_hour]
    mean_d = d.mean()
    return (1 - lam_inf - lam_train) * d + lam_inf * work_hourly * mean_d + lam_train * mean_d


def operator_sharpen_emp(d: np.ndarray, local_hour: np.ndarray, lam_inf: float,
                          lam_train: float = 0.0,
                          empirical_shape: np.ndarray | None = None) -> np.ndarray:
    """AI-inference shape operator using BurstGPT's empirical 24-h profile
    directly, removing the Gaussian-fitting assumption from the chain.

    The empirical shape (token-weighted, 10.1M Azure OpenAI GPT requests) is
    bimodal — peaks at hours 16-17 and 19-20 — and not well-summarized by a
    single Gaussian. Only the INFERENCE share (lam_inf) takes this shape; the
    TRAINING share (lam_train) is flat baseload. See SI § S2.7.
    """
    shape = empirical_shape if empirical_shape is not None else _load_burstgpt_empirical_shape()
    # Normalize so mean = 1 (rather than sum = 1), to match the equal-energy
    # alignment used by operator_sharpen.
    shape_norm = shape * 24.0 / shape.sum()
    work_hourly = shape_norm[local_hour]
    mean_d = d.mean()
    return (1 - lam_inf - lam_train) * d + lam_inf * work_hourly * mean_d + lam_train * mean_d


def operator_burst(d: np.ndarray, lam_inf: float, lam_train: float = 0.0,
                   gamma: float = P3_GAMMA) -> np.ndarray:
    """Amplify the top-1% of demand hours by gamma*lam_inf — bursting is an
    INFERENCE phenomenon (event-driven viral/news spikes in serving), so it
    scales with the inference share only. The TRAINING share (lam_train) is
    flattened to baseload and does not burst.

    Default gamma = 4.84 from BurstGPT v2 hourly P99/mean = 9.31 mapped via
    per-country median d_burst/mean = 1.72 (Cloudflare). Stylized gamma = 3.0
    kept for §4.6 sensitivity.
    """
    mean_d = d.mean()
    d_train_flat = (1 - lam_train) * d + lam_train * mean_d
    thresh = np.quantile(d, 0.99)
    burst_mask = (d >= thresh).astype(float)
    if burst_mask.sum() == 0:
        return d_train_flat
    amp = burst_mask * d * (gamma * lam_inf)
    return d_train_flat + amp


def operator_mix(d: np.ndarray, local_hour: np.ndarray, lam_inf: float,
                 lam_train: float = 0.0,
                 alpha_batch: float = PMIX_ALPHA_BATCH,
                 alpha_interactive: float = PMIX_ALPHA_INTERACTIVE,
                 alpha_burst: float = PMIX_ALPHA_BURST,
                 peak_hour: float = P2_PEAK_HOUR,
                 sigma_h: float = P2_SIGMA_H,
                 gamma: float = P3_GAMMA) -> np.ndarray:
    """Composite AI workload mix: literature-anchored linear combination of a
    batch (flat) component and a demand-following component shaped by the
    BurstGPT v2 evening profile.

    Defaults match Carbon Responder (Xing et al. 2023 Meta) adapted to AI-
    inference dominance: 25% batch + 75% demand-following (the former 65%
    interactive + 10% event share, now sharing the SAME smooth evening shape).
    The event share no longer adds a top-1%-hour amplification spike: that spike
    concentrated all event load into the single evening peak hour and injected
    extra energy straight into the post-sunset deficit window. Event-driven
    bursting is instead studied on its own in the dedicated P3 burst sensitivity
    scenario, rather than baked into the headline Mix. The composite shape is
    applied to the INFERENCE share (lam_inf); the TRAINING share (lam_train) is
    added as flat baseload. `gamma` is retained for call-signature compatibility
    but is no longer used.
    """
    assert abs(alpha_batch + alpha_interactive + alpha_burst - 1.0) < 1e-6, "mix weights must sum to 1"
    mean_d = float(d.mean())
    h = np.arange(24)
    work = np.exp(-((h - peak_hour) ** 2) / (2 * sigma_h ** 2))
    work = work / work.mean()
    work_hourly = work[local_hour]
    # Batch = flat baseload; the interactive AND event shares both follow the
    # same smooth BurstGPT evening profile (no top-1% spike). All pieces are
    # mean-preserving, so the composite stacks correctly under equal-energy
    # alignment and adds no extra energy into the evening deficit window.
    d_batch = mean_d  # scalar broadcast
    d_shaped = work_hourly * mean_d
    d_ai = alpha_batch * d_batch + (alpha_interactive + alpha_burst) * d_shaped
    return (1 - lam_inf - lam_train) * d + lam_inf * d_ai + lam_train * mean_d


# -------------------- R3 cost translation --------------------

def gap_to_storage_cost(uncovered_share: float, capex_yr: int, sensitivity: str = "mid",
                        plant_mw: float = 100.0) -> float:
    """Convert annual uncovered share (as fraction of mean demand) for a
    `plant_mw` MW plant to annualized storage cost in USD/year.

    Annual MWh of supply needed: uncovered_share * 8760 * plant_mw.
    """
    capex = NREL_BATTERY_CAPEX[capex_yr][sensitivity]
    cost_per_mwh = battery_annualized_cost_per_mwh(capex)
    annual_mwh = uncovered_share * 8760 * plant_mw
    return cost_per_mwh * annual_mwh / STORAGE_ROUND_TRIP


# ==================== main pipeline ====================

def run_r1(curves: dict) -> pd.DataFrame:
    rows = []
    for cc, c in curves.items():
        d, s, lh = as_arr(c["d"]), as_arr(c["s"]), as_arr(c["local_hour"])
        wm = waveform_mismatch(d, s)
        rl = run_length_distribution(d, s)
        rows.append({
            "country": cc,
            "tong_country": c["tong_country"],
            "region": c["region"],
            "continent": c["continent"],
            "uncovered_share_local": wm["uncovered_share"],
            "excess_share_local": wm["excess_share"],
            "max_uncovered_x_mean": wm["max_uncovered"],
            "n_runs": rl["n_runs"],
            "median_run_h": rl["median_run_hours"],
            "p95_run_h": rl["p95_run_hours"],
            "max_run_h": rl["max_run_hours"],
            "peak_demand_local_h": peak_local_hour(d, lh),
            "peak_supply_local_h": peak_local_hour(s, lh),
            "phase_lag_h": diurnal_phase_lag(d, s, lh),
            "spectral_diurnal_d": spectral_share(d, 22, 26),
            "spectral_diurnal_s": spectral_share(s, 22, 26),
            "spectral_weekly_d": spectral_share(d, 24*6, 24*8),
            "spectral_weekly_s": spectral_share(s, 24*6, 24*8),
            "spectral_seasonal_d": spectral_share(d, 24*60, 24*200),
            "spectral_seasonal_s": spectral_share(s, 24*60, 24*200),
        })
    return pd.DataFrame(rows).sort_values("uncovered_share_local", ascending=False)


def run_r2(curves: dict, panel_meta: pd.DataFrame,
           rtt_matrix: pd.DataFrame | None = None,
           distance_matrix: pd.DataFrame | None = None,
           hvdc_max_km: float = DEFAULT_HVDC_MAX_KM,
           corridor_set: set[tuple[str, str]] | None = None,
           political_exclusions: set[tuple[str, str]] | None = None,
           use_corridor_list: bool = True,
           political_level_label: str = DEFAULT_POLITICAL_LEVEL) -> pd.DataFrame:
    """Sweep the full 4 × 4 = 16 power × workload-class matrix per country.

    By default uses the 2025 HVDC corridor list and LOOSE political mask.
    """
    if rtt_matrix is None:
        rtt_matrix = build_rtt_matrix(list(panel_meta["cf_location"]))
    if distance_matrix is None:
        distance_matrix = build_distance_matrix(list(panel_meta["cf_location"]))
    if corridor_set is None and use_corridor_list:
        corridor_set = hvdc_corridor_set(2025)
    rows = []
    for cc in curves:
        for power in POWER_SCOPES:
            for workload in WORKLOAD_SCENARIO_CODES:
                rec = evaluate_scenario_2d(cc, power, workload, curves, panel_meta,
                                           rtt_matrix=rtt_matrix,
                                           distance_matrix=distance_matrix,
                                           hvdc_max_km=hvdc_max_km,
                                           corridor_set=corridor_set,
                                           political_exclusions=political_exclusions,
                                           use_corridor_list=use_corridor_list)
                rec["region"] = curves[cc]["region"]
                rec["continent"] = curves[cc]["continent"]
                rec["hvdc_max_km"] = hvdc_max_km
                rec["political_level"] = political_level_label
                rows.append(rec)
    return pd.DataFrame(rows)


def run_political_sensitivity(curves: dict, panel_meta: pd.DataFrame,
                              rtt_matrix: pd.DataFrame,
                              distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Sweep 4 political mask relaxation levels × 4 corner scenarios."""
    levels = list(POLITICAL_MASK_LEVELS.keys())
    corner_scenarios = ["L0_W0", "L3_W0", "L0_W3", "L3_W3"]
    corridor_set_2025 = hvdc_corridor_set(2025)
    rows = []
    for lvl in levels:
        pol = POLITICAL_MASK_LEVELS[lvl]
        for cc in curves:
            for sc in corner_scenarios:
                power, workload = sc.split("_")
                rec = evaluate_scenario_2d(cc, power, workload, curves, panel_meta,
                                           rtt_matrix=rtt_matrix,
                                           distance_matrix=distance_matrix,
                                           corridor_set=corridor_set_2025,
                                           political_exclusions=pol,
                                           use_corridor_list=True)
                rec["political_level"] = lvl
                rec["n_political_pairs"] = len(pol)
                rec["scenario"] = sc
                rec["region"] = curves[cc]["region"]
                rows.append(rec)
    return pd.DataFrame(rows)


def run_r3(curves: dict, panel_meta: pd.DataFrame,
           rtt_matrix: pd.DataFrame | None = None,
           distance_matrix: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rtt_matrix is None:
        rtt_matrix = build_rtt_matrix(list(panel_meta["cf_location"]))
    if distance_matrix is None:
        distance_matrix = build_distance_matrix(list(panel_meta["cf_location"]))
    perturbations = ["P0_baseline", "P1_flatten", "P2_sharpen", "P3_burst", "P_mix"]
    lambdas = AI_SHARE
    # Defaults come from the module-level P2_*, P3_GAMMA, and PMIX_* constants
    # which were fitted from BurstGPT v2 (~10.1M req) + per-country Cloudflare
    # d_burst/mean. See top-of-module docstring and SI Section S2-S3.
    burst_gammas = {"P3_burst": P3_GAMMA}
    years = [2025, 2030, 2050]
    scenarios = R3_SCENARIO_SUBSET  # 4 corner combinations

    per_country_rows = []
    for year in years:
        lam_inf = AI_SHARE_INFERENCE[year]
        lam_train = AI_SHARE_TRAINING[year]
        lam = lambdas[year]  # total AI-related share, for the `ai_share` column
        for cc in curves:
            d_base_s = curves[cc]["d"]
            d_base = as_arr(d_base_s)
            lh = as_arr(curves[cc]["local_hour"])
            home_idx = d_base_s.index
            for op in perturbations:
                if op == "P0_baseline":
                    d_pert = d_base_s
                elif op == "P1_flatten":
                    d_pert = pd.Series(operator_flatten(d_base, lam_inf, lam_train), index=home_idx)
                elif op == "P2_sharpen":
                    d_pert = pd.Series(operator_sharpen(d_base, lh, lam_inf, lam_train), index=home_idx)
                elif op == "P3_burst":
                    d_pert = pd.Series(operator_burst(d_base, lam_inf, lam_train, burst_gammas["P3_burst"]), index=home_idx)
                elif op == "P_mix":
                    d_pert = pd.Series(operator_mix(d_base, lh, lam_inf, lam_train), index=home_idx)
                # Year-dependent HVDC corridor list + distance cap (compatible fallback)
                hvdc_cap_yr = HVDC_MAX_KM_BY_YEAR.get(year, DEFAULT_HVDC_MAX_KM)
                corridor_yr = hvdc_corridor_set(year)
                for sc in scenarios:
                    parts = sc.split("_")
                    power, workload = parts[0], parts[1]
                    rec = evaluate_scenario_2d(cc, power, workload, curves, panel_meta,
                                                rtt_matrix=rtt_matrix,
                                                distance_matrix=distance_matrix,
                                                hvdc_max_km=hvdc_cap_yr,
                                                corridor_set=corridor_yr,
                                                use_corridor_list=True,
                                                perturbed_demand=d_pert)
                    rec["scenario"] = sc
                    rec["hvdc_max_km"] = hvdc_cap_yr
                    rec["n_hvdc_corridors_active"] = len(corridor_yr) // 2  # undirected
                    share = rec.get("uncovered_share", float("nan"))
                    if np.isnan(share):
                        continue
                    # use the appropriate year's capex; for 2040 fall back to 2030
                    capex_yr = year if year in NREL_BATTERY_CAPEX else 2030
                    storage_cost_mid = gap_to_storage_cost(share, capex_yr=capex_yr, sensitivity="mid")
                    storage_cost_low = gap_to_storage_cost(share, capex_yr=capex_yr, sensitivity="low")
                    per_country_rows.append({
                        "year": year,
                        "lambda": lam,
                        "lambda_inf": lam_inf,
                        "lambda_train": lam_train,
                        "country": cc,
                        "region": curves[cc]["region"],
                        "continent": curves[cc]["continent"],
                        "perturbation": op,
                        "scenario": sc,
                        "uncovered_share": share,
                        "annual_unmet_mwh_per_100mw": share * 8760 * 100,
                        "storage_cost_usd_yr_per_100mw_mid": storage_cost_mid,
                        "storage_cost_usd_yr_per_100mw_low": storage_cost_low,
                    })

    per_country_df = pd.DataFrame(per_country_rows)

    # Global aggregate: average uncovered share across countries x fleet GW
    agg_rows = []
    for year in years:
        fleet_gw = GLOBAL_FLEET_GW[year]
        n_units = fleet_gw * 1000.0 / 100.0  # number of 100 MW units
        for op in per_country_df["perturbation"].unique():
            for sc in per_country_df["scenario"].unique():
                sub = per_country_df[(per_country_df["year"] == year) &
                                     (per_country_df["perturbation"] == op) &
                                     (per_country_df["scenario"] == sc)]
                if sub.empty:
                    continue
                mean_share = sub["uncovered_share"].mean()
                annual_mwh = mean_share * 8760 * 100 * n_units
                capex_yr = year if year in NREL_BATTERY_CAPEX else 2030
                cost_mid = battery_annualized_cost_per_mwh(NREL_BATTERY_CAPEX[capex_yr]["mid"]) * annual_mwh / STORAGE_ROUND_TRIP
                cost_low = battery_annualized_cost_per_mwh(NREL_BATTERY_CAPEX[capex_yr]["low"]) * annual_mwh / STORAGE_ROUND_TRIP
                agg_rows.append({
                    "year": year,
                    "perturbation": op,
                    "scenario": sc,
                    "mean_uncovered_share": mean_share,
                    "global_annual_unmet_gwh": annual_mwh / 1000.0,
                    "global_storage_cost_musd_yr_mid": cost_mid / 1e6,
                    "global_storage_cost_usd_yr_low": cost_low / 1e6,
                    "global_fleet_gw": fleet_gw,
                    "ai_share": AI_SHARE[year],
                    "capex_year_used": capex_yr,
                })
    return per_country_df, pd.DataFrame(agg_rows)


def main():
    print(f"loading panel: {PANEL_PATH}")
    panel = load_panel()
    meta = country_table(panel)
    curves = per_country_curves(panel)
    print(f"countries with curves: {len(curves)}")

    print("R1 ...")
    r1 = run_r1(curves)
    r1.to_csv(OUT_DATA / "r1_country_mismatch.csv", index=False)
    print(f"  -> {OUT_DATA / 'r1_country_mismatch.csv'}  ({len(r1)} rows)")

    print("RTT matrix (Azure measured, mapped to 31 countries) ...")
    rtt_matrix = build_rtt_matrix_azure(list(meta["cf_location"]))
    rtt_matrix.to_csv(OUT_DATA / "rtt_matrix_ms.csv")
    print(f"  -> {OUT_DATA / 'rtt_matrix_ms.csv'}  shape: {rtt_matrix.shape}")
    # Sanity-check some published anchors
    sample_pairs = [("US", "GB"), ("US", "JP"), ("US", "DE"), ("JP", "AU"),
                    ("DE", "FR"), ("BR", "DE"), ("US", "BR")]
    print("  RTT sanity check (estimated vs published cloud-region anchors):")
    for a, b in sample_pairs:
        if a in rtt_matrix.index and b in rtt_matrix.columns:
            print(f"    {a}-{b}: {rtt_matrix.at[a, b]:.0f} ms")

    distance_matrix_pre = build_distance_matrix(list(meta["cf_location"]))
    corridor_2025 = hvdc_corridor_set(2025)
    print(f"HVDC corridor set 2025: {len(corridor_2025)//2} unique pairs")
    print("R2 (Azure RTT + 2025 HVDC corridor list + LOOSE political mask) ...")
    r2 = run_r2(curves, meta, rtt_matrix=rtt_matrix,
                distance_matrix=distance_matrix_pre,
                hvdc_max_km=HVDC_MAX_KM_BY_YEAR[2025],
                corridor_set=corridor_2025,
                political_exclusions=POLITICAL_MASK_LEVELS[DEFAULT_POLITICAL_LEVEL],
                use_corridor_list=True,
                political_level_label=DEFAULT_POLITICAL_LEVEL)
    r2.to_csv(OUT_DATA / "r2_portfolio_scenarios.csv", index=False)
    print(f"  -> {OUT_DATA / 'r2_portfolio_scenarios.csv'}  ({len(r2)} rows)")

    print("Political-mask sensitivity sweep (4 levels × 4 corner scenarios) ...")
    pol_sens = run_political_sensitivity(curves, meta, rtt_matrix, distance_matrix_pre)
    pol_sens.to_csv(OUT_DATA / "r2_political_sensitivity.csv", index=False)
    pol_summary = (pol_sens.groupby(["political_level", "scenario"])["uncovered_share"]
                    .median().unstack("scenario") * 100).round(2)
    print("  median uncovered share by political level × scenario (%):")
    print(pol_summary.to_string())

    # Dump the L3_W0 (global power, no compute migration) weight matrix
    # under the realistic 2025 HVDC distance cap (3,000 km) and political mask.
    # Used by Fig 2(b)(c) for the hub-and-spoke supply network visualisation.
    print("R2 L3_W0 weights (3,000 km HVDC cap) ...")
    s4_rows = []
    for cc in curves:
        home_idx = curves[cc]["d"].index
        d = curves[cc]["d"].values
        pool = power_pool(meta, cc, "L3",
                          distance_matrix=distance_matrix_pre,
                          hvdc_max_km=HVDC_MAX_KM_BY_YEAR[2025])
        S, labels = supply_basis(curves, pool, home_idx)
        if S.size == 0:
            continue
        for j, lab in enumerate(labels):
            src = lab.split(":")[0]
            if src != cc:
                S[:, j] = S[:, j] * TRANSMISSION_EFFICIENCY
        S_eq = equal_energy_align(S, float(np.mean(d)))
        w, share, fit = portfolio_nnls(d, S_eq)
        for j, lab in enumerate(labels):
            src, tech = lab.split(":")
            s4_rows.append({
                "demand_country": cc,
                "supply_country": src,
                "tech": tech,
                "weight": float(w[j]),
                "is_home": (src == cc),
            })
    s4_df = pd.DataFrame(s4_rows)
    s4_df.to_csv(OUT_DATA / "r2_s4_weight_matrix.csv", index=False)
    print(f"  -> {OUT_DATA / 'r2_s4_weight_matrix.csv'}  ({len(s4_df)} rows)")

    distance_matrix_pre.to_csv(OUT_DATA / "distance_matrix_km.csv")
    print(f"  -> {OUT_DATA / 'distance_matrix_km.csv'}  shape: {distance_matrix_pre.shape}")

    print("R3 ...")
    r3_country, r3_global = run_r3(curves, meta, rtt_matrix=rtt_matrix,
                                   distance_matrix=distance_matrix_pre)
    r3_country.to_csv(OUT_DATA / "r3_perturbation_cost.csv", index=False)
    r3_global.to_csv(OUT_DATA / "r3_global_aggregate.csv", index=False)
    print(f"  -> {OUT_DATA / 'r3_perturbation_cost.csv'}  ({len(r3_country)} rows)")
    print(f"  -> {OUT_DATA / 'r3_global_aggregate.csv'}  ({len(r3_global)} rows)")

    # also save the per-country diurnal profiles for figure use
    diurnal_rows = []
    for cc, c in curves.items():
        lh = as_arr(c["local_hour"])
        d = as_arr(c["d"]); s = as_arr(c["s"]); pv = as_arr(c["pv"]); wd = as_arr(c["wind"])
        for h in range(24):
            mask = lh == h
            if not mask.any():
                continue
            diurnal_rows.append({
                "country": cc,
                "region": c["region"],
                "local_hour": h,
                "demand": float(d[mask].mean()),
                "supply_vre": float(s[mask].mean()),
                "supply_pv": float(pv[mask].mean()),
                "supply_wind": float(wd[mask].mean()),
            })
    pd.DataFrame(diurnal_rows).to_csv(OUT_DATA / "r1_diurnal_profiles.csv", index=False)
    print(f"  -> {OUT_DATA / 'r1_diurnal_profiles.csv'}")

    # quick sanity check
    print("\nR1 head:")
    print(r1[["country", "uncovered_share_local", "phase_lag_h", "p95_run_h"]].head(8).to_string(index=False))
    print("\nR2 mean uncovered by scenario:")
    print(r2.groupby("scenario")["uncovered_share"].agg(["median", "mean", "max"]).round(3).to_string())
    print("\nR3 global aggregate (corner scenarios, P0_baseline):")
    sel = r3_global[(r3_global["scenario"].isin(R3_SCENARIO_SUBSET)) & (r3_global["perturbation"] == "P0_baseline")]
    print(sel[["year", "scenario", "mean_uncovered_share", "global_storage_cost_musd_yr_mid"]].to_string(index=False))

    # dump assumptions with literature provenance
    workload_sets = {k: sorted(list(v)) for k, v in WORKLOAD_SCENARIOS.items()}
    (OUT_DATA / "assumptions.json").write_text(json.dumps({
        "AI_SHARE": AI_SHARE,
        "AI_SHARE_INFERENCE": AI_SHARE_INFERENCE,
        "AI_SHARE_TRAINING": AI_SHARE_TRAINING,
        "GLOBAL_FLEET_GW": GLOBAL_FLEET_GW,
        "NREL_BATTERY_CAPEX_USD_PER_KWH": NREL_BATTERY_CAPEX,
        "DISCOUNT_RATE": DISCOUNT_RATE,
        "BATTERY_LIFE_YR": BATTERY_LIFE_YR,
        "FIXED_OM_FRAC": FIXED_OM_FRAC,
        "TRANSMISSION_EFFICIENCY": TRANSMISSION_EFFICIENCY,
        "STORAGE_ROUND_TRIP": STORAGE_ROUND_TRIP,
        "CONTINENT": CONTINENT,
        "burst_gamma_P3": P3_GAMMA,
        "burst_gamma_P3_stylized_legacy": P3_GAMMA_STYLIZED,
        "P2_sharpen_peak_hour": P2_PEAK_HOUR,
        "P2_sharpen_sigma_h": P2_SIGMA_H,
        "P2_sharpen_peak_hour_stylized_legacy": P2_PEAK_HOUR_STYLIZED,
        "P2_sharpen_sigma_h_stylized_legacy": P2_SIGMA_H_STYLIZED,
        "P_mix_alpha_batch": PMIX_ALPHA_BATCH,
        "P_mix_alpha_interactive": PMIX_ALPHA_INTERACTIVE,
        "P_mix_alpha_burst": PMIX_ALPHA_BURST,
        "workload_shape_empirical_sources": {
            "primary_diurnal_and_burst": {
                "dataset": "BurstGPT v2.0 (Azure OpenAI GPT serving trace, 10.1M cleaned requests, 335 days)",
                "url": "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0",
                "paper": "Wang et al. 2024, arXiv:2401.17644",
                "fit_script": "scripts/fit_burstgpt_workload_shape_v2.py",
                "fit_output": "reports/ai_inference_growth_curve_fit/burstgpt_workload_shape_params_v2.json",
                "bootstrap_ci_iters": 500,
            },
            "burst_ratio_d_burst_over_mean": {
                "source": "Cloudflare Radar 104-country hourly human-volume proxy",
                "median_across_countries": 1.716,
                "fit_script": "scripts/compute_country_burst_ratios.py",
                "fit_output": "reports/ai_inference_growth_curve_fit/country_burst_ratio_summary.json",
            },
            "cross_validation": {
                "dataset": "Azure LLM Inference Dataset 2024 (~44M req, 7-10 day window)",
                "url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
                "fit_script": "scripts/fit_azure_llm_inference_2024.py",
                "fit_output": "reports/ai_inference_growth_curve_fit/azure_llm_inference_params.json",
                "note": "Combined Azure (conv+code) Gaussian: peak_utc=17.22h sigma=3.17h - sigma matches BurstGPT 3.79 closely; burst stats lower (1.5-3.1 vs 9.31), reflecting enterprise-vs-consumer service mix.",
            },
        },
        "lambda_multi_source": {
            "note": ("Re-based onto a consistent ENERGY (TWh/TWh) basis and split "
                     "into inference (drives shape operators) vs training (flat "
                     "baseload). Replaces the prior single AI-related ensemble "
                     "(0.231/0.402/0.699) that conflated training into the diurnal "
                     "peak. lambda_AI = energy-basis AI-related share; "
                     "lambda_inf = lambda_AI * f_inf; lambda_train = lambda_AI * (1-f_inf)."),
            "basis": "energy (TWh AI-servers / TWh total data-center electricity)",
            "lambda_AI_total_mid": {"2025": 0.20, "2030": 0.44, "2050": 0.70},
            "lambda_AI_total_band": {"2025": [0.15, 0.25], "2030": [0.35, 0.50], "2050": [0.60, 0.80]},
            "inference_fraction_f_inf": {"2025": 0.65, "2030": 0.70, "2050": 0.80},
            "inference_fraction_band": {"2025": [0.60, 0.70], "2030": [0.60, 0.75], "2050": [0.70, 0.85]},
            "lambda_inf_mid": AI_SHARE_INFERENCE,
            "lambda_train_mid": AI_SHARE_TRAINING,
            "energy_basis_sources": ["Gartner (AI-optimized-server TWh share 21%->44%)",
                                     "IEA-4E/EDNA (AI-related DC energy share 5-15%->35-50%)",
                                     "IEA Energy and AI 2025 (total DC 415->945 TWh)"],
            "inference_split_sources": ["Google/Patterson et al. 2022 (measured fleet 60/40 energy)",
                                        "IEA-4E/EDNA 2025 (60-70% inference / 20-40% training of ML energy)",
                                        "EPRI (~60% inference / ~30% training)"],
            "capacity_basis_excluded": ("McKinsey ~70% AI of DC GW and Goldman 39% of DC power are "
                                        "CAPACITY/power-demand shares (run higher than energy shares); "
                                        "reserved for the 2050 ceiling, not used for lambda values."),
            "ceiling_2050": "lambda_AI saturates at 0.70 (range 0.60-0.80); f_inf ceiling 0.80 (range 0.70-0.85).",
            "fit_script": "scripts/fit_lambda_triangulation.py",
            "fit_output": "reports/ai_inference_growth_curve_fit/lambda_triangulation_summary.json",
        },
        "n_countries": len(curves),
        "workload_class_latency_budget_ms": WORKLOAD_LATENCY_MS,
        "workload_class_share": WORKLOAD_SHARE,
        "workload_scenarios": workload_sets,
        "latency_model": {
            "speed_fiber_km_per_ms": LIGHT_SPEED_FIBER_KM_PER_MS,
            "switching_overhead_ms": SWITCHING_OVERHEAD_MS,
            "wan_inflation_factor": WAN_INFLATION_FACTOR,
            "formula": "RTT = 2 * (d/200 + 20) * 1.4",
            "source": "Luo & Yang (2026) Section 5.4",
        },
        "literature_anchors": {
            "workload_class_taxonomy": "Luo & Yang (2026) Table 4 (arXiv:2604.27855)",
            "workload_share": "Xing et al. (2023) Carbon Responder Fig 1 (Meta)",
            "deferrable_workload_anchor": "Google CICM (Radovanovic et al. 2022); ~20% deferrable in 24h",
            "rtt_anchor": "Azure inter-region RTT public reference; Luo & Yang (2026) Section 5.4",
            "transmission_efficiency_eta": "HVDC literature ~0.85-0.95 cross-border loss range",
        },
        "feasibility_masks": {
            "hvdc_max_km_by_year": HVDC_MAX_KM_BY_YEAR,
            "hvdc_max_km_anchor": "2025 cap = 3,000 km. Belo Monte 2,500 km commercial; Xlinks Morocco-UK 3,800 km under construction. 5,000 km = UHVDC theoretical upper bound.",
            "hvdc_corridor_counts_by_year": {y: len(hvdc_corridor_set(y)) // 2
                                              for y in HVDC_CORRIDORS_BY_YEAR},
            "hvdc_corridor_source": "ENTSO-E grid map, IEA Global Energy and CO2 Status Report, Wikipedia 'List of HVDC projects'",
            "political_mask_levels": {lvl: sorted(list({tuple(sorted(p)) for p in pairs}))
                                      for lvl, pairs in POLITICAL_MASK_LEVELS.items()},
            "political_default_level": DEFAULT_POLITICAL_LEVEL,
            "political_note": "31-country sample does not include the most prominent geopolitical pairs (CN/RU/IR/IL/SA/PK/TW). The mask shown reflects in-scope tensions; absolute political-mask impact on 24/7 CFE would be larger with those countries added.",
            "hyperscaler_available_countries": sorted(list(HYPERSCALER_AVAILABLE)),
            "hyperscaler_unavailable_countries": sorted(list(set(meta["cf_location"]) - HYPERSCALER_AVAILABLE)),
            "hyperscaler_anchor": "AWS/Azure/GCP public region map as of 2025. Countries without major hyperscaler region cannot RECEIVE migrated compute.",
            "rtt_source": "Azure inter-region latency P50 (June 30, 2025 snapshot) mapped to 31 countries via COUNTRY_TO_AZURE_REGION nearest-region table",
        },
    }, indent=2))


if __name__ == "__main__":
    main()
