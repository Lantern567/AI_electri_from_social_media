#!/usr/bin/env python3
"""Verified full-system cost parameters for the Result-3 cost extension.

Single source of truth, used by both decompose_fullsystem_portfolio.py (to
co-optimize the VRE overbuild factor) and layer_fullsystem_cost.py (to price the
cached physical decomposition). Every figure was sourced and adversarially
cross-checked (workflow fullsystem-cost-params-and-audit, 2026-06):

  GENERATION  NREL ATB 2024 v3.0.0 (constant 2022 USD, CRP 30 yr). Utility PV
              capex/FOM per kW_AC (ILR 1.34); land-based wind Class-4 per kW_AC.
              Conservative/Moderate/Advanced -> our high/central/low.
  TRANSMISSION HVDC, 2024 real USD (corrected UP from 2014-2018 EU figures via
              MISO MTEP24 + realised links Viking/NeuConnect). Overhead line
              700 USD/MW/km (440-1100); converter 200 USD/kW/terminal x2
              (120-320); FOM 1%/yr; life 40 yr; route tortuosity x1.3; losses
              ~3%/1000 km line + ~0.8%/terminal (distance-dependent, one-way).
  STORAGE     duration-aware Li-ion (NREL ATB 2024 4h) + LDES (public anchors),
              cheaper-of-two per gap (same model as recompute_fig3c_storage.py).
  DATACENTRE  FACILITY shell+power+cooling 10.7 MUSD/MW (JLL 2025; 9-12),
              non-energy opex 300 USD/kW-yr (Epoch; 200-450), facility life
              15 yr. IT (servers/GPU) 23 MUSD/MW (18-30) at 4-yr refresh, kept
              SEPARATE (facility-only vs facility+IT split).
  ANCILLARY   residual balancing/reserve 3 USD/MWh of VRE generation (1.5-6;
              IEA Wind Task 25 / Hirth-Ueckerdt). NOT the full 25-35 USD/MWh
              integration total -> that would double-count bulk storage+transmission.

CASES: 'central' | 'low' | 'high'. low = cheapest (optimistic), high = priciest.
All capex in real USD; mixed vintages (gen 2022$, tx 2024$, DC 2025$) are within
~10% and reported as a single real-USD figure (documented; deflation < model noise).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = (SCRIPT_DIR.parents[1] / "collective_attention_research_plan" / "reports"
        / "cfe_geographic_portfolio_ai" / "data_globalsites")

E_D_MWH = 100.0 * 8760.0           # annual demand energy per 100 MW-mean unit (876 GWh)
CASES = ["central", "low", "high"]

# ----------------------------------------------------------------- discount/CRF
DISCOUNT = 0.07
DISCOUNT_SENS = {"central": 0.07, "low": 0.05, "high": 0.09}   # for r-sensitivity


def crf(life: float, r: float = DISCOUNT) -> float:
    return r * (1 + r) ** life / ((1 + r) ** life - 1)


# ----------------------------------------------------------------- generation
GEN_LIFE = 30.0
# {year: (central, low, high)} ; USD(2022)/kW_AC
PV_CAPEX = {2025: (1491.64, 1456.24, 1540.63), 2030: (1193.48, 1069.57, 1364.94),
            2040: (824.51, 642.79, 1091.27), 2050: (682.90, 562.57, 895.32)}
WIND_CAPEX = {2025: (1569.10, 1544.24, 1632.64), 2030: (1407.95, 1341.67, 1577.39),
              2040: (1261.43, 1175.93, 1466.91), 2050: (1114.91, 1010.18, 1356.43)}
PV_FOM = {2025: (21.00, 20.60, 21.50), 2030: (17.99, 16.64, 19.72),
          2040: (14.26, 12.30, 16.98), 2050: (12.76, 11.38, 15.02)}      # USD/kW_AC-yr
WIND_FOM = {2025: (31.25, 30.14, 31.87), 2030: (29.26, 26.31, 30.91),
            2040: (27.26, 21.35, 29.67), 2050: (25.25, 16.39, 28.43)}
_CASE_IDX = {"central": 0, "low": 1, "high": 2}


def _interp_year(table: dict, year: float, case: str) -> float:
    xs = sorted(table)
    ys = [table[x][_CASE_IDX[case]] for x in xs]
    return float(np.interp(year, xs, ys))


def gen_annual_cost(np_pv_mw: float, np_wind_mw: float, year: float,
                    case: str = "central", r: float = DISCOUNT) -> float:
    """Annualised generation capex + fixed O&M (USD/yr) for the installed
    nameplate (already overbuild-scaled)."""
    c = crf(GEN_LIFE, r)
    pv_cap = _interp_year(PV_CAPEX, year, case)
    wd_cap = _interp_year(WIND_CAPEX, year, case)
    pv_fom = _interp_year(PV_FOM, year, case)
    wd_fom = _interp_year(WIND_FOM, year, case)
    ann_pv = np_pv_mw * 1000.0 * (pv_cap * c + pv_fom)
    ann_wd = np_wind_mw * 1000.0 * (wd_cap * c + wd_fom)
    return ann_pv + ann_wd


# ----------------------------------------------------------------- transmission
TX_LIFE = 40.0
TX_LINE_USD_PER_MW_KM = {"central": 700.0, "low": 440.0, "high": 1100.0}    # overhead, 2024$
TX_CONV_USD_PER_KW_TERM = {"central": 200.0, "low": 120.0, "high": 320.0}   # per terminal
TX_FOM_FRAC = 0.01                       # of total link capex per year
TX_TORTUOSITY = 1.3                      # great-circle -> route length
TX_LINE_LOSS_PER_1000KM = 0.03           # one-way, line/cable
TX_CONV_LOSS_PER_TERM = 0.008            # one-way, per terminal (LCC ~0.7-1%)


def hvdc_efficiency(great_circle_km: float) -> float:
    """One-way delivered/sent efficiency for a dedicated import link, distance-
    dependent (route length = tortuosity x great-circle). 2 converter terminals
    + ohmic line loss. Floor at 0.30 (ultra-long intercontinental links are then
    flagged as barely viable)."""
    route = TX_TORTUOSITY * great_circle_km
    loss = 2 * TX_CONV_LOSS_PER_TERM + TX_LINE_LOSS_PER_1000KM * route / 1000.0
    return float(max(1.0 - loss, 0.30))


def tx_annual_cost(mwkm: float, conv_mw: float, case: str = "central",
                   r: float = DISCOUNT) -> float:
    """Annualised HVDC capex + FOM (USD/yr). mwkm already includes the x1.3 route
    tortuosity; conv_mw = summed corridor MW (2 terminals applied here)."""
    line_capex = mwkm * TX_LINE_USD_PER_MW_KM[case]
    conv_capex = conv_mw * 1000.0 * 2.0 * TX_CONV_USD_PER_KW_TERM[case]
    capex = line_capex + conv_capex
    return capex * (crf(TX_LIFE, r) + TX_FOM_FRAC)


# ----------------------------------------------------------------- storage
# Li-ion 4h: NREL ATB 2024 annual USD/kWh (same CSV the project already uses).
# LDES: public anchors. Cheaper-of-two per gap, identical to recompute_fig3c.
_ATB = pd.read_csv(DATA / "nrel_atb_2024_battery_4h.csv")
_ATB_YR = _ATB["year"].to_numpy(dtype=float)
_ATB_COL = {"high": "Conservative", "central": "Moderate", "low": "Advanced"}
_ATB_VAL = {case: _ATB[col].to_numpy(dtype=float) for case, col in _ATB_COL.items()}
P_LI, RTE_LI, LIFE_LI = 280.0, 0.86, 15
LDES_ANCHOR_YR = [2025, 2030, 2050]
E_LDES = {"high": [120.0, 90.0, 60.0], "central": [80.0, 45.0, 25.0],
          "low": [60.0, 30.0, 12.0]}
P_LDES, RTE_LDES, LIFE_LDES = 1000.0, 0.60, 20
STORE_FOM = 0.025
_CRF_LI, _CRF_LDES = crf(LIFE_LI), crf(LIFE_LDES)


def _e_li(case, year):
    return float(np.interp(year, _ATB_YR, _ATB_VAL[case])) - P_LI / 4.0


def _e_ldes(case, year):
    return float(np.interp(year, LDES_ANCHOR_YR, E_LDES[case]))


def storage_annual_cost(p_peak_mw: float, e_max_mwh: float, annual_unmet_mwh: float,
                        case: str = "central", year: float = 2030) -> tuple[float, str]:
    """Annualised firming-storage cost (USD/yr) under the cheaper of Li-ion vs
    LDES, sized to the P95 gap (p_peak MW, e_max MWh)."""
    if annual_unmet_mwh <= 0 or p_peak_mw <= 0:
        return 0.0, "none"
    eL, eD = _e_li(case, year), _e_ldes(case, year)
    cap_li = (P_LI * p_peak_mw + eL * e_max_mwh) * 1000.0
    cap_ld = (P_LDES * p_peak_mw + eD * e_max_mwh) * 1000.0
    lcos_li = (_CRF_LI + STORE_FOM) * cap_li / (annual_unmet_mwh * RTE_LI)
    lcos_ld = (_CRF_LDES + STORE_FOM) * cap_ld / (annual_unmet_mwh * RTE_LDES)
    if lcos_li <= lcos_ld:
        return annual_unmet_mwh * lcos_li, "Li-ion"
    return annual_unmet_mwh * lcos_ld, "LDES"


# ----------------------------------------------------------------- data centre
DC_FACILITY_LIFE = 15.0
DC_FACILITY_USD_PER_MW = {"central": 10.7e6, "low": 9.0e6, "high": 12.0e6}
DC_OPEX_USD_PER_KW_YR = {"central": 300.0, "low": 200.0, "high": 450.0}
IT_LIFE = 4.0
IT_CAPEX_USD_PER_MW = {"central": 23.0e6, "low": 18.0e6, "high": 30.0e6}


def dc_facility_annual_cost(case: str = "central", r: float = DISCOUNT,
                            mw: float = 100.0) -> float:
    """Annualised DC FACILITY capex + non-energy opex (USD/yr) for a `mw`-MW unit.
    Scenario-invariant (the building is built regardless of dispatch)."""
    cap = DC_FACILITY_USD_PER_MW[case] * mw * crf(DC_FACILITY_LIFE, r)
    opex = DC_OPEX_USD_PER_KW_YR[case] * mw * 1000.0
    return cap + opex


def it_annual_cost(case: str = "central", r: float = DISCOUNT, mw: float = 100.0) -> float:
    """Annualised IT (server/GPU) capex (USD/yr), 4-yr refresh. Context only."""
    return IT_CAPEX_USD_PER_MW[case] * mw * crf(IT_LIFE, r)


# ----------------------------------------------------------------- ancillary
ANCILLARY_USD_PER_MWH_VRE = {"central": 3.0, "low": 1.5, "high": 6.0}


def ancillary_annual_cost(gen_busbar_mwh: float, case: str = "central") -> float:
    """Residual balancing/reserve cost (USD/yr) on VRE generation throughput."""
    return gen_busbar_mwh * ANCILLARY_USD_PER_MWH_VRE[case]


if __name__ == "__main__":
    # quick self-check at a representative L3 case (per 100 MW unit, 2030 central)
    print("CRF: gen", round(crf(GEN_LIFE), 4), "tx", round(crf(TX_LIFE), 4),
          "dc", round(crf(DC_FACILITY_LIFE), 4), "it", round(crf(IT_LIFE), 4))
    print("hvdc eff @2000/4000/11700 km:",
          round(hvdc_efficiency(2000), 3), round(hvdc_efficiency(4000), 3),
          round(hvdc_efficiency(11700), 3))
    print("DC facility ann/100MW (M$/yr):", round(dc_facility_annual_cost() / 1e6, 1),
          " IT ann (M$/yr):", round(it_annual_cost() / 1e6, 1))
