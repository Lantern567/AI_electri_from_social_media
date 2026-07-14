# -*- coding: utf-8 -*-
"""Hourly storage DISPATCH + SIZING LP, replacement for the P95 gap-geometry heuristic.

Per country-scenario, given effective demand d_eff(t) [MW, mean=100] and OB=1 portfolio
supply fit(t) [MW, mean=100], co-optimise generation overbuild ob and Li/LDES power+energy
with explicit hourly SOC dispatch, round-trip efficiency, curtailment, cyclic (annual) SOC,
and unserved-energy priced at VOLL (endogenous reliability). Pure LP -> scipy HiGHS.

Variable vector x (length 5 + 8T):
  0: ob
  1: P_Li  2: E_Li  3: P_LDES  4: E_LDES
  blocks of T (base=5): chg_Li, dis_Li, soc_Li, chg_LDES, dis_LDES, soc_LDES, curtail, unserved
Objective (USD/yr): (GEN1+ANC1)*ob + sum_k (CRF_k+FOM)*(Pcost_k*P_k+Ecost_k*E_k)*1000
                    + VOLL*sum_t unserved_t
Balance (t): fit[t]*ob + dis_Li+dis_LDES - chg_Li-chg_LDES - curtail + unserved = d_eff[t]
SOC_k (t):   soc_k[t] - soc_k[t-1] - etac*chg_k[t] + dis_k[t]/etad = 0   (t-1 cyclic mod T)
Caps:        chg_k<=P_k, dis_k<=P_k, soc_k<=E_k
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

FOM = 0.025
def crf(life, r=0.07):
    return r * (1 + r) ** life / ((1 + r) ** life - 1)
CRF_LI, CRF_LDES = crf(15), crf(20)
P_LI, P_LDES = 280.0, 1000.0          # USD/kW
RTE_LI, RTE_LDES = 0.86, 0.60
ETAc_LI, ETAd_LI = RTE_LI ** 0.5, RTE_LI ** 0.5
ETAc_LD, ETAd_LD = RTE_LDES ** 0.5, RTE_LDES ** 0.5


def solve_dispatch(d_eff, fit, gen1, anc1, ecost_li, ecost_ldes,
                   voll=10000.0, ob_max=5.0, rel_target=1.0, cyclic=True):
    """Return dict with ob, P/E per tech, hourly-derived shares, and annual costs (USD/yr).

    rel_target: energy-reliability target (fraction of demand served); if <1 a hard cap
                sum(unserved) <= (1-rel_target)*sum(d_eff) is added (VOLL still prices the
                residual unserved within that budget). rel_target=1.0 -> ~100% served.
    cyclic:     if True enforce annual SOC balance soc[0] linked to soc[T-1]; if False the
                storage starts empty (soc[0] = charge/discharge at t=0 only)."""
    d_eff = np.asarray(d_eff, float); fit = np.asarray(fit, float)
    T = d_eff.size
    base = 5
    N = base + 8 * T
    # column offsets
    cCHGli, cDISli, cSOCli = base + 0*T, base + 1*T, base + 2*T
    cCHGld, cDISld, cSOCld = base + 3*T, base + 4*T, base + 5*T
    cCUR, cUNS = base + 6*T, base + 7*T

    # objective
    c = np.zeros(N)
    c[0] = gen1 + anc1
    c[1] = (CRF_LI + FOM) * P_LI * 1000.0
    c[2] = (CRF_LI + FOM) * ecost_li * 1000.0
    c[3] = (CRF_LDES + FOM) * P_LDES * 1000.0
    c[4] = (CRF_LDES + FOM) * ecost_ldes * 1000.0
    c[cUNS:cUNS + T] = voll

    rows, cols, vals = [], [], []
    def add(r, cc, v):
        rows.append(r); cols.append(cc); vals.append(v)

    # --- A_eq: balance (T) + SOC_Li (T) + SOC_LDES (T) ---
    beq = np.zeros(3 * T)
    # balance rows 0..T-1
    t = np.arange(T)
    add_r = 0
    for i in range(T):
        add(i, 0, fit[i])                 # fit*ob
        add(i, cDISli + i, 1.0); add(i, cDISld + i, 1.0)
        add(i, cCHGli + i, -1.0); add(i, cCHGld + i, -1.0)
        add(i, cCUR + i, -1.0); add(i, cUNS + i, 1.0)
    beq[:T] = d_eff
    # SOC recursion (cyclic) for each tech
    for (r0, cCHG, cDIS, cSOC, etac, etad) in [
        (T,    cCHGli, cDISli, cSOCli, ETAc_LI, ETAd_LI),
        (2*T,  cCHGld, cDISld, cSOCld, ETAc_LD, ETAd_LD)]:
        for i in range(T):
            add(r0 + i, cSOC + i, 1.0)
            if cyclic or i > 0:
                add(r0 + i, cSOC + ((i - 1) % T), -1.0)
            add(r0 + i, cCHG + i, -etac)
            add(r0 + i, cDIS + i, 1.0 / etad)
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(3 * T, N)).tocsr()

    # --- A_ub: caps chg<=P, dis<=P, soc<=E for each tech (6T rows) ---
    ru, cu, vu = [], [], []
    def addu(r, cc, v):
        ru.append(r); cu.append(cc); vu.append(v)
    r = 0
    for (cCHG, cDIS, cSOC, colP, colE) in [
        (cCHGli, cDISli, cSOCli, 1, 2),
        (cCHGld, cDISld, cSOCld, 3, 4)]:
        for i in range(T):
            addu(r, cCHG + i, 1.0); addu(r, colP, -1.0); r += 1
        for i in range(T):
            addu(r, cDIS + i, 1.0); addu(r, colP, -1.0); r += 1
        for i in range(T):
            addu(r, cSOC + i, 1.0); addu(r, colE, -1.0); r += 1
    nub = 6 * T
    if rel_target < 1.0:                              # reliability cap: sum(unserved) <= budget
        for i in range(T):
            addu(nub, cUNS + i, 1.0)
        nub += 1
    A_ub = sparse.coo_matrix((vu, (ru, cu)), shape=(nub, N)).tocsr()
    b_ub = np.zeros(nub)
    if rel_target < 1.0:
        b_ub[-1] = (1.0 - rel_target) * float(d_eff.sum())

    bounds = [(1.0, ob_max)] + [(0, None)] * (N - 1)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=beq,
                  bounds=bounds, method='highs')
    if not res.success:
        return {'ok': False, 'msg': res.message}
    x = res.x
    ob = x[0]; P_li, E_li, P_ld, E_ld = x[1:5]
    unmet = x[cUNS:cUNS + T].sum()
    dis_li = x[cDISli:cDISli + T].sum(); dis_ld = x[cDISld:cDISld + T].sum()
    cur = x[cCUR:cCUR + T].sum()
    store_li = (CRF_LI + FOM) * (P_LI * P_li + ecost_li * E_li) * 1000.0
    store_ld = (CRF_LDES + FOM) * (P_LDES * P_ld + ecost_ldes * E_ld) * 1000.0
    E_D = 100.0 * T
    return {'ok': True, 'ob': ob, 'P_li': P_li, 'E_li': E_li, 'P_ld': P_ld, 'E_ld': E_ld,
            'store_usd_yr': store_li + store_ld, 'gen_usd_yr': gen1 * ob, 'anc_usd_yr': anc1 * ob,
            'unmet_mwh': unmet, 'curtail_mwh': cur, 'dis_li_mwh': dis_li, 'dis_ld_mwh': dis_ld,
            'ldes_energy_share': dis_ld / max(dis_li + dis_ld, 1e-9),
            'lcoe_store': (store_li + store_ld) / E_D, 'lcoe_gen': gen1 * ob / E_D,
            'lcoe_anc': anc1 * ob / E_D,
            'lcoe_elec': (gen1 * ob + anc1 * ob + store_li + store_ld) / E_D,
            'firm_share': 1.0 - unmet / (100.0 * T)}


if __name__ == '__main__':
    import time
    # synthetic 1-year: solar-like supply (noon bump), evening-peak demand
    T = 8760
    h = np.arange(T) % 24
    sun = np.clip(np.cos((h - 13) / 24 * 2 * np.pi), 0, None)      # noon peak
    fit = sun / sun.mean() * 100.0                                  # mean 100
    dem = 1.0 + 0.6 * np.exp(-((h - 20) % 24 - 0)**2 / 8)          # evening peak ~20h
    d_eff = dem / dem.mean() * 100.0                                # mean 100
    # 2030 central-ish cost anchors
    gen1 = 74.0 * 100.0 * 8760 / 1000 * 1000  # placeholder ~ gen at ob=1 (USD/yr) => set so lcoe_gen~74
    gen1 = 74.0 * 100.0 * 8760                 # USD/yr s.t. /E_D = 74
    anc1 = 3.0 * 100.0 * 8760                  # USD/yr s.t. /E_D = 3
    t0 = time.time()
    r = solve_dispatch(d_eff, fit, gen1, anc1, ecost_li=255.0, ecost_ldes=45.0)
    dt = time.time() - t0
    if r['ok']:
        print(f"solve {dt:.1f}s | ob={r['ob']:.2f} | P_li={r['P_li']:.1f} E_li={r['E_li']:.0f} "
              f"P_ld={r['P_ld']:.1f} E_ld={r['E_ld']:.0f}")
        print(f"  lcoe_gen={r['lcoe_gen']:.1f} store={r['lcoe_store']:.1f} anc={r['lcoe_anc']:.1f} "
              f"elec={r['lcoe_elec']:.1f} USD/MWh | ldes_share={r['ldes_energy_share']:.2f} "
              f"curtail={r['curtail_mwh']:.0f} unmet={r['unmet_mwh']:.1f} firm={r['firm_share']*100:.2f}%")
    else:
        print('FAILED', r['msg'])
