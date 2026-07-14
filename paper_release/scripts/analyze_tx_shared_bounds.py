#!/usr/bin/env python3
"""Transmission-cost BOUNDS for Result 3: dedicated point-to-point (UPPER) vs a
shared region-backbone (LOWER), following the inter-regional UHVDC-corridor logic
of Guo et al. 2022 (Nature Energy, doi:10.1038/s41560-022-01136-0) and the meshed
"enter-and-transmit-anywhere" backbone of Chatzivasileiadis et al. 2013 (The Global
Grid, doi:10.1016/j.renene.2013.01.032).

Both bounds reuse the EXACT physical solve of decompose_fullsystem_portfolio.py
(per-country NNLS portfolio over reachable national PV/wind curves, equal-energy,
distance-dependent HVDC efficiency). The only thing that differs is how the hourly
import flows are turned into transmission infrastructure:

  UPPER  dedicated : every (importing country, source country) corridor is its own
                     HVDC link, sized to that pair's P95 sent flow over its own
                     great-circle route. (= current fullsystem_cost_table.csv)
  LOWER  shared    : all flow between an importing region and an exporting region
                     rides ONE shared backbone corridor, sized to the P95 of the
                     AGGREGATE hourly flow (de-coincidence across countries AND
                     sources) over the energy-weighted mean route; the corridor cost
                     is then spread over the delivered energy. This is the region-node
                     network used by capacity-expansion models (PyPSA, NEWS, LUT).

Each bound is priced under three parameter sets:
  ours      line 700 USD/MW/km, converter 200 USD/kW/term, 7% discount (paper's central)
  lit       line 350,            converter 100,             5% discount (Global Grid / Hartel benchmark)
  lit3      line 350,            converter 100,             3% discount (Global Grid social discount)

Output: prints an L0-L3 table of tx LCOE and total electricity LCOE for every
(topology x price) cell, plus the L0/L3 reversal ratio. Generation / storage /
ancillary are taken (central) from fullsystem_cost_table.csv so only tx changes.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("EXPANDED_DEMAND", "1")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_global_sites as G            # noqa: E402
import analyze_cfe_geographic_portfolio_ai as base  # noqa: E402
import fullsystem_cost_params as P              # noqa: E402

DATA = P.DATA
E_D = P.E_D_MWH                                  # 876,000 MWh/yr per 100 MW-mean unit
TORT = P.TX_TORTUOSITY                           # 1.3 great-circle -> route
LIFE = P.TX_LIFE                                 # 40 yr
FOM = P.TX_FOM_FRAC                              # 0.01/yr

YEAR = 2030
OP = "P_mix"
SCOPES = ["L0", "L1", "L2", "L3"]

# (line USD/MW/km, converter USD/kW/terminal, discount) — see module docstring
PRICE = {
    "ours": (700.0, 200.0, 0.07),
    "lit":  (350.0, 100.0, 0.05),
    "lit3": (350.0, 100.0, 0.03),
}


def tx_annual(mwkm, conv_mw, line_usd_mwkm, conv_usd_kw_term, disc):
    """Annualised HVDC capex+FOM (USD/yr). 2 converter terminals per corridor."""
    capex = mwkm * line_usd_mwkm + conv_mw * 1000.0 * 2.0 * conv_usd_kw_term
    return capex * (P.crf(LIFE, disc) + FOM)


def solve_country_flows(cc, power, demand, supply, smeta, dist_cache, d_pert):
    """Re-run the W0 NNLS portfolio for one country/scope and return, per FOREIGN
    source country, the hourly SENT flow (MW, pre-loss) and the great-circle km.
    Mirrors decompose_fullsystem_portfolio.decompose (W0 branch)."""
    home_idx = demand[cc]["d"].index
    d_base = pd.Series(np.asarray(d_pert), index=home_idx)
    pool = G.global_pool(cc, power, supply, smeta)
    S, labels = base.supply_basis(supply, pool, home_idx, fixed=(power == "L0"))
    if S.size == 0:
        return {}
    if power != "L0":
        for j, lab in enumerate(labels):
            if lab.split(":")[0] != cc:
                S[:, j] = S[:, j] * G.ETA
    S1 = base.equal_energy_align(S, float(d_base.mean()))
    w, share, fit = base.portfolio_nnls(d_base.to_numpy(), S1)
    scale = 100.0 / float(d_base.mean())          # MW per curve-unit (mean = 100 MW)

    src_cols: dict[str, list[int]] = {}
    for j, lab in enumerate(labels):
        src = lab.split(":")[0]
        if src != cc and float(w[j]) > 1e-6:
            src_cols.setdefault(src, []).append(j)
    flows = {}
    for src, cols in src_cols.items():
        gc = dist_cache[(cc, src)]
        eta = P.hvdc_efficiency(gc)
        wsrc = np.array([w[j] for j in cols])
        delivered = (S1[:, cols] @ wsrc) * scale   # MW delivered to home (S1 already x ETA)
        sent = delivered / eta                      # MW sent (pre-loss) at the exporting end
        flows[src] = (sent, gc)
    return flows


def main():
    t0 = time.time()
    print("loading expanded inputs ...")
    _, demand, demand_meta, supply, smeta = G.load_inputs_expanded()
    countries = list(demand)
    N = len(countries)
    print(f"  {N} demand countries, {len(supply)} supply countries")

    dist_cache = {}
    for cc in countries:
        for src in supply:
            if src in smeta:
                dist_cache[(cc, src)] = base.great_circle_km(
                    smeta[cc]["lon"], smeta[cc]["lat"], smeta[src]["lon"], smeta[src]["lat"])

    def region_of(x):
        r = smeta[x].get("region_code")
        if r in (None, "", "nan") or (isinstance(r, float)):
            return str(smeta[x].get("continent_block", "??"))
        return str(r)

    lam_i = float(np.interp(YEAR, sorted(base.AI_SHARE_INFERENCE),
                            [base.AI_SHARE_INFERENCE[k] for k in sorted(base.AI_SHARE_INFERENCE)]))
    lam_t = float(np.interp(YEAR, sorted(base.AI_SHARE_TRAINING),
                            [base.AI_SHARE_TRAINING[k] for k in sorted(base.AI_SHARE_TRAINING)]))
    cc_d = {cc: base.as_arr(demand[cc]["d"]).astype(float) for cc in countries}
    cc_lh = {cc: base.as_arr(demand[cc]["local_hour"]).astype(int) for cc in countries}

    # results[scope][topology][price] = global tx LCOE (USD/MWh)
    res = {sc: {} for sc in SCOPES}
    for power in SCOPES:
        # --- dedicated (UPPER): per-country sum of per-pair links ---
        ded_mwkm = {pk: 0.0 for pk in PRICE}     # not needed; price per pair directly
        ded_capex_terms = []                     # (mwkm, conv_mw) per pair, summed later
        # --- shared (LOWER): aggregate hourly flow per (Rimp, Rexp) corridor ---
        corr_sent = {}                            # corridor -> summed hourly sent MW
        corr_e = {}                               # corridor -> annual sent energy (MWh)
        corr_edist = {}                           # corridor -> energy-weighted route accumulator
        T = None
        ded_pairs = []                            # (mwkm, conv_mw) per pair for dedicated
        for cc in countries:
            d_pert = base.operator_mix(cc_d[cc], cc_lh[cc], lam_i, lam_t)
            flows = solve_country_flows(cc, power, demand, supply, smeta, dist_cache, d_pert)
            rimp = region_of(cc)
            for src, (sent, gc) in flows.items():
                if T is None:
                    T = len(sent)
                route = TORT * gc
                p95 = float(np.percentile(sent, 95))
                ded_pairs.append((p95 * route, p95))           # dedicated link for this pair
                # shared corridor accumulation
                key = (rimp, region_of(src))
                if key not in corr_sent:
                    corr_sent[key] = np.zeros(len(sent))
                    corr_e[key] = 0.0
                    corr_edist[key] = 0.0
                corr_sent[key] += sent
                e = float(sent.sum())
                corr_e[key] += e
                corr_edist[key] += e * route
        # price dedicated
        for pk, (ln, cv, dsc) in PRICE.items():
            tx = sum(tx_annual(mwkm, mw, ln, cv, dsc) for mwkm, mw in ded_pairs)
            res[power].setdefault("dedicated", {})[pk] = tx / (N * E_D)
        # price shared
        for pk, (ln, cv, dsc) in PRICE.items():
            tx = 0.0
            for key, ser in corr_sent.items():
                p95 = float(np.percentile(ser, 95))
                mean_route = corr_edist[key] / corr_e[key] if corr_e[key] > 0 else 0.0
                tx += tx_annual(p95 * mean_route, p95, ln, cv, dsc)
            res[power].setdefault("shared", {})[pk] = tx / (N * E_D)
        n_corr = len(corr_sent)
        print(f"  {power}: {len(ded_pairs):5d} dedicated pairs -> {n_corr:3d} shared corridors  "
              f"({time.time()-t0:.0f}s)")

    # --- assemble with gen/store/anc (central) from the cost table ---
    tbl = pd.read_csv(DATA / "fullsystem_cost_table.csv")
    base_lc = tbl[(tbl.year == YEAR) & (tbl.workload_scenario == "W0")].groupby(
        "power_scope")[["lcoe_gen", "lcoe_store", "lcoe_anc", "lcoe_tx"]].mean()

    print("\n" + "=" * 92)
    print(f"TRANSMISSION LCOE (USD/MWh) — {OP}, {YEAR}, W0 — by topology x price, across scope")
    print("=" * 92)
    hdr = "scope " + "".join(f"{t}/{p:>4}" .rjust(13) for t in ("ded", "shr") for p in PRICE)
    print(f"{'':6}" + "".join(f"{('ded·'+p):>11}" for p in PRICE)
          + "".join(f"{('shr·'+p):>11}" for p in PRICE) + "   (check tx_tbl)")
    for sc in SCOPES:
        row = "".join(f"{res[sc]['dedicated'][p]:11.1f}" for p in PRICE) \
            + "".join(f"{res[sc]['shared'][p]:11.1f}" for p in PRICE)
        print(f"{sc:<6}{row}   {base_lc.loc[sc,'lcoe_tx']:9.1f}")

    print("\n" + "=" * 92)
    print("TOTAL electricity LCOE (USD/MWh) = gen+store+anc(central) + tx(variant); L0/L3 reversal")
    print("=" * 92)
    gsa = (base_lc["lcoe_gen"] + base_lc["lcoe_store"] + base_lc["lcoe_anc"])
    for topo in ("dedicated", "shared"):
        for p in PRICE:
            tot = {sc: gsa[sc] + res[sc][topo][p] for sc in SCOPES}
            ratio = tot["L3"] / tot["L0"]
            cells = "  ".join(f"{sc}={tot[sc]:6.0f}" for sc in SCOPES)
            print(f"  {topo:9} · {p:5}:  {cells}   |  L3/L0 = {ratio:.2f}x")

    # --- tidy CSV for the figure (panel c band) ---
    recs = []
    for sc in SCOPES:
        for topo in ("dedicated", "shared"):
            for p in PRICE:
                tx = res[sc][topo][p]
                recs.append({
                    "scope": sc, "topology": topo, "price": p,
                    "lcoe_tx": tx,
                    "lcoe_gen": float(base_lc.loc[sc, "lcoe_gen"]),
                    "lcoe_store": float(base_lc.loc[sc, "lcoe_store"]),
                    "lcoe_anc": float(base_lc.loc[sc, "lcoe_anc"]),
                    "lcoe_total": float(gsa[sc]) + tx,
                })
    out = pd.DataFrame(recs)
    out.to_csv(DATA / "r3_tx_bounds.csv", index=False)
    print(f"\nwrote {DATA / 'r3_tx_bounds.csv'} ({len(out)} rows)")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
