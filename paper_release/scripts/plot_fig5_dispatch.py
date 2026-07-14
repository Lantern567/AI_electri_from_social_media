# -*- coding: utf-8 -*-
"""Regenerate Fig 5 (fig4_routing_cost) from the hourly storage-dispatch LP cost data.

Combines the 2030 6x6 (tau,phi) grid (r3_dispatch_grid_2030.csv, for panels a/b) with the
phi=1 x 6-tau x all-years table (r3_dispatch_yearly_phi1.csv, for panels c/d), monkey-patches
R3L.load to return this dispatch dataframe, and calls the existing render_results() verbatim.
All panels use Pi = (lcoe_elec - lcoe_gen)/lcoe_gen (convention B), matching the dispatch cost
columns directly. The mechanism figure (fig3_ai_mechanism, gap-based) is unchanged and NOT redrawn.
"""
import sys
from pathlib import Path
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import plot_r3_latency_fig3 as R3L
import plot_r3_split_figs as SPLIT

DATA = R3L.DATA
KEEP = ["country", "tau_ms", "phi", "year", "perturbation",
        "lcoe_gen", "lcoe_store", "lcoe_anc", "lcoe_elec", "uncovered_share", "region"]

def load_dispatch():
    parts = []
    for name in ("r3_dispatch_grid_2030.csv", "r3_dispatch_yearly_phi1.csv"):
        p = DATA / name
        if p.exists():
            parts.append(pd.read_csv(p))
        else:
            print(f"  WARNING: {name} missing")
    df = pd.concat(parts, ignore_index=True)
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["tau_ms"] = df["tau_ms"].astype(int)
    df["phi"] = df["phi"].astype(float)
    df["year"] = df["year"].astype(int)
    df = df.drop_duplicates(subset=["country", "tau_ms", "phi", "year", "perturbation"], keep="first")
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    print(f"  dispatch df: {len(df)} rows | years {sorted(df.year.unique())} | "
          f"tau {sorted(df.tau_ms.unique())} | phi {sorted(df.phi.unique())}")
    return df, dn

if __name__ == "__main__":
    R3L.load = load_dispatch                 # monkey-patch the data source
    SPLIT.R3L.load = load_dispatch           # (SPLIT references R3L.load at call time)
    SPLIT.render_results()                   # -> figures_globalsites/fig4_routing_cost.{png,pdf}
    print("Fig 5 regenerated from dispatch LP data.")
