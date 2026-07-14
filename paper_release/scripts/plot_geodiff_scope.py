#!/usr/bin/env python3
"""Figure for the geo-differentiated scope result (r4_geodiff_scope.csv)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import plot_fig3_redesign as R  # choropleth helper

OUT = R.DATA
FIG = R.FIG
df = pd.read_csv(OUT / "r4_geodiff_fullchain.csv")
SC = ["L0", "T1", "T2", "T3"]
COL = {"L0": "#2C7FB8", "T1": "#7FCDBB", "T2": "#FEB24C", "T3": "#F03B20"}
INK = "#1F2937"
R.set_style(11.0)

fig = plt.figure(figsize=(13.0, 9.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0],
                      left=0.06, right=0.97, top=0.93, bottom=0.08, hspace=0.30, wspace=0.22)

# (a) optimal-scope world map
axm = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
val = {r.iso2: SC.index(r.opt_scope) for r in df.itertuples()}
cmap = mpl.colors.ListedColormap([COL[s] for s in SC])
norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
R._country_choropleth(axm, val, cmap, norm)
axm.set_title("(a) Cost-optimal dispatch reach per country (geo-differentiated cost)",
              fontsize=12.5, fontweight="bold", color=INK, loc="left")
names = {"L0": "L0 national", "T1": "T1 ≤1500 km", "T2": "T2 ≤3000 km", "T3": "T3 continental"}
axm.legend(handles=[mpatches.Patch(facecolor=COL[s], edgecolor="white", label=names[s]) for s in SC],
           loc="lower left", fontsize=10, frameon=False, ncol=2)

# (b) mean cost-vs-scope + archetypes
axb = fig.add_subplot(gs[1, 0])
x = np.arange(4)
mean = [df[f"tot_{s}"].mean() for s in SC]
axb.plot(x, mean, "-o", color=INK, lw=2.4, ms=7, zorder=5, label="104-country mean")
arche = {"CN": "#2C7FB8", "JP": "#F03B20", "NG": "#B45309", "US": "#7FCDBB"}
for cc, col in arche.items():
    row = df[df.iso2 == cc]
    if len(row):
        axb.plot(x, [row[f"tot_{s}"].iloc[0] for s in SC], "-o", color=col, lw=1.4,
                 ms=4, alpha=0.9, label=cc)
axb.set_xticks(x); axb.set_xticklabels(["L0", "T1", "T2", "T3"])
axb.set_ylabel("Full-system cost (USD/MWh)")
axb.set_xlabel("Dispatch reach")
axb.set_title("(b) Cost vs reach — interior optimum at national/near-neighbour",
              fontsize=11.5, fontweight="bold", color=INK, loc="left")
axb.axvspan(-0.2, 1.2, color="#7FCDBB", alpha=0.10, zorder=0)
axb.legend(fontsize=9, frameon=False, loc="upper left")
R.yonly(axb)

# (c) the two drivers: resource adequacy (unc_L0) vs WACC, marker=opt scope
axc = fig.add_subplot(gs[1, 1])
for s in SC:
    sub = df[df.opt_scope == s]
    axc.scatter(sub.unc_L0, sub.wacc * 100, s=34, color=COL[s], edgecolor="white",
                linewidth=0.4, label=names[s], alpha=0.9)
axc.set_xlabel("National uncovered share L0 (%) — resource adequacy →")
axc.set_ylabel("Country WACC (%)")
axc.set_title("(c) Who reaches out: poor domestic resource &/or it pays off",
              fontsize=11.5, fontweight="bold", color=INK, loc="left")
axc.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
R.yonly(axc); axc.grid(True, axis="x", color=R.GRID, lw=0.6, alpha=0.6)

fig.suptitle("Station-level dispatch + distance-bounded scope (no global) + geo-differentiated transmission cost",
             fontsize=13.5, fontweight="bold", y=0.985)
for ext in (".png", ".pdf"):
    fig.savefig(FIG / f"fig_geodiff_scope{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "fig_geodiff_scope.png")
