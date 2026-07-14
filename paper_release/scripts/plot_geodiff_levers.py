#!/usr/bin/env python3
"""Scope x workload picture for the 16-scenario geodiff recompute."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import plot_fig3_redesign as R

OUT, FIG = R.DATA, R.FIG
df = pd.read_csv(OUT / "r4_geodiff_16scen.csv")
SC = ["L0", "T1", "T2", "T3"]
WL = ["W0", "W1", "W2", "W3"]
SCN = {"L0": "L0 national", "T1": "T1 ≤1.5Mm", "T2": "T2 ≤3Mm", "T3": "T3 continental"}
WN = {"W0": "0%", "W1": "~10%", "W2": "~30%", "W3": "~60%"}
INK = "#1F2937"
R.set_style(11.0)
M = np.array([[df[f"tot_{s}_{w}"].mean() for w in WL] for s in SC])

fig = plt.figure(figsize=(12.4, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.08, right=0.97,
                      top=0.86, bottom=0.14, wspace=0.30)

# (a) heatmap scope x workload
axa = fig.add_subplot(gs[0, 0])
im = axa.imshow(M, cmap="RdYlGn_r", aspect="auto")
for i in range(4):
    for j in range(4):
        axa.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=12,
                 color="white" if M[i, j] > 170 else INK, fontweight="bold")
imin = np.unravel_index(np.argmin(M), M.shape)
axa.add_patch(plt.Rectangle((imin[1] - 0.5, imin[0] - 0.5), 1, 1, fill=False,
                            edgecolor="#0B6E4F", lw=3))
axa.set_xticks(range(4)); axa.set_xticklabels([WN[w] for w in WL])
axa.set_yticks(range(4)); axa.set_yticklabels([SCN[s] for s in SC])
axa.set_xlabel("Workload migration →"); axa.set_ylabel("Power-dispatch reach ↑")
axa.set_title("(a) Mean full-system cost (USD/MWh) — cheapest = L0·W3 (move compute, stay local)",
              fontsize=11, fontweight="bold", color=INK, loc="left")

# (b) the two levers (from the cheapest corner is L0/W3)
axb = fig.add_subplot(gs[0, 1])
x = np.arange(4)
axb.plot(x, M[:, 0], "-o", color="#B91C1C", lw=2.2, ms=6,
         label="Power lever (W0): widen reach → costlier")
axb.plot(x, M[0, :], "-o", color="#0B6E4F", lw=2.2, ms=6,
         label="Workload lever (L0): more migration → cheaper")
axb.set_xticks(x)
axb.set_xticklabels(["0", "1", "2", "3"])
axb.set_xlabel("Lever step (reach L0→T3  /  migration W0→W3)")
axb.set_ylabel("Mean full-system cost (USD/MWh)")
axb.set_title("(b) Once transmission is priced, the two levers split:\nmove compute (cheap) beats move power (HVDC)",
              fontsize=11, fontweight="bold", color=INK, loc="left")
axb.legend(fontsize=9.5, frameon=False, loc="upper left")
R.yonly(axb)
axb.annotate("national + max migration\n= cost-optimal (101)", xy=(0, M[0, 3]), xytext=(0.6, 130),
             fontsize=9, color="#0B6E4F", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#0B6E4F", lw=1.0))

fig.suptitle("Full 16-scenario geodiff recompute: power-dispatch reach × workload migration (P_mix-2030, geo-diff cost)",
             fontsize=12.5, fontweight="bold", y=0.985)
for ext in (".png", ".pdf"):
    fig.savefig(FIG / f"fig_geodiff_levers{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "fig_geodiff_levers.png")
