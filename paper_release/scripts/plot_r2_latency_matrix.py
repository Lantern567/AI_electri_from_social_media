#!/usr/bin/env python3
"""R2 v2 matrix (latency tolerance x routable fraction) in the original Fig 2(a) style.

rows  = routable fraction phi (100% top -> 0% bottom)
cols  = latency tolerance tau in ms (tight/local left -> loose/global right), with reach km
cells = cross-country median uncovered share (%)
Marginals: top = pure latency effect (at phi=100%), right = pure routability effect (at tau=500 ms).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plot_cfe_geographic_portfolio_ai as P

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
P.set_style()

DATA = P.REPORT / "data_globalsites_stations_expanded"
FIG = P.REPORT / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)
FIG_PA, PRIMARY, SECONDARY = P.FIG_PA, P.PRIMARY, P.SECONDARY

ROWS_PHI = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]                 # top -> bottom
COLS_TAU = [50, 100, 150, 200, 300, 500]                  # left -> right
PHI_LABEL = {1.0: "100%", 0.8: "80%", 0.6: "60%", 0.4: "40%", 0.2: "20%", 0.0: "0%\n(none)"}
REACH = {50: "local", 100: "≈3000 km", 150: "≈6700 km", 200: "≈10000 km",
         300: "≈17000 km", 500: "global"}
TAU_LABEL = {t: f"{t} ms\n{REACH[t]}" for t in COLS_TAU}


def main():
    g = pd.read_csv(DATA / "r2_latency_grid.csv")
    med = (g.groupby(["phi", "tau_ms"]).uncovered_share.median().unstack() * 100)
    mat = med.reindex(index=ROWS_PHI, columns=COLS_TAU).values
    nr, nc = mat.shape
    bottom = nr - 1                                        # phi=0 row index

    fig = plt.figure(figsize=(11.6, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.0], height_ratios=[1.0, 3.0],
                          wspace=0.06, hspace=0.06, left=0.11, right=0.93, top=0.92, bottom=0.13)
    ax_top = fig.add_subplot(gs[0, 0]); ax_a = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1]); ax_corner = fig.add_subplot(gs[0, 1]); ax_corner.axis("off")
    fs = 13
    vmin, vmax = mat.min() * 0.95, mat.max() * 1.02
    im = ax_a.imshow(mat, aspect="auto", cmap=FIG_PA, vmin=vmin, vmax=vmax)
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            fc = FIG_PA((v - vmin) / (vmax - vmin))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs,
                      weight="bold", color="white" if lum < 0.55 else "#111827")
    ax_a.set_xticks(range(nc)); ax_a.set_xticklabels([TAU_LABEL[t] for t in COLS_TAU], fontsize=fs - 1.5)
    ax_a.set_yticks(range(nr)); ax_a.set_yticklabels([PHI_LABEL[p] for p in ROWS_PHI], fontsize=fs)
    ax_a.set_xlabel("Latency tolerance  (RTT budget → migration reach)", fontsize=fs)
    ax_a.set_ylabel("Routable fraction of load", fontsize=fs)
    ax_a.grid(False)
    for (ri, ci) in [(bottom, 0), (0, nc - 1)]:
        ax_a.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                                          edgecolor="#111827", linewidth=1.6, zorder=4))
    fx = [mpl.patheffects.withStroke(linewidth=2.4, foreground="#111827")]
    ax_a.text(0, bottom + 0.33, "Status quo", fontsize=fs, weight="bold", color="white",
              ha="center", va="center", path_effects=fx, zorder=5)
    ax_a.text(nc - 1, 0.33, "Upper bound", fontsize=fs, weight="bold", color="white",
              ha="center", va="center", path_effects=fx, zorder=5)

    # top marginal: pure latency effect at phi=100% (top row, i=0)
    pure_t = [mat[0, j] for j in range(nc)]
    ax_top.bar(range(nc), [pure_t[0] - v for v in pure_t], color=SECONDARY, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_t):
        d = pure_t[0] - v
        ax_top.text(i, d + 0.1, f"{d:.1f}" if d >= 0.05 else "0.0", ha="center",
                    va="bottom", fontsize=fs, color="#111827")
    ax_top.set_ylabel("Reduction vs\nlocal (latency\nonly, pp)", fontsize=fs)
    P.panel_letter(ax_top, "a", x=-0.24, y=1.02)
    ax_top.tick_params(axis="x", labelbottom=False); ax_top.tick_params(axis="y", labelsize=fs)
    ax_top.set_ylim(0, max(pure_t[0] - min(pure_t), 0.1) * 1.3)
    P.soften_axes(ax_top)

    # right marginal: pure routability effect at tau=500 ms (right col, j=nc-1)
    pure_s = [mat[i, nc - 1] for i in range(nr)]
    ax_right.barh(range(nr), [pure_s[bottom] - v for v in pure_s], color=PRIMARY, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_s):
        d = pure_s[bottom] - v
        ax_right.text(d + 0.1, i, f"{d:.1f}" if d >= 0.05 else "0.0", ha="left",
                      va="center", fontsize=fs, color="#111827")
    ax_right.set_xlabel("Reduction vs\nno routing\n(routability only, pp)", fontsize=fs)
    ax_right.tick_params(axis="y", labelleft=False); ax_right.tick_params(axis="x", labelsize=fs)
    ax_right.invert_yaxis()
    ax_right.set_xlim(0, max(pure_s[bottom] - min(pure_s), 0.1) * 1.35)
    P.soften_axes(ax_right)

    cax = ax_corner.inset_axes([0.08, 0.48, 0.86, 0.15])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Median uncovered\ndemand share (%)", fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    out = FIG / "fig2_latency_matrix"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
