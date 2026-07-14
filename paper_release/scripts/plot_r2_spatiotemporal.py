#!/usr/bin/env python3
"""Figure for the REDESIGNED Result 2: compute spatiotemporal flexibility grid.

Panels:
  (a) median uncovered-share heatmap, rows = max migration distance (spatial),
      cols = delay grace / deadline slack (temporal).
  (b) single-lever response curves: temporal-only (in-country, vs grace) and
      spatial-only (same-hour, vs distance) on a shared uncovered-share axis.
  (c) cross-country uncovered-share distribution at the four key operating points.
  (d) lever decomposition: single-lever vs combined gap reduction (sub-additivity).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import plot_cfe_geographic_portfolio_ai as P

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
P.set_style()

REPORT = P.REPORT
DATA = REPORT / "data_globalsites_stations_expanded"
FIG = REPORT / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)

CMAP = P.FIG_PA
SPATIAL_C = "#6D28D9"   # purple  = distance / spatial lever
TEMPORAL_C = "#D97706"  # amber   = grace / temporal lever
INK = "#111827"

DIST_ORDER = ["D0", "D1", "D2", "D3"]
DIST_TICK = ["In-country\n(0 km)", "Metro\n(≤500 km)", "Continental\n(≤3000 km)", "Global\n(no cap)"]
GRACE = [0, 1, 3, 6, 12, 24, 168]
GRACE_TICK = ["0", "1 h", "3 h", "6 h", "12 h", "24 h", "7 d"]


def load():
    g = pd.read_csv(DATA / "r2_spatiotemporal_grid.csv")
    med = (g.groupby(["dist_ring", "grace_h"]).uncovered_share.median().unstack() * 100)
    med = med.reindex(index=DIST_ORDER, columns=GRACE)
    return g, med


def panel_a(ax, med):
    mat = med.to_numpy()
    im = ax.imshow(mat, aspect="auto", cmap=CMAP, vmin=15, vmax=45)
    ax.set_xticks(range(len(GRACE)))
    ax.set_xticklabels(GRACE_TICK)
    ax.set_yticks(range(len(DIST_ORDER)))
    ax.set_yticklabels(DIST_TICK)
    ax.set_xlabel("Delay grace (deadline slack) → temporal flexibility")
    ax.set_ylabel("Max migration distance → spatial flexibility")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10.5,
                    color="white" if 25 <= v <= 39.5 else INK, fontweight="medium")
    # mark baseline corner and best corner
    ax.add_patch(mpl.patches.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=INK, lw=2.2))
    ax.add_patch(mpl.patches.Rectangle((len(GRACE) - 1.5, len(DIST_ORDER) - 1.5), 1, 1,
                                       fill=False, ec=P.POSITIVE, lw=2.4))
    ax.text(0, -0.78, "status quo", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(len(GRACE) - 1, len(DIST_ORDER) - 0.5 + 0.30, "both levers", ha="center",
            va="top", fontsize=9.5, color=P.POSITIVE)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Median uncovered share (%)", fontsize=10.5)
    ax.set_title("(a) Compute spatiotemporal flexibility grid", loc="left", pad=10)


def panel_b(ax, med):
    # family of curves: one line per distance ring, x = grace. Curve spacing = the
    # spatial lever; each curve's downward slope = the temporal lever.
    xt = np.arange(len(GRACE))
    purples = ["#C9B8E6", "#9B7FC4", "#7C4DB8", "#4C1D95"]  # light->dark = near->global
    ring_lab = {"D0": "In-country (0 km)", "D1": "≤500 km", "D2": "≤3000 km", "D3": "Global"}
    for ring, col in zip(DIST_ORDER, purples):
        ax.plot(xt, med.loc[ring].to_numpy(), "-o", color=col, lw=2.4, ms=5.5,
                zorder=3, label=ring_lab[ring])
    ax.set_xticks(xt)
    ax.set_xticklabels(GRACE_TICK)
    ax.set_xlabel("Delay grace (deadline slack) → temporal flexibility")
    ax.set_ylabel("Median uncovered share (%)")
    ax.set_xlim(-0.3, len(GRACE) - 0.4)

    # spatial-lever annotation: vertical span between D0 and D3 at g=0
    y0, y3 = med.loc["D0", 0], med.loc["D3", 0]
    ax.annotate("", xy=(0.06, y3), xytext=(0.06, y0),
                arrowprops=dict(arrowstyle="<->", color=SPATIAL_C, lw=1.6))
    ax.text(0.18, (y0 + y3) / 2, "spatial\nlever", color=SPATIAL_C, fontsize=9,
            va="center", ha="left", fontweight="bold")
    # temporal-lever annotation along the global curve
    ax.annotate("temporal lever → saturates by ~24 h", xy=(4.0, med.loc["D3", 12]),
                xytext=(2.2, med.loc["D3", 12] - 6.5), fontsize=9, color=TEMPORAL_C,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEMPORAL_C, lw=1.4))
    both = med.loc["D3", 168]
    ax.plot(len(GRACE) - 1, both, "*", color=P.POSITIVE, ms=16, zorder=4)
    ax.legend(title="Max migration distance", loc="upper right", frameon=False,
              fontsize=9.5, title_fontsize=9.5)
    ax.set_title("(b) Curve spacing = spatial lever; slope = temporal lever", loc="left", pad=10)


def panel_c(ax, g):
    cells = [("D0", 0, "Status quo", P.NEUTRAL),
             ("D0", 24, "Temporal max\n(in-country, 24 h)", TEMPORAL_C),
             ("D3", 0, "Spatial max\n(global, 0 h)", SPATIAL_C),
             ("D3", 168, "Both max\n(global, 7 d)", P.POSITIVE)]
    data, labels, colors = [], [], []
    for ring, gh, lab, col in cells:
        arr = g[(g.dist_ring == ring) & (g.grace_h == gh)].uncovered_share.to_numpy() * 100
        data.append(arr); labels.append(lab); colors.append(col)
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for b, col in zip(parts["bodies"], colors):
        b.set_facecolor(col); b.set_alpha(0.28); b.set_edgecolor(col); b.set_linewidth(1.2)
    bp = ax.boxplot(data, widths=0.18, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor("white"); patch.set_edgecolor(col); patch.set_linewidth(1.4)
    for i, arr in enumerate(data):
        ax.text(i + 1, np.median(arr) + 1.5, f"{np.median(arr):.1f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color=colors[i])
    ax.set_xticks(range(1, len(cells) + 1))
    ax.set_xticklabels(labels, fontsize=9.3)
    ax.set_ylabel("Uncovered share across 104 countries (%)")
    ax.set_ylim(0, 62)
    ax.set_title("(c) Both levers shift the whole national distribution", loc="left", pad=10)


def panel_d(ax, med):
    base = med.loc["D0", 0]
    temporal = base - med.loc["D0", 24]
    spatial = base - med.loc["D3", 0]
    combined = base - med.loc["D3", 168]
    ssum = temporal + spatial
    labels = ["Temporal only\n(in-country, 24 h)", "Spatial only\n(global, 0 h)",
              "Naive sum", "Actual combined\n(global, 7 d)"]
    vals = [temporal, spatial, ssum, combined]
    cols = [TEMPORAL_C, SPATIAL_C, P.NEUTRAL, P.POSITIVE]
    y = np.arange(len(labels))[::-1]
    for yi, v, c, lab in zip(y, vals, cols, labels):
        hatch = "//" if lab == "Naive sum" else None
        ax.barh(yi, v, color=("white" if hatch else c), edgecolor=c, lw=1.6,
                hatch=hatch, alpha=(1.0 if not hatch else 0.9), height=0.62)
        ax.text(v + 0.3, yi, f"−{v:.1f} pp", va="center", ha="left",
                fontsize=10.5, color=c, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Gap reduction vs status quo (percentage points)")
    ax.set_xlim(0, ssum * 1.22)
    # interaction-loss annotation between naive sum and actual combined
    ax.annotate("", xy=(combined, 0.5), xytext=(ssum, 0.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text((combined + ssum) / 2, 0.72, f"interaction\nloss {ssum - combined:.1f} pp",
            ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("(d) Levers are sub-additive (spatial dominates)", loc="left", pad=10)


def main():
    g, med = load()
    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26,
                          left=0.075, right=0.965, top=0.93, bottom=0.075)
    panel_a(fig.add_subplot(gs[0, 0]), med)
    panel_b(fig.add_subplot(gs[0, 1]), med)
    panel_c(fig.add_subplot(gs[1, 0]), g)
    panel_d(fig.add_subplot(gs[1, 1]), med)
    fig.suptitle("Result 2 — compute spatiotemporal flexibility under fixed national electricity supply",
                 x=0.075, ha="left", fontsize=15, fontweight="bold")
    out = FIG / "fig2_spatiotemporal"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
