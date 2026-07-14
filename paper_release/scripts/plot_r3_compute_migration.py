#!/usr/bin/env python3
"""Result 3 figure in the ORIGINAL Fig-3 template, recomputed for COMPUTE migration.

Same 5-panel layout/style as plot_fig3_redesign.render_fig3, but:
  - geographic axis = compute MIGRATION distance (D0-D3), not electricity scope;
  - full-system cost has THREE components (generation + firming storage + ancillary),
    no transmission (compute travels over the internet);
  - panel (c) is MONOTONE DOWN: cost falls as migration range widens, global cheapest
    -- overturning the old electricity-transmission 'U-shape / global most expensive'.
Data: data_globalsites/r3_cm_cost_table.csv (built by analyze_cfe_r3_compute_migration).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_geographic_portfolio_ai as base
import plot_fig3_redesign as R3P
import plot_fig3b_polar_rose as rose

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

DATA = R3P.DATA
FIG = R3P.FIG
INK, GRID = R3P.INK, R3P.GRID

DIST = ["D0", "D1", "D2", "D3"]
DIST_NAME = {"D0": "In-country", "D1": "≤500 km", "D2": "≤3000 km", "D3": "Global"}
GRACE = [0, 6, 24, 168]
GLAB = {0: "0 h", 6: "6 h", 24: "24 h", 168: "7 d"}
CMAP_PA = mpl.colors.LinearSegmentedColormap.from_list(
    "fig_pa", ["#F2EFF7", "#B6A8D0", "#8A6FB0", "#C75D88", "#E8A33D"])
WORSE, BETTER = "#C2410C", "#0F766E"

# 3-component cost (no transmission) — reuse the rose's muted blue->lilac ramp
CM_COMPONENTS = [("lcoe_gen", "Generation", "#6E7FA8"),
                 ("lcoe_store", "Firming storage", "#B6A8D0"),
                 ("lcoe_anc", "Ancillary", "#D8D1E6")]


def load():
    df = pd.read_csv(DATA / "r3_cm_cost_table.csv")
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    return df, dn


def unc(df, year, op, dist, grace):
    r = df[(df.year == year) & (df.perturbation == op) & (df.dist_ring == dist)
           & (df.grace_h == grace)]
    return float(r["uncovered_share"].mean()) * 100 if len(r) else np.nan


# ============================ panel (a) ============================
def draw_a(fig, cell, df, dn, op="P_mix", fs=11.0):
    sub = cell.subgridspec(2, 2, height_ratios=[0.40, 1.0], width_ratios=[1.0, 0.42],
                           hspace=0.22, wspace=0.07)
    ax = fig.add_subplot(sub[0, 0]); axh = fig.add_subplot(sub[1, 0])
    axl = fig.add_subplot(sub[1, 1])
    h = np.arange(24)
    avg = dn.groupby("local_hour")["demand"].mean().reindex(range(24)).to_numpy()
    d = avg / avg.mean()
    years, alphas = [2025, 2030, 2050], [0.32, 0.60, 1.0]

    def shp(y):
        li, lt = base.AI_SHARE_INFERENCE[y], base.AI_SHARE_TRAINING[y]
        return base.operator_mix(d, h, li, lt)
    ax.axvspan(18, 23, color="#64748B", alpha=0.12, zorder=0)
    ax.plot(h, d, color="#9CA3AF", lw=0.8, ls=":", alpha=0.9, zorder=2)
    for y, al in zip(years, alphas):
        ax.plot(h, shp(y), color="#7C3AED", alpha=al, lw=1.5 if y == 2030 else 1.0,
                zorder=3, label=str(y))
    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18]); ax.set_ylim(0, 3.2)
    ax.set_yticks([0, 1, 2, 3]); ax.set_xlabel("Local hour", fontsize=fs, labelpad=1.5)
    ax.set_ylabel("Demand\n(mean = 1)", fontsize=fs)
    ax.set_title("Mix", fontsize=fs + 1.5, loc="left", color="#7C3AED", fontweight="bold", pad=2)
    R3P.yonly(ax)
    ax.legend(title="by year", fontsize=fs - 0.5, title_fontsize=fs - 0.5, loc="upper left",
              ncol=3, columnspacing=0.7, handlelength=0.9, borderaxespad=0.12)

    # heatmap: distance (rows, D0 bottom) x grace (cols)
    allv = [unc(df, 2030, o, D, g) for o in ["P0_baseline", "P_mix"] for D in DIST for g in GRACE]
    vmax = float(np.ceil(np.nanmax(allv) / 5) * 5)
    norm = mpl.colors.Normalize(0, vmax)
    for i, D in enumerate(DIST):                         # i=0 (D0) at bottom
        for j, g in enumerate(GRACE):
            v = unc(df, 2030, op, D, g)
            fc = CMAP_PA(norm(v))
            axh.add_patch(mpatches.FancyBboxPatch((j - 0.46, i - 0.46), 0.92, 0.92,
                          boxstyle="round,pad=0,rounding_size=0.14", facecolor=fc,
                          edgecolor="white", linewidth=0.6, mutation_aspect=1))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            axh.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs + 4.5,
                     color="white" if lum < 0.55 else "#1F2937", fontweight="bold", zorder=4)
    axh.set_xlim(-0.6, 3.6); axh.set_ylim(-0.6, 3.6)
    axh.set_xticks(range(4)); axh.set_xticklabels([GLAB[g] for g in GRACE], fontsize=fs)
    axh.set_yticks(range(4)); axh.set_yticklabels(DIST, fontsize=fs)
    axh.set_ylabel("Migration distance ↑", fontsize=fs)
    axh.set_xlabel("Delay grace →", fontsize=fs)
    axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    # lollipop: Δ uncovered share vs no-AI (P0) by distance (mean over grace)
    base_row = {D: np.mean([unc(df, 2030, "P0_baseline", D, g) for g in GRACE]) for D in DIST}
    diff = {D: np.mean([unc(df, 2030, op, D, g) for g in GRACE]) - base_row[D] for D in DIST}
    dlim = max(1.0, np.ceil(max(abs(v) for v in diff.values())))
    axl.set_ylim(-0.6, 3.6); axl.set_xlim(-dlim, dlim)
    axl.axvline(0, color="#9CA3AF", lw=0.7)
    for i, D in enumerate(DIST):
        dv = diff[D]
        mc = WORSE if dv > 0.05 else (BETTER if dv < -0.05 else "#9CA3AF")
        axl.plot([0, dv], [i, i], color="#CBD5E1", lw=1.3)
        axl.plot(dv, i, "o", color=mc, ms=4.6, markeredgecolor="white", markeredgewidth=0.5)
    axl.set_yticks([]); axl.set_xticks([-int(dlim), 0, int(dlim)])
    for sp in ("top", "right", "left"):
        axl.spines[sp].set_visible(False)
    axl.spines["bottom"].set_color("#9CA3AF")
    axl.set_xlabel("Δ share\nvs no-AI (pp)", fontsize=fs - 0.5, labelpad=1.5)
    axl.grid(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP_PA); sm.set_array([])
    cax = axh.inset_axes([0.0, -0.30, 1.0, 0.045])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Uncovered demand share (%, 2030)", fontsize=fs - 1)
    cb.ax.tick_params(labelsize=fs - 1)


# ============================ panel (b): rose ============================
def draw_b(fig, cell, df, fs=10.5):
    sub = df[(df.year == 2030) & (df.perturbation == "P_mix")]
    m = sub.groupby(["dist_ring", "grace_h"])[[c for c, _, _ in CM_COMPONENTS]].mean()
    rows = [(D, GLAB[g], m.loc[(D, g)]) for D in DIST for g in GRACE]
    rose.COMPONENTS = CM_COMPONENTS
    rose.L_ORDER = DIST
    rose.W_ORDER = [GLAB[g] for g in GRACE]
    rose.L_NAME = DIST_NAME
    rose.W_NAME = {GLAB[g]: GLAB[g] for g in GRACE}
    rose.GROUP_COLOR = {"D0": "#AEB4C2", "D1": "#9DA1C4", "D2": "#867BAE", "D3": "#6A5E96"}
    rose.draw_rose_into(fig, cell, rows=rows, fs=fs)


# ============================ panel (c): cost vs distance ============================
def draw_c(ax, df, grace=0):
    sub = df[(df.year == 2030) & (df.perturbation == "P_mix") & (df.grace_h == grace)]
    g = sub.groupby("dist_ring")[["lcoe_gen", "lcoe_store", "lcoe_anc", "lcoe_elec"]].mean().reindex(DIST)
    x = np.arange(4)
    gen, sto, anc, tot = g.lcoe_gen.values, g.lcoe_store.values, g.lcoe_anc.values, g.lcoe_elec.values
    ax.fill_between(x, 0, gen, color="#6E7FA8", alpha=0.85, lw=0, label="Generation", zorder=2)
    ax.fill_between(x, gen, gen + sto, color="#B6A8D0", alpha=0.9, lw=0, label="Firming storage", zorder=2)
    ax.fill_between(x, gen + sto, gen + sto + anc, color="#D8D1E6", alpha=0.95, lw=0, label="Ancillary", zorder=2)
    ax.plot(x, tot, color="#1F2937", lw=2.6, marker="s", ms=7, mec="white", mew=0.9,
            zorder=6, label="Full-system total")
    for xi, tv in zip(x, tot):
        ax.annotate(f"{tv:.0f}", (xi, tv), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color="#1F2937", fontweight="bold")
    # contrast with the old move-power result (global most expensive ~285)
    ax.axhline(285, color="#B91C1C", lw=1.3, ls=(0, (5, 2)), zorder=3)
    ax.text(0.04, 288, "move-power global ≈ 285 (old: most expensive)", fontsize=8.0,
            color="#B91C1C", ha="left", va="bottom")
    ax.annotate(f"global cheapest\n{tot[-1]:.0f} (≈{285/tot[-1]:.1f}× below move-power)",
                xy=(3, tot[-1]), xytext=(2.0, tot[-1] + 70),
                fontsize=8.6, color="#0F766E", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels([f"{D}\n{DIST_NAME[D]}" for D in DIST], fontsize=9.3)
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(0, 305)
    ax.set_xlabel("Compute migration distance", fontsize=11.0)
    ax.set_ylabel("Full-system cost\n(USD/MWh, 2030)", fontsize=11.0)
    R3P.yonly(ax)
    ax.legend(fontsize=8.2, frameon=False, loc="center left", labelspacing=0.35, handlelength=1.4)


# ============================ panel (d): ridgeline ============================
RXG = np.linspace(50, 250, 240)


def _kde(a):
    a = a[np.isfinite(a)]
    if len(a) < 5 or np.std(a) < 1e-6:
        return np.zeros_like(RXG)
    return gaussian_kde(a, bw_method=0.45)(RXG)


def draw_d(ax, df):
    sub = df[(df.perturbation == "P_mix") & (df.grace_h == 0)]
    col = {"D0": "#9AA7B8", "D1": "#8C84C4", "D2": "#7A4FA0", "D3": "#46235F"}
    years = [2025, 2030, 2035, 2040, 2045, 2050]
    n, gap, scale = len(years), 1.0, 1.48

    def arr(D, yr):
        return sub[(sub.dist_ring == D) & (sub.year == yr)]["lcoe_elec"].to_numpy()
    dmax = max(_kde(arr(D, y)).max() for D in DIST for y in years)
    for row, y in enumerate(years):
        b = (n - 1 - row) * gap; z = (row + 1) * 10
        for D in ["D3", "D2", "D1", "D0"]:
            a = arr(D, y); dd = _kde(a) / dmax * scale
            ax.fill_between(RXG, b, b + dd, color=col[D], alpha=0.5, lw=0, zorder=z + DIST.index(D))
            ax.plot(RXG, b + dd, color=col[D], lw=1.1, zorder=z + DIST.index(D) + 4)
            med = float(np.median(a)); hpk = dd[np.argmin(np.abs(RXG - med))]
            ax.plot([med, med], [b, b + hpk], color="white", lw=1.0, zorder=z + DIST.index(D) + 5)
        ax.text(248, b + 0.06, str(y), ha="right", va="bottom", fontsize=9.5,
                color=INK, fontweight="bold", zorder=z + 30)
    d0 = np.median(arr("D0", 2025)); g50 = np.median(arr("D3", 2050))
    ax.annotate(f"Global always cheapest →\nfalls to ~{g50:.0f} by 2050", xy=(g50, 0.5),
                xytext=(120, 2.0), fontsize=8.4, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.8))
    ax.set_yticks([]); ax.set_xlim(50, 252); ax.set_ylim(-0.15, (n - 1) * gap + scale + 0.25)
    ax.set_xlabel("Full-system electricity cost (USD/MWh, per country)", fontsize=11.0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6)
    handles = [mpatches.Patch(facecolor=col[D], edgecolor="white", alpha=0.7,
                              label=f"{D} {DIST_NAME[D]}") for D in DIST]
    ax.legend(handles=handles, title="Compute migration distance", fontsize=8.4,
              title_fontsize=8.6, frameon=False, loc="upper right", ncol=2,
              columnspacing=1.0, handlelength=1.3, labelspacing=0.3)


# ============================ assemble ============================
def main():
    R3P.set_style(11.0)
    plt.close("all")
    df, dn = load()
    years = [2025, 2030, 2050]
    fig = plt.figure(figsize=(13.2, 16.0))
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.955, top=0.965, bottom=0.032,
                             height_ratios=[1.04, 1.30], hspace=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.16)
    draw_a(fig, top[0, 0], df, dn)
    draw_b(fig, top[0, 1], df)

    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.92], wspace=0.10)
    cd = bot[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.40)
    draw_c(fig.add_subplot(cd[0, 0]), df)
    draw_d(fig.add_subplot(cd[1, 0]), df)

    em = bot[0, 1].subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.09], hspace=0.14)
    cmap = CMAP_PA; cmap.set_bad(alpha=0)
    sub = df[(df.perturbation == "P_mix") & (df.grace_h == 0) & (df.dist_ring == "D3")]
    vby = {y: dict(zip(sub[sub.year == y]["country"], sub[sub.year == y]["lcoe_elec"]))
           for y in years}
    allv = np.array([v for y in years for v in vby[y].values()])
    vmin = float(np.floor(np.percentile(allv, 4) / 10) * 10)
    vmax = float(np.ceil(np.percentile(allv, 96) / 10) * 10)
    norm = mpl.colors.Normalize(vmin, vmax)
    for j, y in enumerate(years):
        axm = fig.add_subplot(em[j, 0], projection=ccrs.Robinson())
        R3P._country_choropleth(axm, vby[y], cmap, norm)
        axm.text(0.005, 0.90, str(y), transform=axm.transAxes, fontsize=12.5,
                 fontweight="bold", color=INK)
        if j == 0:
            axm.legend(handles=[mpatches.Patch(facecolor="#C7CCD1", edgecolor="none",
                       label="No data")], loc="lower left", fontsize=11.0, frameon=False,
                       handlelength=1.1, handleheight=1.1, borderaxespad=0.0)
    cax = fig.add_subplot(em[3, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Per-country full-system cost (USD/MWh · global compute migration)", fontsize=11.0)
    cb.ax.tick_params(labelsize=11.0)

    for cell, lab in [(top[0, 0], "a"), (top[0, 1], "b"), (cd[0, 0], "c"),
                      (cd[1, 0], "d"), (em[0, 0], "e")]:
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.030, bb.y1 + 0.005, lab, fontweight="bold", fontsize=16,
                 ha="left", va="bottom")
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig3_compute_migration{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG/'fig3_compute_migration.png'}")


if __name__ == "__main__":
    main()
