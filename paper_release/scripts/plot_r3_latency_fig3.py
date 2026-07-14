#!/usr/bin/env python3
"""Result 3 (latency x routable-fraction) in the ORIGINAL Fig-3 template.

Same 5-panel layout as plot_fig3_redesign.render_fig3, compute-migration cost
(generation + firming storage + ancillary, NO transmission), geographic lever =
latency tolerance tau (ms, RTT-gated reach), flexibility lever = routable fraction phi.
Panel (c) is MONOTONE DOWN in tau: looser latency -> wider reach -> cheaper, global
cheapest. Data: data_globalsites/r3_latency_cost_table.csv.
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

DATA = R3P.DATA; FIG = R3P.FIG; INK, GRID = R3P.INK, R3P.GRID

TAU = [50, 100, 150, 200, 300, 500]      # aligned with R2 (6 levels)
TAU_NAME = {50: "50 ms\nlocal", 100: "100 ms\n≈3 Mm", 150: "150 ms\n≈6.7 Mm",
            200: "200 ms\n≈10 Mm", 300: "300 ms\n≈17 Mm", 500: "500 ms\nglobal"}
PHI_ROWS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]   # top -> bottom (6 levels)
PHI_LAB = {1.0: "100%", 0.8: "80%", 0.6: "60%", 0.4: "40%", 0.2: "20%", 0.0: "0%"}
TAU_SUB = [50, 150, 300, 500]            # 4-level subset for the rose (b) and ridgeline (d)
PHI_SUB = [1.0, 0.8, 0.4, 0.0]           # 4-level subset for the rose (b)
CMAP_PA = mpl.colors.LinearSegmentedColormap.from_list(
    "fig_pa", ["#F2EFF7", "#B6A8D0", "#8A6FB0", "#C75D88", "#E8A33D"])
WORSE, BETTER = "#C2410C", "#0F766E"
CM_COMPONENTS = [("lcoe_gen", "Generation", "#3D348B"),
                 ("lcoe_store", "Firming storage", "#9085B8"),
                 ("lcoe_anc", "Ancillary", "#D9D4EA")]


def load():
    return pd.read_csv(DATA / "r3_waveform_cost_table.csv"), pd.read_csv(DATA / "r1_diurnal_profiles.csv")


def unc(df, year, op, phi, tau):
    r = df[(df.year == year) & (df.perturbation == op) & (df.phi == phi) & (df.tau_ms == tau)]
    return float(r["uncovered_share"].mean()) * 100 if len(r) else np.nan


# panel (a)
def draw_a(fig, cell, df, dn, op="P_mix", fs=11.0):
    sub = cell.subgridspec(2, 2, height_ratios=[0.40, 1.0], width_ratios=[1.0, 0.42],
                           hspace=0.22, wspace=0.07)
    ax = fig.add_subplot(sub[0, 0]); axh = fig.add_subplot(sub[1, 0]); axl = fig.add_subplot(sub[1, 1])
    h = np.arange(24)
    avg = dn.groupby("local_hour")["demand"].mean().reindex(range(24)).to_numpy(); d = avg / avg.mean()
    for y, al in zip([2025, 2030, 2050], [0.32, 0.60, 1.0]):
        li, lt = base.AI_SHARE_INFERENCE[y], base.AI_SHARE_TRAINING[y]
        ax.plot(h, base.operator_mix(d, h, li, lt), color="#7C3AED", alpha=al,
                lw=1.5 if y == 2030 else 1.0, zorder=3, label=str(y))
    ax.axvspan(18, 23, color="#64748B", alpha=0.12, zorder=0)
    ax.plot(h, d, color="#9CA3AF", lw=0.8, ls=":", alpha=0.9, zorder=2)
    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18]); ax.set_ylim(0, 3.2); ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Local hour", fontsize=fs, labelpad=1.5); ax.set_ylabel("Demand\n(mean = 1)", fontsize=fs)
    ax.set_title("Mix", fontsize=fs + 1.5, loc="left", color="#7C3AED", fontweight="bold", pad=2)
    R3P.yonly(ax)
    ax.legend(title="by year", fontsize=fs - 0.5, title_fontsize=fs - 0.5, loc="upper left", ncol=3,
              columnspacing=0.7, handlelength=0.9, borderaxespad=0.12)
    allv = [unc(df, 2030, o, p, t) for o in ["P0_baseline", "P_mix"] for p in PHI_ROWS for t in TAU]
    vmax = float(np.ceil(np.nanmax(allv) / 5) * 5); norm = mpl.colors.Normalize(0, vmax)
    for i, p in enumerate(PHI_ROWS):
        for j, t in enumerate(TAU):
            v = unc(df, 2030, op, p, t); fc = CMAP_PA(norm(v))
            axh.add_patch(mpatches.FancyBboxPatch((j - 0.46, i - 0.46), 0.92, 0.92,
                          boxstyle="round,pad=0,rounding_size=0.14", facecolor=fc,
                          edgecolor="white", linewidth=0.6, mutation_aspect=1))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            axh.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs + 0.5,
                     color="white" if lum < 0.55 else "#1F2937", fontweight="bold", zorder=4)
    axh.set_xlim(-0.6, len(TAU) - 0.4); axh.set_ylim(-0.6, len(PHI_ROWS) - 0.4)
    axh.set_xticks(range(len(TAU))); axh.set_xticklabels([str(t) for t in TAU], fontsize=fs - 1)
    axh.set_yticks(range(len(PHI_ROWS))); axh.set_yticklabels([PHI_LAB[p] for p in PHI_ROWS], fontsize=fs - 1)
    axh.set_ylabel("Routable fraction ↑", fontsize=fs); axh.set_xlabel("Latency tolerance (ms) →", fontsize=fs)
    axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    base_row = {p: np.mean([unc(df, 2030, "P0_baseline", p, t) for t in TAU]) for p in PHI_ROWS}
    diff = {p: np.mean([unc(df, 2030, op, p, t) for t in TAU]) - base_row[p] for p in PHI_ROWS}
    dlim = max(1.0, np.ceil(max(abs(v) for v in diff.values())))
    axl.set_ylim(-0.6, len(PHI_ROWS) - 0.4); axl.set_xlim(-dlim, dlim); axl.axvline(0, color="#9CA3AF", lw=0.7)
    for i, p in enumerate(PHI_ROWS):
        dv = diff[p]; mc = WORSE if dv > 0.05 else (BETTER if dv < -0.05 else "#9CA3AF")
        axl.plot([0, dv], [i, i], color="#CBD5E1", lw=1.3)
        axl.plot(dv, i, "o", color=mc, ms=4.6, markeredgecolor="white", markeredgewidth=0.5)
    axl.set_yticks([]); axl.set_xticks([-int(dlim), 0, int(dlim)])
    for sp in ("top", "right", "left"):
        axl.spines[sp].set_visible(False)
    axl.spines["bottom"].set_color("#9CA3AF")
    axl.set_xlabel("Δ share\nvs no-AI (pp)", fontsize=fs - 0.5, labelpad=1.5); axl.grid(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP_PA); sm.set_array([])
    cax = axh.inset_axes([0.0, -0.30, 1.0, 0.045]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Uncovered demand share (%, 2030)", fontsize=fs - 1); cb.ax.tick_params(labelsize=fs - 1)


# panel (b) rose — full 6 latency x 6 routable = 36 sectors
ROSE_GROUP = ["#C2C6D2", "#AEB4C8", "#9DA1C4", "#867BAE", "#6A5E96", "#4C1D95"]


def draw_b(fig, cell, df, fs=10.5):
    # phase-firming premium Pi = firming(storage+ancillary) cost as % of the
    # generation floor; two radial components (storage premium, ancillary premium).
    sub = df[(df.year == 2030) & (df.perturbation == "P_mix")]
    g = sub.groupby(["tau_ms", "phi"])[["lcoe_gen", "lcoe_store", "lcoe_anc"]].mean()
    m = pd.DataFrame({"pi_store": g["lcoe_store"] / g["lcoe_gen"] * 100.0,
                      "pi_anc": g["lcoe_anc"] / g["lcoe_gen"] * 100.0})
    pi_components = [("pi_store", "Firming storage", "#9085B8"),
                    ("pi_anc", "Ancillary", "#D9D4EA")]
    phi_asc = PHI_ROWS[::-1]
    rows = [(t, PHI_LAB[p], m.loc[(t, p)]) for t in TAU for p in phi_asc]
    rose.COMPONENTS = pi_components
    rose.L_ORDER = TAU
    rose.W_ORDER = [PHI_LAB[p] for p in phi_asc]
    rose.L_NAME = {t: f"{t}ms" for t in TAU}
    rose.W_NAME = {PHI_LAB[p]: PHI_LAB[p] for p in PHI_ROWS}
    rose.GROUP_COLOR = {t: ROSE_GROUP[i] for i, t in enumerate(TAU)}
    _orig = rose.sector_angles
    rose.sector_angles = lambda n_groups=len(TAU), n_per=len(phi_asc), gap=np.pi / 60: _orig(n_groups, n_per, gap)
    ax = fig.add_subplot(cell, projection="polar")
    handles, labels = rose.draw_rose_on_ax(ax, rows, fs=fs, sector_labels=False, center_label=False)
    ax.text(0, 0, "Phase-firming\npremium (% of gen)", ha="center", va="center",
            fontsize=fs - 1.0, fontweight="bold", color="#555555")
    ax.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=len(handles), fontsize=fs - 1.0, frameon=False, columnspacing=0.8,
              handlelength=1.0, handletextpad=0.35, borderaxespad=0.0)
    rose.sector_angles = _orig


# panel (c/a) — full-system cost waterfall vs latency tolerance.
# The total steps DOWN from the local total (125) to the global total (95) as
# reach widens; every descending step is a firming-storage saving (the
# cross-hemisphere ~200 ms jump is the largest, and 300/500 ms coincide), while
# a flat reference line marks the unchanged generation floor (~73). Replaces the
# old stacked area, where the constant generation block buried the shrinking
# storage band.
def draw_c(ax, df, phi=1.0):
    sub = df[(df.year == 2030) & (df.perturbation == "P_mix") & (df.phi == phi)].copy()
    sub["pi"] = (sub["lcoe_elec"] - sub["lcoe_gen"]) / sub["lcoe_gen"] * 100.0
    tot = sub.groupby("tau_ms")["pi"].median().reindex(TAU).values
    x = np.arange(len(TAU)); w = 0.62
    C_ANCHOR, C_SAVE, C_FLOOR = "#7C57C9", BETTER, "#6E7FA8"
    floor = 0.0                                       # generation floor = Pi = 0
    ax.axhline(floor, color=C_FLOOR, lw=1.2, ls=(0, (4, 2)), zorder=1)
    ax.text(len(TAU) - 0.45, 3.5, "generation floor (Π = 0)", color=C_FLOOR,
            fontsize=8.0, fontweight="bold", va="bottom", ha="right")
    # anchor bar at the local premium
    ax.bar(0, tot[0], w, color=C_ANCHOR, alpha=0.85, zorder=3)
    ax.annotate(f"{tot[0]:.0f}%", (0, tot[0]), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9.0, color=INK, fontweight="bold")
    # floating saving steps between consecutive premiums
    for i in range(1, len(TAU)):
        lo, hi = min(tot[i], tot[i - 1]), max(tot[i], tot[i - 1])
        drop = tot[i - 1] - tot[i]
        col = C_SAVE if drop > 0.15 else "#CBD5E1"
        ax.bar(i, hi - lo, w, bottom=lo, color=col, alpha=0.85, zorder=3)
        ax.plot([i - 1 + w / 2, i - w / 2], [tot[i - 1], tot[i - 1]], color="#94A3B8",
                lw=0.9, ls=":", zorder=2)
        ax.annotate(f"{tot[i]:.0f}%", (i, lo), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=8.6, color=INK, fontweight="bold")
        if drop > 2:                                  # only the meaningful steps
            ax.annotate(f"−{drop:.0f}", (i, (lo + hi) / 2), ha="center", va="center",
                        fontsize=8.2, color="white", fontweight="bold", zorder=6)
    ax.annotate("crossing a hemisphere:\nbiggest storage saving", xy=(3, max(tot[2], tot[3]) + 1),
                xytext=(3.05, 92), fontsize=8.2, color=C_SAVE, fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_SAVE, lw=1.1))
    ax.set_xticks(x); ax.set_xticklabels([TAU_NAME[t] for t in TAU], fontsize=8.0)
    ax.set_xlim(-0.6, len(TAU) - 0.15); ax.set_ylim(0, 100)
    ax.set_xlabel("Latency tolerance (RTT → reach)", fontsize=11.0)
    ax.set_ylabel("Phase-firming premium\n(% of generation, 2030, φ=100%)", fontsize=11.0)
    R3P.yonly(ax)
    ax.set_axisbelow(True)


# panel (d/c) — two faceted ridgelines (local vs global) + all-band median inset.
# The six latency bands collapse into two clusters (50/100/150 vs 200/300/500,
# and 300 ms == 500 ms), so overlaying all six per year piled into mud. We split
# the two anchor bands into clean per-year ridges and keep every band as a line
# in the median-vs-year inset, so nothing is dropped.
RXG = np.linspace(0, 140, 260)
RIDGE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def _kde(a):
    a = a[np.isfinite(a)]
    if len(a) < 5 or np.std(a) < 1e-6:
        return np.zeros_like(RXG)
    return gaussian_kde(a, bw_method=0.42)(RXG)


def _facet_ridge(ax, sub, tau, base_col, title):
    """One clean 6-year ridgeline for a single latency band: pale->saturated by
    year, white median sticks, and a dashed median trajectory labelled at the
    2025/2050 endpoints."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("f", ["#E5E7EB", base_col])
    gap, scale = 1.0, 1.5
    yrs = RIDGE_YEARS

    def arr(y):
        s = sub[(sub.tau_ms == tau) & (sub.year == y)]
        return ((s["lcoe_elec"] - s["lcoe_gen"]) / s["lcoe_gen"] * 100.0).to_numpy()
    dmax = max(_kde(arr(y)).max() for y in yrs)
    meds = []
    for row, y in enumerate(yrs):
        b = (len(yrs) - 1 - row) * gap
        dd = _kde(arr(y)) / dmax * scale
        c = cmap(0.32 + 0.68 * row / (len(yrs) - 1))
        ax.fill_between(RXG, b, b + dd, color=c, alpha=0.82, lw=0, zorder=10 + row)
        ax.plot(RXG, b + dd, color="white", lw=0.8, zorder=10 + row + 1)
        m = float(np.median(arr(y))); meds.append((m, b))
        hpk = dd[np.argmin(np.abs(RXG - m))]
        ax.plot([m, m], [b, b + hpk], color="white", lw=1.1, zorder=10 + row + 2)
        ax.text(132, b + 0.08, str(y), ha="right", va="bottom", fontsize=8.2,
                color=INK, fontweight="bold")
    mx, my = zip(*meds)
    ax.plot(mx, my, color=base_col, lw=1.4, ls="--", marker="o", ms=4.2,
            mec="white", mew=0.7, zorder=60)
    ax.annotate(f"{mx[0]:.0f}", (mx[0], my[0]), xytext=(6, 0), textcoords="offset points",
                fontsize=7.8, color=base_col, va="center", fontweight="bold")
    ax.annotate(f"{mx[-1]:.0f}", (mx[-1], my[-1]), xytext=(-6, 0), textcoords="offset points",
                fontsize=7.8, color=base_col, va="center", ha="right", fontweight="bold")
    ax.set_title(title, fontsize=9.5, color=base_col, fontweight="bold", loc="left", pad=3)
    ax.set_yticks([]); ax.set_xlim(0, 134)
    ax.set_ylim(-0.2, (len(yrs) - 1) * gap + scale + 0.15)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.5)
    ax.set_xlabel("Per-country premium (% of gen)", fontsize=9.0)


def draw_d(fig, cell, df):
    sub = df[(df.perturbation == "P_mix") & (df.phi == 1.0)]
    inner = cell.subgridspec(2, 2, height_ratios=[1.0, 0.42], width_ratios=[1, 1],
                             hspace=0.34, wspace=0.07)
    axL = fig.add_subplot(inner[0, 0]); axG = fig.add_subplot(inner[0, 1])
    _facet_ridge(axL, sub, 50, WORSE, "Local reach (50 ms)")
    _facet_ridge(axG, sub, 500, BETTER, "Global reach (500 ms)")
    # median-vs-year inset spanning both columns — keeps all six latency bands
    axm = fig.add_subplot(inner[1, :])
    band_cmap = mpl.colors.LinearSegmentedColormap.from_list("b", [WORSE, "#9A6A8A", BETTER])
    yrs = RIDGE_YEARS

    def med(t, y):
        s = sub[(sub.tau_ms == t) & (sub.year == y)]
        return float(np.median((s["lcoe_elec"] - s["lcoe_gen"]) / s["lcoe_gen"] * 100.0))
    for k, t in enumerate(TAU):
        c = band_cmap(k / (len(TAU) - 1)); lw = 2.2 if t in (50, 500) else 1.1
        axm.plot(yrs, [med(t, y) for y in yrs], color=c, lw=lw, marker="o", ms=3.4,
                 mec="white", mew=0.5, zorder=5 if t in (50, 500) else 3)
    axm.text(2050.4, med(50, 2050), "local", color=WORSE, fontsize=8.0,
             fontweight="bold", va="center", ha="left")
    axm.text(2050.4, med(500, 2050), "global", color=BETTER, fontsize=8.0,
             fontweight="bold", va="center", ha="left")
    axm.set_xlim(2025, 2052); axm.set_xlabel("Year", fontsize=9.0)
    axm.set_ylabel("Median premium\n(% of gen)", fontsize=8.6)
    axm.set_title("Median across 104 countries — all six latency bands",
                  fontsize=8.6, color="#475569", loc="left", pad=4)
    axm.grid(True, color=GRID, lw=0.6, alpha=0.5)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)


def main():
    R3P.set_style(11.0); plt.close("all")
    df, dn = load(); years = [2025, 2030, 2050]
    fig = plt.figure(figsize=(13.2, 16.0))
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.955, top=0.965, bottom=0.032,
                             height_ratios=[1.04, 1.30], hspace=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.16)
    draw_a(fig, top[0, 0], df, dn); draw_b(fig, top[0, 1], df)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.92], wspace=0.10)
    cd = bot[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.40)
    draw_c(fig.add_subplot(cd[0, 0]), df); draw_d(fig, cd[1, 0], df)
    em = bot[0, 1].subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.09], hspace=0.14)
    cmap = CMAP_PA; cmap.set_bad(alpha=0)
    sub = df[(df.perturbation == "P_mix") & (df.phi == 1.0) & (df.tau_ms == 500)]
    vby = {y: dict(zip(sub[sub.year == y]["country"], sub[sub.year == y]["lcoe_elec"])) for y in years}
    allv = np.array([v for y in years for v in vby[y].values()])
    vmin = float(np.floor(np.percentile(allv, 4) / 10) * 10); vmax = float(np.ceil(np.percentile(allv, 96) / 10) * 10)
    norm = mpl.colors.Normalize(vmin, vmax)
    for j, y in enumerate(years):
        axm = fig.add_subplot(em[j, 0], projection=ccrs.Robinson())
        R3P._country_choropleth(axm, vby[y], cmap, norm)
        axm.text(0.005, 0.90, str(y), transform=axm.transAxes, fontsize=12.5, fontweight="bold", color=INK)
        if j == 0:
            axm.legend(handles=[mpatches.Patch(facecolor="#C7CCD1", edgecolor="none", label="No data")],
                       loc="lower left", fontsize=11.0, frameon=False, handlelength=1.1, handleheight=1.1, borderaxespad=0.0)
    cax = fig.add_subplot(em[3, 0]); sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Per-country full-system cost (USD/MWh · global reach · φ=100%)", fontsize=11.0)
    cb.ax.tick_params(labelsize=11.0)
    for cell, lab in [(top[0, 0], "a"), (top[0, 1], "b"), (cd[0, 0], "c"), (cd[1, 0], "d"), (em[0, 0], "e")]:
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.030, bb.y1 + 0.005, lab, fontweight="bold", fontsize=16, ha="left", va="bottom")
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig3_latency{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG/'fig3_latency.png'}")


if __name__ == "__main__":
    main()
