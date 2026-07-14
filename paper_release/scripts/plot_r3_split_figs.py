#!/usr/bin/env python3
"""Split the Result-3 figure into a MECHANISM figure and a RESULTS figure.

The former single Fig. 3 (plot_r3_latency_fig3.py) carried two stories at once:
how AI reshapes data-centre demand, and the full-system cost of routing compute.
This script splits them.

fig3_ai_mechanism — why the AI-reshaped demand barely moves the gap:
  (a) the Mix day-shape across 2025/2030/2050 against the no-AI baseline;
  (b) NEW: the Mix and no-AI demand laid over the national clean-supply curve,
      shading where AI reshaping ADDS vs REMOVES residual gap — near zero-sum
      against a solar-driven supply trough that no demand shape can fill;
  (c) the six AI demand shapes cluster within ~1 pp at both national and global
      reach, dwarfed by the national->global routing lever.

fig4_routing_cost — routing reachability sets the gap and the cost:
  (a) routable-fraction x latency-tolerance uncovered-share heatmap;
  (b) full-system cost rose (36 scenarios); (c) cost vs latency tolerance;
  (d) 2025-2050 ridgeline; (e) three world maps.

Built on plot_r3_latency_fig3 (the live Fig-3 pipeline) — reuses its rose/cost/
ridgeline/map panels verbatim. academic-paper-figures skill: English-only,
colorblind-safe, matplotlib + cartopy. Data: data_globalsites/.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_geographic_portfolio_ai as base   # noqa: E402
import plot_r3_latency_fig3 as R3L                    # noqa: E402  (live Fig-3 pipeline)

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
warnings.filterwarnings("ignore", message=".*facecolor will have no effect.*")

DATA, FIG = R3L.DATA, R3L.FIG
INK, GRID = R3L.INK, R3L.GRID
CMAP_PA = R3L.CMAP_PA

# Reader-facing names for the six AI demand shapes (never the code keys).
SHAPE_ORDER = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
SHAPE_NAME = {"P0_baseline": "No AI (today)", "P1_flatten": "Batch & training",
              "P2_sharpen": "Consumer chat", "P2_emp": "Measured chat",
              "P3_burst": "Viral spikes", "P_mix": "Mix"}
SHAPE_COLOR = {"P0_baseline": "#64748B", "P1_flatten": "#059669",
               "P2_sharpen": "#F59E0B", "P2_emp": "#B45309",
               "P3_burst": "#DC2626", "P_mix": "#7C3AED"}

# palette
C_SUPPLY = "#2563EB"      # clean supply available
C_SUPPLY_FILL = "#DBEAFE"
C_NOAI = "#6B7280"        # no-AI demand
C_MIX = "#7C3AED"         # AI (Mix) demand
C_ADD = "#D97706"         # AI raises residual gap
C_REMOVE = "#0F766E"      # AI lowers residual gap


def _save(fig, fname):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"{fname}{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG / (fname + '.png')}")


def _diurnal(dn, col):
    """Cross-country mean diurnal profile, equal-energy normalised (mean = 1)."""
    a = dn.groupby("local_hour")[col].mean().reindex(range(24)).to_numpy()
    return a / a.mean()


# ============================ MECHANISM FIGURE ============================
def panel_decomp(ax, dn, fs=11.0):
    """(a) Anatomy of the AI-reshaped demand. The 2050 Mix curve is the sum of a
    flat training+batch baseload, the residual human-traffic shape, and an evening
    inference peak. The AI pieces pile into the night and the post-sunset window,
    never onto the midday solar peak — which is why reshaping cannot fill the gap."""
    h = np.arange(24)
    d = _diurnal(dn, "demand")
    li, lt = base.AI_SHARE_INFERENCE[2050], base.AI_SHARE_TRAINING[2050]
    ab = base.PMIX_ALPHA_BATCH
    ash = base.PMIX_ALPHA_INTERACTIVE + base.PMIX_ALPHA_BURST
    work = np.exp(-((h - base.P2_PEAK_HOUR) ** 2) / (2 * base.P2_SIGMA_H ** 2))
    work = work / work.mean()
    flat = (lt + li * ab) * np.ones(24)            # training + batch baseload (flat)
    human = (1 - li - lt) * d                       # residual human traffic
    infer = li * ash * work                         # evening inference peak
    c_flat, c_human, c_infer = "#94A3B8", "#C7B9E6", "#6D28D9"
    ax.axvspan(18, 23, color="#64748B", alpha=0.08, zorder=0)
    ax.fill_between(h, 0, flat, color=c_flat, lw=0, zorder=2,
                    label="Training + batch (flat)")
    ax.fill_between(h, flat, flat + human, color=c_human, lw=0, zorder=2,
                    label="Human traffic")
    ax.fill_between(h, flat + human, flat + human + infer, color=c_infer, alpha=0.92,
                    lw=0, zorder=2, label="Inference (evening)")
    ax.plot(h, d, color="#1F2937", lw=1.4, ls=":", zorder=5, label="No AI")
    # midday solar peak — abundant supply; the AI pieces add nothing here
    ax.axvline(12, color="#EA8A0B", lw=1.5, ls=(0, (4, 2)), zorder=4)
    ax.text(11.6, 1.99, "midday\nsolar peak", color="#B45309", fontsize=fs - 2.2,
            ha="right", va="top", fontweight="bold")
    # the inference peak (deep purple) clearly sits to the right of the solar line
    ax.annotate("evening\nAI peak", xy=(18, flat[18] + human[18] + infer[18] - 0.06),
                xytext=(20.6, 1.62), fontsize=fs - 2.2, color="#4C1D95", ha="center",
                va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#6D28D9", lw=0.9))
    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18]); ax.set_xticklabels(["0", "6", "12", "18"])
    ax.set_ylim(0, 2.05); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Local hour", fontsize=fs)
    ax.set_ylabel("Data-centre demand\n(equal-energy, mean = 1)", fontsize=fs)
    R3L.R3P.yonly(ax)
    ax.legend(title="AI demand (Mix), 2050", fontsize=fs - 2.2, title_fontsize=fs - 2.0,
              loc="upper left", ncol=1, labelspacing=0.3, handlelength=1.2,
              borderaxespad=0.2).get_title().set_color("#475569")


def panel_zerosum(ax, dn, df, fs=11.0):
    """(b) NEW — AI reshaping is near zero-sum against the clean-supply trough.
    Mix and no-AI demand over the national VRE supply; shaded where AI ADDS vs
    REMOVES residual gap. The headline net change is the real model value at
    national reach (no routing)."""
    h = np.arange(24)
    d = _diurnal(dn, "demand")
    s = _diurnal(dn, "supply_vre")
    years = [2025, 2030, 2050]
    alphas = [0.32, 0.62, 1.0]
    mixes = {y: base.operator_mix(d, h, base.AI_SHARE_INFERENCE[y],
                                  base.AI_SHARE_TRAINING[y]) for y in years}
    mix30 = mixes[2030]

    # clean-supply envelope (year-independent under equal-energy normalisation)
    ax.fill_between(h, 0, s, color=C_SUPPLY_FILL, zorder=0)
    # residual-gap change for the 2030 reference = (Mix vs no-AI) above supply
    add = mix30 > d
    ax.fill_between(h, np.maximum(d, s), mix30, where=add & (mix30 > s),
                    color=C_ADD, alpha=0.55, lw=0, zorder=2)
    ax.fill_between(h, np.maximum(mix30, s), d, where=(~add) & (d > s),
                    color=C_REMOVE, alpha=0.55, lw=0, zorder=2)
    # curves: supply + no-AI + the Mix family across years (matching panel a)
    h_sup, = ax.plot(h, s, color=C_SUPPLY, lw=1.9, zorder=4)
    h_noai, = ax.plot(h, d, color=C_NOAI, lw=1.3, ls=":", zorder=5)
    h_yr = []
    for y, al in zip(years, alphas):
        ln, = ax.plot(h, mixes[y], color=C_MIX, alpha=al,
                      lw=2.1 if y == 2030 else 1.3, zorder=6, label=str(y))
        h_yr.append(ln)

    # real model net change at national reach (tau = 50 ms, 2030 reference)
    dpp = R3L.unc(df, 2030, "P_mix", 1.0, 50) - R3L.unc(df, 2030, "P0_baseline", 1.0, 50)
    ax.annotate("solar trough:\nno clean supply", xy=(4, s[4]), xytext=(6.6, 0.46),
                fontsize=fs - 2.0, color="#1E40AF", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_SUPPLY, lw=0.8))
    ax.text(0.985, 0.975, f"net residual-gap change at\nnational reach (2030): {dpp:+.1f} pp",
            transform=ax.transAxes, fontsize=fs - 2.2, color=INK, ha="right", va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", lw=0.7))

    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18]); ax.set_xticklabels(["0", "6", "12", "18"])
    ax.set_ylim(0, 2.7); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Local hour", fontsize=fs)
    ax.set_ylabel("Power (equal-energy, mean = 1)", fontsize=fs)
    R3L.R3P.yonly(ax)
    # two compact legends: the Mix year family (like panel a) + the supply/shading key
    leg1 = ax.legend(handles=h_yr, title="AI (Mix) demand, by year", fontsize=fs - 2.2,
                     title_fontsize=fs - 2.2, loc="upper left", ncol=3, columnspacing=0.6,
                     handlelength=1.0, borderaxespad=0.25)
    leg1.get_title().set_color("#475569")
    ax.add_artist(leg1)
    keys = [h_sup, h_noai, mpatches.Patch(facecolor=C_ADD, alpha=0.7),
            mpatches.Patch(facecolor=C_REMOVE, alpha=0.7)]
    ax.legend(keys, ["Clean supply", "No-AI demand", "AI adds gap", "AI removes gap"],
              fontsize=fs - 2.2, loc="upper left", bbox_to_anchor=(0.0, 0.82),
              labelspacing=0.28, handlelength=1.2, borderaxespad=0.25)


def panel_shapes(ax, df, fs=11.0):
    """(c) Dumbbell — each AI demand shape's 104-country mean uncovered share at
    national reach (50 ms) vs global reach (500 ms). The six shapes line up in two
    tight columns (shape barely matters) and every shape drops ~16 pp from national
    to global (the routing lever dominates). phi = 100%."""
    order = ["P0_baseline", "P1_flatten", "P2_emp", "P2_sharpen", "P3_burst", "P_mix"]
    loc = {o: R3L.unc(df, 2030, o, 1.0, 50) for o in order}
    glo = {o: R3L.unc(df, 2030, o, 1.0, 500) for o in order}
    ys = list(range(len(order)))[::-1]                 # first shape at the top
    C_NAT, C_GLO = "#D97706", "#0F766E"
    nat, gl = list(loc.values()), list(glo.values())
    ax.axvspan(min(nat), max(nat), color=C_NAT, alpha=0.10, zorder=0)
    ax.axvspan(min(gl), max(gl), color=C_GLO, alpha=0.10, zorder=0)
    for o, y in zip(order, ys):
        ms = 9.5 if o == "P_mix" else 8.0
        ax.plot([glo[o], loc[o]], [y, y], color="#CBD5E1", lw=2.8, zorder=2,
                solid_capstyle="round")
        ax.plot(glo[o], y, "o", color=C_GLO, ms=ms, mec="white", mew=1.1, zorder=4)
        ax.plot(loc[o], y, "o", color=C_NAT, ms=ms, mec="white", mew=1.1, zorder=4)
    ax.set_yticks(ys)
    labs = ax.set_yticklabels([SHAPE_NAME[o] for o in order], fontsize=fs - 1.5)
    for o, t in zip(order, labs):
        if o == "P_mix":
            t.set_color(C_MIX); t.set_fontweight("bold")
    ytop = ys[0]
    ax.set_ylim(-1.35, ytop + 1.55)
    ax.set_xlim(0, max(nat) * 1.16)
    ax.set_xlabel("Uncovered demand share (%, 2030)", fontsize=fs)
    # column headers
    ax.text(np.mean(gl), ytop + 0.75, "Global reach\n(500 ms)", ha="center", va="bottom",
            fontsize=fs - 1.5, color=C_GLO, fontweight="bold")
    ax.text(np.mean(nat), ytop + 0.75, "National reach\n(50 ms)", ha="center", va="bottom",
            fontsize=fs - 1.5, color=C_NAT, fontweight="bold")
    # shape barely matters — the two clusters are narrow
    ax.text(0.6, 4.25, "6 AI shapes\nwithin ~3 pp", ha="left", va="center",
            fontsize=fs - 2.0, color="#475569", style="italic")
    # routing lever — every dumbbell spans ~16 pp
    yb = -0.7
    ax.annotate("", xy=(np.mean(gl), yb), xytext=(np.mean(nat), yb),
                arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.5))
    ax.text((np.mean(gl) + np.mean(nat)) / 2, yb - 0.12,
            f"routing lever:  −{np.mean(nat) - np.mean(gl):.0f} pp", ha="center",
            va="top", fontsize=fs - 1.2, color="#111827", fontweight="bold")
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6)
    ax.grid(False, axis="y")
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def render_mechanism():
    R3L.R3P.set_style(11.0)
    plt.close("all")
    df, dn = R3L.load()
    fig = plt.figure(figsize=(13.6, 10.2))
    outer = fig.add_gridspec(2, 1, left=0.105, right=0.985, top=0.945, bottom=0.062,
                             height_ratios=[1.0, 1.02], hspace=0.30)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.12)
    ax_a = fig.add_subplot(top[0, 0]); panel_decomp(ax_a, dn)        # a anatomy
    ax_b = fig.add_subplot(top[0, 1]); panel_zerosum(ax_b, dn, df)   # b zero-sum
    ax_c = fig.add_subplot(bot[0, 0]); panel_shapes(ax_c, df)        # c six shapes
    panel_heatmap(fig, bot[0, 1], df, fs=10.0)                       # d reach heatmap
    for cell, lab in [(top[0, 0], "a"), (top[0, 1], "b"), (bot[0, 0], "c"), (bot[0, 1], "d")]:
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.040, bb.y1 + 0.012, lab, fontweight="bold", fontsize=16,
                 ha="left", va="bottom")
    _save(fig, "fig3_ai_mechanism")


# ============================ RESULTS FIGURE ============================
def panel_heatmap(fig, cell, df, op="P_mix", fs=11.0):
    """(a) Routable-fraction x latency-tolerance uncovered-share heatmap (Mix,
    2030). Same axes as the mechanism figure's reach panel, full size."""
    sub = cell.subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.06)
    ax = fig.add_subplot(sub[0, 0])
    allv = [R3L.unc(df, 2030, op, p, t) for p in R3L.PHI_ROWS for t in R3L.TAU]
    vmax = float(np.ceil(np.nanmax(allv) / 5) * 5)
    norm = mpl.colors.Normalize(0, vmax)
    for i, p in enumerate(R3L.PHI_ROWS):
        for j, t in enumerate(R3L.TAU):
            v = R3L.unc(df, 2030, op, p, t); fc = CMAP_PA(norm(v))
            ax.add_patch(mpatches.FancyBboxPatch(
                (j - 0.46, i - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0,rounding_size=0.14", facecolor=fc,
                edgecolor="white", linewidth=0.7, mutation_aspect=1))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs + 2.5,
                    color="white" if lum < 0.55 else "#1F2937", fontweight="bold", zorder=4)
    ax.set_xlim(-0.6, len(R3L.TAU) - 0.4); ax.set_ylim(-0.6, len(R3L.PHI_ROWS) - 0.4)
    ax.set_aspect("equal", adjustable="box")           # square cells
    ax.set_xticks(range(len(R3L.TAU)))
    ax.set_xticklabels([str(t) for t in R3L.TAU], fontsize=fs)
    ax.set_yticks(range(len(R3L.PHI_ROWS)))
    ax.set_yticklabels([R3L.PHI_LAB[p] for p in R3L.PHI_ROWS], fontsize=fs)
    ax.set_ylabel("Routable fraction ↑", fontsize=fs)
    ax.set_xlabel("Latency tolerance (ms) →", fontsize=fs)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.0, len(R3L.PHI_ROWS) - 0.35, "Mix · 2030", fontsize=fs + 0.5,
            color="#475569", fontweight="bold", ha="left", va="bottom")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP_PA); sm.set_array([])
    cax = fig.add_subplot(sub[0, 1])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("Uncovered demand share (%, 2030)", fontsize=fs - 1)
    cb.ax.tick_params(labelsize=fs - 1)


def render_results():
    R3L.R3P.set_style(11.0)
    plt.close("all")
    df, _dn = R3L.load()
    years = [2025, 2030, 2050]
    fig = plt.figure(figsize=(13.2, 12.8))
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.955, top=0.935, bottom=0.052,
                             height_ratios=[1.0, 1.16], hspace=0.21)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
    R3L.draw_c(fig.add_subplot(top[0, 0]), df)   # a cost vs latency
    R3L.draw_b(fig, top[0, 1], df)               # b rose
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.12)
    R3L.draw_d(fig, bot[0, 0], df)               # c ridgeline (local/global facets)
    em = bot[0, 1].subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.09], hspace=0.14)
    cmap = CMAP_PA; cmap.set_bad(alpha=0)
    sub = df[(df.perturbation == "P_mix") & (df.phi == 1.0) & (df.tau_ms == 500)].copy()
    sub["pi"] = (sub["lcoe_elec"] - sub["lcoe_gen"]) / sub["lcoe_gen"] * 100.0
    vby = {y: dict(zip(sub[sub.year == y]["country"], sub[sub.year == y]["pi"]))
           for y in years}
    allv = np.array([v for y in years for v in vby[y].values()])
    vmin = float(np.floor(np.percentile(allv, 4) / 10) * 10)
    vmax = float(np.ceil(np.percentile(allv, 96) / 10) * 10)
    norm = mpl.colors.Normalize(vmin, vmax)
    for j, y in enumerate(years):
        axm = fig.add_subplot(em[j, 0], projection=ccrs.Robinson())
        R3L.R3P._country_choropleth(axm, vby[y], cmap, norm)
        axm.text(0.005, 0.90, str(y), transform=axm.transAxes, fontsize=12.5,
                 fontweight="bold", color=INK)
        if j == 0:
            axm.legend(handles=[mpatches.Patch(facecolor="#C7CCD1", edgecolor="none",
                                               label="No data")],
                       loc="lower left", fontsize=11.0, frameon=False,
                       handlelength=1.1, handleheight=1.1, borderaxespad=0.0)
    cax = fig.add_subplot(em[3, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Per-country phase-firming premium Π (% of generation · global reach)",
                 fontsize=11.0)
    cb.ax.tick_params(labelsize=11.0)
    for cell, lab in [(top[0, 0], "a"), (top[0, 1], "b"), (bot[0, 0], "c"),
                      (em[0, 0], "d")]:
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.030, bb.y1 + 0.005, lab, fontweight="bold", fontsize=16,
                 ha="left", va="bottom")
    _save(fig, "fig4_routing_cost")


def main():
    render_mechanism()
    render_results()


if __name__ == "__main__":
    main()
