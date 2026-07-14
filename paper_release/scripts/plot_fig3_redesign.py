#!/usr/bin/env python3
"""Redesigned Figure 3 + appendix figures.

Fig. 3 (main figure, realistic blend Mix only):
  (a) day-shape curve over the L x W uncovered-share heatmap (value printed in
      every cell) plus a vertical "vs no-AI" lollipop;
  (b) zero-baseline grouped bars: the 2030 global firming-storage bill across
      all 16 dispatch scenarios (L0-L3 groups x W0-W3 bars, shared y-limit);
  (c) dispatch-lever waterfall: status-quo bill -> power dispatch -> workload
      migration -> irreducible floor (L3 x W3);
  (d) duration-aware bill trajectory 2025-2050 with storage-technology and
      AI-shape uncertainty bands;
  (e) three frykit world maps of the per-country firming bill (2025/2030/2050).

Appendix (two single-page figures over the five non-Mix operators):
  figS_ai_operators_a — day-shape + heatmap + lollipop composites (2-col grid,
      shared colorbar in the spare slot);
  figS_ai_operators_b — grouped-bar bill panels (2-col grid, migration legend
      in the spare slot).

academic-paper-figures skill: English-only, restrained colorblind-safe palette,
matplotlib + frykit. Data: data_globalsites/ (104 countries, full 16-scenario R3).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import frykit.plot as fplt
import frykit.shp as fshp
from shapely.geometry import Point
from shapely.strtree import STRtree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_geographic_portfolio_ai as base  # noqa: E402
import plot_fig3b_polar_rose as rose  # noqa: E402  (full-system cost rose for panel b)

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
warnings.filterwarnings("ignore", message=".*facecolor will have no effect.*")

ROOT = base.ROOT
REPORT = ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai"
DATA = REPORT / "data_globalsites"
FIG = REPORT / "figures_globalsites"
COUNTRY_META = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites" / "country_meta.csv"
FIG.mkdir(parents=True, exist_ok=True)

OP_COLOR = {
    "P0_baseline": "#64748B", "P1_flatten": "#059669", "P2_sharpen": "#F59E0B",
    "P2_emp": "#B45309", "P3_burst": "#DC2626", "P_mix": "#7C3AED",
}
OP_ORDER = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
# panel-title hue: operator colour, with a dark legible variant for the amber one
TITLE_COLOR = {**OP_COLOR, "P2_sharpen": "#B45309"}
INK = "#111827"
GRID = "#E5E7EB"


def set_style(fs=11.0):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.titlesize": fs + 1.5, "axes.titleweight": "bold", "axes.labelsize": fs,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#374151", "axes.labelcolor": INK,
        "xtick.color": "#374151", "ytick.color": "#374151",
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.6,
    })


def yonly(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.grid(False, axis="x")


def load():
    g = pd.read_csv(DATA / "r3_global_aggregate.csv")
    g["bill_mid"] = g["global_storage_cost_musd_yr_mid"] / 1000.0   # B USD/yr
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    pc = pd.read_csv(DATA / "r3_perturbation_cost.csv")
    return g, dn, pc


def bill(g, year, op, sc):
    r = g[(g.year == year) & (g.perturbation == op) & (g.scenario == sc)]
    return float(r["bill_mid"].iloc[0]) if len(r) else np.nan


def share(g, year, op, sc):
    r = g[(g.year == year) & (g.perturbation == op) & (g.scenario == sc)]
    return float(r["mean_uncovered_share"].iloc[0]) * 100 if len(r) else np.nan


# ===================== panel (a): curves + heatmaps + lollipops =====================
WORSE = "#C2410C"   # AI raises mismatch vs the no-AI curve
BETTER = "#0F766E"  # AI lowers mismatch vs the no-AI curve


SCEN_TITLE = {
    "P0_baseline": "Today",
    "P1_flatten": "Batch & training",
    "P2_sharpen": "Consumer chat",
    "P2_emp": "Measured chat",
    "P3_burst": "Viral spikes",
    "P_mix": "Mix",
}


def draw_ab(fig, gs_cell, dn, g, ops=None, ncols=3, fs=11.0, cell_hspace=0.28):
    """AI-workload operators laid out in an `ncols`-wide grid. Each cell is a small
    composite: a slim day-shape curve on top (width = heatmap width), a large L x W
    uncovered-share heatmap with the value printed in every cell (L0/W0 at the
    BOTTOM-LEFT, matching Fig. 2), and — running vertically beside it — a lollipop
    of the per-power-scope change vs the no-AI (today's) curve. The shared colorbar
    goes in the spare grid slot when one exists, else in a strip below."""
    ops = ops or OP_ORDER
    nrows = (len(ops) + ncols - 1) // ncols
    spare = nrows * ncols - len(ops)
    # ---- curve setup ----
    h = np.arange(24)
    avg = dn.groupby("local_hour")["demand"].mean().reindex(range(24)).to_numpy()
    d = avg / avg.mean()
    years = [2025, 2030, 2050]
    alphas = [0.32, 0.60, 1.0]

    def shp(op, y):
        li, lt = base.AI_SHARE_INFERENCE[y], base.AI_SHARE_TRAINING[y]
        if op == "P0_baseline":
            return d
        if op == "P1_flatten":
            return base.operator_flatten(d, li, lt)
        if op == "P2_sharpen":
            return base.operator_sharpen(d, h, li, lt)
        if op == "P2_emp":
            return base.operator_sharpen_emp(d, h, li, lt)
        if op == "P3_burst":
            return base.operator_burst(d, li, lt)
        return base.operator_mix(d, h, li, lt)

    # ---- heatmap + lollipop setup (colour & diff scales over ALL six operators
    #      so the main figure and the appendix stay directly comparable) ----
    yr = 2030
    Ls, Ws = ["L0", "L1", "L2", "L3"], ["W0", "W1", "W2", "W3"]
    allv = [share(g, yr, o, f"{L}_{W}") for o in OP_ORDER for L in Ls for W in Ws]
    vmax = float(np.ceil(max(allv) / 5.0) * 5.0)
    norm = mpl.colors.Normalize(0, vmax)
    # two-pole ramp (lilac -> purple -> amber): purple stays cohesive with panel b's
    # rose, the amber high-end pole adds contrast so the heatmap is not mono-purple
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "fig_pa", ["#F2EFF7", "#B6A8D0", "#8A6FB0", "#C75D88", "#E8A33D"])
    base_row = {L: np.mean([share(g, yr, "P0_baseline", f"{L}_{W}") for W in Ws])
                for L in Ls}
    diff = {o: {L: np.mean([share(g, yr, o, f"{L}_{W}") for W in Ws]) - base_row[L]
                for L in Ls} for o in OP_ORDER}
    dlim = np.ceil(max(abs(v) for o in OP_ORDER for v in diff[o].values()))

    if spare:
        outer = gs_cell.subgridspec(nrows, ncols, hspace=0.40, wspace=0.26)
    else:
        outer = gs_cell.subgridspec(nrows + 1, ncols,
                                    height_ratios=[1.0] * nrows + [0.05],
                                    hspace=0.28, wspace=0.26)
    leg_shown = False

    for k, op in enumerate(ops):
        r, c = k // ncols, k % ncols
        bottom_row = (k + ncols >= len(ops))   # bottom-most occupied cell per column
        col = OP_COLOR[op]
        cell = outer[r, c].subgridspec(2, 2, height_ratios=[0.46, 1.0],
                                       width_ratios=[1.0, 0.40],
                                       hspace=cell_hspace, wspace=0.06)
        ax = fig.add_subplot(cell[0, 0])      # slim day-shape curve
        axh = fig.add_subplot(cell[1, 0])     # large L x W heatmap (numbers inside)
        axl = fig.add_subplot(cell[1, 1])     # vertical lollipop (diff vs no-AI)

        # --- day-shape curve (slim, width matches the heatmap) ---
        ax.axvspan(18, 23, color="#64748B", alpha=0.12, zorder=0)
        if op == "P0_baseline":
            ax.plot(h, d, color=col, lw=1.6, zorder=3)
            pk, hp = float(d.max()), int(np.argmax(d))
        else:
            ax.plot(h, d, color="#9CA3AF", lw=0.8, ls=":", alpha=0.9, zorder=2)
            curves = [shp(op, y) for y in years]
            for cv, y, al in zip(curves, years, alphas):
                ax.plot(h, cv, color=col, alpha=al,
                        lw=1.5 if y == 2030 else 1.0, zorder=3, label=str(y))
            imax = int(np.argmax([cv.max() for cv in curves]))
            pk, hp = float(curves[imax].max()), int(np.argmax(curves[imax]))
        if pk > 3.2:   # shared ylim clips the spike — print the true peak value
            ax.plot(hp, 3.2, marker="^", color=col, ms=7, markeredgecolor="white",
                    markeredgewidth=0.6, clip_on=False, zorder=6)
            ax.text(hp - 1.2, 3.05, f"peak {pk:.1f}", ha="right", va="top",
                    fontsize=fs, color="#475569")
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18])
        if bottom_row:
            ax.set_xticklabels(["0", "6", "12", "18"])
            ax.set_xlabel("Local hour", fontsize=fs, labelpad=1.5)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 3.2)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_title(SCEN_TITLE[op], fontsize=fs + 1.5, loc="left",
                     color=TITLE_COLOR[op], fontweight="bold", pad=2)
        ax.tick_params(labelsize=fs, length=2)
        if c == 0:
            ax.set_ylabel("Demand\n(mean = 1)", fontsize=fs)
        yonly(ax)
        if not leg_shown and op != "P0_baseline":
            leg = ax.legend(title="by year", fontsize=fs, title_fontsize=fs,
                            loc="upper left", ncol=3, columnspacing=0.8,
                            handlelength=0.9, borderaxespad=0.12)
            leg.get_title().set_color("#475569")
            leg_shown = True

        # --- L x W uncovered-share heatmap; L0/W0 at the BOTTOM-LEFT (Fig. 2) ---
        for i, L in enumerate(Ls):
            for j, W in enumerate(Ws):
                v = float(share(g, yr, op, f"{L}_{W}"))
                fc = cmap(norm(v))
                box = mpatches.FancyBboxPatch(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    boxstyle="round,pad=0,rounding_size=0.14",
                    facecolor=fc, edgecolor="white", linewidth=0.6,
                    mutation_aspect=1)
                axh.add_patch(box)
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                tc = "white" if lum < 0.55 else "#1F2937"
                axh.text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=fs + 5.5, color=tc, fontweight="bold", zorder=4)
        axh.set_xlim(-0.6, 3.6)
        axh.set_ylim(-0.6, 3.6)             # L0 (i=0) at the bottom, L3 at the top
        axh.set_xticks(range(4))
        axh.set_xticklabels(Ws, fontsize=fs)
        if c == 0:
            axh.set_yticks(range(4))
            axh.set_yticklabels(Ls, fontsize=fs)
            axh.set_ylabel("Power scope ↑", fontsize=fs)
        else:
            axh.set_yticks([])
        if bottom_row:
            axh.set_xlabel("Workload migration →", fontsize=fs)
        axh.tick_params(length=0)
        for sp in axh.spines.values():
            sp.set_visible(False)

        # --- vertical lollipop: uncovered-share change vs the no-AI curve ---
        axl.set_ylim(-0.6, 3.6)            # share the (flipped) heatmap row coords
        axl.set_xlim(-dlim, dlim)
        axl.axvline(0, color="#9CA3AF", lw=0.7, zorder=1)
        for i, L in enumerate(Ls):
            dv = diff[op][L]
            mc = WORSE if dv > 0.05 else (BETTER if dv < -0.05 else "#9CA3AF")
            axl.plot([0, dv], [i, i], color="#CBD5E1", lw=1.3, zorder=2)
            axl.plot(dv, i, "o", color=mc, ms=4.6, markeredgecolor="white",
                     markeredgewidth=0.5, zorder=3)
        axl.set_yticks([])
        for sp in ("top", "right", "left"):
            axl.spines[sp].set_visible(False)
        axl.spines["bottom"].set_color("#9CA3AF")
        if bottom_row:
            axl.set_xticks([-int(dlim), 0, int(dlim)])
            axl.tick_params(axis="x", labelsize=fs, length=2, colors="#6B7280")
            axl.set_xlabel("Δ uncovered share\nvs no-AI (pp)", fontsize=fs,
                           labelpad=1.5)
        else:
            axl.set_xticks([])
        axl.grid(False)

    # shared colorbar — in the spare grid slot when one exists, else in a strip
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    if spare:
        host = fig.add_subplot(outer[nrows - 1, ncols - 1]); host.axis("off")
        cax = host.inset_axes([0.10, 0.55, 0.80, 0.05])
    else:
        cax_host = fig.add_subplot(outer[nrows, :]); cax_host.axis("off")
        cax = cax_host.inset_axes([0.30, 0.0, 0.40, 1.0])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Uncovered demand share (%, 2030)", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)


# ============================ panel (c) ============================
def draw_c(ax, pc):
    op, yr, sc = "P_mix", 2030, "L1_W2"
    sub = pc[(pc.year == yr) & (pc.perturbation == op) & (pc.scenario == sc)]
    cmeta = pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""])
    cen = {row.iso2: (float(row.lon), float(row.lat)) for row in cmeta.itertuples()}
    val_by_iso = dict(zip(sub["country"], sub["uncovered_share"] * 100))
    vv = np.array(list(val_by_iso.values()))
    vmin = max(0.0, np.floor(np.percentile(vv, 5) / 2) * 2)
    vmax = np.ceil(np.percentile(vv, 95) / 2) * 2

    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    ax.set_frame_on(False)
    ax.set_aspect("auto")
    geoms = fshp.get_countries()
    tree = STRtree(geoms)
    arr = np.full(len(geoms), np.nan)
    for cc, v in val_by_iso.items():
        if cc not in cen:
            continue
        p = Point(cen[cc][0], cen[cc][1])
        placed = False
        for gi in np.atleast_1d(tree.query(p)):
            gi = int(gi)
            if geoms[gi].covers(p):
                arr[gi] = v if np.isnan(arr[gi]) else max(arr[gi], v)
                placed = True
                break
        if not placed:
            gi = int(tree.nearest(p))
            if np.isnan(arr[gi]):
                arr[gi] = v
    fplt.add_geometries(ax, geoms, fc="#C7CCD1", ec="none",
                        crs=fplt.PLATE_CARREE, zorder=1)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad(alpha=0)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fplt.add_geometries(ax, geoms, array=np.ma.masked_invalid(arr),
                        cmap=cmap, norm=norm, crs=fplt.PLATE_CARREE,
                        ec="#FFFFFF", lw=0.18, zorder=2)
    top5 = sub.nlargest(5, "uncovered_share")
    for _, row in top5.iterrows():
        cc = row["country"]
        if cc in cen:
            lon, lat = cen[cc]
            ax.scatter(lon, lat, s=52, marker="D", facecolor="#7C3AED",
                       edgecolor="white", linewidth=0.8, zorder=10,
                       transform=ccrs.PlateCarree())
            ax.text(lon, lat + 6, f"{cc} {row['uncovered_share']*100:.0f}%",
                    fontsize=6.6, color="#4C1D95", ha="center", fontweight="bold",
                    transform=ccrs.PlateCarree(), zorder=11)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = ax.inset_axes([0.32, -0.02, 0.40, 0.028])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Hourly mismatch, realistic case (% of year)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.8)


# ============================ panel (b): per-operator 16-scenario bill bars ============================
W_ORDER = ["W0", "W1", "W2", "W3"]
L_ORDER = ["L0", "L1", "L2", "L3"]
# workload-migration bars: monotone-lightness purples (greyscale-safe)
W_COLOR = {W: plt.cm.Purples(t) for W, t in zip(W_ORDER, (0.35, 0.55, 0.75, 0.95))}
W_LABEL = {"W0": "W0 (none)", "W1": "W1 (~10%)", "W2": "W2 (~30%)", "W3": "W3 (~60%)"}


def draw_d(fig, gs_cell, g, ops=None, ncols=3, fs=11.0):
    """Panel (b): zero-baseline grouped bars of the 2030 global firming-storage
    bill — four power-scope groups (L0-L3), four workload-migration bars (W0-W3)
    inside each. The y-limit is shared across ALL six operators so the main
    figure and the appendix panels compare directly."""
    ops = ops or OP_ORDER
    nrows = (len(ops) + ncols - 1) // ncols
    spare = nrows * ncols - len(ops)
    yr = 2030
    ymax = float(np.ceil(max(bill(g, yr, o, f"{L}_{W}") for o in OP_ORDER
                             for L in L_ORDER for W in W_ORDER) / 10.0) * 10.0)
    sub = gs_cell.subgridspec(nrows, ncols, hspace=0.42, wspace=0.22)
    single = (len(ops) == 1)
    bw = 0.19
    for k, op in enumerate(ops):
        r, c = k // ncols, k % ncols
        ax = fig.add_subplot(sub[r, c])
        for li, L in enumerate(L_ORDER):
            for wi, W in enumerate(W_ORDER):
                ax.bar(li + (wi - 1.5) * bw, bill(g, yr, op, f"{L}_{W}"),
                       width=bw * 0.92, color=W_COLOR[W], edgecolor="white",
                       linewidth=0.4, zorder=3)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(L_ORDER)
        ax.tick_params(labelsize=fs, length=2.5)
        if k + ncols >= len(ops):
            ax.set_xlabel("Power scope", fontsize=fs)
        if c == 0:
            ax.set_ylabel("Global firming-storage bill\n(B USD/yr, 2030)",
                          fontsize=fs)
        yonly(ax)
        if single:
            ax.text(0.975, 0.985, "Mix · 2030", transform=ax.transAxes,
                    ha="right", va="top", fontsize=fs + 1.0, color="#475569",
                    fontweight="bold")
        else:
            ax.set_title(SCEN_TITLE[op], fontsize=fs + 1.5, loc="left",
                         color=TITLE_COLOR[op], fontweight="bold", pad=3)

    handles = [mpatches.Patch(facecolor=W_COLOR[W], edgecolor="white",
                              label=W_LABEL[W]) for W in W_ORDER]
    if single:
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93),
                  fontsize=fs, title="Workload migration", title_fontsize=fs,
                  frameon=False, labelspacing=0.45, handlelength=1.4)
    elif spare:   # compact key in the spare grid slot
        ax_leg = fig.add_subplot(sub[nrows - 1, ncols - 1]); ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="center", fontsize=fs + 0.5,
                      title="Workload migration", title_fontsize=fs + 0.5,
                      frameon=False, labelspacing=0.7, handlelength=1.7,
                      handleheight=1.3)
    else:
        ax.legend(handles=handles, loc="upper right", fontsize=fs,
                  title="Workload migration", title_fontsize=fs, frameon=False,
                  labelspacing=0.45, handlelength=1.4)


# ==================== panels (c, d): full-system cost (matches panel b) ====================
# Components and colours match panel b's rose (inner->outer: gen, tx, store, anc).
FS_E_D = 876_000.0                                   # MWh/yr delivered per 100 MW-mean unit
FS_FLEET_GW = {2025: 82.3, 2030: 219.0, 2035: 285.0, 2040: 415.0, 2050: 700.0}
FS_COMPONENTS = [
    ("lcoe_gen",   "Generation",      "#6E7FA8"),
    ("lcoe_tx",    "Transmission",    "#9085B8"),
    ("lcoe_store", "Firming storage", "#B6A8D0"),
    ("lcoe_anc",   "Ancillary",       "#D8D1E6"),
]
# strategy lines (panel d): teal = the efficient "move compute" option, purple = the
# transmission-heavy "expand grid" option, grey dashed = status-quo reference.
STRAT = [("L3", "W0", "Expand grid (global pool)",       "#7C3AED", "-"),
         ("L0", "W3", "Move compute (local + migration)", "#0F766E", "-"),
         ("L0", "W0", "Status quo (local, no shift)",     "#9CA3AF", (0, (5, 2)))]


def _fs_units(year):
    xs = sorted(FS_FLEET_GW)
    gw = float(np.interp(year, xs, [FS_FLEET_GW[x] for x in xs]))
    return gw * 1000.0 / 100.0


def _fs_table():
    """Per-(year, scope, workload) mean full-system LCOE components ($/MWh)."""
    df = pd.read_csv(DATA / "fullsystem_cost_table.csv")
    cols = [c for c, _, _ in FS_COMPONENTS] + ["lcoe_elec"]
    return df.groupby(["year", "power_scope", "workload_scenario"])[cols].mean()


def _fs_bill(tbl, year, scope, workload, comp=None):
    """Global full-system bill (B USD/yr); comp=None gives the electricity total."""
    lcoe = tbl.loc[(year, scope, workload)]
    val = lcoe["lcoe_elec"] if comp is None else lcoe[comp]
    return float(val) * FS_E_D * _fs_units(year) / 1e9


FS_BOX_SCENARIO = ("L0", "W3", "Move compute\n(local + migration)")   # strategy shown in d


# Transmission cost to data centres over time = the "ride-the-grid" credit (option B).
# The cross-regional HVDC grid is built out for the WHOLE decarbonisation, so 24/7-CFE
# pooling increasingly rides shared, already-built infrastructure instead of self-building
# dedicated links. We model this as a declining ATTRIBUTABLE SHARE of the (shared-backbone,
# literature-cost) transmission: ~1.0 through 2025-2030 (no intercontinental grid yet, must
# self-build) falling to ~0.3 by 2050 (grid largely built; data centres pay a marginal/wheeling
# share). Grounded in Guo et al. 2022 (Nat. Energy, doi:10.1038/s41560-022-01136-0), who find
# UHVDC investment is OFFSET by avoided generation/storage at the system level — so charging one
# user full greenfield capex overstates its cost. Anchored at 2030 = 1.0 for consistency with
# panel c. The 2050 share is an explicit ASSUMPTION: central 0.30; sensitivity ~0.50 (conservative)
# to ~0.10 (transmission nearly fully offset). HVDC unit cost itself barely falls (line-dominated
# commodity; see hvdc-cost-learning-grounding workflow) and is a separate, minor effect, omitted here.
TX_CREDIT = {2025: 1.00, 2030: 1.00, 2035: 0.84, 2040: 0.66, 2045: 0.47, 2050: 0.30}


def _tx_credit(year):
    xs = sorted(TX_CREDIT)
    return float(np.interp(year, xs, [TX_CREDIT[x] for x in xs]))


def _fs_country_tot(df, scope, year, fac):
    """Per-country full-system electricity cost (USD/MWh) for one power scope and
    year: generation + firming + ancillary + transmission at the credible central
    (per-scope `fac`) with the ride-the-grid credit. Workload W0 (no migration)."""
    sub = df[(df.workload_scenario == "W0") & (df.power_scope == scope)
             & (df.year == year)]
    tot = (sub.lcoe_gen + sub.lcoe_store + sub.lcoe_anc
           + sub.lcoe_tx * fac[scope] * _tx_credit(year))
    return dict(zip(sub.country, tot))


RIDGE_XGRID = np.linspace(50, 350, 260)


def _ridge_kde(a):
    a = a[np.isfinite(a)]
    if len(a) < 5 or np.std(a) < 1e-6:
        return np.zeros_like(RIDGE_XGRID)
    return gaussian_kde(a, bw_method=0.45)(RIDGE_XGRID)


def draw_future(ax, g):
    """Panel (d) — RIDGELINE of the 104-country full-system electricity-cost distribution
    for each power-dispatch scope, stacked by year 2025-2050. In 2025 the Global pool sits
    far to the right (expensive) while the other scopes cluster; year by year the Global
    distribution marches left and the four scopes CONVERGE, as the cross-regional grid is
    built out for the whole decarbonisation and data centres ride it — the transmission
    attributable share falls ~1.0→0.3 by 2050 (an explicit SCENARIO grounded in Guo et al.
    2022's system-level transmission offset and shallow-connection socialisation). Each ridge:
    filled density + outline per scope, white tick = cross-country median.
    Data: fullsystem_cost_table.csv + r3_tx_bounds.csv."""
    scopes = ["L0", "L1", "L2", "L3"]
    names = {"L0": "National", "L1": "≤1500 km", "L2": "≤3000 km", "L3": "Global"}
    col = {"L0": "#9AA7B8", "L1": "#8C84C4", "L2": "#7A4FA0", "L3": "#46235F"}
    fb = pd.read_csv(DATA / "r3_tx_bounds.csv")

    def _tx(topo, price, sc):
        r = fb[(fb.topology == topo) & (fb.price == price) & (fb.scope == sc)]
        return float(r["lcoe_tx"].iloc[0]) if len(r) else 0.0
    fac = {sc: (_tx("shared", "lit", sc) / _tx("dedicated", "ours", sc))
           if _tx("dedicated", "ours", sc) > 1e-9 else 0.0 for sc in scopes}

    df = pd.read_csv(DATA / "fullsystem_cost_table.csv")
    sub = df[df.workload_scenario == "W0"].copy()
    sub["txc"] = sub["power_scope"].map(fac) * sub["lcoe_tx"] * sub["year"].map(_tx_credit)
    sub["tot"] = sub["lcoe_gen"] + sub["lcoe_store"] + sub["lcoe_anc"] + sub["txc"]

    def arr(sc, yr):
        return sub[(sub.power_scope == sc) & (sub.year == yr)]["tot"].to_numpy()

    years = [2025, 2030, 2035, 2040, 2045, 2050]
    n = len(years); gap = 1.0; scale = 1.48
    dmax = max(_ridge_kde(arr(s, y)).max() for s in scopes for y in years)
    for row, y in enumerate(years):                       # row 0 = 2025 at top
        base = (n - 1 - row) * gap
        z = (row + 1) * 10
        for sc in ["L3", "L2", "L1", "L0"]:               # darkest first, lighter on top
            a = arr(sc, y); d = _ridge_kde(a) / dmax * scale
            ax.fill_between(RIDGE_XGRID, base, base + d, color=col[sc], alpha=0.50,
                            lw=0, zorder=z + scopes.index(sc))
            ax.plot(RIDGE_XGRID, base + d, color=col[sc], lw=1.1,
                    zorder=z + scopes.index(sc) + 4)
            med = float(np.median(a)); hpk = d[np.argmin(np.abs(RIDGE_XGRID - med))]
            ax.plot([med, med], [base, base + hpk], color="white", lw=1.0,
                    zorder=z + scopes.index(sc) + 5)
        ax.text(348, base + 0.06, str(y), ha="right", va="bottom", fontsize=9.5,
                color=INK, fontweight="bold", zorder=z + 30)

    g25 = np.median(arr("L3", 2025)); g50 = np.median(arr("L3", 2050)); n50 = np.median(arr("L0", 2050))
    ax.annotate(f"Global {g25:.0f}", xy=(g25, (n - 1) * gap + scale * 0.5), xytext=(6, 6),
                textcoords="offset points", fontsize=8.2, color=col["L3"], fontweight="bold")
    ax.annotate(f"converge ~{n50:.0f}–{g50:.0f}", xy=(g50, 0.55), xytext=(10, 16),
                textcoords="offset points", fontsize=8.2, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.8))

    ax.set_yticks([]); ax.set_xlim(50, 352)
    ax.set_ylim(-0.15, (n - 1) * gap + scale + 0.25)
    ax.set_xlabel("Full-system electricity cost (USD/MWh, per country)", fontsize=11.0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6)
    handles = [mpatches.Patch(facecolor=col[s], edgecolor="white", alpha=0.7, label=names[s])
               for s in scopes]
    ax.legend(handles=handles, title="Power-dispatch scope", fontsize=8.4, title_fontsize=8.6,
              frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
              handlelength=1.3, labelspacing=0.3)


# ============================ panel (c): present mechanism ============================
def draw_mechanism(ax, g):
    """Panel (c) — full-system cost of widening the power-dispatch scope. Firming storage
    (green line) falls as the pool grows, but the inter-regional transmission it needs is
    uncertain. The CREDIBLE inner band uses literature HVDC unit costs (350 USD/MW·km,
    100 USD/kW·terminal) at a 3–5% discount over either a shared region-backbone or
    dedicated links, following the inter-regional UHVDC-corridor accounting of Guo et al.
    (Nat. Energy 2022) and the meshed Global Grid of Chatzivasileiadis et al. (2013); the
    central line is the shared backbone at a 5% market discount. The lighter outer zone is
    the paper's PRIOR costing (700/200·kW, 7%, dedicated links) — a pessimistic ceiling.
    The global pool stays above national supply throughout — the reversal is robust:
    ~2x at the central estimate (1.8–2.3x credible), up to ~4.3x under the prior costing.
    Data: data_globalsites/r3_tx_bounds.csv."""
    b = pd.read_csv(DATA / "r3_tx_bounds.csv")
    scopes = ["L0", "L1", "L2", "L3"]
    x = np.arange(len(scopes))

    def tot(topo, price):
        s = b[(b.topology == topo) & (b.price == price)].set_index("scope")["lcoe_total"]
        return s.reindex(scopes).to_numpy()

    # credible literature-grounded range = unit costs 350/100 USD, 3-5% discount,
    # either dedicated or shared topology (the inner band). The central line is the
    # shared backbone at a 5% market discount. The paper's prior costing (700/200,
    # 7%, dedicated links) sits well above and is shown as a lighter "prior" zone.
    lit = np.array([tot("shared", "lit3"), tot("shared", "lit"),
                    tot("dedicated", "lit3"), tot("dedicated", "lit")])
    inner_lo, inner_hi = lit.min(axis=0), lit.max(axis=0)
    central = tot("shared", "lit")
    prior_hi = tot("dedicated", "ours")
    store = (b[(b.topology == "shared") & (b.price == "lit")]
             .set_index("scope")["lcoe_store"].reindex(scopes).to_numpy())
    base0 = central[0]
    labels = ["National\n(0 km)", "≤1500 km", "≤3000 km", "Global"]
    C_TOTAL, C_INNER, C_OUTER, C_STORE = "#1F2937", "#B6A8D0", "#E7E2F0", "#6FA08C"

    ax.fill_between(x, inner_hi, prior_hi, color=C_OUTER, lw=0, zorder=1,
                    label="Prior dedicated-link costing (700/200·kW, 7%)")
    ax.fill_between(x, inner_lo, inner_hi, color=C_INNER, alpha=0.60, lw=0, zorder=2,
                    label="Credible range (shared↔dedicated · 3–5%)")
    ax.axhline(base0, color="#9CA3AF", lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.text(0.04, base0 + 9, f"national baseline = {base0:.0f}", fontsize=7.8,
            color="#5B6470", ha="left", va="bottom")
    ax.plot(x, central, color=C_TOTAL, lw=2.6, marker="s", ms=6, mec="white", mew=0.8,
            zorder=6, label="Full-system total (central)")
    ax.plot(x, store, color=C_STORE, lw=1.8, marker="o", ms=4.5, mec="white", mew=0.6,
            zorder=5, label="Firming storage (what pooling saves)")
    for xi, tv in zip(x, central):
        ax.annotate(f"{tv:.0f}", xy=(xi, tv), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.4, color=C_TOTAL, fontweight="bold")
    ax.annotate(f"prior {prior_hi[-1]:.0f}", xy=(x[-1], prior_hi[-1]), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=7.6,
                color="#8C84A8")
    ax.annotate(f"{inner_lo[-1]:.0f}", xy=(x[-1], inner_lo[-1]), xytext=(-4, -1),
                textcoords="offset points", ha="right", va="top", fontsize=7.6, color="#7A6CA6")
    ax.text(0.965, 0.60, f"L3 vs national\n{central[-1]/base0:.1f}× central\n"
            f"({inner_lo[-1]/base0:.1f}–{inner_hi[-1]/base0:.1f}× credible)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=C_TOTAL,
            fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.2, len(scopes) - 0.8)
    ax.set_ylim(0, prior_hi.max() * 1.13)
    ax.set_xlabel("Power-dispatch scope", fontsize=11.0)
    ax.set_ylabel("Full-system cost\n(USD/MWh, 2030)", fontsize=11.0)
    ax.tick_params(axis="x", length=0, labelsize=9.5)
    ax.tick_params(axis="y", labelsize=11.0)
    yonly(ax)
    ax.legend(fontsize=8.0, frameon=False, loc="upper left", labelspacing=0.36,
              handlelength=1.5, borderaxespad=0.2)


# main figure (Fig. 3) tells the present-day story with the realistic blend only
# (Mix); the four other AI operators stay in the appendix figure
MAIN_OPS = ["P_mix"]
APPENDIX_OPS = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst"]


def _save_fig(fig, fig_dir, fname):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(fig_dir / f"{fname}{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {fig_dir / (fname + '.png')}")


def render_fig3(data_dir=DATA, fig_dir=FIG, fname="fig3_ai_perturbation_cost"):
    """Figure 3 — the realistic blend (Mix) only. TOP = present + mechanism:
    (a) day-shape curve over the L×W uncovered-share heatmap (+ vs-no-AI lollipop);
    (b) full-system cost rose at the credible central with a prior-tx ceiling band;
    (c) full-system cost vs power-dispatch scope (storage saving vs transmission).
    BOTTOM = future outlook: (d) the 2025-2050 ridgeline of the per-country
    full-system cost distribution by scope (ride-the-grid convergence); (e) three
    world maps of the per-country full-system cost under the global pool."""
    set_style(11.0)
    plt.close("all")
    global DATA
    DATA = data_dir
    g, dn, pc = load()
    years = [2025, 2030, 2050]

    fig = plt.figure(figsize=(13.2, 16.0))
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.955, top=0.965, bottom=0.032,
                             height_ratios=[1.04, 1.30], hspace=0.12)

    # ---- TOP row: a | b ----
    top = outer[0].subgridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.16)
    draw_ab(fig, top[0, 0], dn, g, ops=MAIN_OPS, ncols=1, cell_hspace=0.20)  # a
    rose.draw_rose_band_into(fig, top[0, 1], fs=10.5)  # b (central full-system rose + prior-tx band)

    # ---- BOTTOM: left column = c (storage↓ vs transmission↑) over d (box plot); right = e ----
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.92], wspace=0.10)
    cd = bot[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.40)
    draw_mechanism(fig.add_subplot(cd[0, 0]), g)               # c (storage vs transmission)
    draw_future(fig.add_subplot(cd[1, 0]), g)                  # d (box plot)

    em = bot[0, 1].subgridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.09], hspace=0.14)
    # two-pole ramp (lilac -> purple -> amber), same as panel a; purple ties to the
    # rose, the amber high-end pole gives the maps contrast instead of mono-purple.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "fig_pa", ["#F2EFF7", "#B6A8D0", "#8A6FB0", "#C75D88", "#E8A33D"])
    cmap.set_bad(alpha=0)
    # panel e = per-country FULL-SYSTEM electricity cost under the global pool (L3,
    # no migration), transmission at the credible central with the ride-the-grid
    # credit (same basis as panels b, c, d). The maps lighten and even out as the
    # cross-regional grid is built and the import premium erodes towards 2050.
    fst = pd.read_csv(DATA / "fullsystem_cost_table.csv")
    fac = rose.central_factor()
    vby = {y: _fs_country_tot(fst, "L3", y, fac) for y in years}
    allv = np.array([v for y in years for v in vby[y].values()])
    vmin = float(np.floor(np.percentile(allv, 4) / 25) * 25)
    vmax = float(np.ceil(np.percentile(allv, 96) / 25) * 25)
    norm = mpl.colors.Normalize(vmin, vmax)
    for j, y in enumerate(years):                              # e: three maps
        axm = fig.add_subplot(em[j, 0], projection=ccrs.Robinson())
        _country_choropleth(axm, vby[y], cmap, norm)
        axm.text(0.005, 0.90, str(y), transform=axm.transAxes, fontsize=12.5,
                 fontweight="bold", color=INK)
        if j == 0:
            axm.legend(handles=[mpatches.Patch(facecolor="#C7CCD1",
                                               edgecolor="none", label="No data")],
                       loc="lower left", fontsize=11.0, frameon=False,
                       handlelength=1.1, handleheight=1.1, borderaxespad=0.0)
    cax = fig.add_subplot(em[3, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Per-country full-system electricity cost "
                   "(USD/MWh · global pool · central tx)", fontsize=11.0)
    cticks = [t for t in (100, 150, 200, 250, 300, 350, 400) if vmin <= t <= vmax]
    cbar.set_ticks(cticks)
    cbar.ax.tick_params(labelsize=11.0)

    for cell, lab in [(top[0, 0], "a"), (top[0, 1], "b"), (cd[0, 0], "c"),
                      (cd[1, 0], "d"), (em[0, 0], "e")]:
        bb = cell.get_position(fig)
        fig.text(bb.x0 - 0.030, bb.y1 + 0.005, lab, fontweight="bold", fontsize=16,
                 ha="left", va="bottom")
    _save_fig(fig, fig_dir, fname)


def render_appendix(data_dir=DATA, fig_dir=FIG):
    """Appendix: the five AI operators NOT featured in the main figure, split into
    two single-page figures — figS_ai_operators_a (day-shape + L x W heatmap +
    lollipop composites, shared colorbar in the spare slot) and figS_ai_operators_b
    (grouped-bar bill panels, migration legend in the spare slot)."""
    set_style(9.5)
    plt.close("all")
    global DATA
    DATA = data_dir
    g, dn, pc = load()

    fig = plt.figure(figsize=(11.0, 12.0))
    gs = fig.add_gridspec(1, 1, left=0.075, right=0.965, top=0.975, bottom=0.045)
    draw_ab(fig, gs[0], dn, g, ops=APPENDIX_OPS, ncols=2, fs=9.5)
    fig.text(0.012, 0.988, "a", fontweight="bold", fontsize=14,
             ha="left", va="top")
    _save_fig(fig, fig_dir, "figS_ai_operators_a")

    fig = plt.figure(figsize=(11.0, 10.0))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.965, top=0.96, bottom=0.055)
    draw_d(fig, gs[0], g, ops=APPENDIX_OPS, ncols=2, fs=9.5)
    fig.text(0.012, 0.988, "b", fontweight="bold", fontsize=14,
             ha="left", va="top")
    _save_fig(fig, fig_dir, "figS_ai_operators_b")


# ============================ Figure 4: future outlook ============================
def _country_choropleth(ax, val_by_iso, cmap, norm, framed=False):
    """Colour countries by val_by_iso (iso2 -> value) on a frykit/Robinson map.
    framed=True draws the full globe (incl. Antarctica) with the Robinson frame."""
    cmeta = pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""])
    cen = {row.iso2: (float(row.lon), float(row.lat)) for row in cmeta.itertuples()}
    if framed:
        ax.set_global()                       # full globe, incl. Antarctica
        ax.spines["geo"].set_visible(True)
        ax.spines["geo"].set_edgecolor("#94A3B8")
        ax.spines["geo"].set_linewidth(0.7)
        ax.set_frame_on(True)
    else:
        ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)
        ax.set_frame_on(False)
        ax.set_aspect("auto")
    geoms = fshp.get_countries()
    tree = STRtree(geoms)
    arr = np.full(len(geoms), np.nan)
    for cc, v in val_by_iso.items():
        if cc not in cen:
            continue
        p = Point(cen[cc][0], cen[cc][1])
        placed = False
        for gi in np.atleast_1d(tree.query(p)):
            gi = int(gi)
            if geoms[gi].covers(p):
                arr[gi] = v if np.isnan(arr[gi]) else max(arr[gi], v)
                placed = True
                break
        if not placed:
            gi = int(tree.nearest(p))
            if np.isnan(arr[gi]):
                arr[gi] = v
    fplt.add_geometries(ax, geoms, fc="#C7CCD1", ec="none",
                        crs=fplt.PLATE_CARREE, zorder=1)
    fplt.add_geometries(ax, geoms, array=np.ma.masked_invalid(arr), cmap=cmap,
                        norm=norm, crs=fplt.PLATE_CARREE, ec="#FFFFFF", lw=0.15,
                        zorder=2)


def render_fig4(data_dir=DATA, fig_dir=FIG, fname="fig4_future_outlook"):
    """Figure 4 — FUTURE outlook (technology progress): (a) the global firming-bill
    trajectory with the storage-technology + AI-shape uncertainty bands; (b) three
    world maps of the per-country firming bill at 2025 / 2030 / 2050 (Mix, central
    storage), showing how the geographic burden evolves."""
    set_style()
    plt.close("all")
    global DATA
    DATA = data_dir
    g, dn, pc = load()
    cdf = pd.read_csv(DATA / "r3c_country_bill.csv")

    fig = plt.figure(figsize=(11.0, 8.8))
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.95, top=0.95, bottom=0.06,
                          height_ratios=[1.0, 1.12], hspace=0.20)

    ax_line = fig.add_subplot(gs[0])
    draw_future(ax_line, g)

    years = [2025, 2030, 2050]
    sub = gs[1].subgridspec(2, 3, height_ratios=[1.0, 0.07], wspace=0.04, hspace=0.06)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad(alpha=0)
    cen = cdf[(cdf.storage_scenario == "Central") & (cdf.year.isin(years))]
    vmax = float(np.ceil(np.percentile(cen.country_bill_musd, 95) / 50) * 50)
    norm = mpl.colors.Normalize(0, vmax)
    for j, y in enumerate(years):
        axm = fig.add_subplot(sub[0, j], projection=ccrs.Robinson())
        s = cdf[(cdf.year == y) & (cdf.storage_scenario == "Central")]
        _country_choropleth(axm, dict(zip(s.country, s.country_bill_musd)), cmap, norm)
        axm.set_title(str(y), fontsize=9.5, fontweight="bold", color=INK, pad=2)
    cax = fig.add_subplot(sub[1, 1])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Per-country firming bill (M USD/yr · Mix · central storage)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.8)

    for cell, lab in [(gs[0], "a"), (gs[1], "b")]:
        bb = cell.get_position(fig)
        fig.text(0.04, bb.y1 + 0.005, lab, fontweight="bold", fontsize=13,
                 ha="left", va="bottom")
    _save_fig(fig, fig_dir, fname)


def main():
    render_fig3(DATA, FIG)        # now also contains the former Fig-4 panels (d, e)
    render_appendix(DATA, FIG)


if __name__ == "__main__":
    main()
