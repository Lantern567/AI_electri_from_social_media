#!/usr/bin/env python3
"""Result 3, panel (b): full-system cost as a polar stacked-bar (Nightingale)
rose, styled after the green-methanol supply-chain rose charts
(products/supply_chain_optimization/visualization/polar_stacked_bar_visualization.py).

16 angular sectors = the 16 dispatch scenarios (4 power-dispatch scopes x 4
workload-migration shares). Radius = full-system electricity LCOE (USD/MWh of
delivered demand); each sector is stacked radially by the four cost components
(generation, transmission, firming storage, ancillary). The outer coloured arc
groups sectors by power-dispatch scope (national -> global). A companion
percentage version normalises each sector to 100% to show composition.

Data: data_globalsites/fullsystem_cost_table.csv (P_mix; year 2030 by default).
Outputs: figures_globalsites/fig3b_polar_rose_cost.{png,pdf}
         figures_globalsites/fig3b_polar_rose_share.{png,pdf}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import fullsystem_cost_params as P  # noqa: E402

REPORT = P.DATA.parent
FIG = REPORT / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)

YEAR = 2030
OPERATOR = "P_mix"

L_ORDER = ["L0", "L1", "L2", "L3"]
W_ORDER = ["W0", "W1", "W2", "W3"]
L_NAME = {"L0": "National", "L1": "≤1.5 Mm", "L2": "≤3 Mm", "L3": "Global"}
W_NAME = {"W0": "0%", "W1": "10%", "W2": "30%", "W3": "60%"}   # migration share

# four cost components (inner -> outer): a soft, low-saturation blue -> lilac ramp
# (muted/"elegant" — no bright violet, no dark navy-purple). Lightness increases
# outward; transmission is a muted periwinkle, distinct from the dusty-blue base.
COMPONENTS = [
    ("lcoe_gen",   "Generation",      "#6E7FA8"),  # muted dusty blue
    ("lcoe_tx",    "Transmission",    "#9085B8"),  # muted periwinkle-violet
    ("lcoe_store", "Firming storage", "#B6A8D0"),  # soft mauve
    ("lcoe_anc",   "Ancillary",       "#D8D1E6"),  # pale lilac
]
# power-scope group-ring colours: soft blue-grey -> muted lavender as pooling widens
# (deepest stays medium, never dark).
GROUP_COLOR = {"L0": "#AEB4C2", "L1": "#9DA1C4", "L2": "#867BAE", "L3": "#6A5E96"}

INK = "#1F2937"


def set_style(fs=13.0):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.edgecolor": "#374151", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
    })


def load_matrix():
    df = pd.read_csv(P.DATA / "fullsystem_cost_table.csv")
    df = df[(df.year == YEAR) & (df.perturbation == OPERATOR)] if "perturbation" in df else df[df.year == YEAR]
    g = df.groupby(["power_scope", "workload_scenario"])[
        [c for c, _, _ in COMPONENTS]].mean()
    # ordered 16 rows: L0(W0..W3), L1(...), ...
    rows = []
    for L in L_ORDER:
        for W in W_ORDER:
            rows.append((L, W, g.loc[(L, W)]))
    return rows


def sector_angles(n_groups=4, n_per=4, gap=np.pi / 36):
    """Clockwise from top; 4 groups of 4 with inter-group gaps. Returns angles,
    width, and per-group (min,max) angular ranges for the outer ring."""
    n = n_groups * n_per
    avail = 2 * np.pi - n_groups * gap
    ang_per = avail / n
    width = ang_per * 0.92
    angles, group_ranges = [], {}
    cur = np.pi / 2
    for gi, L in enumerate(L_ORDER):
        temp = []
        for _ in range(n_per):
            angles.append(cur)
            temp.append(cur)
            cur -= ang_per
        group_ranges[L] = (min(temp) - ang_per / 2, max(temp) + ang_per / 2)
        cur -= gap
    return np.array(angles), width, ang_per, group_ranges


def draw_rose_on_ax(ax, rows, percentage=False, fs=13.0, sector_labels=True,
                    center_label=True):
    """Draw the polar stacked-bar rose onto an existing polar `ax`. Returns the
    (handles, labels) for a cost-component legend so the caller can place it."""
    comp_vals = np.array([[r[2][c] for c, _, _ in COMPONENTS] for r in rows])  # 16x4
    if percentage:
        comp_vals = 100.0 * comp_vals / comp_vals.sum(axis=1, keepdims=True)
    max_total = comp_vals.sum(axis=1).max()

    angles, width, ang_per, group_ranges = sector_angles()
    inner = max_total * 0.35
    ax.set_facecolor("white")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    bottom = np.full(len(rows), inner, dtype=float)
    handles = []
    for ci, (col, name, color) in enumerate(COMPONENTS):
        ax.bar(angles, comp_vals[:, ci], width=width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="white", label=name))
        bottom += comp_vals[:, ci]

    ring_gap = max_total * 0.03
    ring_w = max_total * 0.11
    ring_start = bottom.max() + ring_gap
    for L, (a0, a1) in group_ranges.items():
        center = (a0 + a1) / 2.0
        ax.bar(center, ring_w, width=(a1 - a0), bottom=ring_start,
               color=GROUP_COLOR[L], edgecolor="white", linewidth=1.0, alpha=0.95, zorder=3)
        deg = np.degrees(center) % 360
        rot = deg - 90 + (180 if 90 < deg < 270 else 0)
        ax.text(center, ring_start + ring_w / 2, L_NAME[L], ha="center", va="center",
                fontsize=fs + 1, fontweight="bold", color="white", rotation=rot,
                rotation_mode="anchor", zorder=5)

    if sector_labels:
        for ang, r in zip(angles, rows):
            deg = np.degrees(ang) % 360
            rot = deg - 90 + (180 if 90 < deg < 270 else 0)
            ax.text(ang, bottom.max() + ring_gap * 0.5, W_NAME[r[1]], ha="center",
                    va="center", fontsize=fs - 3.5, color="#475569", rotation=rot,
                    rotation_mode="anchor", zorder=5)

    span = bottom.max() - inner
    if percentage:
        tick_vals = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    else:
        step = next(s for s in (50, 100, 150, 200, 300, 500, 1000) if span / s <= 5)
        tick_vals = np.arange(step, span + 1e-6, step)
    ax.set_yticks(tick_vals + inner)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#aaaaaa", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, ring_start + ring_w * 1.05)
    suffix = "%" if percentage else ""
    for val, r in zip(tick_vals, tick_vals + inner):
        ax.text(np.pi / 2, r, f"{int(val)}{suffix}", ha="center", va="center",
                fontsize=fs - 2, color="#666666", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4), zorder=6)

    if center_label:
        ctr = "Cost share\n(%)" if percentage else "Full-system\ncost (USD/MWh)"
        ax.text(0, 0, ctr, ha="center", va="center", fontsize=fs, fontweight="bold",
                color="#555555")
    return handles, [h.get_label() for h in handles]


def draw_rose_into(fig, subspec, rows=None, percentage=False, fs=10.0):
    """Embed the rose into a gridspec cell of an existing figure (for Fig. 3 b).
    The power-scope group ring is labelled in-place, so only a compact
    cost-component legend is added (lower-left corner of the cell)."""
    if rows is None:
        rows = load_matrix()
    ax = fig.add_subplot(subspec, projection="polar")
    handles, labels = draw_rose_on_ax(ax, rows, percentage=percentage, fs=fs,
                                      sector_labels=True)
    # horizontal component legend ABOVE the rose; no title (the swatches+labels
    # are self-explanatory) to keep the strip narrow and avoid occlusion.
    ax.legend(handles=handles, labels=labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=fs - 0.5,
              frameon=False, columnspacing=1.2, handlelength=1.1,
              handletextpad=0.4, borderaxespad=0.0)
    return ax


# ---------------------------------------------------------------------------
# central-transmission re-basing + prior-ceiling band (Fig. 3 panel b)
# ---------------------------------------------------------------------------
def central_factor():
    """Per-scope multiplier that turns the table's PRIOR dedicated-link lcoe_tx into
    the credible shared-backbone literature CENTRAL used by Fig. 3 panels c/d, read
    from data_globalsites/r3_tx_bounds.csv (central / prior transmission LCOE)."""
    b = pd.read_csv(P.DATA / "r3_tx_bounds.csv")

    def tx(topo, price, sc):
        r = b[(b.topology == topo) & (b.price == price) & (b.scope == sc)]
        return float(r["lcoe_tx"].iloc[0]) if len(r) else 0.0
    return {sc: (tx("shared", "lit", sc) / tx("dedicated", "ours", sc))
            if tx("dedicated", "ours", sc) > 1e-9 else 0.0 for sc in L_ORDER}


def _scale_tx(rows, fac):
    out = []
    for L, W, s in rows:
        s2 = s.copy()
        s2["lcoe_tx"] = s["lcoe_tx"] * fac[L]
        out.append((L, W, s2))
    return out


PRIOR_BAND_COLOR = "#9CA3AF"


def draw_rose_band_on_ax(ax, central_rows, prior_rows, fs=13.0, sector_labels=True,
                         center_label=True):
    """Like draw_rose_on_ax, but the transmission ring is the CENTRAL estimate and a
    faint hatched band runs from each sector's central total out to its PRIOR total —
    the transmission upper bound (dedicated links, 700/200·kW, 7%). Returns the
    (handles, labels) for the legend, including the prior-ceiling band patch."""
    comp = np.array([[r[2][c] for c, _, _ in COMPONENTS] for r in central_rows])
    prior_tot = np.array([sum(r[2][c] for c, _, _ in COMPONENTS) for r in prior_rows])
    max_total = float(prior_tot.max())
    angles, width, ang_per, group_ranges = sector_angles()
    inner = max_total * 0.35
    ax.set_facecolor("white")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    bottom = np.full(len(central_rows), inner, dtype=float)
    handles = []
    for ci, (col, name, color) in enumerate(COMPONENTS):
        ax.bar(angles, comp[:, ci], width=width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="white", label=name))
        bottom += comp[:, ci]
    ghost = np.clip((inner + prior_tot) - bottom, 0, None)
    ax.bar(angles, ghost, width=width, bottom=bottom, color=PRIOR_BAND_COLOR,
           edgecolor="white", linewidth=0.3, alpha=0.30, hatch="////", zorder=2)
    handles.append(mpatches.Patch(facecolor=PRIOR_BAND_COLOR, alpha=0.34,
                                  hatch="////", edgecolor="white",
                                  label="Prior tx ceiling"))
    top = inner + prior_tot

    ring_gap = max_total * 0.03
    ring_w = max_total * 0.11
    ring_start = top.max() + ring_gap
    for L, (a0, a1) in group_ranges.items():
        center = (a0 + a1) / 2.0
        ax.bar(center, ring_w, width=(a1 - a0), bottom=ring_start,
               color=GROUP_COLOR[L], edgecolor="white", linewidth=1.0,
               alpha=0.95, zorder=3)
        deg = np.degrees(center) % 360
        rot = deg - 90 + (180 if 90 < deg < 270 else 0)
        ax.text(center, ring_start + ring_w / 2, L_NAME[L], ha="center", va="center",
                fontsize=fs + 1, fontweight="bold", color="white", rotation=rot,
                rotation_mode="anchor", zorder=5)
    if sector_labels:
        for ang, r in zip(angles, central_rows):
            deg = np.degrees(ang) % 360
            rot = deg - 90 + (180 if 90 < deg < 270 else 0)
            ax.text(ang, top.max() + ring_gap * 0.5, W_NAME[r[1]], ha="center",
                    va="center", fontsize=fs - 3.5, color="#475569", rotation=rot,
                    rotation_mode="anchor", zorder=5)
    span = max_total
    step = next(s for s in (50, 100, 150, 200, 300, 500, 1000) if span / s <= 5)
    tick_vals = np.arange(step, span + 1e-6, step)
    ax.set_yticks(tick_vals + inner)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#aaaaaa", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, ring_start + ring_w * 1.05)
    for val, r in zip(tick_vals, tick_vals + inner):
        ax.text(np.pi / 2, r, f"{int(val)}", ha="center", va="center", fontsize=fs - 2,
                color="#666666", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
                zorder=6)
    if center_label:
        ax.text(0, 0, "Full-system\ncost (USD/MWh)", ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#555555")
    return handles, [h.get_label() for h in handles]


def draw_rose_band_into(fig, subspec, fs=10.0):
    """Embed the CENTRAL rose + prior-tx ceiling band into a gridspec cell (Fig. 3 b)."""
    prior_rows = load_matrix()
    central_rows = _scale_tx(prior_rows, central_factor())
    ax = fig.add_subplot(subspec, projection="polar")
    handles, labels = draw_rose_band_on_ax(ax, central_rows, prior_rows, fs=fs,
                                           sector_labels=True)
    ax.legend(handles=handles, labels=labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=len(handles), fontsize=fs - 1.0,
              frameon=False, columnspacing=0.8, handlelength=1.0,
              handletextpad=0.35, borderaxespad=0.0)
    return ax


def draw_rose(rows, percentage=False):
    set_style()
    fig, ax = plt.subplots(figsize=(12.5, 12.5), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("white")
    handles, labels = draw_rose_on_ax(ax, rows, percentage=percentage, fs=13.0)
    leg1 = ax.legend(handles=handles, labels=labels, loc="upper left",
                     bbox_to_anchor=(1.04, 1.0), fontsize=12.5, title="Cost component",
                     title_fontsize=13, frameon=False, labelspacing=0.7)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    gh = [mpatches.Patch(facecolor=GROUP_COLOR[L], edgecolor="white", label=L_NAME[L])
          for L in L_ORDER]
    leg2 = ax.legend(handles=gh, loc="lower left", bbox_to_anchor=(1.04, 0.0),
                     fontsize=12.5, title="Power-dispatch scope", title_fontsize=13,
                     frameon=False, labelspacing=0.6)
    leg2.get_title().set_fontweight("bold")
    fname = "fig3b_polar_rose_share" if percentage else "fig3b_polar_rose_cost"
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"{fname}{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG / (fname + '.png')}")


def main():
    rows = load_matrix()
    draw_rose(rows, percentage=False)
    draw_rose(rows, percentage=True)


if __name__ == "__main__":
    main()
