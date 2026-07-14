#!/usr/bin/env python3
"""Appendix figure S3 (figS_ai_operators) — the AI demand shapes NOT carried as
the headline Mix, in the NEW latency x routable caliber.

Replaces the old L x W heatmap + storage-bill-bar version. Two single-page views:

  figS_ai_operators_a — for the no-AI baseline (Today) + the four deferred
      operators (batch & training, measured chat, consumer chat, viral spikes):
      each panel shows the operator's diurnal day-shape (2025/2030/2050 vs no-AI)
      above a 6x6 latency-tolerance x routable-fraction uncovered-share heatmap
      (2030). Shared colorbar in the spare slot.
  figS_ai_operators_b — the full-system cost view: every shape's national vs
      global full-system cost (USD/MWh, 2030), a dumbbell per shape, showing the
      shapes cluster tightly and the national->global routing lever dominates.

English-only, colorblind-safe, matplotlib. Data: data_globalsites/
{r3_latency_cost_table.csv, r1_diurnal_profiles.csv}. Consistent with figs 4/5
(reuses plot_r3_latency_fig3 helpers + plot_r3_split_figs shape names/colours).
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import analyze_cfe_geographic_portfolio_ai as base   # noqa: E402
import plot_r3_latency_fig3 as R3L                    # noqa: E402

warnings.filterwarnings("ignore")

DATA, FIG = R3L.DATA, R3L.FIG
INK, GRID = R3L.INK, R3L.GRID
CMAP_PA = R3L.CMAP_PA
TAU, PHI_ROWS, PHI_LAB = R3L.TAU, R3L.PHI_ROWS, R3L.PHI_LAB

# deferred operators (everything except the headline Mix), reader-facing names
OPS = ["P0_baseline", "P1_flatten", "P2_emp", "P2_sharpen", "P3_burst"]
NAME = {"P0_baseline": "No AI (Today)", "P1_flatten": "Batch & training",
        "P2_emp": "Measured chat", "P2_sharpen": "Consumer chat",
        "P3_burst": "Viral spikes"}
COL = {"P0_baseline": "#64748B", "P1_flatten": "#059669", "P2_emp": "#B45309",
       "P2_sharpen": "#F59E0B", "P3_burst": "#DC2626", "P_mix": "#7C3AED"}


def _ai(t, y):
    xs = sorted(t)
    return float(np.interp(y, xs, [t[x] for x in xs]))


def _diurnal_demand(dn):
    a = dn.groupby("local_hour")["demand"].mean().reindex(range(24)).to_numpy()
    return a / a.mean()


def _shape(op, d, h, y):
    li, lt = _ai(base.AI_SHARE_INFERENCE, y), _ai(base.AI_SHARE_TRAINING, y)
    if op == "P0_baseline":
        return d
    if op == "P1_flatten":
        return base.operator_flatten(d, li, lt)
    if op == "P2_emp":
        return base.operator_sharpen_emp(d, h, li, lt)
    if op == "P2_sharpen":
        return base.operator_sharpen(d, h, li, lt)
    if op == "P3_burst":
        return base.operator_burst(d, li, lt)
    return base.operator_mix(d, h, li, lt)


def _heatmap(ax, df, op, fs=7.0):
    allv = [R3L.unc(df, 2030, op, p, t) for p in PHI_ROWS for t in TAU]
    vmax = float(np.ceil(np.nanmax([v for v in allv if np.isfinite(v)]) / 5) * 5)
    norm = mpl.colors.Normalize(0, vmax)
    for i, p in enumerate(PHI_ROWS):
        for j, t in enumerate(TAU):
            v = R3L.unc(df, 2030, op, p, t)
            fc = CMAP_PA(norm(v))
            ax.add_patch(mpatches.FancyBboxPatch(
                (j - 0.46, i - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0,rounding_size=0.12", facecolor=fc,
                edgecolor="white", linewidth=0.5, mutation_aspect=1))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs,
                    color="white" if lum < 0.55 else "#1F2937", fontweight="bold")
    ax.set_xlim(-0.6, len(TAU) - 0.4); ax.set_ylim(-0.6, len(PHI_ROWS) - 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(len(TAU))); ax.set_xticklabels([str(t) for t in TAU], fontsize=fs)
    ax.set_yticks(range(len(PHI_ROWS))); ax.set_yticklabels([PHI_LAB[p] for p in PHI_ROWS], fontsize=fs)
    ax.set_xlabel("Latency tolerance (ms) →", fontsize=fs + 0.5)
    ax.set_ylabel("Routable fraction ↑", fontsize=fs + 0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return norm


def _dayshape(ax, dn, op, fs=7.5):
    h = np.arange(24)
    d = _diurnal_demand(dn)
    ax.axvspan(18, 23, color="#64748B", alpha=0.10, zorder=0)
    if op != "P0_baseline":
        for y, al in zip([2025, 2030, 2050], [0.34, 0.62, 1.0]):
            ax.plot(h, _shape(op, d, h, y), color=COL[op], alpha=al,
                    lw=1.5 if y == 2030 else 1.0, zorder=3)
    ax.plot(h, d, color="#9CA3AF", lw=1.0, ls=":", zorder=2)
    ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18]); ax.set_xticklabels([0, 6, 12, 18], fontsize=fs)
    ax.set_ylim(0, max(3.2, ax.get_ylim()[1]))
    ax.set_yticks([0, 1, 2, 3]); ax.tick_params(labelsize=fs)
    ax.set_ylabel("Demand\n(mean=1)", fontsize=fs)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(NAME[op], fontsize=fs + 2.0, loc="left", color=COL[op], fontweight="bold", pad=2)


def render_a(df, dn):
    R3L.R3P.set_style(9.5)
    plt.close("all")
    fig = plt.figure(figsize=(12.6, 9.4))
    outer = fig.add_gridspec(2, 3, left=0.055, right=0.97, top=0.94, bottom=0.07,
                             hspace=0.34, wspace=0.26)
    norm = None
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for op, (r, c) in zip(OPS, cells):
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[0.55, 1.0], hspace=0.34)
        _dayshape(fig.add_subplot(inner[0, 0]), dn, op)
        norm = _heatmap(fig.add_subplot(inner[1, 0]), df, op)
    # shared colorbar + legend in the spare slot (1,2)
    sl = outer[1, 2].subgridspec(3, 1, height_ratios=[1.0, 0.12, 1.0], hspace=0.3)
    cax = fig.add_subplot(sl[1, 0])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP_PA); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Uncovered demand share (%, 2030)", fontsize=9.0)
    cb.ax.tick_params(labelsize=8.5)
    ax_note = fig.add_subplot(sl[0, 0]); ax_note.axis("off")
    ax_note.text(0.0, 0.95,
                 "Each panel: the AI demand shape's diurnal\nprofile (2025/2030/2050 vs no-AI, dotted)\n"
                 "above its latency × routable uncovered-\nshare heatmap (2030, % of demand).\n\n"
                 "The shape barely moves the gap: every\noperator collapses along the same\n"
                 "national→global routing lever as the\nheadline Mix (Fig. 4c,d).",
                 transform=ax_note.transAxes, fontsize=9.0, va="top", ha="left", color="#334155")
    for op, (r, c) in zip(OPS, cells):
        bb = outer[r, c].get_position(fig)
        fig.text(bb.x0 - 0.028, bb.y1 + 0.004, "abcde"[OPS.index(op)], fontweight="bold",
                 fontsize=14, ha="left", va="bottom")
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"figS_ai_operators_a{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG / 'figS_ai_operators_a.png'}")


def _corner(df, op, tau, phi, col="lcoe_elec"):
    s = df[(df.year == 2030) & (df.perturbation == op) & (df.tau_ms == tau) & (df.phi == phi)]
    return float(s[col].median())


def render_b(df):
    R3L.R3P.set_style(10.5)
    plt.close("all")
    allops = ["P1_flatten", "P2_emp", "P3_burst", "P0_baseline", "P_mix", "P2_sharpen"]
    nat = {o: _corner(df, o, 50, 0.0) for o in allops}
    glo = {o: _corner(df, o, 500, 1.0) for o in allops}
    order = sorted(allops, key=lambda o: glo[o])          # cheapest global at top
    ys = list(range(len(order)))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    C_NAT, C_GLO = "#C2410C", "#0F766E"
    nv, gv = [nat[o] for o in order], [glo[o] for o in order]
    ax.axvspan(min(nv), max(nv), color=C_NAT, alpha=0.08, zorder=0)
    ax.axvspan(min(gv), max(gv), color=C_GLO, alpha=0.08, zorder=0)
    for o, y in zip(order, ys):
        ax.plot([glo[o], nat[o]], [y, y], color="#CBD5E1", lw=3.0, zorder=2, solid_capstyle="round")
        ax.plot(glo[o], y, "o", color=C_GLO, ms=10 if o == "P_mix" else 8.5, mec="white", mew=1.2, zorder=4)
        ax.plot(nat[o], y, "o", color=C_NAT, ms=10 if o == "P_mix" else 8.5, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"{glo[o]:.0f}", (glo[o], y), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.4, color=C_GLO, fontweight="bold")
        ax.annotate(f"{nat[o]:.0f}", (nat[o], y), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.4, color=C_NAT, fontweight="bold")
    ax.set_yticks(ys)
    labs = ax.set_yticklabels([NAME.get(o, "Mix (headline)") for o in order], fontsize=10.0)
    for o, t in zip(order, labs):
        if o == "P_mix":
            t.set_color(COL["P_mix"]); t.set_fontweight("bold")
    ytop = ys[0]
    ax.text(np.mean(gv), ytop + 0.7, "Global reach\n(500 ms, full routing)", ha="center", va="bottom",
            fontsize=9.0, color=C_GLO, fontweight="bold")
    ax.text(np.mean(nv), ytop + 0.7, "National reach\n(50 ms, no routing)", ha="center", va="bottom",
            fontsize=9.0, color=C_NAT, fontweight="bold")
    yb = -0.85
    ax.annotate("", xy=(np.mean(gv), yb), xytext=(np.mean(nv), yb),
                arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.6))
    ax.text((np.mean(gv) + np.mean(nv)) / 2, yb - 0.12,
            f"routing lever:  −{np.mean(nv) - np.mean(gv):.0f} USD/MWh", ha="center", va="top",
            fontsize=9.0, color="#111827", fontweight="bold")
    ax.text(0.015, 0.02, f"six shapes span only ~{max(gv) - min(gv):.0f} USD/MWh at each reach",
            transform=ax.transAxes, fontsize=8.6, color="#475569", style="italic", ha="left", va="bottom")
    ax.set_xlim(min(gv) - 12, max(nv) + 10); ax.set_ylim(yb - 0.9, ytop + 1.7)
    ax.set_xlabel("Full-system electricity cost (USD/MWh, 2030, median of 104 countries)", fontsize=10.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(0.012, 0.985, "f", fontweight="bold", fontsize=14, ha="left", va="top")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.13)
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"figS_ai_operators_b{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG / 'figS_ai_operators_b.png'}")


def main():
    df, dn = R3L.load()
    render_a(df, dn)
    render_b(df)


if __name__ == "__main__":
    main()
