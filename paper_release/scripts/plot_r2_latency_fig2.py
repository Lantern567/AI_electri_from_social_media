#!/usr/bin/env python3
"""R2 v2 (latency tolerance x routable fraction), 3-panel Fig 2 layout.

  (a) matrix heatmap : latency tolerance tau (x) x routable fraction phi (y)
  (b) violin         : residual uncovered share across tau (at phi=100%)
  (c) scatter        : sender demand-peak hour vs receiver solar hour when compute runs
The sender->receiver pairing panels (chord + flow map) live in the dedicated pairing
figure (plot_r2_pairing.py -> fig3_pairing), which shows them across several tau scenarios.
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
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
P.set_style()

DATA = P.REPORT / "data_globalsites_stations_expanded"
FIG = P.REPORT / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)
FIG_PA, PRIMARY, SECONDARY = P.FIG_PA, P.PRIMARY, P.SECONDARY
COUNTRY_META = P.COUNTRY_META

ROWS_PHI = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
COLS_TAU = [50, 100, 150, 200, 300, 500]
PHI_LABEL = {1.0: "100%", 0.8: "80%", 0.6: "60%", 0.4: "40%", 0.2: "20%", 0.0: "0%"}
REACH = {50: "local", 100: "≈3 Mm", 150: "≈6.7 Mm", 200: "≈10 Mm", 300: "≈17 Mm", 500: "global"}
TAU_LABEL = {t: f"{t} ms\n{REACH[t]}" for t in COLS_TAU}
ZONE_COLOR = {"Americas": "#0072B2", "Europe / Africa": "#E69F00", "Asia-Pacific": "#009E73"}
ZONE_ORDER = ["Americas", "Europe / Africa", "Asia-Pacific"]
REGIONS = ["North America", "Latin America", "Europe", "Africa/MENA", "Asia-Pacific"]
REG_ZONE = {"North America": "Americas", "Latin America": "Americas", "Europe": "Europe / Africa",
            "Africa/MENA": "Europe / Africa", "Asia-Pacific": "Asia-Pacific"}


def lon_zone(lon):
    if -170.0 <= lon < -30.0:
        return "Americas"
    if -30.0 <= lon < 60.0:
        return "Europe / Africa"
    return "Asia-Pacific"


# ---------------------------------------------------------------- panel (a)
def panel_a(fig, gs_cell, med):
    mat = med.reindex(index=ROWS_PHI, columns=COLS_TAU).values
    nr, nc = mat.shape
    bottom = nr - 1
    gs_a = gs_cell.subgridspec(2, 2, width_ratios=[3.0, 1.0], height_ratios=[1.0, 3.0],
                               wspace=0.06, hspace=0.06)
    ax_top = fig.add_subplot(gs_a[0, 0]); ax_a = fig.add_subplot(gs_a[1, 0])
    ax_right = fig.add_subplot(gs_a[1, 1]); ax_corner = fig.add_subplot(gs_a[0, 1]); ax_corner.axis("off")
    fs = 15
    vmin, vmax = mat.min() * 0.95, mat.max() * 1.02
    im = ax_a.imshow(mat, aspect="auto", cmap=FIG_PA, vmin=vmin, vmax=vmax)
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            fc = FIG_PA((v - vmin) / (vmax - vmin))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs - 1,
                      weight="bold", color="white" if lum < 0.55 else "#111827")
    ax_a.set_xticks(range(nc)); ax_a.set_xticklabels([TAU_LABEL[t] for t in COLS_TAU], fontsize=fs - 2.5)
    ax_a.set_yticks(range(nr)); ax_a.set_yticklabels([PHI_LABEL[p] for p in ROWS_PHI], fontsize=fs)
    ax_a.set_xlabel("Latency tolerance (RTT → reach)", fontsize=fs)
    ax_a.set_ylabel("Routable fraction of load", fontsize=fs)
    ax_a.grid(False)
    for (ri, ci) in [(bottom, 0), (0, nc - 1)]:
        ax_a.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                                          edgecolor="#111827", linewidth=1.6, zorder=4))
    fx = [mpl.patheffects.withStroke(linewidth=2.4, foreground="#111827")]
    ax_a.text(0, bottom + 0.34, "Status quo", fontsize=fs - 1, weight="bold", color="white",
              ha="center", va="center", path_effects=fx, zorder=5)
    ax_a.text(nc - 0.55, 0.34, "Follow-sun floor", fontsize=fs - 1, weight="bold", color="white",
              ha="right", va="center", path_effects=fx, zorder=20, clip_on=False)
    pure_t = [mat[0, j] for j in range(nc)]
    ax_top.bar(range(nc), [pure_t[0] - v for v in pure_t], color=SECONDARY, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_t):
        d = pure_t[0] - v
        ax_top.text(i, d + 0.1, f"{d:.0f}" if d >= 0.05 else "0", ha="center", va="bottom",
                    fontsize=fs - 1, color="#111827")
    ax_top.set_ylabel("Δ vs local\n(latency, pp)", fontsize=fs - 1)
    P.panel_letter(ax_top, "a", x=-0.27, y=1.02)
    ax_top.tick_params(axis="x", labelbottom=False); ax_top.tick_params(axis="y", labelsize=fs - 1)
    ax_top.set_ylim(0, max(pure_t[0] - min(pure_t), 0.1) * 1.3)
    P.soften_axes(ax_top)
    pure_s = [mat[i, nc - 1] for i in range(nr)]
    ax_right.barh(range(nr), [pure_s[bottom] - v for v in pure_s], color=PRIMARY, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_s):
        d = pure_s[bottom] - v
        ax_right.text(d + 0.1, i, f"{d:.0f}" if d >= 0.05 else "0", ha="left", va="center",
                      fontsize=fs - 1, color="#111827")
    ax_right.set_xlabel("Δ vs no routing\n(routability, pp)", fontsize=fs - 1)
    ax_right.tick_params(axis="y", labelleft=False); ax_right.tick_params(axis="x", labelsize=fs - 1)
    ax_right.invert_yaxis()
    ax_right.set_xlim(0, max(pure_s[bottom] - min(pure_s), 0.1) * 1.35)
    P.soften_axes(ax_right)
    cax = ax_corner.inset_axes([0.08, 0.48, 0.86, 0.15])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Median uncovered\nshare (%)", fontsize=fs - 1)
    cb.ax.tick_params(labelsize=fs - 1)
    return ax_a


# ---------------------------------------------------------------- panel (c) scatter
def draw_compute_followsun(ax, ax_top, ax_right):
    d = pd.read_csv(DATA / "r2_wf_followsun.csv")
    b, a = np.polyfit(d["x_peak"], d["y_solar"], 1)
    yhat = a + b * d["x_peak"]
    r2 = 1 - ((d["y_solar"] - yhat) ** 2).sum() / ((d["y_solar"] - d["y_solar"].mean()) ** 2).sum()
    eve = d[(d["x_peak"] >= 17) | (d["x_peak"] <= 1)]
    reach_med = eve["reach"].median()
    m = d["phase_lag"].notna() & (d["x_peak"] >= 12)
    rr = float(np.corrcoef(d.loc[m, "reach"], d.loc[m, "phase_lag"])[0, 1]) if m.sum() > 2 else float("nan")
    ybar = float(d["y_solar"].mean())
    xcross = a / (1 - b) if b != 1 else 12
    FS = 13; XLIM = (8.5, 23.8)
    xw = np.linspace(xcross, XLIM[1], 50)
    ax.fill_between(xw, a + b * xw, xw, color=PRIMARY, alpha=0.08, lw=0, zorder=1)
    ax.text(16.4, 14.4, "compute runs west (sunlit)", fontsize=FS - 2, color="#3B5BA5",
            ha="center", va="center", style="italic", rotation=27, rotation_mode="anchor")
    ax.plot([0, 23], [0, 23], color="#9CA3AF", lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.text(17.2, 17.7, "run-at-home (slope +1.00)", fontsize=FS, color="#6B7280", ha="left",
            va="bottom", style="italic", rotation=45, rotation_mode="anchor")
    ax.axhspan(11, 13, color=SECONDARY, alpha=0.08, zorder=0)
    ax.text(8.8, 12.3, "solar noon", fontsize=FS, color="#B45309", ha="left", va="center")
    xx = np.array([d["x_peak"].min() - 0.5, d["x_peak"].max() + 0.5])
    ax.plot(xx, a + b * xx, color="#111827", lw=2.2, zorder=4, label=f"fit: slope {b:+.2f}  (R²={r2:.2f})")
    rng = np.random.default_rng(0); jit = rng.uniform(-0.22, 0.22, len(d))
    ax.scatter(d["x_peak"] + jit, d["y_solar"], s=62, c=PRIMARY, alpha=0.5, edgecolor="white",
               linewidth=0.5, zorder=5, label=f"one country (n = {len(d)})")
    xr = 22.7
    ax.annotate("", xy=(xr, xr), xytext=(xr, ybar), arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.6), zorder=6)
    ax.text(xr - 0.45, 19.7, f"reach ≈ {reach_med:.0f} h\n(west, r {rr:.2f})", fontsize=FS - 1,
            color="#111827", ha="right", va="center", linespacing=1.05)
    ax.set_xlim(*XLIM); ax.set_ylim(*XLIM); ax.set_aspect("equal")
    ax.set_xticks(range(9, 24, 3)); ax.set_yticks(range(9, 24, 3))
    ax.set_xlabel("Demand peak local hour", fontsize=FS)
    ax.set_ylabel("Receiver solar hour when compute runs", fontsize=FS)
    ax.tick_params(axis="both", labelsize=FS)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#EEF1F4", lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=FS - 1, frameon=False, labelspacing=0.4, handletextpad=0.4, borderpad=0.2)
    bins = np.arange(8.5, 24.5, 1)
    ax_top.hist(d["x_peak"], bins=bins, color="#B7C0CA", edgecolor="white", linewidth=0.6)
    ax_top.set_xlim(*XLIM); ax_top.axis("off")
    ax_top.text(0.0, 0.92, "demand peaks spread to evening", transform=ax_top.transAxes,
                fontsize=FS - 3, color="#6B7280", ha="left", va="top")
    ax_right.hist(d["y_solar"], bins=bins, orientation="horizontal", color=SECONDARY, alpha=0.55,
                  edgecolor="white", linewidth=0.6)
    ax_right.set_ylim(*XLIM); ax_right.axis("off")
    ax_right.text(0.95, 0.02, "compute piles\nat noon", transform=ax_right.transAxes, fontsize=FS - 3,
                  color="#B45309", ha="right", va="bottom", rotation=270, linespacing=1.0)


# ---------------------------------------------------------------- panel (b) violin across tau
def draw_tau_distribution(ax, grid):
    FS = 13
    taus = [50, 100, 200, 500]
    labels = ["50 ms\nlocal", "100 ms\n≈3 Mm", "200 ms\n≈10 Mm", "500 ms\nglobal"]
    g1 = grid[grid["phi"] == 1.0]
    data = [(g1[g1["tau_ms"] == t]["uncovered_share"] * 100).values for t in taus]
    pos = list(range(4))
    fills = ["#E5A9A2", "#EAD0A0", "#A9D2CA", "#A7C4E6"]
    parts = ax.violinplot(data, positions=pos, widths=0.9, showextrema=False)
    for pc, c in zip(parts["bodies"], fills):
        pc.set_facecolor(c); pc.set_alpha(0.40); pc.set_edgecolor("none")
    bp = ax.boxplot(data, positions=pos, widths=0.24, patch_artist=True, showfliers=False,
                    manage_ticks=False, medianprops=dict(color="#111827", lw=1.6),
                    whiskerprops=dict(color="#111827", lw=1.4), capprops=dict(color="#111827", lw=1.4),
                    boxprops=dict(edgecolor="#111827", lw=1.4))
    for patch, c in zip(bp["boxes"], fills):
        patch.set_facecolor(c); patch.set_alpha(0.95)
    means = [float(np.mean(x)) for x in data]
    ax.scatter(pos, means, marker="D", s=55, facecolor="white", edgecolor="#111827", linewidth=1.4, zorder=7)
    for i, x in enumerate(data):
        ax.annotate(f"{np.median(x):.1f}", (i + 0.16, np.median(x)), textcoords="offset points",
                    xytext=(2, 0), fontsize=FS - 1, fontweight="bold", color="#111827", va="center")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=FS - 2)
    ax.set_ylabel("Residual uncovered share (%)", fontsize=FS)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 62); ax.tick_params(axis="y", labelsize=FS)
    ax.text(0.98, 0.97, "φ = 100% routable", transform=ax.transAxes, fontsize=FS - 3,
            color="#6B7280", ha="right", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", color="#EEF1F4", lw=0.5); ax.set_axisbelow(True)


def main():
    g = pd.read_csv(DATA / "r2_waveform_grid.csv")
    med = (g.groupby(["phi", "tau_ms"]).uncovered_share.median().unstack() * 100)
    fig = plt.figure(figsize=(15.0, 11.2))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.985, top=0.935, bottom=0.065,
                          hspace=0.40, wspace=0.16, width_ratios=[1.12, 1.0],
                          height_ratios=[0.66, 1.08])
    # (a) latency tolerance x routable fraction heatmap, spanning the left column
    ax_a = panel_a(fig, gs[:, 0], med)
    # (b) residual uncovered share across tau (top-right)
    ax_b = fig.add_subplot(gs[0, 1])
    draw_tau_distribution(ax_b, g)
    P.panel_letter(ax_b, "b", x=-0.09, y=1.05)
    # (c) follow-the-sun scatter with marginal histograms (bottom-right)
    cell = gs[1, 1].get_position(fig)
    gap = 0.006
    s_x = cell.width * 0.90
    s_y = s_x * fig.get_figwidth() / fig.get_figheight()
    mh = cell.height * 0.15
    if s_y + gap + mh > cell.height:
        s_y = cell.height - gap - mh
        s_x = s_y * fig.get_figheight() / fig.get_figwidth()
    mw = s_x * 0.15
    bx0 = cell.x0 + 0.015
    by0 = cell.y0
    ax_c = fig.add_axes([bx0, by0, s_x, s_y])
    ax_ctop = fig.add_axes([bx0, by0 + s_y + gap, s_x, mh])
    ax_cright = fig.add_axes([bx0 + s_x + gap, by0, mw, s_y])
    draw_compute_followsun(ax_c, ax_ctop, ax_cright)
    fig.text(bx0 - 0.032, by0 + s_y + mh + gap, "c", fontsize=20, fontweight="bold",
             ha="left", va="bottom")
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig2_latency_full{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_latency_full")


if __name__ == "__main__":
    main()
