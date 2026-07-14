#!/usr/bin/env python3
"""Redesigned Fig 2 in the ORIGINAL 5-panel layout/style, with new data.

  (a) matrix heatmap : migration distance x delay grace (compute flexibility)
  (b) chord          : compute-receiver hubs x sender regions
  (c) violin         : residual uncovered share across D0->D3 (g=0, spatial only)
  (d) map            : sender -> receiver compute-migration flows (follow-the-sun)
  (e) scatter        : sender demand-peak hour vs receiver solar hour when compute runs

Panels (b)/(d)/(e) use the spatial upper bound (D3 global, g=0) so the
follow-the-sun-via-compute story is isolated from temporal deferral.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Geod

import plot_cfe_geographic_portfolio_ai as P
import cartopy.crs as ccrs
import frykit.plot as fplt
import frykit.shp as fshp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
P.set_style()

DATA = P.REPORT / "data_globalsites_stations_expanded"
FIG = P.REPORT / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)
FIG_PA, PRIMARY, SECONDARY = P.FIG_PA, P.PRIMARY, P.SECONDARY
COUNTRY_META = P.COUNTRY_META

ROWS_D = ["D3", "D2", "D1", "D0"]
COLS_G = [0, 1, 3, 6, 12, 24, 168]
GRACE_TICK = {0: "0", 1: "1 h", 3: "3 h", 6: "6 h", 12: "12 h", 24: "24 h", 168: "7 d"}
DIST_LABEL = {"D3": "D3 Global\n(no cap)", "D2": "D2 ≤3000 km",
              "D1": "D1 ≤500 km", "D0": "D0 In-country\n(no migration)"}
ZONE_COLOR = {"Americas": "#3A7CA5", "Europe / Africa": "#E0A33C", "Asia-Pacific": "#2A9D8F"}
ZONE_ORDER = ["Americas", "Europe / Africa", "Asia-Pacific"]
REGIONS = ["North America", "Latin America", "Europe", "Africa/MENA", "Asia-Pacific"]
REG_ZONE = {"North America": "Americas", "Latin America": "Americas",
            "Europe": "Europe / Africa", "Africa/MENA": "Europe / Africa",
            "Asia-Pacific": "Asia-Pacific"}


def lon_zone(lon):
    if -170.0 <= lon < -30.0:
        return "Americas"
    if -30.0 <= lon < 60.0:
        return "Europe / Africa"
    return "Asia-Pacific"


# ---------------------------------------------------------------- panel (a)
def panel_a(fig, gs_cell, med):
    mat = med.reindex(index=ROWS_D, columns=COLS_G).values
    nr, nc = mat.shape
    bottom = nr - 1
    gs_a = gs_cell.subgridspec(2, 2, width_ratios=[3.0, 1.0], height_ratios=[1.0, 3.0],
                               wspace=0.06, hspace=0.06)
    ax_top = fig.add_subplot(gs_a[0, 0])
    ax_a = fig.add_subplot(gs_a[1, 0])
    ax_right = fig.add_subplot(gs_a[1, 1])
    ax_corner = fig.add_subplot(gs_a[0, 1]); ax_corner.axis("off")
    fs_a = 13
    vmin, vmax = mat.min() * 0.95, mat.max() * 1.02
    im = ax_a.imshow(mat, aspect="auto", cmap=FIG_PA, vmin=vmin, vmax=vmax)
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            fc = FIG_PA((v - vmin) / (vmax - vmin))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs_a,
                      weight="bold", color="white" if lum < 0.55 else "#111827")
    ax_a.set_xticks(range(nc)); ax_a.set_xticklabels([GRACE_TICK[c] for c in COLS_G], fontsize=fs_a)
    ax_a.set_yticks(range(nr)); ax_a.set_yticklabels([DIST_LABEL[r] for r in ROWS_D], fontsize=fs_a)
    ax_a.set_xlabel("Delay grace (deadline slack)", fontsize=fs_a)
    ax_a.set_ylabel("Compute migration distance", fontsize=fs_a)
    ax_a.grid(False)
    for (ri, ci) in [(bottom, 0), (0, nc - 1)]:
        ax_a.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                                          edgecolor="#111827", linewidth=1.6, zorder=4))
    tag_fx = [mpl.patheffects.withStroke(linewidth=2.4, foreground="#111827")]
    ax_a.text(0, bottom + 0.33, "Status quo", fontsize=fs_a, weight="bold", color="white",
              ha="center", va="center", path_effects=tag_fx, zorder=5)
    ax_a.text(nc - 1, 0.33, "Upper bound", fontsize=fs_a, weight="bold", color="white",
              ha="center", va="center", path_effects=tag_fx, zorder=5)

    pure_t = [mat[bottom, j] for j in range(nc)]
    ax_top.bar(range(nc), [pure_t[0] - v for v in pure_t], color=SECONDARY, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_t):
        d = pure_t[0] - v
        ax_top.text(i, d + 0.1, f"{d:.1f}" if d >= 0.05 else "0.0", ha="center",
                    va="bottom", fontsize=fs_a, color="#111827")
    ax_top.set_ylabel("Reduction vs\nstatus quo\n(temporal only, pp)", fontsize=fs_a)
    P.panel_letter(ax_top, "a", x=-0.26, y=1.02)
    ax_top.tick_params(axis="x", labelbottom=False); ax_top.tick_params(axis="y", labelsize=fs_a)
    ax_top.set_ylim(0, max(pure_t[0] - min(pure_t), 0.1) * 1.3)
    P.soften_axes(ax_top)

    pure_s = [mat[i, 0] for i in range(nr)]
    ax_right.barh(range(nr), [pure_s[bottom] - v for v in pure_s], color=PRIMARY, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_s):
        d = pure_s[bottom] - v
        ax_right.text(d + 0.1, i, f"{d:.1f}" if d >= 0.05 else "0.0", ha="left",
                      va="center", fontsize=fs_a, color="#111827")
    ax_right.set_xlabel("Reduction vs\nstatus quo\n(spatial only, pp)", fontsize=fs_a)
    ax_right.tick_params(axis="y", labelleft=False); ax_right.tick_params(axis="x", labelsize=fs_a)
    ax_right.invert_yaxis()
    ax_right.set_xlim(0, max(pure_s[bottom] - min(pure_s), 0.1) * 1.35)
    P.soften_axes(ax_right)

    cax = ax_corner.inset_axes([0.08, 0.48, 0.86, 0.15])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Median uncovered\ndemand share (%)", fontsize=fs_a)
    cb.ax.tick_params(labelsize=fs_a)
    return ax_a


# ---------------------------------------------------------------- panel (d) map
def panel_d_map(fig, ax_b, flows, cen):
    ax_b.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax_b.spines["geo"].set_visible(False); ax_b.set_frame_on(False); ax_b.set_facecolor("white")
    fplt.add_geometries(ax_b, fshp.get_countries(), fc="#EDF0F3", ec="white",
                        lw=0.3, crs=fplt.PLATE_CARREE, zorder=1)
    # keep each sender's top-2 receiver arcs
    e = flows[flows["sender_country"] != flows["receiver_country"]].copy()
    e = e.sort_values("weight", ascending=False).groupby("sender_country").head(2)
    e = e[e["weight"] > 0.01].copy()
    e["zone"] = e["recv_lon"].map(lon_zone)
    wmax = float(np.percentile(e["weight"], 97)) if len(e) else 1.0
    e["wc"] = e["weight"].clip(upper=wmax)
    e["lw"] = 0.25 + (e["wc"] / wmax) * 1.6
    e["alpha"] = 0.20 + (e["wc"] / wmax) * 0.55
    e["color"] = e["zone"].map(ZONE_COLOR).fillna("#9CA3AF")
    geo = Geod(ellps="WGS84"); geod = ccrs.Geodetic(); K = 10
    for r in e.sort_values("weight").itertuples():
        # thick at RECEIVER end (clean-surplus source) -> thin at sender end
        pts = ([(r.recv_lon, r.recv_lat)]
               + geo.npts(r.recv_lon, r.recv_lat, r.send_lon, r.send_lat, K - 1)
               + [(r.send_lon, r.send_lat)])
        for k in range(K):
            frac = k / (K - 1)
            ax_b.plot([pts[k][0], pts[k + 1][0]], [pts[k][1], pts[k + 1][1]],
                      color=r.color, linewidth=r.lw * (1.0 - 0.55 * frac), alpha=r.alpha,
                      solid_capstyle="round", transform=geod, zorder=3)
    # receiver hubs (size = inbound weight), coloured by longitude zone
    rt = flows.groupby("receiver_country").agg(
        wt=("weight", "sum"), lon=("recv_lon", "first"), lat=("recv_lat", "first")).reset_index()
    rt = rt[rt["wt"] > 0.05]
    smax = float(rt["wt"].max()) if len(rt) else 1.0
    rt["color"] = rt["lon"].map(lon_zone).map(ZONE_COLOR).fillna("#9CA3AF")
    ax_b.scatter(rt["lon"], rt["lat"], s=45 + (rt["wt"] / smax) * 255, c=list(rt["color"]),
                 edgecolor="white", linewidth=0.6, alpha=0.95, zorder=6, transform=ccrs.PlateCarree())
    # sender dots (small)
    sd = flows.drop_duplicates("sender_country")
    ax_b.scatter(sd["send_lon"], sd["send_lat"], s=12, marker="o",
                 c=[ZONE_COLOR.get(lon_zone(x), "#9CA3AF") for x in sd["send_lon"]],
                 edgecolor="white", linewidth=0.4, alpha=0.9, zorder=5, transform=ccrs.PlateCarree())
    # label top receiver hubs
    dodge = {"CL": (-3, -8), "MX": (-4, -8), "NZ": (5, -6), "NO": (-7, 7), "US": (-2, -7.5),
             "SE": (1, 8.5), "BR": (3, -8), "JP": (5, -6), "CA": (-2, -7.5), "FI": (9, 6)}
    for cc in rt.sort_values("wt", ascending=False)["receiver_country"].head(8):
        if cc not in cen:
            continue
        lon, lat = cen[cc]; dx, dy = dodge.get(cc, (0, -6))
        ax_b.text(lon + dx, lat + dy, cc, fontsize=13, weight="bold", color="#1F2937",
                  ha="center", va="top", transform=ccrs.PlateCarree(), zorder=11,
                  path_effects=[mpl.patheffects.withStroke(linewidth=2.2, foreground="white")])
    P.panel_letter(ax_b, "d", x=-0.01, y=1.0)
    leg = [plt.Line2D([], [], color="#9AA7B4", lw=3.0, label="Flow: thick = receiver, thin = sender"),
           plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                      markeredgecolor="white", markersize=12, label="Receiver hub (size = compute in)"),
           plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                      markeredgecolor="white", markersize=5, label="Sender country (small)")]
    zoneh = [mpatches.Patch(facecolor=ZONE_COLOR[z], label=z) for z in ZONE_ORDER]
    ax_b.legend(handles=leg + zoneh, loc="upper left", bbox_to_anchor=(0.0, -0.02), fontsize=13,
                frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.0, borderpad=0.0,
                title="Colour = receiver longitude zone", title_fontsize=13)


# ---------------------------------------------------------------- panel (b) chord
def panel_b_chord(fig, gs, ax_a, flows, region_map, grid):
    ax_ch = fig.add_subplot(gs[1, :])
    flows = flows.copy()
    flows["s_reg"] = flows["sender_country"].map(region_map)
    rt = flows.groupby("receiver_country")["weight"].sum().sort_values(ascending=False)
    cen = {row.iso2: (float(row.lon), float(row.lat))
           for row in pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""]).itertuples()}
    hubs = [h for h in rt.index if h in cen][:12]
    ri = {r: i for i, r in enumerate(REGIONS)}
    hub_i = {h: i for i, h in enumerate(hubs)}
    M = np.zeros((len(hubs), len(REGIONS)))
    for row in flows.itertuples():
        if row.receiver_country in hub_i and row.s_reg in ri:
            M[hub_i[row.receiver_country]][ri[row.s_reg]] += row.weight
    hub_zone = {h: lon_zone(cen[h][0]) for h in hubs}
    zrank = {"Americas": 0, "Europe / Africa": 1, "Asia-Pacific": 2}
    order = sorted(range(len(hubs)), key=lambda i: (zrank[hub_zone[hubs[i]]], -M[i].sum()))
    hubs = [hubs[i] for i in order]; M = M[order]
    sup_color = [ZONE_COLOR[hub_zone[h]] for h in hubs]
    reg_color = [ZONE_COLOR[REG_ZONE[r]] for r in REGIONS]
    reg_lab = ["N.America", "L.America", "Europe", "Africa/MENA", "Asia-Pacific"]
    P.draw_bipartite_chord(ax_ch, M, hubs, sup_color, reg_lab, reg_color, sup_color,
                           lim=1.30, gap_bottom=0.58, label_size=14)

    # outer ring: cross-country mean migrated compute share (%) as distance widens (g=0)
    g0 = grid[grid["grace_h"] == 0]
    mig = (g0.groupby("dist_ring")["migrated_share"].mean() * 100).reindex(["D0", "D1", "D2", "D3"]).values
    div_lab = ["D0", "D1", "D2", "D3"]
    div_col = ["#94A3B8", "#7DA9C9", "#5A86B0", "#274C77"]
    dmax = max(float(np.max(mig)), 1e-6); base = 0.86
    for v, ang, col, sl in zip(mig, np.radians(np.linspace(-99, -81, 4)), div_col, div_lab):
        ln = 0.05 + (v / dmax) * 0.26
        ax_ch.plot([base * np.cos(ang), (base + ln) * np.cos(ang)],
                   [base * np.sin(ang), (base + ln) * np.sin(ang)],
                   color=col, lw=12, solid_capstyle="round", zorder=8)
        ax_ch.text((base + ln + 0.07) * np.cos(ang), (base + ln + 0.07) * np.sin(ang),
                   f"{v:.0f}%\n{sl}", ha="center", va="center", fontsize=11, weight="bold",
                   color=col, zorder=9, linespacing=0.95)
    zoneh = [mpatches.Patch(facecolor=ZONE_COLOR[z], label=z) for z in ZONE_ORDER]
    ax_ch.legend(handles=zoneh, loc="upper left", bbox_to_anchor=(0.41, 0.95),
                 bbox_transform=fig.transFigure, fontsize=14, frameon=False,
                 handletextpad=0.6, labelspacing=0.45)
    ax_ch.set_xlim(-1.27, 1.27); ax_ch.set_ylim(-1.40, 1.14); ax_ch.set_aspect("auto")
    return ax_ch


# ---------------------------------------------------------------- panel (e) scatter
def draw_compute_followsun(ax, ax_top, ax_right):
    d = pd.read_csv(DATA / "r2_st_followsun.csv")
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
    ax.text(17.2, 17.7, "run-at-home (slope +1.00)", fontsize=FS, color="#6B7280",
            ha="left", va="bottom", style="italic", rotation=45, rotation_mode="anchor")
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
    ax.legend(loc="lower right", fontsize=FS - 1, frameon=False, labelspacing=0.4,
              handletextpad=0.4, borderpad=0.2)
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


# ---------------------------------------------------------------- panel (c) violin
def draw_dist_distribution(ax, grid):
    FS = 13
    labels = ["D0\nIn-country", "D1\n≤500 km", "D2\n≤3000 km", "D3\nGlobal"]
    g0 = grid[grid["grace_h"] == 0]
    data = [(g0[g0["dist_ring"] == s]["uncovered_share"] * 100).values for s in ["D0", "D1", "D2", "D3"]]
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
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=FS - 1)
    ax.set_ylabel("Residual uncovered share (%)", fontsize=FS)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 62); ax.tick_params(axis="y", labelsize=FS)
    ax.text(0.98, 0.97, "g = 0 (no temporal deferral)", transform=ax.transAxes,
            fontsize=FS - 3, color="#6B7280", ha="right", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", color="#EEF1F4", lw=0.5); ax.set_axisbelow(True)


def main():
    g = pd.read_csv(DATA / "r2_spatiotemporal_grid.csv")
    med = (g.groupby(["dist_ring", "grace_h"]).uncovered_share.median().unstack() * 100)
    flows = pd.read_csv(DATA / "r2_st_compute_flows.csv")
    region_map = g.drop_duplicates("country").set_index("country")["region"].to_dict()
    cen = {row.iso2: (float(row.lon), float(row.lat))
           for row in pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""]).itertuples()}

    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(2, 2, left=0.015, right=0.98, top=0.95, bottom=0.04,
                          hspace=0.16, wspace=0.18, width_ratios=[1.0, 1.65], height_ratios=[0.46, 1.0])
    ax_a = panel_a(fig, gs[0, 0], med)
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=10))
    panel_d_map(fig, ax_b, flows, cen)
    ax_ch = panel_b_chord(fig, gs, ax_a, flows, region_map, g)

    # ---- layout (verbatim from original plot_fig2): chord top-right, map bottom-left,
    #      scatter bottom-right, violin left-middle ----
    _bw, _bh = 0.591, 0.50
    _top = gs[0, 1].get_position(fig); _bot = gs[1, :].get_position(fig)
    ax_ch.set_position([0.985 - _bw, _top.y1 - _bh - 0.06, _bw, _bh])
    chp = ax_ch.get_position()
    fig.text(chp.x0 - 0.03, _top.y1, "b", fontsize=20, fontweight="bold", ha="left", va="bottom")

    d_w = 0.355; d_h = d_w * fig.get_figwidth() / fig.get_figheight()
    bx0 = 0.985 - d_w; by0 = _bot.y0 + 0.02; gap = 0.006
    s_x = d_w * 0.87; s_y = s_x * fig.get_figwidth() / fig.get_figheight()
    mh = d_h - gap - s_y; mw = d_w - gap - s_x
    ax_d = fig.add_axes([bx0, by0, s_x, s_y])
    ax_dtop = fig.add_axes([bx0, by0 + s_y + gap, s_x, mh])
    ax_dright = fig.add_axes([bx0 + s_x + gap, by0, mw, s_y])
    draw_compute_followsun(ax_d, ax_dtop, ax_dright)
    fig.text(bx0 - 0.10 * d_w, by0 + d_h, "e", fontsize=20, fontweight="bold", ha="left", va="bottom")

    ax_b.set_position([0.012, 0.04, 0.591, 0.30])
    mb = ax_b.get_position(); ap = ax_a.get_position()
    e_x0, e_x1 = 0.012, chp.x0 - 0.004
    e_y0, e_y1 = mb.y1 + 0.055, ap.y0 - 0.055
    ax_e = fig.add_axes([e_x0, e_y0, e_x1 - e_x0, e_y1 - e_y0])
    draw_dist_distribution(ax_e, g)
    ep = ax_e.get_position()
    fig.text(ep.x0 - 0.015, ep.y1, "c", fontsize=20, fontweight="bold", ha="left", va="bottom")

    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig2_st_full{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_st_full")


if __name__ == "__main__":
    main()
