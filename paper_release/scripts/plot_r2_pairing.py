#!/usr/bin/env python3
"""Fig 3: sender -> receiver compute-migration PAIRING across latency-tolerance scenarios.

Four rows (latency tolerance tau = 100 / 150 / 200 ms and global reach) x two columns:
  left  = flow map        : where each sender's movable compute is routed
  right = bipartite chord  : receiver hubs x sender regions (the same pairing, aggregated)
As tau loosens, the round-trip-time gate widens and the pairing grows from regional to
global. This is the b/d content split out of Fig 2 and expanded to several scenarios.

Data (data_globalsites_stations_expanded/):
  r2_st_compute_flows_tau{100,150,200}.csv  + r2_st_compute_flows.csv (global)
  r2_latency_grid.csv  (mean routed share annotation, phi = 100%)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Geod

import plot_cfe_geographic_portfolio_ai as P
import plot_r2_latency_fig2 as F2
import cartopy.crs as ccrs
import frykit.plot as fplt
import frykit.shp as fshp

warnings.filterwarnings("ignore")
P.set_style()

DATA, FIG = F2.DATA, F2.FIG
ZONE_COLOR, ZONE_ORDER = F2.ZONE_COLOR, F2.ZONE_ORDER
REGIONS, REG_ZONE = F2.REGIONS, F2.REG_ZONE
lon_zone = F2.lon_zone

# (row title, flow file, grid tau key for routed-share annotation, reach caption)
SCEN = [
    ("τ = 100 ms", "r2_wf_compute_flows_tau100.csv", 100, "≈3 Mm · regional"),
    ("τ = 150 ms", "r2_wf_compute_flows_tau150.csv", 150, "≈6.7 Mm · cross-continental"),
    ("τ = 200 ms", "r2_wf_compute_flows_tau200.csv", 200, "≈10 Mm · half the globe"),
    ("τ ≥ 300 ms", "r2_wf_compute_flows.csv", 500, "global reach"),
]

DODGE = {"CL": (-3, -8), "MX": (-4, -8), "NZ": (6, -5), "NO": (-6, 7), "US": (-3, -7),
         "SE": (2, 8), "BR": (4, -8), "JP": (6, -5), "CA": (-3, -7), "FI": (9, 6),
         "DK": (3, 8), "ES": (-6, -7), "IN": (5, -6), "AU": (5, -7), "ZA": (2, -8),
         "AR": (-3, -8), "PE": (-5, -6), "DE": (-2, 8), "GB": (-7, 5), "FR": (-5, -7)}


def draw_flow_map(ax, flows, cen, show_legend):
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False); ax.set_frame_on(False); ax.set_facecolor("white")
    fplt.add_geometries(ax, fshp.get_countries(), fc="#EDF0F3", ec="white", lw=0.3,
                        crs=fplt.PLATE_CARREE, zorder=1)
    e = flows[flows["sender_country"] != flows["receiver_country"]].copy()
    e = e.sort_values("weight", ascending=False).groupby("sender_country").head(2)
    e = e[e["weight"] > 0.01].copy()
    e["zone"] = e["recv_lon"].map(lon_zone)
    wmax = float(np.percentile(e["weight"], 97)) if len(e) else 1.0
    e["wc"] = e["weight"].clip(upper=wmax)
    e["lw"] = 0.25 + (e["wc"] / wmax) * 1.5
    e["alpha"] = 0.20 + (e["wc"] / wmax) * 0.55
    e["color"] = e["zone"].map(ZONE_COLOR).fillna("#9CA3AF")
    geo = Geod(ellps="WGS84"); geod = ccrs.Geodetic(); K = 10
    for r in e.sort_values("weight").itertuples():
        pts = ([(r.recv_lon, r.recv_lat)] + geo.npts(r.recv_lon, r.recv_lat, r.send_lon, r.send_lat, K - 1)
               + [(r.send_lon, r.send_lat)])
        for k in range(K):
            frac = k / (K - 1)
            ax.plot([pts[k][0], pts[k + 1][0]], [pts[k][1], pts[k + 1][1]], color=r.color,
                    linewidth=r.lw * (1.0 - 0.55 * frac), alpha=r.alpha, solid_capstyle="round",
                    transform=geod, zorder=3)
    rt = flows.groupby("receiver_country").agg(wt=("weight", "sum"), lon=("recv_lon", "first"),
                                               lat=("recv_lat", "first")).reset_index()
    rt = rt[rt["wt"] > 0.05]
    smax = float(rt["wt"].max()) if len(rt) else 1.0
    rt["color"] = rt["lon"].map(lon_zone).map(ZONE_COLOR).fillna("#9CA3AF")
    ax.scatter(rt["lon"], rt["lat"], s=30 + (rt["wt"] / smax) * 210, c=list(rt["color"]),
               edgecolor="white", linewidth=0.6, alpha=0.95, zorder=6, transform=ccrs.PlateCarree())
    sd = flows.drop_duplicates("sender_country")
    ax.scatter(sd["send_lon"], sd["send_lat"], s=9, marker="o",
               c=[ZONE_COLOR.get(lon_zone(x), "#9CA3AF") for x in sd["send_lon"]],
               edgecolor="white", linewidth=0.3, alpha=0.9, zorder=5, transform=ccrs.PlateCarree())
    for cc in rt.sort_values("wt", ascending=False)["receiver_country"].head(5):
        if cc not in cen:
            continue
        lon, lat = cen[cc]; dx, dy = DODGE.get(cc, (0, -6))
        ax.text(lon + dx, lat + dy, cc, fontsize=8.5, weight="bold", color="#1F2937", ha="center",
                va="top", transform=ccrs.PlateCarree(), zorder=11,
                path_effects=[mpl.patheffects.withStroke(linewidth=1.8, foreground="white")])
    if show_legend:
        leg = [plt.Line2D([], [], color="#9AA7B4", lw=3.0, label="Flow: thick = receiver, thin = sender"),
               plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                          markeredgecolor="white", markersize=11, label="Receiver hub (size = compute in)"),
               plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                          markeredgecolor="white", markersize=5, label="Sender country")]
        zoneh = [mpatches.Patch(facecolor=ZONE_COLOR[z], label=z) for z in ZONE_ORDER]
        ax.legend(handles=leg + zoneh, loc="upper left", bbox_to_anchor=(0.0, -0.04), fontsize=9,
                  frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.0, borderpad=0.0,
                  title="Colour = receiver longitude zone", title_fontsize=9)


def draw_pairing_chord(ax, flows, region_map, cen, mean_share):
    flows = flows.copy()
    flows["s_reg"] = flows["sender_country"].map(region_map)
    rt = flows.groupby("receiver_country")["weight"].sum().sort_values(ascending=False)
    hubs = [h for h in rt.index if h in cen][:8]
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
    reg_lab = ["N.Am", "L.Am", "Europe", "Afr/MENA", "Asia-Pac"]
    P.draw_bipartite_chord(ax, M, hubs, sup_color, reg_lab, reg_color, sup_color,
                           lim=1.34, gap_bottom=0.5, label_size=9)
    ax.text(0, -1.34, f"mean routed share {mean_share:.0f}%", ha="center", va="center",
            fontsize=10, weight="bold", color="#374151")
    ax.set_xlim(-1.34, 1.34); ax.set_ylim(-1.46, 1.18); ax.set_aspect("equal")


def main():
    cen = {row.iso2: (float(row.lon), float(row.lat))
           for row in pd.read_csv(F2.COUNTRY_META, keep_default_na=False, na_values=[""]).itertuples()}
    g = pd.read_csv(DATA / "r2_waveform_grid.csv")
    region_map = g.drop_duplicates("country").set_index("country")["region"].to_dict()
    g1 = g[g["phi"] == 1.0]

    fig = plt.figure(figsize=(12.0, 20.0))
    gs = fig.add_gridspec(4, 2, left=0.015, right=0.985, top=0.965, bottom=0.04,
                          hspace=0.10, wspace=0.03, width_ratios=[1.18, 1.0])
    letters = "abcdefgh"
    for i, (tlab, fname, gtau, reach) in enumerate(SCEN):
        flows = pd.read_csv(DATA / fname)
        ax_m = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson(central_longitude=10))
        draw_flow_map(ax_m, flows, cen, show_legend=(i == len(SCEN) - 1))
        P.panel_letter(ax_m, letters[2 * i], x=0.0, y=1.02, size=17)
        ax_m.text(0.085, 1.072, f"{tlab}", transform=ax_m.transAxes, fontsize=12, weight="bold",
                  color="#111827", ha="left", va="top")
        ax_m.text(0.085, 1.018, reach, transform=ax_m.transAxes, fontsize=9.5,
                  color="#6B7280", ha="left", va="top")
        mean_share = float(g1[g1["tau_ms"] == gtau]["migrated_share"].mean()) * 100
        ax_c = fig.add_subplot(gs[i, 1])
        draw_pairing_chord(ax_c, flows, region_map, cen, mean_share)
        P.panel_letter(ax_c, letters[2 * i + 1], x=-0.02, y=1.02, size=17)
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig3_pairing{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_pairing")


if __name__ == "__main__":
    main()
