#!/usr/bin/env python3
"""Publication figures for the cfe_geographic_portfolio_ai report.

Three multi-panel figures following the academic-paper-figures skill:
  fig1  National-level demand-supply mismatch (R1)
  fig2  Cross-country supply portfolio scenarios (R2)
  fig3  AI demand perturbations vs storage cost (R3)

Style rules
-----------
* English-only text in axes, legends, panel titles, annotations.
* Palette: #2563EB (primary), #F59E0B (secondary), #DC2626 (highlight),
  #64748B (neutral), #059669 (positive supporting).
* Clean white background; top and right spines removed; subtle gridlines.
* Panel titles are domain claims, not metric names.
* Consistent line widths/marker sizes; no 3D, no gradients, no decoration.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import frykit.plot as fplt
import frykit.shp as fshp
from shapely.geometry import Point
from shapely.strtree import STRtree

warnings.filterwarnings("error")  # promote font-fallback etc. so we can fix them
# allow cartopy's harmless download warning
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
warnings.filterwarnings("ignore", message=".*facecolor will have no effect.*")

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "collective_attention_research_plan"
    / "reports"
    / "cfe_geographic_portfolio_ai"
)
DATA = REPORT / "data"
COUNTRY_META = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites" / "country_meta.csv"
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette (academic-paper-figures skill defaults)
PRIMARY = "#2563EB"
SECONDARY = "#F59E0B"
HIGHLIGHT = "#DC2626"
NEUTRAL = "#64748B"
POSITIVE = "#059669"

PV_COLOR = "#F59E0B"    # warm = solar PV (circle markers)
WIND_COLOR = "#2563EB"  # cool = wind (triangle markers)
EVENING_KW = dict(color=NEUTRAL, alpha=0.12)  # evening window 18-23 h shading

# shared two-pole heat ramp (lilac -> purple -> amber), matching Fig. 3 a/e — used
# for every uncovered-share / deficit map so the whole figure set is colour-consistent.
FIG_PA = mpl.colors.LinearSegmentedColormap.from_list(
    "fig_pa", ["#F2EFF7", "#B6A8D0", "#8A6FB0", "#C75D88", "#E8A33D"])

# Okabe-Ito colorblind-safe region palette
REGION_COLORS = {
    "Africa/MENA": "#E69F00",
    "Asia-Pacific": "#D55E00",
    "Europe": "#0072B2",
    "Latin America": "#009E73",
    "North America": "#CC79A7",
}

POWER_SCOPES = ["L0", "L1", "L2", "L3"]
WORKLOAD_SCOPES = ["W0", "W1", "W2", "W3"]
POWER_LABEL = {
    "L0": "L0 National\n(own VRE only)",
    "L1": "L1 ≤1500 km",
    "L2": "L2 ≤3000 km",
    "L3": "L3 Global",
}
WORKLOAD_LABEL = {
    "W0": "W0 None\n(0% migratable)",
    "W1": "W1 D batch\n(~10%, τ=∞)",
    "W2": "W2 C+D\n(~30%, τ≥60s)",
    "W3": "W3 B+C+D\n(~60%, τ≥1.5s)",
}
SCENARIO_2D_LABEL = {
    f"{p}_{w}": f"{p} × {w}" for p in POWER_SCOPES for w in WORKLOAD_SCOPES
}
# R3 corner scenarios used by Fig 3
R3_CORNER_SCENARIOS = ["L0_W0", "L3_W0", "L0_W3", "L3_W3"]
R3_CORNER_LABEL = {
    "L0_W0": "Status quo (L0 × W0)",
    "L3_W0": "Power-only (L3 × W0)",
    "L0_W3": "Workload-only (L0 × W3)",
    "L3_W3": "Both (L3 × W3)",
}
R3_CORNER_COLOR = {
    "L0_W0": NEUTRAL,
    "L3_W0": PRIMARY,
    "L0_W3": SECONDARY,
    "L3_W3": POSITIVE,
}

PERTURB_ORDER = ["P0_baseline", "P1_flatten", "P2_sharpen", "P2_emp", "P3_burst", "P_mix"]
PERTURB_LABEL = {
    "P0_baseline": "P0 Baseline",
    "P1_flatten": "P1 Flatten",
    "P2_sharpen": "P2 Sharpen (Gaussian)",
    "P2_emp": "P2_emp (BurstGPT shape)",
    "P3_burst": "P3 Burst-amplify",
    "P_mix": "P_mix Composite",
}
PERTURB_COLOR = {
    "P0_baseline": "#111827",
    "P1_flatten": POSITIVE,
    "P2_sharpen": SECONDARY,
    "P2_emp": "#B45309",   # darker amber - empirical-shape variant of P2
    "P3_burst": HIGHLIGHT,
    "P_mix": "#7C3AED",
}


# ---------- global style ----------
def set_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
    })


def soften_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#E5E7EB", linewidth=0.6, alpha=0.7)


def panel_letter(ax, letter, x=-0.06, y=1.04, size=20):
    """Bare bold lowercase panel letter, uniform across figures. ``size`` lets a
    narrower/wider figure match the others' on-page letter size."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", ha="left", va="bottom")


def _arc_xy(r, a0, a1, n=12):
    t = np.linspace(a0, a1, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def draw_chord(ax, M, labels, node_color, gap_deg=5.0, R=1.0, band=0.055,
               r_rib=0.93, label_color="#1F2937", label_pad=0.16):
    """Directed chord diagram. ``M[i][j]`` is the supply weight flowing from
    region i to region j. Node arc length is proportional to total supplied
    (row sum); each cross-region ribbon is tapered by direction and coloured by
    the dominant supplier. The within-region (diagonal) weight is left as an
    un-ribboned lighter arc segment, so a region's "stays home" share reads as
    the bare part of its arc."""
    import matplotlib.path as mpath
    M = np.asarray(M, float)
    n = len(M)
    out = M.sum(1)
    total = out.sum()
    gap = np.radians(gap_deg)
    span = out / total * (2 * np.pi - n * gap)

    node_a = np.zeros((n, 2))
    sub = [[None] * n for _ in range(n)]   # sub[i][j] = (ang0, ang1)
    a = np.pi / 2.0
    for i in range(n):
        node_a[i] = (a, a - span[i])
        s = a
        for j in range(n):
            frac = (M[i][j] / out[i]) if out[i] > 0 else 0.0
            sub[i][j] = (s, s - frac * span[i])
            s -= frac * span[i]
        a = a - span[i] - gap

    # node arc bands + labels
    for i in range(n):
        outer = _arc_xy(R, node_a[i][0], node_a[i][1])
        inner = _arc_xy(R - band, node_a[i][1], node_a[i][0])
        ax.add_patch(mpatches.Polygon(np.vstack([outer, inner]), closed=True,
                     facecolor=node_color[i], edgecolor="white", linewidth=0.6,
                     zorder=6))
        # bare diagonal (within-region) segment: thin desaturated overlay
        d0, d1 = sub[i][i]
        do = _arc_xy(R, d0, d1)
        di = _arc_xy(R - band, d1, d0)
        ax.add_patch(mpatches.Polygon(np.vstack([do, di]), closed=True,
                     facecolor="white", edgecolor="none", alpha=0.45, zorder=7))
        mid = node_a[i].mean()
        ax.text((R + label_pad) * np.cos(mid), (R + label_pad) * np.sin(mid),
                labels[i], ha="left" if np.cos(mid) >= 0 else "right",
                va="center", fontsize=9.5, weight="bold", color=label_color,
                zorder=9)

    Path = mpath.Path
    for i in range(n):
        for j in range(n):
            if j <= i or (M[i][j] + M[j][i]) <= 0:
                continue
            ai = _arc_xy(r_rib, *sub[i][j], n=10)
            aj = _arc_xy(r_rib, *sub[j][i], n=10)
            verts, codes = [tuple(ai[0])], [Path.MOVETO]
            for p in ai[1:]:
                verts.append(tuple(p)); codes.append(Path.LINETO)
            verts += [(0, 0), tuple(aj[0])]; codes += [Path.CURVE3, Path.CURVE3]
            for p in aj[1:]:
                verts.append(tuple(p)); codes.append(Path.LINETO)
            verts += [(0, 0), tuple(ai[0])]; codes += [Path.CURVE3, Path.CURVE3]
            src = i if M[i][j] >= M[j][i] else j
            ax.add_patch(mpatches.PathPatch(
                Path(verts, codes), facecolor=node_color[src], edgecolor="none",
                alpha=0.50, zorder=3))
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.40, 1.40)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_chord_circos(ax, M, order, node_color, node_label, R=1.0, band=0.045,
                      r_rib=0.955, gap_total=0.16, min_ribbon=0.05,
                      label_color="#1F2937", lim=1.34):
    """Circos-style directed chord for many (country) nodes. ``M[i][j]`` is the
    supply weight from node i to node j; nodes are placed in ``order`` around the
    circle (ordered by longitude). Arc length is proportional to total
    involvement (supply + demand). Ribbons with weight >= ``min_ribbon`` are
    drawn, coloured by the supplier's node colour; only nodes with a non-empty
    ``node_label`` get a radially rotated label."""
    import matplotlib.path as mpath
    M = np.asarray(M, float)
    n = len(order)
    inv = M.sum(1) + M.sum(0)
    total = float(sum(inv[i] for i in order))
    gap = gap_total * 2 * np.pi / n
    avail = 2 * np.pi - n * gap

    arc = {}
    a = np.pi / 2
    for idx in order:
        sp = inv[idx] / total * avail if total > 0 else 0.0
        arc[idx] = (a, a - sp)
        a = a - sp - gap
    # ribbon endpoint sub-segments: outgoing (to each j) then incoming (from j)
    seg = {}
    for idx in order:
        s = arc[idx][0]
        for kind, getv in (("o", lambda j: M[idx][j]), ("i", lambda j: M[j][idx])):
            for j in order:
                v = getv(j)
                if v > 0:
                    w = v / total * avail
                    seg[(idx, kind, j)] = (s, s - w)
                    s -= w

    Path = mpath.Path
    pairs = [(i, j) for i in order for j in order
             if i != j and M[i][j] >= min_ribbon
             and (i, "o", j) in seg and (j, "i", i) in seg]
    pairs.sort(key=lambda p: M[p[0]][p[1]])   # weak ribbons first
    wmax = max((M[i][j] for i, j in pairs), default=1.0)
    for i, j in pairs:
        ai = _arc_xy(r_rib, *seg[(i, "o", j)], n=6)
        aj = _arc_xy(r_rib, *seg[(j, "i", i)], n=6)
        verts, codes = [tuple(ai[0])], [Path.MOVETO]
        for p in ai[1:]:
            verts.append(tuple(p)); codes.append(Path.LINETO)
        verts += [(0, 0), tuple(aj[0])]; codes += [Path.CURVE3, Path.CURVE3]
        for p in aj[1:]:
            verts.append(tuple(p)); codes.append(Path.LINETO)
        verts += [(0, 0), tuple(ai[0])]; codes += [Path.CURVE3, Path.CURVE3]
        alpha = 0.16 + 0.50 * min(M[i][j] / wmax, 1.0)
        ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                     facecolor=node_color[i], edgecolor="none", alpha=alpha,
                     zorder=3))

    for idx in order:
        a0, a1 = arc[idx]
        outer = _arc_xy(R, a0, a1, 8)
        inner = _arc_xy(R - band, a1, a0, 8)
        ax.add_patch(mpatches.Polygon(np.vstack([outer, inner]), closed=True,
                     facecolor=node_color[idx], edgecolor="white", linewidth=0.3,
                     zorder=6))
        lab = node_label[idx]
        if lab:
            mid = (a0 + a1) / 2
            deg = np.degrees(mid) % 360
            rot, ha = (deg + 180, "right") if 90 < deg < 270 else (deg, "left")
            ax.text((R + 0.04) * np.cos(mid), (R + 0.04) * np.sin(mid), lab,
                    rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                    fontsize=10, weight="bold", color=label_color, zorder=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_bipartite_chord(ax, M, sup_labels, sup_color, reg_labels, reg_color,
                         ribbon_color, gap_group=0.06, gap_node=0.006, R=1.0,
                         band=0.055, r_rib=0.945, label_color="#1F2937", lim=1.40,
                         gap_bottom=None, label_size=13):
    """Bipartite directed chord. Rows of ``M`` are supply hubs, columns are
    demand regions; ``M[i][r]`` is the weight hub i sends to region r. Supply
    hubs occupy the first half of the circle and demand regions the second half,
    so every ribbon crosses from a hub to a region; ribbons are coloured by
    ``ribbon_color[i]`` (the hub's longitude zone)."""
    import matplotlib.path as mpath
    M = np.asarray(M, float)
    nS, nR = M.shape
    vals = list(M.sum(1)) + list(M.sum(0))
    valsum = float(sum(vals))
    n = nS + nR
    if gap_bottom is None:
        gap_bottom = gap_group
    avail = 2 * np.pi - (gap_group + gap_bottom + (n - 2) * gap_node)

    arcs = []
    a = np.pi / 2
    for k in range(n):
        sp = vals[k] / valsum * avail if valsum > 0 else 0.0
        arcs.append((a, a - sp))
        g = gap_bottom if k == nS - 1 else (gap_group if k == n - 1 else gap_node)
        a -= sp + g

    sup_seg, reg_seg = {}, {}
    for i in range(nS):
        s = arcs[i][0]
        for r in range(nR):
            w = M[i][r] / valsum * avail
            sup_seg[(i, r)] = (s, s - w); s -= w
    for r in range(nR):
        s = arcs[nS + r][0]
        for i in range(nS):
            w = M[i][r] / valsum * avail
            reg_seg[(i, r)] = (s, s - w); s -= w

    Path = mpath.Path
    wmax = float(M.max()) if M.size else 1.0
    # full-width band per flow; only the receiving (region) end is capped with a
    # triangle pointing at the region, so hub -> region direction reads clearly.
    r_in = r_rib - 0.075  # region end pulled inward so the (gentle) triangle stays
    for i in range(nS):    # in the open interior, not hidden under the node band
        for r in range(nR):
            if M[i][r] <= 0:
                continue
            ai = _arc_xy(r_rib, *sup_seg[(i, r)], n=8)
            j0, j1 = reg_seg[(i, r)]
            c1 = (r_in * np.cos(j0), r_in * np.sin(j0))
            c2 = (r_in * np.cos(j1), r_in * np.sin(j1))
            tip = (r_rib * np.cos((j0 + j1) / 2.0), r_rib * np.sin((j0 + j1) / 2.0))
            verts, codes = [tuple(ai[0])], [Path.MOVETO]
            for p in ai[1:]:
                verts.append(tuple(p)); codes.append(Path.LINETO)
            verts += [(0, 0), c1]; codes += [Path.CURVE3, Path.CURVE3]
            verts += [tip, c2]; codes += [Path.LINETO, Path.LINETO]
            verts += [(0, 0), tuple(ai[0])]; codes += [Path.CURVE3, Path.CURVE3]
            ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                         facecolor=ribbon_color[i], edgecolor="none",
                         alpha=0.34 + 0.42 * min(M[i][r] / wmax, 1.0), zorder=3))

    allcol = list(sup_color) + list(reg_color)
    alllab = list(sup_labels) + list(reg_labels)
    for k in range(n):
        a0, a1 = arcs[k]
        if abs(a0 - a1) < 1e-9:
            continue
        outer = _arc_xy(R, a0, a1, 8)
        inner = _arc_xy(R - band, a1, a0, 8)
        ax.add_patch(mpatches.Polygon(np.vstack([outer, inner]), closed=True,
                     facecolor=allcol[k], edgecolor="white", linewidth=0.5,
                     zorder=6))
        mid = (a0 + a1) / 2
        deg = np.degrees(mid) % 360
        rot, ha = (deg + 180, "right") if 90 < deg < 270 else (deg, "left")
        ax.text((R + 0.04) * np.cos(mid), (R + 0.04) * np.sin(mid), alllab[k],
                rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=label_size, weight="bold", color=label_color, zorder=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------- Figure 1 ----------
PANEL_PATH = (
    ROOT
    / "collective_attention_research_plan"
    / "reports"
    / "prime_time_cfe_penalty_ptp_r2_r3"
    / "ptp_country_hourly_cfe_panel.csv.gz"
)


def compute_local_hour_mismatch_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a (country x 24) matrix of mean uncovered intensity per local hour."""
    df = panel.copy()
    # equal-energy normalise within each country
    g = df.groupby("cf_location")
    df["d"] = g["demand_scaled"].transform(lambda x: x)
    df["s"] = g["supply_vre_cf"].transform(lambda x: x)
    # rescale s so mean(s) == mean(d) per country
    g2 = df.groupby("cf_location")
    s_mean = g2["s"].transform("mean")
    d_mean = g2["d"].transform("mean")
    ratio = (d_mean / s_mean.where(s_mean > 0)).fillna(1.0)
    df["s_eq"] = df["s"] * ratio
    df["unmet"] = np.maximum(df["d"] - df["s_eq"], 0)
    mat = (df.groupby(["cf_location", "local_hour"])["unmet"].mean()
             .unstack("local_hour").reindex(columns=range(24)))
    return mat


def compute_fingerprint_table(r1: pd.DataFrame, dn: pd.DataFrame) -> pd.DataFrame:
    """Per-country fingerprint with 5 dimensions for the heatmap in panel (d)."""
    rows = []
    for cc, row in r1.set_index("country").iterrows():
        sub = dn[dn["country"] == cc].set_index("local_hour")
        if sub.empty:
            continue
        d_prof = sub["demand"].reindex(range(24)).interpolate().bfill().ffill()
        s_prof = sub["supply_vre"].reindex(range(24)).interpolate().bfill().ffill()
        if s_prof.mean() > 0:
            s_eq = s_prof * (d_prof.mean() / s_prof.mean())
        else:
            s_eq = s_prof
        d_amp = d_prof.max() - d_prof.min()
        s_amp = s_eq.max() - s_eq.min()
        amp_ratio = d_amp / s_amp if s_amp > 0 else np.nan
        day_idx = list(range(8, 18))
        day_excess = float(np.maximum(s_eq.loc[day_idx] - d_prof.loc[day_idx], 0).sum()
                           / d_prof.sum())
        rows.append({
            "country": cc,
            "region": row["region"],
            "uncovered": row["uncovered_share_local"] * 100,
            "phase_lag": row["phase_lag_h"],
            "amplitude_ratio": float(amp_ratio),
            "p95_run": row["p95_run_h"],
            "daytime_excess": day_excess * 100,
        })
    out = pd.DataFrame(rows)
    return out


def plot_fig1():
    # mmc2-style 5-panel layout: full-width map on top + 2x2 envelope-band grid below
    r1 = pd.read_csv(DATA / "r1_country_mismatch.csv",
                     keep_default_na=False, na_values=[""])
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv",
                     keep_default_na=False, na_values=[""])
    cmeta = pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""])
    cen = {row.iso2: (float(row.lon), float(row.lat)) for row in cmeta.itertuples()}

    fig = plt.figure(figsize=(14.5, 12.6))
    # Row 1: panel (a) — full-width world map with a longitude-capacity strip on
    # top and a latitude-capacity strip on the right (placed by a draw callback).
    # Row 2: (c) vertical country x local-hour deficit heatmap on the left, and on
    # the right (b) the merged diurnal panel over (d) the decomposition scatter.
    outer = fig.add_gridspec(2, 1, left=0.05, right=0.965, top=0.965,
                             bottom=0.055, height_ratios=[1.0, 1.04], hspace=0.13)
    gs_a = outer[0].subgridspec(3, 2, height_ratios=[0.13, 1.0, 0.07],
                                width_ratios=[1.0, 0.055], hspace=0.0, wspace=0.0)
    # 3 columns: narrow c | b-over-d (original width) | right-side spacer, so the
    # whitespace from narrowing c lands on the right margin, not between c and b/d.
    gs_bot = outer[1].subgridspec(1, 3, width_ratios=[0.85, 1.0, 0.40], wspace=0.30)
    gs_right = gs_bot[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.42)

    # ----- (a) World map: choropleth of uncovered share -----
    # rectangular PlateCarree base map: square frame, light-blue ocean,
    # straight graticule with lon/lat labels (the lon ticks also key the
    # capacity strip above)
    ax_a = fig.add_subplot(gs_a[1, 0], projection=ccrs.PlateCarree())
    ax_a.set_anchor("NW")   # snap the aspect-locked map to the cell TOP-LEFT; the
    # horizontal colorbar then occupies the slack below instead of dead space
    ax_a.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax_a.spines["geo"].set_visible(True)
    ax_a.spines["geo"].set_edgecolor("#6B7280")
    ax_a.spines["geo"].set_linewidth(0.8)
    ax_a.set_facecolor("#E3EDF6")
    # country choropleth: match frykit country polygons to demand values via centroids
    geoms = fshp.get_countries()
    tree = STRtree(geoms)
    val_by_iso = dict(zip(r1["country"], r1["uncovered_share_local"] * 100))
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
    # mmc2-style base: medium-grey land for all countries, then colored overlay
    # for the 104 sample countries (no white hairline borders that wash out the map)
    fplt.add_geometries(ax_a, geoms, fc="#C7CCD1", ec="none",
                        crs=fplt.PLATE_CARREE, zorder=1)
    cmap_a = FIG_PA.copy(); cmap_a.set_bad(alpha=0)
    norm_a = mpl.colors.Normalize(vmin=15, vmax=56)
    fplt.add_geometries(ax_a, geoms, array=np.ma.masked_invalid(arr),
                        cmap=cmap_a, norm=norm_a, crs=fplt.PLATE_CARREE,
                        ec="#FFFFFF", lw=0.18, zorder=2)
    gl = ax_a.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=0.4, color="#9CA3AF", alpha=0.55,
                        linestyle=(0, (2, 2)), zorder=3,
                        xlocs=range(-180, 181, 60), ylocs=[-30, 0, 30, 60])
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False
    gl.xlabel_style = {"size": 9.5, "color": "#374151"}
    gl.ylabel_style = {"size": 9.5, "color": "#374151"}
    sm = mpl.cm.ScalarMappable(cmap=cmap_a, norm=norm_a); sm.set_array([])
    cb_host = fig.add_subplot(gs_a[2, 0]); cb_host.axis("off")
    cax = cb_host.inset_axes([0.20, 0.32, 0.60, 0.40])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Share of annual digital demand not met by home renewables "
                   "in the same hour (%)", fontsize=12.5)
    cbar.ax.tick_params(labelsize=11)

    # slim marginal above the map: global installed PV/wind capacity by 15°
    # longitude band (WRI sites), aligned with the map's equatorial x-scale —
    # capacity spans every longitude, the resource base of the follow-the-sun
    # pooling in Result 2
    sites = pd.read_csv(
        ROOT / "collective_attention_research_plan" / "data"
        / "global_renewable_sites" / "representative_sites.csv",
        keep_default_na=False, na_values=[""],
    )
    CAP_BAR, CAP_LINE = "#9085B8", "#5E5290"   # coordinate with the figure's purple
    ax_t = fig.add_subplot(gs_a[0, 0])
    lon_bins = np.arange(-180, 181, 5)
    centers = (lon_bins[:-1] + lon_bins[1:]) / 2.0
    gw, _ = np.histogram(sites["lon"], bins=lon_bins,
                         weights=sites["wri_capacity_mw"] / 1000.0)
    ax_t.bar(centers, gw, width=5.0 * 0.9, color=CAP_BAR,
             edgecolor="white", linewidth=0.2)
    # smooth envelope curve tracing the longitude distribution of the bars
    from scipy.interpolate import make_interp_spline
    from scipy.ndimage import gaussian_filter1d
    xf = np.linspace(centers[0], centers[-1], 720)
    spl = make_interp_spline(centers, gaussian_filter1d(gw, 2.0, mode="nearest"), k=3)
    ax_t.plot(xf, np.clip(spl(xf), 0, None), color=CAP_LINE, linewidth=1.8,
              alpha=0.9, zorder=4)
    # no y-axis: the three dominant peaks carry their value (units on the
    # tallest, peaks >=20° apart), a short tag over the empty Pacific bins
    # names the quantity; the full definition lives in the caption
    v_max = float(gw.max())
    picked = []
    for i in np.argsort(gw)[::-1]:
        if gw[i] < 0.4 * v_max or len(picked) == 3:
            break
        if all(abs(centers[i] - centers[j]) >= 20 for j in picked):
            picked.append(i)
    for i in picked:
        lab = f"{gw[i]:.0f} GW" if gw[i] == v_max else f"{gw[i]:.0f}"
        ax_t.text(centers[i], gw[i] + v_max * 0.05, lab, ha="center",
                  va="bottom", fontsize=10, color="#475569")
    ax_t.text(0.008, 0.90, "Installed capacity (GW)",
              transform=ax_t.transAxes, fontsize=10, color="#475569",
              ha="left", va="top")
    ax_t.set_xlim(ax_a.get_xlim())
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.set_ylim(0, gw.max() * 1.25)
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.grid(False)
    panel_letter(ax_t, "a", x=-0.022, y=1.02)

    # right marginal: installed capacity by 5° LATITUDE band (companion to the top
    # longitude strip). The map is aspect-locked and does not fill its cell, so this
    # strip is pinned to the map's actual bbox by a draw callback; same purple colours.
    ax_r = fig.add_axes([0.5, 0.5, 0.05, 0.2])   # placeholder; positioned in _align_lat
    lat_bins = np.arange(-60, 86, 5)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2.0
    gw_lat, _ = np.histogram(sites["lat"], bins=lat_bins,
                             weights=sites["wri_capacity_mw"] / 1000.0)
    ax_r.barh(lat_centers, gw_lat, height=5.0 * 0.9, color=CAP_BAR,
              edgecolor="white", linewidth=0.2)
    yf = np.linspace(lat_centers[0], lat_centers[-1], 720)
    spl_y = make_interp_spline(
        lat_centers, gaussian_filter1d(gw_lat, 2.0, mode="nearest"), k=3)
    ax_r.plot(np.clip(spl_y(yf), 0, None), yf, color=CAP_LINE, linewidth=1.8,
              alpha=0.9, zorder=4)
    iy = int(np.argmax(gw_lat))
    ax_r.text(gw_lat[iy] + gw_lat.max() * 0.06, lat_centers[iy], f"{gw_lat[iy]:.0f}",
              ha="left", va="center", fontsize=10, color="#475569")
    ax_r.set_ylim(ax_a.get_ylim())
    ax_r.set_xlim(0, gw_lat.max() * 1.30)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    for sp in ax_r.spines.values():
        sp.set_visible(False)
    ax_r.grid(False)

    def _align_lat_marginal(event=None):
        p = ax_a.get_position()
        ax_r.set_position([p.x1 + 0.004, p.y0, p.width * 0.052, p.height])
    _align_lat_marginal()
    fig.canvas.mpl_connect("draw_event", _align_lat_marginal)

    # ----- (b–d) Stacked local-hour panels across 104 countries ----------------
    def envelope_band(ax, x, mat, fill_color, line_color):
        mn = np.nanmin(mat, axis=0); mx = np.nanmax(mat, axis=0)
        p10 = np.nanpercentile(mat, 10, axis=0)
        p90 = np.nanpercentile(mat, 90, axis=0)
        med = np.nanmedian(mat, axis=0)
        ax.fill_between(x, mn, mx, color=fill_color, alpha=0.15, linewidth=0)
        ax.fill_between(x, p10, p90, color=fill_color, alpha=0.42, linewidth=0)
        ax.plot(x, med, color=line_color, linewidth=1.8)

    countries = sorted(dn["country"].unique())
    cmat_demand = np.full((len(countries), 24), np.nan)
    cmat_supply = np.full((len(countries), 24), np.nan)
    cmat_unmet = np.full((len(countries), 24), np.nan)
    for i, cc in enumerate(countries):
        sub = dn[dn["country"] == cc].set_index("local_hour").reindex(range(24))
        d = sub["demand"].to_numpy(); s = sub["supply_vre"].to_numpy()
        dm = float(np.nanmean(d))
        sm_ = float(np.nanmean(s))
        if not np.isfinite(dm) or dm <= 0:
            continue
        s_eq = s * (dm / sm_) if (np.isfinite(sm_) and sm_ > 0) else np.zeros_like(s)
        cmat_demand[i, :] = d / dm
        cmat_supply[i, :] = s_eq / dm
        cmat_unmet[i, :] = np.maximum(d - s_eq, 0) / dm * 100

    # semantic families shared with figs 2-3: warm = demand / deficit
    # (YlOrRd map, fig2d demand), blue = renewable supply (wind, power links)
    DEM_FILL = "#FCA5A5"; DEM_LINE = HIGHLIGHT
    SUP_FILL = "#93C5FD"; SUP_LINE = PRIMARY
    ORANGE_FILL = "#FED7AA"; ORANGE_LINE = "#C2410C"

    # (c) MERGED diurnal panel (replaces the former b/c/d stack): demand & supply
    # median curves with their cross-country 10-90% bands on the left axis, plus
    # the SIGNED net mismatch (demand - supply) as bars + median curve on the
    # right axis — positive = evening deficit (amber), negative = midday surplus
    # (teal). The midday-surplus / evening-deficit pattern is the dispatch case.
    DEF_POS, SUR_NEG, NETLINE = "#E8A33D", "#2F9C8B", "#7C2D12"
    H = np.arange(24)
    dem_med = np.nanmedian(cmat_demand, axis=0)
    sup_med = np.nanmedian(cmat_supply, axis=0)
    dem_p10 = np.nanpercentile(cmat_demand, 10, axis=0)
    dem_p90 = np.nanpercentile(cmat_demand, 90, axis=0)
    sup_p10 = np.nanpercentile(cmat_supply, 10, axis=0)
    sup_p90 = np.nanpercentile(cmat_supply, 90, axis=0)
    net = (cmat_demand - cmat_supply) * 100.0
    net_med = np.nanmedian(net, axis=0)
    net_p10 = np.nanpercentile(net, 10, axis=0)
    net_p90 = np.nanpercentile(net, 90, axis=0)

    ax_b = fig.add_subplot(gs_right[0])
    ax_b.axvspan(18, 23, zorder=0, **EVENING_KW)
    axr = ax_b.twinx()
    nmax = max(np.nanmax(net_p90), -np.nanmin(net_p10))
    axr.set_ylim(-nmax * 1.15, nmax * 1.15)
    axr.axhline(0, color="#6B7280", lw=0.9, zorder=1)
    axr.fill_between(H, net_p10, net_p90, color="#E7E2D5", alpha=0.55, lw=0, zorder=1)
    bar_cols = [DEF_POS if v >= 0 else SUR_NEG for v in net_med]
    axr.bar(H, net_med, width=0.84, color=bar_cols, alpha=0.85, edgecolor="white",
            linewidth=0.3, zorder=2)
    axr.plot(H, net_med, color=NETLINE, lw=1.6, ls=(0, (4, 1.5)), zorder=3,
             label="Net mismatch (median)")
    axr.set_ylabel("Net mismatch, demand − supply\n(% of mean demand;  + deficit / − surplus)",
                   fontsize=11)
    axr.tick_params(axis="y", labelsize=10)
    axr.text(20.5, np.nanmax(net_med[18:24]) * 0.5, "deficit", color=NETLINE,
             fontsize=9.5, ha="center", va="center", fontweight="bold")
    axr.text(12, np.nanmin(net_med[10:15]) * 0.55, "surplus", color="#1F6E63",
             fontsize=9.5, ha="center", va="center", fontweight="bold")

    ax_b.set_zorder(axr.get_zorder() + 1)
    ax_b.patch.set_visible(False)
    ax_b.fill_between(H, dem_p10, dem_p90, color=DEM_FILL, alpha=0.20, lw=0, zorder=4)
    ax_b.fill_between(H, sup_p10, sup_p90, color=SUP_FILL, alpha=0.20, lw=0, zorder=4)
    ax_b.plot(H, dem_med, color=DEM_LINE, lw=2.6, zorder=6, label="Digital demand")
    ax_b.plot(H, sup_med, color=SUP_LINE, lw=2.6, zorder=6, label="Home renewable supply")
    ax_b.set_xlim(0, 23); ax_b.set_xticks(range(0, 24, 3))
    ax_b.set_ylim(0, np.nanmax(sup_p90) * 1.05)
    ax_b.set_yticks([0, 1, 2, 3])
    ax_b.set_xlabel("Local hour of day", fontsize=12)
    ax_b.set_ylabel("Demand / supply (mean = 1)", fontsize=12)
    ax_b.text(20.5, np.nanmax(sup_p90) * 1.0, "Evening\n(18-23 h)", ha="center",
              va="top", fontsize=9.5, color="#475569")
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9.5, frameon=False)
    ax_b.spines["top"].set_visible(False)
    axr.spines["top"].set_visible(False)
    panel_letter(ax_b, "c", x=-0.13, y=0.98)

    region_of = dn.drop_duplicates("country").set_index("country")["region"].to_dict()

    # ----- (b) VERTICAL country x local-hour deficit heatmap: countries run up the
    # y-axis (sorted by annual uncovered share); the evening deficit forms a vertical
    # stripe (18-23 h) aligned across all 104 countries (timing universal) while its
    # intensity grows up the sort (amplitude graded)
    from scipy import stats as sp_stats
    ax_e = fig.add_subplot(gs_bot[0, 0])
    uncov = r1.set_index("country")["uncovered_share_local"]
    order = [i for i in np.argsort([uncov.get(cc, np.nan) for cc in countries])
             if np.isfinite(cmat_unmet[i]).any()]
    mat_e = cmat_unmet[order, :]            # rows = countries, cols = 24 local hours
    hm_vmax = 150.0
    pm = ax_e.pcolormesh(np.arange(25), np.arange(len(order) + 1), mat_e,
                         cmap=FIG_PA, vmin=0.0, vmax=hm_vmax, rasterized=True)
    ax_e.axvline(18, color="white", lw=1.0, ls=(0, (3, 2)))
    ax_e.axvline(23, color="white", lw=1.0, ls=(0, (3, 2)))
    ax_e.text(20.5, len(order) * 0.98, "Evening\n18-23 h", fontsize=9.5,
              color="white", ha="center", va="top",
              path_effects=[mpl.patheffects.withStroke(linewidth=1.8,
                                                       foreground="#7F1D1D")])
    ordered_cc = [countries[i] for i in order]
    anchors = [cc for cc in ("US", "IN", "JP", "PS") if cc in ordered_cc]
    ax_e.set_yticks([ordered_cc.index(cc) + 0.5 for cc in anchors])
    ax_e.set_yticklabels(anchors, fontsize=10)
    ax_e.set_xticks([0, 6, 12, 18])
    ax_e.set_xlabel("Local hour", fontsize=12)
    ax_e.set_ylabel("Country", fontsize=12)
    ax_e.grid(False)
    cax_e = ax_e.inset_axes([1.08, 0.05, 0.022, 0.90])
    cb_e = plt.colorbar(pm, cax=cax_e, extend="max")
    cb_e.ax.set_title("Deficit (%)", fontsize=9.5, color="#374151", pad=6)
    cb_e.ax.tick_params(labelsize=9.5)
    panel_letter(ax_e, "b", x=-0.12, y=0.98)

    # ----- (d) decomposition regression: the clock-only uncovered share
    # (recomputed from the 24-h mean profiles alone; cmat_unmet rows are in %
    # of mean demand, so the row mean IS that share) against the true annual
    # share — the diurnal clock dominates, weather/seasons only amplify
    from scipy import stats as sp_stats
    ax_g = fig.add_subplot(gs_right[1])
    gpts = []
    for i, cc in enumerate(countries):
        u_clock = np.nanmean(cmat_unmet[i, :])
        u_ann = uncov.get(cc, np.nan)
        if np.isfinite(u_clock) and np.isfinite(u_ann):
            gpts.append((cc, u_clock, u_ann * 100, region_of.get(cc)))
    xg = np.array([p[1] for p in gpts]); yg = np.array([p[2] for p in gpts])
    lim = (0, max(xg.max(), yg.max()) * 1.08)
    # shade the weather + season excess zone (above the y = x clock-only line)
    ax_g.fill_between(lim, lim, lim[1], color="#F0E7D6", alpha=0.6, lw=0, zorder=0)
    ax_g.plot(lim, lim, color="#9CA3AF", lw=1.0, ls=(0, (3, 2)), zorder=1)
    lr = sp_stats.linregress(xg, yg)
    xx = np.linspace(*lim, 50)
    ax_g.plot(xx, lr.intercept + lr.slope * xx, color="#374151", lw=1.6,
              ls="--", zorder=2)
    exc = yg - xg
    sc = ax_g.scatter(xg, yg, s=42, c=exc, cmap=FIG_PA, vmin=0,
                      vmax=float(np.percentile(exc, 95)), edgecolor="white",
                      linewidth=0.5, alpha=0.95, zorder=3)
    cax_g = ax_g.inset_axes([1.04, 0.05, 0.035, 0.90])
    cb_g = fig.colorbar(sc, cax=cax_g)
    cb_g.set_label("Weather + season excess (pp)", fontsize=9.5)
    cb_g.ax.tick_params(labelsize=8.5)
    ax_g.text(0.04, 0.96,
              f"R² = {lr.rvalue**2:.2f}\nmedian excess +{np.median(yg - xg):.1f} pp",
              transform=ax_g.transAxes, ha="left", va="top", fontsize=9.5,
              color="#374151")
    for cc in ("US", "PS"):
        m = [p for p in gpts if p[0] == cc]
        if m:
            ax_g.text(m[0][1] + 0.8, m[0][2] - 2.2, cc, fontsize=9.5,
                      color="#111827",
                      path_effects=[mpl.patheffects.withStroke(
                          linewidth=1.8, foreground="white")])
    ax_g.set_xlabel("Clock-only uncovered share (%)\n(mean diurnal profiles)",
                    fontsize=12)
    ax_g.set_ylabel("Annual uncovered share (%)", fontsize=12)
    ax_g.set_xlim(lim); ax_g.set_ylim(lim)
    soften_axes(ax_g)
    panel_letter(ax_g, "d", x=-0.10, y=1.10)

    # nudge panels b and d slightly to the right (the twin axis follows ax_b)
    for _a in (ax_b, ax_g):
        _p = _a.get_position()
        _a.set_position([_p.x0 + 0.028, _p.y0, _p.width, _p.height])
    fig.canvas.draw()        # trigger aspect lock + draw callback so ax_r aligns to the map
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig1_national_mismatch{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1")


def draw_followsun_panel(ax, ax_top=None, ax_right=None):
    """Fig 2 panel (e): per-country "follow-the-sun" scatter with marginals.

    One point per demand country: x = its demand-peak local hour, y = the
    energy-weighted mean SOLAR hour of the L3 global portfolio's supply at that
    peak. The home-only counterfactual is the diagonal (supply solar hour =
    local hour, slope +1); the data instead pin at solar noon regardless of when
    a country peaks. The shaded wedge between the two lines is the westward reach
    the global pool buys. Optional marginal axes (``ax_top``/``ax_right``) carry
    the demand-peak-hour and supply-solar-hour distributions, making the
    "demand spreads to evening, supply stays at noon" decoupling explicit. Reads
    the precomputed r2_d_followsun.csv (built by _try_fig2d_bubble.py from the
    4M-row site_hourly_cf parquet)."""
    csv = DATA.parent / "data_globalsites_stations_expanded" / "r2_d_followsun.csv"
    d = pd.read_csv(csv)
    b, a = np.polyfit(d["x_peak"], d["y_solar"], 1)
    yhat = a + b * d["x_peak"]
    r2 = 1 - ((d["y_solar"] - yhat) ** 2).sum() / (
        (d["y_solar"] - d["y_solar"].mean()) ** 2).sum()
    eve = d[(d["x_peak"] >= 17) | (d["x_peak"] <= 1)]
    reach_med = eve["reach"].median()
    m = d["phase_lag"].notna() & (d["x_peak"] >= 12)
    rr = float(np.corrcoef(d.loc[m, "reach"], d.loc[m, "phase_lag"])[0, 1])
    ybar = float(d["y_solar"].mean())
    xcross = a / (1 - b)                       # fit line meets the diagonal
    FS = 13
    XLIM = (8.5, 23.8)

    # reach wedge: triangle between the home-only diagonal and the flat fit line,
    # right of their crossover; its vertical height == hours of westward reach
    xw = np.linspace(xcross, XLIM[1], 50)
    ax.fill_between(xw, a + b * xw, xw, color=PRIMARY, alpha=0.08, lw=0, zorder=1)
    ax.text(16.4, 14.4, "global pool reaches west", fontsize=FS - 2,
            color="#3B5BA5", ha="center", va="center", style="italic",
            rotation=27, rotation_mode="anchor")

    ax.plot([0, 23], [0, 23], color="#9CA3AF", lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.text(17.2, 17.7, "home-only (slope +1.00)", fontsize=FS, color="#6B7280",
            ha="left", va="bottom", style="italic", rotation=45,
            rotation_mode="anchor")
    ax.axhspan(11, 13, color=SECONDARY, alpha=0.08, zorder=0)
    ax.text(8.8, 12.3, "solar noon", fontsize=FS, color="#B45309",
            ha="left", va="center")

    xx = np.array([d["x_peak"].min() - 0.5, d["x_peak"].max() + 0.5])
    ax.plot(xx, a + b * xx, color="#111827", lw=2.2, zorder=4,
            label=f"fit: slope {b:+.2f}  (R²={r2:.2f})")
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.22, 0.22, len(d))
    ax.scatter(d["x_peak"] + jit, d["y_solar"], s=62, c=PRIMARY, alpha=0.5,
               edgecolor="white", linewidth=0.5, zorder=5,
               label="one country (n = 104)")

    xr = 22.7
    ax.annotate("", xy=(xr, xr), xytext=(xr, ybar),
                arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.6),
                zorder=6)
    ax.text(xr - 0.45, 19.7, f"reach ≈ {reach_med:.0f} h\n(west, r {rr:.2f})",
            fontsize=FS - 1, color="#111827", ha="right", va="center",
            linespacing=1.05)

    ax.set_xlim(*XLIM); ax.set_ylim(*XLIM)
    ax.set_aspect("equal")
    ax.set_xticks(range(9, 24, 3)); ax.set_yticks(range(9, 24, 3))
    ax.set_xlabel("Demand peak local hour", fontsize=FS)
    ax.set_ylabel("Mean supply solar hour at demand peak", fontsize=FS)
    ax.tick_params(axis="both", labelsize=FS)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#EEF1F4", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=FS - 1, frameon=False,
              labelspacing=0.4, handletextpad=0.4, borderpad=0.2)

    # marginal distributions
    bins = np.arange(8.5, 24.5, 1)
    if ax_top is not None:
        ax_top.hist(d["x_peak"], bins=bins, color="#B7C0CA", edgecolor="white",
                    linewidth=0.6)
        ax_top.set_xlim(*XLIM); ax_top.axis("off")
        ax_top.text(0.0, 0.92, "demand peaks spread to evening",
                    transform=ax_top.transAxes, fontsize=FS - 3, color="#6B7280",
                    ha="left", va="top")
    if ax_right is not None:
        ax_right.hist(d["y_solar"], bins=bins, orientation="horizontal",
                      color=SECONDARY, alpha=0.55, edgecolor="white",
                      linewidth=0.6)
        ax_right.set_ylim(*XLIM); ax_right.axis("off")
        ax_right.text(0.95, 0.02, "supply piles\nat noon",
                      transform=ax_right.transAxes, fontsize=FS - 3,
                      color="#B45309", ha="right", va="bottom", rotation=270,
                      linespacing=1.0)


def draw_gap_distribution_panel(ax, r2):
    """Fig 2 extra panel: distribution of the 104-country residual uncovered
    share as the power-dispatch scope widens L0->L3 (at W0, no workload
    migration). Complements the scenario-matrix medians by showing that global
    pooling compresses the WHOLE distribution, not just the median."""
    FS = 13
    labels = ["L0\nNational", "L1\n≤1.5Mm", "L2\n≤3Mm", "L3\nGlobal"]
    w0 = r2[r2["workload_scenario"] == "W0"]
    data = [(w0[w0["power_scope"] == s]["uncovered_share"] * 100).values
            for s in POWER_SCOPES]
    pos = list(range(4))
    # soft sequential fills: warm (L0, high gap) -> cool (L3, low gap)
    fills = ["#E5A9A2", "#EAD0A0", "#A9D2CA", "#A7C4E6"]

    # violin (light, behind) + inner box plot (dark outline) + white diamond mean
    parts = ax.violinplot(data, positions=pos, widths=0.9, showextrema=False)
    for pc, c in zip(parts["bodies"], fills):
        pc.set_facecolor(c); pc.set_alpha(0.40); pc.set_edgecolor("none")
    bp = ax.boxplot(data, positions=pos, widths=0.24, patch_artist=True,
                    showfliers=False, manage_ticks=False,
                    medianprops=dict(color="#111827", lw=1.6),
                    whiskerprops=dict(color="#111827", lw=1.4),
                    capprops=dict(color="#111827", lw=1.4),
                    boxprops=dict(edgecolor="#111827", lw=1.4))
    for patch, c in zip(bp["boxes"], fills):
        patch.set_facecolor(c); patch.set_alpha(0.95)
    means = [float(np.mean(d)) for d in data]
    ax.scatter(pos, means, marker="D", s=55, facecolor="white",
               edgecolor="#111827", linewidth=1.4, zorder=7)
    meds = [float(np.median(d)) for d in data]
    for i, m in enumerate(meds):                      # median value labels
        ax.annotate(f"{m:.1f}", (i + 0.16, m), textcoords="offset points",
                    xytext=(2, 0), fontsize=FS - 1, fontweight="bold",
                    color="#111827", va="center")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=FS - 1)
    ax.set_ylabel("Residual uncovered share (%)", fontsize=FS)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 62)
    ax.tick_params(axis="y", labelsize=FS)
    ax.text(0.98, 0.97, "W0 (no workload migration)", transform=ax.transAxes,
            fontsize=FS - 3, color="#6B7280", ha="right", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", color="#EEF1F4", lw=0.5)
    ax.set_axisbelow(True)


# ---------- Figure 2 ----------
def plot_fig2():
    r1 = pd.read_csv(DATA / "r1_country_mismatch.csv")
    r2 = pd.read_csv(DATA / "r2_portfolio_scenarios.csv")
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    s4 = pd.read_csv(DATA / "r2_s4_weight_matrix.csv")
    # Ensure power_scope/workload_scenario columns exist (split from scenario)
    if "power_scope" not in r2.columns:
        r2["power_scope"] = r2["scenario"].str.split("_").str[0]
    if "workload_scenario" not in r2.columns:
        r2["workload_scenario"] = r2["scenario"].str.split("_").str[1]

    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(2, 2, left=0.015, right=0.98, top=0.95, bottom=0.04,
                          hspace=0.16, wspace=0.18,
                          width_ratios=[1.0, 1.65], height_ratios=[0.46, 1.0])
    region_order = ["Africa/MENA", "Asia-Pacific", "Europe", "Latin America", "North America"]
    region_map = r1.set_index("country")["region"].to_dict()

    # ----- (a) 4 × 4 scenario heatmap with row/column marginals -----
    gs_a = gs[0, 0].subgridspec(2, 2, width_ratios=[3.0, 1.0],
                                 height_ratios=[1.0, 3.0],
                                 wspace=0.06, hspace=0.06)
    ax_top = fig.add_subplot(gs_a[0, 0])   # top marginal: pure-compute effect
    ax_a = fig.add_subplot(gs_a[1, 0])     # main 4x4 heatmap
    ax_right = fig.add_subplot(gs_a[1, 1]) # right marginal: pure-power effect
    # placeholder corner for legend area
    ax_corner = fig.add_subplot(gs_a[0, 1]); ax_corner.axis("off")

    # build 4x4 matrix: rows = power scope (L3 top, L0 bottom), cols = compute scope (M0..M3)
    rows_L = list(reversed(POWER_SCOPES))
    cols_W = list(WORKLOAD_SCOPES)
    med_pivot = (r2.groupby(["power_scope", "workload_scenario"])["uncovered_share"]
                  .median().unstack("workload_scenario") * 100)
    mat = med_pivot.reindex(index=rows_L, columns=cols_W).values
    gap_cmap = FIG_PA
    vmin_a, vmax_a = mat.min() * 0.95, mat.max() * 1.02
    im = ax_a.imshow(mat, aspect="auto", cmap=gap_cmap, vmin=vmin_a, vmax=vmax_a)
    fs_a = 13  # single, uniform font size for every text element in panel (a)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            # luminance-adaptive text colour (the two-pole ramp is darkest mid-range,
            # lightest at both ends, so a value threshold would mis-colour the labels)
            fc = gap_cmap((v - vmin_a) / (vmax_a - vmin_a))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax_a.text(j, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=fs_a, weight="bold",
                      color="white" if lum < 0.55 else "#111827")
    ax_a.set_xticks(range(len(cols_W)))
    short_w = {"W0": "W0\n0%", "W1": "W1\n~10%",
               "W2": "W2\n~30%", "W3": "W3\n~60%"}
    ax_a.set_xticklabels([short_w[c] for c in cols_W], fontsize=fs_a,
                         color="#000000")
    ax_a.set_yticks(range(len(rows_L)))
    ax_a.set_yticklabels([POWER_LABEL[p] for p in rows_L], fontsize=fs_a,
                         color="#000000")
    ax_a.set_xlabel("Workload-class migration policy", fontsize=fs_a)
    ax_a.set_ylabel("Power transmission scope", fontsize=fs_a)
    ax_a.grid(False)
    # annotate diagonal path L0→L3 corner-to-corner
    for k in range(min(len(rows_L), len(cols_W))):
        ax_a.add_patch(mpatches.Rectangle((k - 0.5, len(rows_L) - 1 - k - 0.5), 1, 1,
                                          fill=False, edgecolor="#111827",
                                          linewidth=1.4, zorder=4))
    # corner tags inside the L0xW0 / L3xW3 cells, below the cell value
    tag_fx = [mpl.patheffects.withStroke(linewidth=2.4, foreground="#111827")]
    ax_a.text(0, len(rows_L) - 1 + 0.31, "Status quo", fontsize=fs_a,
              weight="bold", color="white", ha="center", va="center",
              path_effects=tag_fx, zorder=5)
    ax_a.text(len(cols_W) - 1, 0.31, "Upper bound", fontsize=fs_a,
              weight="bold", color="white", ha="center", va="center",
              path_effects=tag_fx, zorder=5)

    # Top marginal: pure compute effect at L0 power
    pure_workload = [mat[len(rows_L) - 1, j] for j in range(len(cols_W))]
    ax_top.bar(range(len(cols_W)),
               [pure_workload[0] - v for v in pure_workload],
               color=SECONDARY, alpha=0.85, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_workload):
        delta = pure_workload[0] - v
        lbl = f"{delta:.1f}" if delta >= 0.05 else "0.0"
        ax_top.text(i, delta + 0.1, lbl, ha="center", va="bottom",
                    fontsize=fs_a, color="#111827")
    ax_top.set_ylabel("Reduction vs L0×W0\n(workload only, pp)", fontsize=fs_a)
    panel_letter(ax_top, "a", x=-0.26, y=1.02)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=fs_a)
    ax_top.set_ylim(0, max(pure_workload[0] - min(pure_workload), 0.1) * 1.3)
    soften_axes(ax_top)

    # Right marginal: pure power effect at W0 workload
    pure_power = [mat[i, 0] for i in range(len(rows_L))]
    ax_right.barh(range(len(rows_L)),
                   [pure_power[len(rows_L) - 1] - v for v in pure_power],
                   color=PRIMARY, alpha=0.85, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(pure_power):
        delta = pure_power[len(rows_L) - 1] - v
        lbl = f"{delta:.1f}" if delta >= 0.05 else "0.0"
        ax_right.text(delta + 0.1, i, lbl, ha="left", va="center",
                       fontsize=fs_a, color="#111827")
    ax_right.set_xlabel("Reduction vs L0×W0\n(power only, pp)", fontsize=fs_a)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=fs_a)
    ax_right.invert_yaxis()
    ax_right.set_xlim(0, max(pure_power[len(rows_L) - 1] - min(pure_power), 0.1) * 1.35)
    soften_axes(ax_right)

    # colorbar lives in the empty corner cell of the composite, so it cannot be
    # misread as belonging to the panel below
    cax_a = ax_corner.inset_axes([0.08, 0.48, 0.86, 0.15])
    cbar_a = plt.colorbar(im, cax=cax_a, orientation="horizontal")
    cbar_a.set_label("Median uncovered\ndemand share (%)", fontsize=fs_a)
    cbar_a.ax.tick_params(labelsize=fs_a)

    # ----- (b) Global L3 follow-the-sun network: supply origin by longitude ----
    # Each demand country in its local evening draws clean power from sites still
    # in daylight at a different longitude. Arcs and hubs are coloured by the
    # supply site's longitude zone, so cross-zone arcs make the east-west
    # "follow-the-sun" complementarity visible. Hub size encodes cumulative weight.
    cmeta_b = pd.read_csv(COUNTRY_META, keep_default_na=False, na_values=[""])
    cen_b = {row.iso2: (float(row.lon), float(row.lat)) for row in cmeta_b.itertuples()}
    sites = pd.read_csv(ROOT / "collective_attention_research_plan" / "data" /
                        "global_renewable_sites" / "representative_sites.csv",
                        keep_default_na=False, na_values=[""])
    site_xy = {row.site_id: (float(row.lon), float(row.lat)) for row in sites.itertuples()}
    site_lon = {row.site_id: float(row.lon) for row in sites.itertuples()}
    if "globalsites" in DATA.name:
        sw_path = DATA.parent / "data_globalsites_stations_expanded" / "r2_l3_station_weights.csv"
    else:
        sw_path = DATA.parent / "data_globalsites_stations" / "r2_l3_station_weights.csv"
    sw_full = pd.read_csv(sw_path)  # demand_country, site_id, supply_country, tech, weight

    def lon_zone(lon):
        if -170.0 <= lon < -30.0:
            return "Americas"
        if -30.0 <= lon < 60.0:
            return "Europe / Africa"
        return "Asia-Pacific"
    ZONE_COLOR = {"Americas": "#3A7CA5", "Europe / Africa": "#E0A33C",
                  "Asia-Pacific": "#2A9D8F"}
    ZONE_ORDER = ["Americas", "Europe / Africa", "Asia-Pacific"]
    # Panels (b) and (c) share ONE colour scheme: the 3 longitude zones
    # (ZONE_COLOR). Colour = the supply longitude zone (where the power comes
    # from); supply vs demand is shown by node SIZE (large hub / small demand
    # dot) and by the flow taper (thick supply end -> thin demand end).

    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=10))
    ax_b.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax_b.spines["geo"].set_visible(False)
    ax_b.set_frame_on(False)
    ax_b.set_facecolor("white")
    base_geoms = fshp.get_countries()
    fplt.add_geometries(ax_b, base_geoms, fc="#EDF0F3", ec="white",
                        lw=0.3, crs=fplt.PLATE_CARREE, zorder=1)

    # Cross-border station weights: keep each demand country's top-2 arcs so the
    # global flow texture stays readable while every demand country is represented;
    # hub sizes below still use each site's full cumulative weight.
    xb = sw_full[(sw_full["supply_country"] != sw_full["demand_country"]) &
                 (sw_full["weight"] > 0)].copy()
    xb = xb[xb["site_id"].isin(site_xy) & xb["demand_country"].isin(cen_b)]
    edges = xb.sort_values("weight", ascending=False).groupby("demand_country").head(2)
    edges = edges[edges["weight"] > 0.03].copy()
    edges["zone"] = edges["site_id"].map(site_lon).map(lon_zone)
    w_max = float(np.percentile(edges["weight"], 97)) if len(edges) else 1.0
    edges["w_clip"] = edges["weight"].clip(upper=w_max)
    edges["lw"] = 0.25 + (edges["w_clip"] / w_max) * 1.5
    edges["alpha"] = 0.20 + (edges["w_clip"] / w_max) * 0.55

    # One colour per flow = the SUPPLY longitude zone (same as its source hub),
    # so a line's colour says where the power comes from. Direction is shown by
    # the taper alone: thick at the supply end, thin at the demand end.
    edges["color"] = edges["zone"].map(ZONE_COLOR).fillna("#9CA3AF")
    from pyproj import Geod
    geo = Geod(ellps="WGS84")
    geod = ccrs.Geodetic()
    K = 10
    for r in edges.sort_values("weight").itertuples():
        s_lon, s_lat = site_xy[r.site_id]
        d_lon, d_lat = cen_b[r.demand_country]
        pts = ([(s_lon, s_lat)]
               + geo.npts(s_lon, s_lat, d_lon, d_lat, K - 1)
               + [(d_lon, d_lat)])
        for k in range(K):
            frac = k / (K - 1)             # 0 at supply, 1 at demand
            ax_b.plot([pts[k][0], pts[k + 1][0]], [pts[k][1], pts[k + 1][1]],
                      color=r.color, linewidth=r.lw * (1.0 - 0.55 * frac),
                      alpha=r.alpha, solid_capstyle="round",
                      transform=geod, zorder=3)

    # Nodes carry COUNTRY identity (coloured by world region), not supply/demand
    # role; supply vs demand is read from node SIZE (large hub vs small demand
    # dot) and the flow taper. Faint grey candidate sites first.
    ax_b.scatter(sites["lon"], sites["lat"], s=2.5, c="#C9D2DA", alpha=0.5,
                 linewidths=0, zorder=2, transform=ccrs.PlateCarree())
    site_tot = sw_full.groupby("site_id")["weight"].sum()
    hub = sites[sites["site_id"].isin(site_tot[site_tot > 0.05].index)].copy()
    hub["wt"] = hub["site_id"].map(site_tot)
    hub["color"] = hub["lon"].map(lon_zone).map(ZONE_COLOR).fillna("#9CA3AF")
    s_max = float(hub["wt"].max()) if len(hub) else 1.0
    ax_b.scatter(hub["lon"], hub["lat"], s=45 + (hub["wt"] / s_max) * 255,
                 c=list(hub["color"]), edgecolor="white", linewidth=0.6,
                 alpha=0.95, zorder=6, transform=ccrs.PlateCarree())

    # demand countries: small circles (same glyph as hubs, role shown by smaller
    # size), coloured by region like the hubs.
    dccs = [c for c in edges["demand_country"].unique() if c in cen_b]
    dx = [cen_b[c][0] for c in dccs]
    dy = [cen_b[c][1] for c in dccs]
    dcol = [ZONE_COLOR.get(lon_zone(cen_b[c][0]), "#9CA3AF") for c in dccs]
    ax_b.scatter(dx, dy, s=14, marker="o", c=dcol, edgecolor="white",
                 linewidth=0.4, alpha=0.95, zorder=5,
                 transform=ccrs.PlateCarree())

    # label the top supply-hub countries by cumulative weight
    sc_tot = sw_full.groupby("supply_country")["weight"].sum().sort_values(ascending=False)
    hub_dodge = {"CN": (8, -7), "US": (0, -7.5), "MX": (-3, -8), "CL": (-2, -8),
                 "AU": (0, -7.5), "ES": (2, 6), "IN": (3, -7.5), "ZA": (0, -7.5)}
    labelled = 0
    for cc in sc_tot.index:
        if cc not in cen_b or labelled >= 8:
            continue
        lon, lat = cen_b[cc]
        dx, dy = hub_dodge.get(cc, (0, -6))
        ax_b.text(lon + dx, lat + dy, cc, fontsize=13, weight="bold",
                  color="#1F2937", ha="center", va="top",
                  transform=ccrs.PlateCarree(), zorder=11,
                  path_effects=[mpl.patheffects.withStroke(linewidth=2.2,
                                                           foreground="white")])
        labelled += 1

    panel_letter(ax_b, "d", x=-0.01, y=1.0)   # map -> "d" (reading order)

    leg_handles = [
        plt.Line2D([], [], color="#9AA7B4", lw=3.0,
                   label="Flow: thick = supply end, thin = demand end"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                   markeredgecolor="white", markersize=12,
                   label="Supply hub (large, size = weight)"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#9CA3AF",
                   markeredgecolor="white", markersize=5,
                   label="Demand country (small)"),
    ]
    zone_handles = [mpatches.Patch(facecolor=ZONE_COLOR[z], label=z)
                    for z in ZONE_ORDER]
    ax_b.legend(handles=leg_handles + zone_handles, loc="upper left",
                bbox_to_anchor=(0.0, -0.02), fontsize=13, frameon=False, ncol=3,
                handletextpad=0.4, columnspacing=1.0, borderpad=0.0,
                title="Colour = supply longitude zone",
                title_fontsize=13)

    # ----- (c) Supply-hub x demand-region chord, with an outer scope ring -----
    ax_ch = fig.add_subplot(gs[1, :])
    # 12 largest supply hubs (individual countries) as one half of the circle,
    # the 5 demand regions as the other half; M[hub][region] is the weight a hub
    # dispatches to a region. Hubs and ribbons are coloured by longitude zone, so
    # each region's incoming ribbons reveal the zone mix of where its clean power
    # is drawn from.
    sup_tot = (s4[s4["supply_country"].apply(lambda x: isinstance(x, str))]
               .groupby("supply_country")["weight"].sum().sort_values(ascending=False))
    hubs = [h for h in sup_tot.index if h in cen_b][:12]
    regions = ["North America", "Latin America", "Europe",
               "Africa/MENA", "Asia-Pacific"]
    ri = {r: i for i, r in enumerate(regions)}
    s4d = s4.copy()
    s4d["d_reg"] = s4d["demand_country"].map(region_map)
    Mb = np.zeros((len(hubs), len(regions)))
    hub_i = {h: i for i, h in enumerate(hubs)}
    for row in s4d.itertuples():
        if row.supply_country in hub_i and row.d_reg in ri:
            Mb[hub_i[row.supply_country]][ri[row.d_reg]] += row.weight
    REG_ZONE = {"North America": "Americas", "Latin America": "Americas",
                "Europe": "Europe / Africa", "Africa/MENA": "Europe / Africa",
                "Asia-Pacific": "Asia-Pacific"}
    zrank = {"Americas": 0, "Europe / Africa": 1, "Asia-Pacific": 2}
    hub_zone = {h: lon_zone(cen_b[h][0]) for h in hubs}
    horder = sorted(range(len(hubs)),
                    key=lambda i: (zrank[hub_zone[hubs[i]]], -Mb[i].sum()))
    hubs = [hubs[i] for i in horder]
    Mb = Mb[horder]
    sup_color = [ZONE_COLOR[hub_zone[h]] for h in hubs]
    reg_color = [ZONE_COLOR[REG_ZONE[r]] for r in regions]
    reg_lab = ["N.America", "L.America", "Europe", "Africa/MENA", "Asia-Pacific"]
    draw_bipartite_chord(ax_ch, Mb, hubs, sup_color, reg_lab, reg_color,
                         sup_color, lim=1.30, gap_bottom=0.58, label_size=14)

    # outer ring (bottom wedge): how many supply sites the NNLS portfolio
    # activates as the dispatch scope widens from home (L0) to global (L3).
    div_sc = ["L0_W0", "L1_W0", "L2_W0", "L3_W0", "L3_W3"]
    div_lab = ["L0", "L1", "L2", "L3", "L3·W3"]
    div = (r2.set_index("scenario").loc[div_sc]
           .groupby(level=0)["n_active_basis"].mean().reindex(div_sc).values)
    div_col = ["#94A3B8", "#7DA9C9", "#5A86B0", "#3A6CA0", "#274C77"]
    dmax = float(np.max(div))
    base = 0.86
    for v, ang, col, sl in zip(div, np.radians(np.linspace(-101, -79, 5)),
                               div_col, div_lab):
        ln = 0.05 + (v / dmax) * 0.26
        ax_ch.plot([base * np.cos(ang), (base + ln) * np.cos(ang)],
                   [base * np.sin(ang), (base + ln) * np.sin(ang)],
                   color=col, lw=12, solid_capstyle="round", zorder=8)
        ax_ch.text((base + ln + 0.07) * np.cos(ang), (base + ln + 0.07) * np.sin(ang),
                   f"{v:.0f}\n{sl}", ha="center", va="center", fontsize=11,
                   weight="bold", color=col, zorder=9, linespacing=0.95)

    zone_handles = [mpatches.Patch(facecolor=ZONE_COLOR[z], label=z)
                    for z in ZONE_ORDER]
    # legend pinned to the figure top (transFigure) so it stays put when the
    # chord wheel is nudged down -- letter & legend do NOT follow the body
    ax_ch.legend(handles=zone_handles, loc="upper left",
                 bbox_to_anchor=(0.41, 0.95), bbox_transform=fig.transFigure,
                 fontsize=14, frameon=False, handletextpad=0.6, labelspacing=0.45)
    ax_ch.set_xlim(-1.27, 1.27)   # x-range == y-range so the wheel stays round
    ax_ch.set_ylim(-1.40, 1.14)
    # aspect="auto" so the square data ranges + square box keep the wheel round.
    ax_ch.set_aspect("auto")
    # ---- b<->c position swap, panel SIZES unchanged ----
    # the chord keeps its big square box but moves to the TOP-RIGHT; the map
    # keeps its size but moves to the BOTTOM-LEFT; the scatter (d) stays
    # bottom-right, placed below the chord.
    _bw, _bh = 0.591, 0.50                      # chord box size (unchanged)
    _top = gs[0, 1].get_position(fig)           # top-right cell (old map slot)
    _bot = gs[1, :].get_position(fig)           # bottom row (old chord slot)
    # nudge the chord DOWN so its body fills the centre band (size unchanged)
    ax_ch.set_position([0.985 - _bw, _top.y1 - _bh - 0.06, _bw, _bh])
    chp = ax_ch.get_position()
    fig.text(chp.x0 - 0.03, _top.y1, "b", fontsize=20, fontweight="bold",
             ha="left", va="bottom")   # "b" pinned to original top (not the body)

    # ----- (e) follow-the-sun scatter + marginals, bottom-right (below chord) --
    d_w = 0.355
    d_h = d_w * fig.get_figwidth() / fig.get_figheight()   # square display box
    bx0 = 0.985 - d_w
    by0 = _bot.y0 + 0.02
    gap = 0.006
    s_x = d_w * 0.87                                        # main scatter width
    s_y = s_x * fig.get_figwidth() / fig.get_figheight()   # square display main
    mh = d_h - gap - s_y                                    # top marginal height
    mw = d_w - gap - s_x                                    # right marginal width
    ax_d = fig.add_axes([bx0, by0, s_x, s_y])
    ax_dtop = fig.add_axes([bx0, by0 + s_y + gap, s_x, mh])
    ax_dright = fig.add_axes([bx0 + s_x + gap, by0, mw, s_y])
    draw_followsun_panel(ax_d, ax_top=ax_dtop, ax_right=ax_dright)
    fig.text(bx0 - 0.10 * d_w, by0 + d_h, "e", fontsize=20,
             fontweight="bold", ha="left", va="bottom")   # scatter -> "e"

    # relocate the map to the BOTTOM-LEFT, keeping its natural size, vertically
    # centred against the scatter
    # enlarge the map; box aspect (w/h ~ 1.97) matches Robinson so the map
    # FILLS the box and its left edge sits flush at the page margin (no inset
    # whitespace, and it lines up with panels a and c).
    ax_b.set_position([0.012, 0.04, 0.591, 0.30])

    # ----- (e) gap-distribution panel filling the left-middle gap ------------
    mb = ax_b.get_position()        # map, now bottom-left
    ap = ax_a.get_position()        # heatmap main axes (top-left)
    e_x0, e_x1 = 0.012, chp.x0 - 0.004   # left edge aligned with a and the map
    e_y0, e_y1 = mb.y1 + 0.055, ap.y0 - 0.055
    ax_e = fig.add_axes([e_x0, e_y0, e_x1 - e_x0, e_y1 - e_y0])
    draw_gap_distribution_panel(ax_e, r2)
    ep = ax_e.get_position()
    fig.text(ep.x0 - 0.015, ep.y1, "c", fontsize=20, fontweight="bold",
             ha="left", va="bottom")   # distribution -> "c" (reading order)

    # figure-level title -> caption only
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig2_portfolio_scenarios{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2")


def plot_appendix_country_diurnal():
    """Appendix figure (moved out of Fig 2 panel d): four representative
    countries' intraday demand-supply curves under the L3 global portfolio."""
    dn = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    examples = [("JP", "Japan (Asia-Pacific)"), ("DE", "Germany (Europe)"),
                ("MX", "Mexico (North America)"), ("LY", "Libya (Africa/MENA)")]
    l3p = DATA / "r2_l3_diurnal.csv"
    l3_diur = pd.read_csv(l3p) if l3p.exists() else pd.DataFrame(
        columns=["country", "local_hour", "demand_orig", "demand_migrated",
                 "portfolio_w0_fit", "portfolio_w3_fit"])
    fig = plt.figure(figsize=(9.6, 7.2))
    gs_d = fig.add_gridspec(2, 2, left=0.09, right=0.97, top=0.86, bottom=0.09,
                            hspace=0.42, wspace=0.22)
    d_axes = []
    for k, (cc, label) in enumerate(examples):
        ax_d = fig.add_subplot(gs_d[k // 2, k % 2])
        d_axes.append(ax_d)
        sub = dn[dn["country"] == cc].set_index("local_hour")
        diur = l3_diur[l3_diur["country"] == cc].set_index("local_hour")
        if sub.empty:
            continue
        if diur.empty:
            d_orig = sub["demand"].reindex(range(24)); d_mig = d_orig.copy()
            port_w3 = pd.Series(np.nan, index=range(24))
        else:
            d_orig = diur["demand_orig"].reindex(range(24))
            d_mig = diur["demand_migrated"].reindex(range(24))
            port_w3 = diur["portfolio_w3_fit"].reindex(range(24))
        home = sub["supply_vre"].reindex(range(24))
        home_eq = home * (d_orig.mean() / home.mean()) if home.mean() > 0 else home
        ax_d.plot(range(24), d_orig.values, color=HIGHLIGHT, linewidth=2.2,
                  label="Demand (original)")
        ax_d.plot(range(24), d_mig.values, color=HIGHLIGHT, linewidth=1.4,
                  linestyle=(0, (4, 2)), label="Demand after W3 migration")
        ax_d.plot(range(24), home_eq.values, color=NEUTRAL, linewidth=1.3,
                  linestyle=":", label="Home VRE (L0)")
        ax_d.plot(range(24), port_w3.values, color=POSITIVE, linewidth=2.0,
                  label="NNLS L3 portfolio")
        ax_d.fill_between(range(24), d_mig.values, port_w3.values,
                          where=(d_mig.values > port_w3.values),
                          color=HIGHLIGHT, alpha=0.10, interpolate=True)
        ax_d.axvspan(18, 23, zorder=0, **EVENING_KW)
        ax_d.set_xticks(range(0, 24, 6)); ax_d.set_xlim(0, 23)
        ax_d.set_title(label, loc="left", fontsize=11.5, weight="normal")
        if k % 2 == 0:
            ax_d.set_ylabel("Demand / supply\n(equal-energy normalized)", fontsize=12)
        if k // 2 == 1:
            ax_d.set_xlabel("Local hour", fontsize=13)
        if k == 0:
            handles, labels = ax_d.get_legend_handles_labels()
            ax_d.legend(handles, labels, loc="upper center",
                        bbox_to_anchor=(1.1, 1.40), ncol=2, fontsize=10.5,
                        frameon=False, columnspacing=0.8, handlelength=1.7)
        soften_axes(ax_d)
    lo = min(ax.get_ylim()[0] for ax in d_axes)
    hi = max(ax.get_ylim()[1] for ax in d_axes)
    for ax in d_axes:
        ax.set_ylim(lo, hi)
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"figA_country_diurnal{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figA_country_diurnal")


# ---------- Figure 3 ----------
def plot_fig3():
    """Redesigned Figure 3 (4 orthogonal panels, no redundant cost-cube heatmaps).

    Delegates to plot_fig3_redesign.render_fig3 so the 31-country
    (data/ + figures/) and 104-country (data_globalsites/ + figures_globalsites/)
    pipelines share one implementation; DATA/FIG are module-level and may be
    overridden by the caller (see plot_global_sites_figs.py)."""
    import plot_fig3_redesign as _r3
    _r3.render_fig3(DATA, FIG)

def main():
    set_style()
    plot_fig1()
    plot_fig2()
    plot_fig3()


if __name__ == "__main__":
    main()
