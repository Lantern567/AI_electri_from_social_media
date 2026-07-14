#!/usr/bin/env python3
"""Supplementary Figure S3 — Cross-country distribution of the human-traffic
peak-to-mean ratio (d_burst/mean) and the derived burst-amplification coefficient.

Panel (a): the 104-country distribution of the hourly human-traffic peak-to-mean
ratio d_burst/mean = P99/mean (Cloudflare Radar), as a histogram with a soft KDE
overlay; the cross-country median (1.716) is marked and the IQR [1.632, 1.885]
is shaded. This median replaces the prior hard-coded constant 1.75.

Panel (b): the mapping to the P3 burst-amplification operator coefficient
gamma_op = (gamma_emp - 1) / (d_burst/mean) with gamma_emp = 9.31 (BurstGPT
pooled tokens). The IQR-derived band [4.41, 5.09] and the recommended default
4.84 at the median 1.716 are marked.

English-only, colorblind-safe, matplotlib. Reuses the repo figure rcParams
conventions (DejaVu Sans, INK/GRID, 8-9 pt fonts, clean spines).
Data: reports/ai_inference_growth_curve_fit/{country_burst_ratios.csv,
country_burst_ratio_summary.json}.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPORT = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"
FIG = ROOT / "collective_attention_research_plan" / "reports" / "cfe_geographic_portfolio_ai" / "figures_globalsites"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style (repo convention)
INK = "#111827"
GRID = "#E5E7EB"
# colorblind-safe accents
C_HIST = "#3B82F6"   # blue  — distribution bars
C_KDE = "#1E3A8A"    # deep blue — density curve
C_MED = "#B45309"    # amber/brown — median marker (legible)
C_IQR = "#94A3B8"    # slate — IQR shade
C_CURVE = "#0F766E"  # teal — gamma_op curve
C_BAND = "#5EEAD4"   # light teal — gamma_op IQR band


def set_style(fs=8.5):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.titlesize": fs + 1.0, "axes.titleweight": "bold", "axes.labelsize": fs,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#374151", "axes.labelcolor": INK,
        "xtick.color": "#374151", "ytick.color": "#374151",
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs - 0.5, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.6,
        "mathtext.default": "regular",
    })


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(REPORT / "country_burst_ratios.csv")
    with open(REPORT / "country_burst_ratio_summary.json") as fh:
        js = json.load(fh)
    return df, js


def main():
    df, js = load()
    x = df["d_burst_over_mean"].to_numpy(dtype=float)
    n = len(x)

    s = js["d_burst_over_mean"]
    p25, med, mean, p75 = s["p25"], s["p50"], s["mean"], s["p75"]
    xmin, xmax = s["min"], s["max"]

    g = js["gamma_op_from_burstgpt"]
    gamma_emp = g["gamma_emp_pooled_tokens"]
    gop_med = g["p50"]            # 4.84 recommended default
    gop_p25, gop_p75 = g["p25"], g["p75"]   # band [4.41, 5.09]

    set_style(8.5)
    plt.close("all")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ===================================================== panel (a): distribution
    lo, hi = 1.30, 3.00
    bins = np.arange(lo, hi + 1e-9, 0.075)
    # IQR shade (behind everything)
    axa.axvspan(p25, p75, color=C_IQR, alpha=0.28, zorder=0, lw=0)
    cnts, edges, _ = axa.hist(x, bins=bins, color=C_HIST, alpha=0.80,
                              edgecolor="white", linewidth=0.4, zorder=2)
    # soft KDE overlay, scaled to the histogram (counts) for the same y-axis
    kde = gaussian_kde(x)
    xs = np.linspace(lo, hi, 400)
    binw = bins[1] - bins[0]
    dens = kde(xs) * n * binw
    axa.plot(xs, dens, color=C_KDE, lw=1.6, zorder=4)
    axa.fill_between(xs, 0, dens, color=C_KDE, alpha=0.06, zorder=1)

    ymax = max(cnts.max(), dens.max()) * 1.18
    # median line + small (non-boxed) label
    axa.axvline(med, color=C_MED, lw=1.8, zorder=5)
    axa.text(med + 0.025, ymax * 0.94, f"median {med:.3f}", color=C_MED,
             fontsize=8.0, fontweight="bold", ha="left", va="top", rotation=0)
    # IQR end-tick labels (small, just inside the axis so they clear the x-ticks)
    axa.text(p25 - 0.012, ymax * 0.035, f"{p25:.2f}", color="#475569", fontsize=6.8,
             ha="right", va="bottom")
    axa.text(p75 + 0.012, ymax * 0.035, f"{p75:.2f}", color="#475569", fontsize=6.8,
             ha="left", va="bottom")
    axa.text((med + p75) / 2.0, ymax * 0.86, "IQR", color="#475569", fontsize=7.0,
             ha="center", va="center", style="italic")

    axa.set_xlim(lo, hi)
    axa.set_ylim(0, ymax)
    axa.set_xlabel(r"Country peak-to-mean ratio  $d_{\mathrm{burst}}/\mathrm{mean}$"
                   "\n(hourly human traffic)", fontsize=8.5)
    axa.set_ylabel("Number of countries", fontsize=8.5)
    axa.grid(True, axis="y")
    axa.grid(False, axis="x")
    # small inline n note (no box)
    axa.text(0.97, 0.97, f"$n$ = {n} countries", transform=axa.transAxes,
             fontsize=7.2, color="#475569", ha="right", va="top")

    # ===================================================== panel (b): gamma_op mapping
    dx = np.linspace(xmin, xmax, 400)
    gop = (gamma_emp - 1.0) / dx
    # band from the IQR of d_burst/mean -> [gop(p75), gop(p25)] = [4.41, 5.09]
    axb.axhspan(gop_p25, gop_p75, color=C_BAND, alpha=0.40, zorder=0, lw=0)
    axb.plot(dx, gop, color=C_CURVE, lw=1.8, zorder=3)

    # recommended default at the median
    axb.plot([med], [gop_med], "o", color=C_MED, ms=6.5, mec="white", mew=1.0, zorder=5)
    # guide lines to the axes
    axb.plot([xmin, med], [gop_med, gop_med], color=C_MED, lw=0.8, ls=":", zorder=2)
    axb.plot([med, med], [axb.get_ylim()[0], gop_med], color=C_MED, lw=0.8, ls=":", zorder=2)
    axb.annotate(rf"$\gamma_{{\mathrm{{op}}}}$ = {gop_med:.2f}",
                 (med, gop_med), xytext=(10, 10), textcoords="offset points",
                 fontsize=8.0, fontweight="bold", color=C_MED, ha="left", va="bottom")

    # band labels (small, near right edge, no box)
    axb.text(xmax, gop_p75, f"{gop_p75:.2f}", color="#0F766E", fontsize=7.0,
             ha="left", va="center")
    axb.text(xmax, gop_p25, f"{gop_p25:.2f}", color="#0F766E", fontsize=7.0,
             ha="left", va="center")
    axb.text(0.96, 0.05,
             rf"$\gamma_{{\mathrm{{op}}}} = (\gamma_{{\mathrm{{emp}}}}-1)\,/\,(d_{{\mathrm{{burst}}}}/\mathrm{{mean}})$"
             "\n"
             rf"$\gamma_{{\mathrm{{emp}}}}$ = {gamma_emp:.2f}",
             transform=axb.transAxes, fontsize=7.2, color="#334155",
             ha="right", va="bottom")

    axb.set_xlim(xmin, xmax)
    axb.set_xlabel(r"Country peak-to-mean ratio  $d_{\mathrm{burst}}/\mathrm{mean}$",
                   fontsize=8.5)
    axb.set_ylabel(r"$\gamma_{\mathrm{op}}$ (P3 burst amplification)", fontsize=8.5)
    axb.grid(True)

    # ---------------------------------------------------------------- panel labels
    for ax, lab in ((axa, "a"), (axb, "b")):
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="bottom")

    fig.tight_layout(w_pad=2.2)

    out_png = FIG / "figS3_burst_ratio_dist.png"
    out_pdf = FIG / "figS3_burst_ratio_dist.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    for p in (out_png, out_pdf):
        print(f"wrote {p}  ({p.stat().st_size/1024:.1f} kB)")
    # echo key numbers for the caption
    print(f"n={n}  min={xmin:.3f} p25={p25:.3f} median={med:.3f} "
          f"mean={mean:.3f} p75={p75:.3f} max={xmax:.3f}")
    print(f"gamma_emp={gamma_emp:.3f}  gamma_op: p25={gop_p25:.3f} "
          f"median(default)={gop_med:.3f} p75={gop_p75:.3f}")


if __name__ == "__main__":
    main()
