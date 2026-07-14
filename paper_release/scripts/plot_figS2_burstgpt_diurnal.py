#!/usr/bin/env python3
"""Supplementary Figure S2 — BurstGPT v2 intraday inference shape.

Two panels (double-column, ~7.0 x 3.0 in):

  (a) empirical 24-hour token-share (bars) overlaid with the Gaussian fit
      (mu = 18.05, sigma = 3.79, token weighting) and a shaded 95% bootstrap
      CI band built by re-evaluating the Gaussian for every bootstrap
      (peak_hour, sigma_h) draw and taking the 2.5-97.5 percentile envelope
      across hours. The empirical evening peak is bimodal (hour 17 leading,
      19-20 secondary); the fitted peak is marked near hour 18.

  (b) robustness dumbbell of the three trace slices in (peak_hour, sigma_h)
      space, with the pooled token fit highlighted and each slice's hourly
      P99/mean burst ratio labelled.

English-only, colorblind-safe, mathtext symbols. Style conventions
(rcParams / INK / GRID / palette) reuse those of plot_figS_ai_operators.py
(via plot_fig3_redesign.set_style). Data under
reports/ai_inference_growth_curve_fit/. Outputs to
reports/cfe_geographic_portfolio_ai/figures_globalsites/.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "collective_attention_research_plan" / "reports" / "ai_inference_growth_curve_fit"
FIG = (ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "figures_globalsites")
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# style — mirrors plot_fig3_redesign.set_style (used by plot_figS_ai_operators)
# ----------------------------------------------------------------------------
INK = "#111827"
GRID = "#E5E7EB"

# colorblind-safe palette (Okabe-Ito flavoured; consistent with repo hues)
C_EMP = "#64748B"     # empirical bars (slate grey)
C_FIT = "#7C3AED"     # Gaussian fit (purple = headline Mix hue)
C_CI = "#C4B5E8"      # CI band (light purple)
C_PEAK = "#DC2626"    # peak marker (red)
C_POOL = "#0F766E"    # pooled value (teal)
C_SLICE = "#B45309"   # trace slices (amber/brown)


def set_style(fs=8.5):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.titlesize": fs + 1.0, "axes.titleweight": "bold",
        "axes.labelsize": fs, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#374151", "axes.labelcolor": INK,
        "xtick.color": "#374151", "ytick.color": "#374151",
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs - 0.5, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def yonly(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.grid(False, axis="x")


def gaussian(h, baseline, amplitude, mu, sigma):
    return baseline + amplitude * np.exp(-((h - mu) ** 2) / (2.0 * sigma ** 2))


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load():
    diur = pd.read_csv(DATA / "burstgpt_diurnal_fit_v2.csv")
    with open(DATA / "burstgpt_workload_shape_params_v2.json", "r", encoding="utf-8") as fh:
        params = json.load(fh)
    boot = pd.read_csv(DATA / "burstgpt_bootstrap_ci.csv")
    stat = pd.read_csv(DATA / "burstgpt_stationarity.csv")
    return diur, params, boot, stat


# ----------------------------------------------------------------------------
# panel (a): empirical share + Gaussian fit + bootstrap CI band
# ----------------------------------------------------------------------------
def draw_a(ax, diur, params, boot):
    tok = params["gaussian_fit"]["tokens"]
    mu, sigma = tok["peak_hour"], tok["sigma_h"]
    baseline, amplitude = tok["baseline_share"], tok["amplitude"]

    hours = diur["hour"].to_numpy()
    share = diur["share_tokens"].to_numpy()

    # smooth grid for the fitted curve and CI envelope
    hg = np.linspace(0.0, 24.0, 481)

    # empirical bars
    ax.bar(hours, share, width=0.86, color=C_EMP, alpha=0.40,
           edgecolor=C_EMP, linewidth=0.5, zorder=2, label="empirical (BurstGPT v2)")

    # bootstrap CI band: re-evaluate the Gaussian for every (peak_hour, sigma_h)
    # draw, keeping the pooled baseline/amplitude, then take the 2.5-97.5
    # percentile envelope across hours.
    curves = np.empty((len(boot), hg.size))
    for k, (ph, sg) in enumerate(zip(boot["peak_hour"], boot["sigma_h"])):
        curves[k] = gaussian(hg, baseline, amplitude, ph, sg)
    lo = np.percentile(curves, 2.5, axis=0)
    hi = np.percentile(curves, 97.5, axis=0)
    ax.fill_between(hg, lo, hi, color=C_CI, alpha=0.65, linewidth=0, zorder=3,
                    label="95% bootstrap CI")

    # fitted Gaussian (pooled token weighting)
    yfit = gaussian(hg, baseline, amplitude, mu, sigma)
    ax.plot(hg, yfit, color=C_FIT, lw=1.8, zorder=5, label="Gaussian fit")

    # mark the fitted peak
    ax.axvline(mu, color=C_PEAK, lw=1.0, ls="--", alpha=0.85, zorder=4)
    ypk = gaussian(mu, baseline, amplitude, mu, sigma)
    ax.plot(mu, ypk, "o", color=C_PEAK, ms=5.0, mec="white", mew=0.9, zorder=6)
    ax.annotate(r"$\mu=%.1f$ h" % mu, xy=(mu, ypk), xytext=(5, 6),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8.0, color=C_PEAK, fontweight="bold")

    # annotate the bimodal empirical structure (leading 17, secondary 19-20)
    i17 = int(np.where(hours == 17)[0][0])
    i19 = int(np.where(hours == 19)[0][0])
    ax.annotate("17", xy=(17, share[i17]), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.0, color="#475569")
    ax.annotate("19-20", xy=(19.5, share[i19]), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.0, color="#475569")

    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0, max(share.max(), hi.max()) * 1.18)
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("Normalized token share")
    yonly(ax)
    ax.tick_params(length=2)

    leg = ax.legend(loc="upper left", handlelength=1.4, labelspacing=0.35,
                    borderaxespad=0.2)
    leg.set_zorder(20)

    return mu, sigma


# ----------------------------------------------------------------------------
# panel (b): stationarity dumbbell of the three trace slices in (mu, sigma)
# ----------------------------------------------------------------------------
def draw_b(ax, params, stat):
    tok = params["gaussian_fit"]["tokens"]
    mu_p, sg_p = tok["peak_hour"], tok["sigma_h"]
    gamma_pool = params["burst_stats_hourly"]["tokens"]["p99_over_mean"]

    # 95% CI rectangle for the pooled fit (mu x sigma)
    ci = params["bootstrap_ci_95pct_tokens"]
    mu_lo, mu_hi = ci["peak_hour_p2_5"], ci["peak_hour_p97_5"]
    sg_lo, sg_hi = ci["sigma_h_p2_5"], ci["sigma_h_p97_5"]
    ax.add_patch(mpatches.Rectangle(
        (mu_lo, sg_lo), mu_hi - mu_lo, sg_hi - sg_lo,
        facecolor=C_POOL, alpha=0.12, edgecolor=C_POOL, lw=0.7, ls="--", zorder=2))

    # the three trace slices
    for row in stat.itertuples():
        ph, sg, gm = row.peak_hour, row.sigma_h, row.p99_over_mean
        # dumbbell connector from the slice to the pooled value
        ax.plot([mu_p, ph], [sg_p, sg], color="#CBD5E1", lw=1.4, zorder=3,
                solid_capstyle="round")
        ax.plot(ph, sg, "o", color=C_SLICE, ms=8.0, mec="white", mew=1.0, zorder=5)
        ax.annotate("slice %s\nP99/mean=%.1f" % (int(row.trace_slice), gm),
                    xy=(ph, sg), xytext=(7, -2), textcoords="offset points",
                    ha="left", va="center", fontsize=6.8, color="#475569")

    # pooled token fit, highlighted
    ax.plot(mu_p, sg_p, "D", color=C_POOL, ms=9.0, mec="white", mew=1.1, zorder=6)
    ax.annotate("pooled fit\nP99/mean=%.1f" % gamma_pool,
                xy=(mu_p, sg_p), xytext=(-7, 9), textcoords="offset points",
                ha="right", va="bottom", fontsize=7.2, color=C_POOL, fontweight="bold")

    ax.set_xlabel(r"Peak hour $\mu$ (h)")
    ax.set_ylabel(r"Width $\sigma$ (h)")
    ax.set_xlim(16.6, 18.9)
    ax.set_ylim(3.0, 4.45)
    ax.set_xticks([17.0, 17.5, 18.0, 18.5])
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(length=2)

    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_POOL,
               markeredgecolor="white", markersize=8, label="pooled token fit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_SLICE,
               markeredgecolor="white", markersize=8, label="trace slices 1-3"),
        mpatches.Patch(facecolor=C_POOL, alpha=0.12, edgecolor=C_POOL, ls="--",
                       label="95% bootstrap CI"),
    ]
    ax.legend(handles=handles, loc="lower left", handlelength=1.2,
              labelspacing=0.4, borderaxespad=0.25)


# ----------------------------------------------------------------------------
def main():
    set_style(8.5)
    diur, params, boot, stat = load()

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.55, 1.0]))

    draw_a(axa, diur, params, boot)
    draw_b(axb, params, stat)

    # bold panel labels, top-left (set_title is never clipped)
    for ax, lab in ((axa, "a"), (axb, "b")):
        ax.set_title(lab, loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)

    fig.tight_layout(w_pad=2.2)

    out_png = FIG / "figS2_burstgpt_diurnal.png"
    out_pdf = FIG / "figS2_burstgpt_diurnal.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    for p in (out_png, out_pdf):
        print("wrote %s (%.1f KB)" % (p, p.stat().st_size / 1024.0))


if __name__ == "__main__":
    main()
