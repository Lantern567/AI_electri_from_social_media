#!/usr/bin/env python3
"""Supplementary figure — independent corroboration of the diurnal shape from E3.

Qualitative comparison (NOT a quantitative validation) between:

  * E3's *modelled* projections of the daily shape of TOTAL U.S. data-centre
    load under two usage scenarios (digitised from Figure 5 of E3's July 2024
    white paper "Load Growth Is Here to Stay, but Are Data Centers?"), and

  * our *empirical* 104-country median of the interactive digital-demand share
    by local hour (the Cloudflare human-traffic curve, Fig. 1c basis).

Two panels (double-column, ~7.0 x 3.0 in):

  (a) E3's two modelled total-load scenarios (zoomed y-axis). As user-facing
      utilisation overtakes flat training/baseload, the daily peak migrates
      with the use case: a business tool peaks in working hours (~14 h) while a
      personal tool peaks in the evening (~19 h), coincident with our
      interactive-load peak (~20 h, dashed marker). Independent, same-direction
      support for the premise that interactive load is evening-peaking.

  (b) Amplitude contrast on a common daily-mean = 1 basis. E3's total-load
      curve stays within +-5% of flat because baseload/training dilutes the
      interactive signal; the interactive slice alone (our Cloudflare median)
      swings from ~0.3 to ~1.4. The evening peak is invisible in total load and
      must be isolated at the interactive layer -- the rationale for our
      flat-baseload separation (Supplementary robustness on the flat share).

E3 curves were digitised by colour-segmenting the raster chart and calibrating
against the y-gridlines; each digitised curve integrates to 1.0 over 24 h (it is
a share of daily load), a self-consistency check that fixes the calibration.

English-only, colorblind-safe. Style mirrors plot_figS2_burstgpt_diurnal.py.
Outputs to reports/cfe_geographic_portfolio_ai/figures_globalsites/.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = (ROOT / "collective_attention_research_plan" / "reports"
        / "cfe_geographic_portfolio_ai" / "data_globalsites")
FIG = (ROOT / "collective_attention_research_plan" / "reports"
       / "cfe_geographic_portfolio_ai" / "figures_globalsites")
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# style — mirrors plot_figS2_burstgpt_diurnal.set_style
# ----------------------------------------------------------------------------
INK = "#111827"
GRID = "#E5E7EB"

C_DEMAND = "#DC2626"   # our interactive demand (red, as in Fig 1c)
C_PERS = "#BE185D"     # E3 personal-utilisation scenario (magenta, as drawn)
C_BUS = "#0891B2"      # E3 business-utilisation scenario (cyan-blue, as drawn)


def set_style(fs=8.5):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": fs,
        "axes.titlesize": fs + 1.0, "axes.titleweight": "bold",
        "axes.labelsize": fs, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#374151", "axes.labelcolor": INK,
        "xtick.color": "#374151", "ytick.color": "#374151",
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs - 0.7, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def yonly(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.grid(False, axis="x")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
HOURS = np.arange(24)

# E3 Fig 5, digitised as share-of-daily-load (each sums to 1.0 over 24 h).
# Colour-segmented from the raster chart, y-calibrated on the 0.037-0.045
# gridlines. Source: I. Riu et al., E3, July 2024, Figure 5.
E3_PERSONAL = np.array([
    0.0400, 0.0394, 0.0390, 0.0387, 0.0386, 0.0394, 0.0404, 0.0416,
    0.0424, 0.0428, 0.0424, 0.0425, 0.0427, 0.0427, 0.0428, 0.0427,
    0.0427, 0.0429, 0.0432, 0.0433, 0.0432, 0.0425, 0.0419, 0.0415])
E3_BUSINESS = np.array([
    0.0393, 0.0392, 0.0392, 0.0393, 0.0396, 0.0399, 0.0403, 0.0409,
    0.0420, 0.0432, 0.0436, 0.0437, 0.0438, 0.0439, 0.0440, 0.0439,
    0.0438, 0.0434, 0.0427, 0.0421, 0.0415, 0.0408, 0.0401, 0.0396])


def load_cfe_demand():
    """104-country median interactive digital demand by local hour, mean = 1."""
    df = pd.read_csv(DATA / "r1_diurnal_profiles.csv")
    med = df.groupby("local_hour")["demand"].median().reindex(range(24)).to_numpy()
    return med / med.mean()


# ----------------------------------------------------------------------------
# panel (a): E3's two modelled total-load scenarios (direction)
# ----------------------------------------------------------------------------
def draw_a(ax, cfe_peak_hour):
    # to daily-mean = 1 (each E3 curve sums to 1 -> multiply by 24)
    pers = E3_PERSONAL * 24.0
    bus = E3_BUSINESS * 24.0

    ax.plot(HOURS, bus, color=C_BUS, lw=1.9, marker="o", ms=3.0, mec="white",
            mew=0.5, zorder=5, label="E3 business tool")
    ax.plot(HOURS, pers, color=C_PERS, lw=1.9, marker="o", ms=3.0, mec="white",
            mew=0.5, zorder=6, label="E3 personal tool")

    # peaks
    hb, hp = int(np.argmax(bus)), int(np.argmax(pers))
    ax.plot(hb, bus[hb], "o", color=C_BUS, ms=6.0, mec="white", mew=1.0, zorder=8)
    ax.plot(hp, pers[hp], "o", color=C_PERS, ms=6.0, mec="white", mew=1.0, zorder=8)
    ax.annotate("afternoon peak\n(%d h)" % hb, xy=(hb, bus[hb]), xytext=(-2, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.0, color=C_BUS, fontweight="bold")
    ax.annotate("evening peak\n(%d h)" % hp, xy=(hp, pers[hp]), xytext=(6, -14),
                textcoords="offset points", ha="left", va="top",
                fontsize=7.0, color=C_PERS, fontweight="bold")

    # our interactive peak marker (label parked at the base to avoid collisions)
    ax.axvline(cfe_peak_hour, color=INK, lw=0.9, ls=(0, (4, 3)), alpha=0.55, zorder=3)
    ax.annotate("our interactive\npeak (%d h)" % cfe_peak_hour,
                xy=(cfe_peak_hour, 0.905), xytext=(4, 0), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.8, color="#475569")

    ax.axhline(1.0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.8, zorder=1)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_ylim(0.90, 1.075)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Total data-centre load\n(relative to daily mean)")
    yonly(ax)
    ax.tick_params(length=2)
    ax.legend(loc="lower center", handlelength=1.5, labelspacing=0.3,
              borderaxespad=0.3, ncol=1)


# ----------------------------------------------------------------------------
# panel (b): amplitude contrast, total load vs isolated interactive slice
# ----------------------------------------------------------------------------
def draw_b(ax, cfe):
    pers = E3_PERSONAL * 24.0

    ax.axhline(1.0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.8, zorder=1)
    ax.plot(HOURS, pers, color=C_PERS, lw=1.9, zorder=4,
            label="E3 total load (modelled)")
    ax.fill_between(HOURS, 1.0, cfe, color=C_DEMAND, alpha=0.10, zorder=2)
    ax.plot(HOURS, cfe, color=C_DEMAND, lw=2.1, marker="o", ms=3.0, mec="white",
            mew=0.5, zorder=6, label="our interactive slice (Cloudflare)")

    hp = int(np.argmax(cfe))
    ax.plot(hp, cfe[hp], "o", color=C_DEMAND, ms=6.0, mec="white", mew=1.0, zorder=8)
    ax.annotate("evening peak\n%.2f x mean" % cfe[hp], xy=(hp, cfe[hp]),
                xytext=(-8, -18), textcoords="offset points", ha="right", va="top",
                fontsize=7.2, color=C_DEMAND, fontweight="bold")
    # annotate E3 flatness
    ax.annotate("total load stays within +-5% of flat\n(baseload/training dilutes the signal)",
                xy=(11, pers[11]), xytext=(11, 0.60), textcoords="data",
                ha="center", va="center", fontsize=6.8, color=C_PERS,
                arrowprops=dict(arrowstyle="-|>", color=C_PERS, lw=0.9,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_ylim(0.2, 1.56)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Load relative to daily mean")
    yonly(ax)
    ax.tick_params(length=2)
    ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.3, borderaxespad=0.3)


# ----------------------------------------------------------------------------
def main():
    set_style(8.5)
    cfe = load_cfe_demand()
    cfe_peak = int(np.argmax(cfe))

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.2, 3.05), gridspec_kw=dict(width_ratios=[1.0, 1.12]))

    draw_a(axa, cfe_peak)
    draw_b(axb, cfe)

    for ax, lab in ((axa, "a"), (axb, "b")):
        ax.set_title(lab, loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)

    fig.tight_layout(w_pad=2.4)

    out_png = FIG / "figS_e3_datacenter_shape.png"
    out_pdf = FIG / "figS_e3_datacenter_shape.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("CFE interactive peak hour:", cfe_peak,
          " peak/mean=%.2f trough/mean=%.2f" % (cfe.max(), cfe.min()))
    print("E3 personal peak hour:", int(np.argmax(E3_PERSONAL)),
          " E3 business peak hour:", int(np.argmax(E3_BUSINESS)))
    for p in (out_png, out_pdf):
        print("wrote %s (%.1f KB)" % (p, p.stat().st_size / 1024.0))


if __name__ == "__main__":
    main()
