#!/usr/bin/env python3
"""Fit literature-anchored AI/data-centre growth curves for PTP Result 3.

The goal is deliberately modest:
1. fit short-horizon curves from published numeric anchors;
2. derive workload shares from those fitted totals;
3. estimate an internal attention-to-human-traffic response curve as a proxy
   for event amplification, without calling it an external AI-inference curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("collective_attention_research_plan")
OUT_DIR = ROOT / "reports" / "ai_inference_growth_curve_fit"
FIG_DIR = OUT_DIR / "figures"

ATTENTION_PANEL = (
    ROOT
    / "reports"
    / "wikimedia_all_project_country_top_attention_event_study"
    / "country_wikimedia_attention_cloudflare_panel.csv.gz"
)
CFE_PANEL = (
    ROOT
    / "reports"
    / "prime_time_cfe_penalty_ptp_r2_r3"
    / "ptp_country_hourly_cfe_panel.csv.gz"
)
SELECTED_BURSTS = (
    ROOT
    / "reports"
    / "prime_time_cfe_penalty_ptp_r2_r3"
    / "ptp_selected_country_attention_bursts.csv"
)


@dataclass(frozen=True)
class CurveAnchor:
    source: str
    curve_id: str
    year: int
    value: float
    unit: str
    note: str
    url: str


def anchors() -> pd.DataFrame:
    rows = [
        # McKinsey Data Center Demand Model, reported in McKinsey article.
        CurveAnchor("McKinsey", "mckinsey_total_dc_capacity_gw", 2025, 82.3, "GW", "Global data centre demand capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_total_dc_capacity_gw", 2030, 219.0, "GW", "Global data centre demand capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_non_ai_capacity_gw", 2025, 38.3, "GW", "Non-AI workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_non_ai_capacity_gw", 2030, 63.5, "GW", "Non-AI workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_ai_inference_capacity_gw", 2025, 20.9, "GW", "AI inference workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_ai_inference_capacity_gw", 2030, 93.3, "GW", "AI inference workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_ai_training_capacity_gw", 2025, 23.1, "GW", "AI training workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        CurveAnchor("McKinsey", "mckinsey_ai_training_capacity_gw", 2030, 62.2, "GW", "AI training workload capacity", "https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-next-big-shifts-in-ai-workloads-and-hyperscaler-strategies"),
        # Gartner press release.
        CurveAnchor("Gartner", "gartner_total_dc_twh", 2025, 448.0, "TWh/yr", "Worldwide data-centre electricity consumption", "https://www.gartner.com/en/newsroom/press-releases/gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030"),
        CurveAnchor("Gartner", "gartner_total_dc_twh", 2030, 980.0, "TWh/yr", "Worldwide data-centre electricity consumption", "https://www.gartner.com/en/newsroom/press-releases/gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030"),
        CurveAnchor("Gartner", "gartner_ai_optimized_server_twh", 2025, 93.0, "TWh/yr", "AI-optimized server electricity use", "https://www.gartner.com/en/newsroom/press-releases/gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030"),
        CurveAnchor("Gartner", "gartner_ai_optimized_server_twh", 2030, 432.0, "TWh/yr", "AI-optimized server electricity use", "https://www.gartner.com/en/newsroom/press-releases/gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030"),
        # IEA Energy and AI.
        CurveAnchor("IEA", "iea_global_dc_twh_base", 2024, 415.0, "TWh/yr", "Base-case global data-centre electricity use", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_twh_base", 2030, 945.0, "TWh/yr", "Base-case global data-centre electricity use", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_twh_base", 2035, 1200.0, "TWh/yr", "Approximate Base Case after 2030", "https://www.iea.org/reports/energy-and-ai/executive-summary"),
        CurveAnchor("IEA", "iea_global_dc_share_percent_base", 2024, 1.5, "%", "Share of global electricity consumption", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_share_percent_base", 2030, 2.9, "%", "Just under 3%; encoded as 2.9 for fitting", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_twh_headwinds", 2035, 700.0, "TWh/yr", "Headwinds sensitivity case", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_twh_high_efficiency", 2035, 970.0, "TWh/yr", "High Efficiency sensitivity case", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_twh_lift_off", 2035, 1700.0, "TWh/yr", "Lift-Off sensitivity case", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_share_percent_high_efficiency", 2035, 2.6, "%", "High Efficiency share of global electricity", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        CurveAnchor("IEA", "iea_global_dc_share_percent_lift_off", 2035, 4.4, "%", "Lift-Off share of global electricity", "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai"),
        # IEA 4E / EDNA critical review.
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_low", 2023, 10.0, "TWh/yr", "Low estimate of current AI-related data-centre electricity", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_mid", 2023, 30.0, "TWh/yr", "Midpoint proxy within 10-50 TWh current range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_high", 2023, 50.0, "TWh/yr", "High estimate of current AI-related data-centre electricity", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_low", 2030, 200.0, "TWh/yr", "Low plausible AI-related 2030 range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_mid", 2030, 300.0, "TWh/yr", "Midpoint proxy within 200-400 TWh plausible range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_twh_high", 2030, 400.0, "TWh/yr", "High plausible AI-related 2030 range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_low", 2023, 5.0, "%", "Low current share", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_mid", 2023, 10.0, "%", "Midpoint proxy within 5-15% current range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_high", 2023, 15.0, "%", "High current share", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_low", 2030, 35.0, "%", "Low plausible 2030 share", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_mid", 2030, 42.5, "%", "Midpoint proxy within 35-50% plausible 2030 range", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
        CurveAnchor("IEA-4E/EDNA", "edna_ai_related_dc_share_high", 2030, 50.0, "%", "High plausible 2030 share", "https://www.iea-4e.org/wp-content/uploads/2025/05/Data-Centre-Energy-Use-Critical-Review-of-Models-and-Results.pdf"),
    ]
    return pd.DataFrame([r.__dict__ for r in rows])


def exponential_curve(anchor_df: pd.DataFrame, curve_id: str) -> pd.DataFrame:
    sub = anchor_df[anchor_df["curve_id"] == curve_id].sort_values("year")
    years = np.arange(int(sub["year"].min()), int(sub["year"].max()) + 1)
    x = sub["year"].to_numpy(dtype=float)
    y = sub["value"].to_numpy(dtype=float)
    pred = np.full(len(years), np.nan, dtype=float)
    cagr = np.full(len(years), np.nan, dtype=float)
    # Piecewise log-linear interpolation. This preserves every published
    # numeric anchor exactly, unlike a global exponential least-squares fit.
    for i in range(len(x) - 1):
        start_y, end_y = x[i], x[i + 1]
        start_v, end_v = y[i], y[i + 1]
        mask = (years >= start_y) & (years <= end_y)
        segment_growth = (end_v / start_v) ** (1.0 / (end_y - start_y)) - 1.0
        pred[mask] = start_v * (1.0 + segment_growth) ** (years[mask] - start_y)
        cagr[mask] = segment_growth
    if len(x) == 1:
        pred[:] = y[0]
        cagr[:] = np.nan
    return pd.DataFrame(
        {
            "curve_id": curve_id,
            "year": years,
            "value": pred,
            "unit": sub["unit"].iloc[0],
            "method": "piecewise log-linear interpolation",
            "annual_growth_rate": cagr,
        }
    )


def logit_share_curve(anchor_df: pd.DataFrame, curve_id: str) -> pd.DataFrame:
    sub = anchor_df[anchor_df["curve_id"] == curve_id].sort_values("year")
    years = np.arange(int(sub["year"].min()), int(sub["year"].max()) + 1)
    x = sub["year"].to_numpy(dtype=float)
    p = sub["value"].to_numpy(dtype=float) / 100.0
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    coef = np.polyfit(x - x[0], logit, 1)
    pred_logit = coef[1] + coef[0] * (years - x[0])
    pred = 100.0 / (1.0 + np.exp(-pred_logit))
    return pd.DataFrame(
        {
            "curve_id": curve_id,
            "year": years,
            "value": pred,
            "unit": "%",
            "method": "linear logit share fit",
            "annual_growth_rate": np.nan,
        }
    )


def build_external_curves(anchor_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exponential_ids = [
        "mckinsey_total_dc_capacity_gw",
        "mckinsey_non_ai_capacity_gw",
        "mckinsey_ai_inference_capacity_gw",
        "mckinsey_ai_training_capacity_gw",
        "gartner_total_dc_twh",
        "gartner_ai_optimized_server_twh",
        "iea_global_dc_twh_base",
        "edna_ai_related_dc_twh_low",
        "edna_ai_related_dc_twh_mid",
        "edna_ai_related_dc_twh_high",
    ]
    share_ids = [
        "iea_global_dc_share_percent_base",
        "edna_ai_related_dc_share_low",
        "edna_ai_related_dc_share_mid",
        "edna_ai_related_dc_share_high",
    ]
    curves = [exponential_curve(anchor_df, cid) for cid in exponential_ids]
    curves.extend(logit_share_curve(anchor_df, cid) for cid in share_ids)
    fitted = pd.concat(curves, ignore_index=True)

    wide = fitted.pivot_table(index="year", columns="curve_id", values="value", aggfunc="first")
    derived = pd.DataFrame({"year": wide.index})
    if {"mckinsey_ai_inference_capacity_gw", "mckinsey_total_dc_capacity_gw"}.issubset(wide.columns):
        derived["mckinsey_inference_share_total_dc_percent"] = (
            100 * wide["mckinsey_ai_inference_capacity_gw"] / wide["mckinsey_total_dc_capacity_gw"]
        ).to_numpy()
    if {"mckinsey_ai_inference_capacity_gw", "mckinsey_ai_training_capacity_gw"}.issubset(wide.columns):
        ai_total = wide["mckinsey_ai_inference_capacity_gw"] + wide["mckinsey_ai_training_capacity_gw"]
        derived["mckinsey_inference_share_ai_percent"] = (
            100 * wide["mckinsey_ai_inference_capacity_gw"] / ai_total
        ).to_numpy()
    if {"mckinsey_ai_inference_capacity_gw", "mckinsey_ai_training_capacity_gw", "mckinsey_total_dc_capacity_gw"}.issubset(wide.columns):
        ai_total = wide["mckinsey_ai_inference_capacity_gw"] + wide["mckinsey_ai_training_capacity_gw"]
        derived["mckinsey_ai_share_total_dc_percent"] = (100 * ai_total / wide["mckinsey_total_dc_capacity_gw"]).to_numpy()
    if {"gartner_ai_optimized_server_twh", "gartner_total_dc_twh"}.issubset(wide.columns):
        derived["gartner_ai_optimized_share_total_dc_percent"] = (
            100 * wide["gartner_ai_optimized_server_twh"] / wide["gartner_total_dc_twh"]
        ).to_numpy()

    derived_long = derived.melt(id_vars="year", var_name="curve_id", value_name="value").dropna()
    derived_long["unit"] = "%"
    derived_long["method"] = "derived ratio from fitted source curves"
    derived_long["annual_growth_rate"] = np.nan
    fitted = pd.concat([fitted, derived_long], ignore_index=True)
    return fitted.sort_values(["curve_id", "year"]).reset_index(drop=True), derived


def fit_attention_response() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(SELECTED_BURSTS, parse_dates=["event_time_utc"])
    events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], utc=True)

    att_cols = [
        "timestamp_utc",
        "cf_location",
        "human_volume_proxy",
        "human_volume_proxy_anomaly",
        "country_wiki_shock_sum",
    ]
    att = pd.read_csv(ATTENTION_PANEL, usecols=att_cols, parse_dates=["timestamp_utc"])
    att["timestamp_utc"] = pd.to_datetime(att["timestamp_utc"], utc=True)
    cfe = pd.read_csv(CFE_PANEL, usecols=["timestamp_utc", "cf_location", "local_hour"], parse_dates=["timestamp_utc"])
    cfe["timestamp_utc"] = pd.to_datetime(cfe["timestamp_utc"], utc=True)
    panel = att.merge(cfe, on=["cf_location", "timestamp_utc"], how="left")
    panel["month"] = panel["timestamp_utc"].dt.month.astype(int)

    baseline = (
        panel.groupby(["cf_location", "month", "local_hour"], as_index=False)["human_volume_proxy"]
        .mean()
        .rename(columns={"human_volume_proxy": "matched_human_baseline"})
    )
    event_panel = events.merge(
        panel,
        left_on=["cf_location", "event_time_utc"],
        right_on=["cf_location", "timestamp_utc"],
        how="left",
        validate="one_to_one",
    )
    event_panel = event_panel.merge(baseline, on=["cf_location", "month", "local_hour"], how="left")
    event_panel["traffic_ratio_matched"] = event_panel["human_volume_proxy"] / event_panel["matched_human_baseline"]
    event_panel["log1p_event_shock"] = np.log1p(event_panel["event_shock"].astype(float))
    event_panel["log_traffic_ratio"] = np.log(event_panel["traffic_ratio_matched"].clip(lower=1e-9))
    event_panel = event_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log1p_event_shock", "log_traffic_ratio", "traffic_ratio_matched"]
    )

    x = event_panel["log1p_event_shock"].to_numpy(dtype=float)
    y = event_panel["log_traffic_ratio"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    r2 = 1 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
    n = len(y)
    sigma2 = float(np.sum(resid**2) / max(n - 2, 1))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    shock_grid = np.quantile(event_panel["event_shock"].to_numpy(dtype=float), np.linspace(0.05, 0.95, 19))
    pred_log = beta[0] + beta[1] * np.log1p(shock_grid)
    curve = pd.DataFrame(
        {
            "shock_quantile": np.linspace(0.05, 0.95, 19),
            "event_shock": shock_grid,
            "predicted_human_traffic_ratio_to_matched_hour": np.exp(pred_log),
        }
    )

    summary = pd.DataFrame(
        [
            {
                "n_events": n,
                "model": "log(traffic_ratio) = alpha + beta * log1p(event_shock)",
                "alpha": beta[0],
                "beta_log1p_event_shock": beta[1],
                "alpha_se": se[0],
                "beta_se": se[1],
                "r_squared": r2,
                "traffic_ratio_median": float(event_panel["traffic_ratio_matched"].median()),
                "traffic_ratio_p75": float(event_panel["traffic_ratio_matched"].quantile(0.75)),
                "traffic_ratio_p90": float(event_panel["traffic_ratio_matched"].quantile(0.90)),
                "traffic_ratio_p95": float(event_panel["traffic_ratio_matched"].quantile(0.95)),
                "traffic_ratio_share_gt_1": float((event_panel["traffic_ratio_matched"] > 1).mean()),
            }
        ]
    )
    keep = [
        "cf_location",
        "tong_country",
        "event_time_utc",
        "event_shock",
        "burst_rank",
        "local_hour",
        "human_volume_proxy",
        "matched_human_baseline",
        "traffic_ratio_matched",
        "human_volume_proxy_anomaly",
    ]
    event_panel = event_panel[keep]
    return summary, curve, event_panel


def plot_curves(fitted: pd.DataFrame, attention_curve: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180)

    ax = axes[0, 0]
    for cid, label in [
        ("mckinsey_non_ai_capacity_gw", "Non-AI"),
        ("mckinsey_ai_inference_capacity_gw", "AI inference"),
        ("mckinsey_ai_training_capacity_gw", "AI training"),
    ]:
        sub = fitted[fitted["curve_id"] == cid]
        ax.plot(sub["year"], sub["value"], marker="o", label=label)
    ax.set_title("McKinsey workload capacity fit")
    ax.set_ylabel("GW")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for cid, label in [
        ("mckinsey_inference_share_total_dc_percent", "Inference / total DC"),
        ("mckinsey_inference_share_ai_percent", "Inference / AI"),
        ("gartner_ai_optimized_share_total_dc_percent", "AI-optimized / total DC"),
        ("edna_ai_related_dc_share_mid", "AI-related / total DC"),
    ]:
        sub = fitted[fitted["curve_id"] == cid]
        ax.plot(sub["year"], sub["value"], marker="o", label=label)
    ax.set_title("AI shares in data-centre demand")
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    for cid, label in [
        ("iea_global_dc_twh_base", "IEA base"),
        ("gartner_total_dc_twh", "Gartner"),
    ]:
        sub = fitted[fitted["curve_id"] == cid]
        ax.plot(sub["year"], sub["value"], marker="o", label=label)
    ax.set_title("Global data-centre electricity fit")
    ax.set_ylabel("TWh/yr")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        attention_curve["event_shock"],
        attention_curve["predicted_human_traffic_ratio_to_matched_hour"],
        marker="o",
        color="#2563EB",
    )
    ax.axhline(1.0, color="#111827", lw=0.8, ls="--")
    ax.set_title("Internal attention-to-traffic response")
    ax.set_xlabel("Wikimedia attention shock")
    ax.set_ylabel("Human traffic ratio")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ai_inference_growth_curve_fits.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ai_inference_growth_curve_fits.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(anchor_df: pd.DataFrame, fitted: pd.DataFrame, attention_summary: pd.DataFrame) -> None:
    def value(curve_id: str, year: int) -> float:
        sub = fitted[(fitted["curve_id"] == curve_id) & (fitted["year"] == year)]
        return float(sub["value"].iloc[0])

    lines = [
        "# AI inference growth curve fit",
        "",
        "## Scope",
        "",
        "This report fits only the curves that have explicit numeric anchors. It does not treat the previous 8x high-AI path as a literature forecast.",
        "",
        "## External fitted curves",
        "",
        "1. McKinsey workload capacity curve: non-AI, AI inference, and AI training capacity from 2025 to 2030. This is the best direct anchor for inference share.",
        f"   - AI inference grows from {value('mckinsey_ai_inference_capacity_gw', 2025):.1f} GW in 2025 to {value('mckinsey_ai_inference_capacity_gw', 2030):.1f} GW in 2030.",
        f"   - Inference share of total data-centre demand rises from {value('mckinsey_inference_share_total_dc_percent', 2025):.1f}% to {value('mckinsey_inference_share_total_dc_percent', 2030):.1f}%.",
        f"   - Inference share within AI demand rises from {value('mckinsey_inference_share_ai_percent', 2025):.1f}% to {value('mckinsey_inference_share_ai_percent', 2030):.1f}%.",
        "",
        "2. Gartner AI-optimized server curve: not pure inference, but a useful AI-server electricity share anchor.",
        f"   - AI-optimized server electricity rises from {value('gartner_ai_optimized_server_twh', 2025):.0f} TWh in 2025 to {value('gartner_ai_optimized_server_twh', 2030):.0f} TWh in 2030.",
        f"   - The fitted AI-optimized share of data-centre electricity rises from {value('gartner_ai_optimized_share_total_dc_percent', 2025):.1f}% to {value('gartner_ai_optimized_share_total_dc_percent', 2030):.1f}%.",
        "",
        "3. IEA global data-centre electricity curve: this is the cleanest source for total data-centre electricity and its global electricity share.",
        f"   - Base global data-centre electricity grows from {value('iea_global_dc_twh_base', 2024):.0f} TWh in 2024 to {value('iea_global_dc_twh_base', 2030):.0f} TWh in 2030 and about {value('iea_global_dc_twh_base', 2035):.0f} TWh in 2035.",
        f"   - The global electricity share rises from {value('iea_global_dc_share_percent_base', 2024):.1f}% in 2024 to about {value('iea_global_dc_share_percent_base', 2030):.1f}% in 2030.",
        "",
        "4. IEA-4E/EDNA AI-related data-centre curve: useful for AI-related share, but not inference-specific.",
        f"   - Midpoint AI-related energy rises from {value('edna_ai_related_dc_twh_mid', 2023):.0f} TWh in 2023 to {value('edna_ai_related_dc_twh_mid', 2030):.0f} TWh in 2030.",
        f"   - Midpoint AI-related share rises from {value('edna_ai_related_dc_share_mid', 2023):.1f}% to {value('edna_ai_related_dc_share_mid', 2030):.1f}%.",
        "",
        "## Internal attention response fit",
        "",
    ]
    s = attention_summary.iloc[0]
    lines.extend(
        [
            "There is no external curve that directly maps public-attention bursts to AI-inference load amplification. As a proxy, the script fits the project panel:",
            "",
            "`log(human traffic ratio to same-country/month/local-hour baseline) = alpha + beta log1p(Wikimedia attention shock)`",
            "",
            f"- Events: {int(s['n_events'])}",
            f"- beta: {s['beta_log1p_event_shock']:.4f} (SE {s['beta_se']:.4f})",
            f"- R2: {s['r_squared']:.4f}",
            f"- Median event traffic ratio: {s['traffic_ratio_median']:.3f}",
            f"- P90 event traffic ratio: {s['traffic_ratio_p90']:.3f}",
            f"- Share of events with traffic above matched baseline: {100*s['traffic_ratio_share_gt_1']:.1f}%",
            "",
            "This should be described as an internal attention-to-traffic proxy, not as an AI-inference literature curve.",
            "",
            "## Recommendation for Result 3",
            "",
            "- Replace the old 2050 8x headline with a 2025-2030 literature-anchored scenario.",
            "- Use McKinsey inference capacity growth as the main inference curve: 20.9 -> 93.3 GW, i.e. 4.46x by 2030.",
            "- Use Gartner or IEA-4E/EDNA only as supporting AI-server / AI-related data-centre share checks.",
            "- Keep any 8x case only as a non-forecast sensitivity appendix.",
            "",
            "## Additional workload-trace literature",
            "",
            "- BurstGPT provides real Azure OpenAI GPT serving traces and documents strong temporal burstiness, daily and weekly periodicity. It is useful for arguing that LLM serving is bursty, but it does not label public-attention events and therefore cannot directly fit an attention-event amplification curve.",
            "  Source: https://arxiv.org/abs/2401.17644 and https://github.com/HPMLL/BurstGPT",
            "- EcoServe reports production traces from two generative-AI services and notes that offline/batch inference can account for a large share of serving capacity. It supports heterogeneity and schedulability of inference, not a public-attention growth curve.",
            "  Source: https://arxiv.org/abs/2502.05043",
            "",
            "## Source links",
            "",
        ]
    )
    for source, sub in anchor_df.groupby("source"):
        urls = sorted(sub["url"].unique())
        lines.append(f"- {source}: " + "; ".join(urls))
    (OUT_DIR / "ai_inference_growth_curve_fit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor_df = anchors()
    fitted, _ = build_external_curves(anchor_df)
    attention_summary, attention_curve, event_panel = fit_attention_response()
    plot_curves(fitted, attention_curve)

    anchor_df.to_csv(OUT_DIR / "curve_source_anchors.csv", index=False)
    fitted.to_csv(OUT_DIR / "fitted_curves_annual.csv", index=False)
    attention_summary.to_csv(OUT_DIR / "attention_burst_response_fit.csv", index=False)
    attention_curve.to_csv(OUT_DIR / "attention_burst_response_curve.csv", index=False)
    event_panel.to_csv(OUT_DIR / "attention_event_traffic_ratios.csv", index=False)
    write_report(anchor_df, fitted, attention_summary)

    print(OUT_DIR / "curve_source_anchors.csv")
    print(OUT_DIR / "fitted_curves_annual.csv")
    print(OUT_DIR / "attention_burst_response_fit.csv")
    print(OUT_DIR / "ai_inference_growth_curve_fit_report.md")


if __name__ == "__main__":
    main()
