#!/usr/bin/env python3
"""Fetch hourly Cloudflare Radar human-traffic proxy for ALL station-supply
countries (ISO-2), to expand the demand panel beyond the original 31.

Demand proxy per country-hour:
    human_volume_proxy = total_volume_norm (MIN0_MAX index) * human_share_pct/100
which is the same shape-based convention used for the original 31-country panel.

Requires CLOUDFLARE_API_TOKEN. Raw JSON is cached per country/endpoint/chunk so
the run is fully resumable.

Input : data/global_renewable_sites/country_capacity.csv  (station-supply ISO-2)
Output: data/cloudflare_global_demand/country_hourly_demand.csv.gz
        data/cloudflare_global_demand/fetch_failures.csv (if any)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_cloudflare_radar_hourly_year import (  # noqa: E402
    chunk_ranges, iso, rename_existing, utc_timestamp, write_csv,
)
from fetch_cloudflare_radar_country_hourly_year_cf42 import (  # noqa: E402
    fetch_series_resumable,
)

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites" / "country_capacity.csv"
OUT = ROOT / "collective_attention_research_plan" / "data" / "cloudflare_global_demand"
START = "2025-05-02T00:00:00Z"
END = "2026-05-01T23:00:00Z"
EXCLUDE = {"AQ"}  # Antarctica: no meaningful traffic


def country_list() -> list[str]:
    cap = pd.read_csv(CAP, keep_default_na=False, na_values=[""])
    return [c for c in sorted(cap["iso2"].dropna().unique()) if c and c not in EXCLUDE]


def main() -> int:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise SystemExit("Set CLOUDFLARE_API_TOKEN before running this script.")
    codes = country_list()
    start, end = utc_timestamp(START), utc_timestamp(END)
    ranges = chunk_ranges(start, end, 31)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"=== Cloudflare Radar global demand fetch: {len(codes)} countries, {len(ranges)} chunks ===", flush=True)

    session = requests.Session()
    frames, fails = [], []
    for k, code in enumerate(codes, 1):
        print(f"[{k}/{len(codes)}] {code}", flush=True)
        try:
            total = fetch_series_resumable(
                session, token, name=f"{code}_total_volume", path="http/timeseries",
                base_params={"location": code}, ranges=ranges, raw_dir=raw, sleep_seconds=0.25)
            total = rename_existing(total, {"values": "total_norm"})
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL total {code}: {exc}", flush=True)
            fails.append({"code": code, "series": "total", "error": repr(exc)})
            continue
        bot = None
        try:
            bot = fetch_series_resumable(
                session, token, name=f"{code}_bot_class", path="http/timeseries_groups/bot_class",
                base_params={"location": code}, ranges=ranges, raw_dir=raw, sleep_seconds=0.25)
            bot = rename_existing(bot, {"human": "human_share_pct", "bot": "bot_share_pct"})
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL bot {code}: {exc}", flush=True)
            fails.append({"code": code, "series": "bot", "error": repr(exc)})

        df = total[["timestamp_utc", "total_norm"]].copy()
        if bot is not None and "human_share_pct" in bot:
            df = df.merge(bot[["timestamp_utc", "human_share_pct"]], on="timestamp_utc", how="left")
            df["human_volume_proxy"] = df["total_norm"] * df["human_share_pct"] / 100.0
        else:
            df["human_share_pct"] = float("nan")
            df["human_volume_proxy"] = df["total_norm"]
        df.insert(0, "cf_location", code)
        frames.append(df)

    if not frames:
        raise SystemExit("No demand data fetched.")
    out = pd.concat(frames, ignore_index=True).sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(out, OUT / "country_hourly_demand.csv.gz")
    if fails:
        pd.DataFrame(fails).to_csv(OUT / "fetch_failures.csv", index=False)
    print(f"DONE. countries fetched: {out['cf_location'].nunique()}/{len(codes)}; failures: {len(fails)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
