#!/usr/bin/env python3
"""
Fetch one year of hourly Cloudflare Radar HTTP data for the 42 countries with
usable country-level solar/wind capacity-factor series.

The API token is read from CLOUDFLARE_API_TOKEN and is never written to disk.
Raw API responses are cached by endpoint, country, and chunk so interrupted
runs can resume without repeating completed requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_cloudflare_radar_hourly_year import (  # noqa: E402
    chunk_ranges,
    iso,
    rename_existing,
    result_to_frame,
    safe_name,
    stamp,
    utc_timestamp,
    write_csv,
)


BASE = "https://api.cloudflare.com/client/v4/radar"
DEFAULT_INVENTORY = Path("country_hourly_generation/data/tong_case_inventory.csv")
DEFAULT_OUT_DIR = Path(
    "/data2/AI_electri_from_social_media/cloudflare_radar/"
    "country_hourly_year_cf42_2025-05-02_2026-05-01"
)
DEFAULT_START = "2025-05-02T00:00:00Z"
DEFAULT_END = "2026-05-01T23:00:00Z"

COUNTRY_TO_CF_LOCATION = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Brazil": "BR",
    "Canada": "CA",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "Egypt": "EG",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Italy": "IT",
    "Japan": "JP",
    "Libya": "LY",
    "Malaysia": "MY",
    "Mexico": "MX",
    "Morocco": "MA",
    "Mozambique": "MZ",
    "New Zealand": "NZ",
    "Nigeria": "NG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Poland": "PL",
    "Russia": "RU",
    "Saudi Arabia": "SA",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sudan": "SD",
    "Sweden": "SE",
    "Thailand": "TH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "GB",
    "United States": "US",
    "Venezuela": "VE",
    "Vietnam": "VN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--skip-bot-class", action="store_true")
    parser.add_argument(
        "--countries",
        nargs="*",
        help="Optional subset of Tong country names or Cloudflare alpha-2 location codes.",
    )
    return parser.parse_args()


def country_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_country_list(inventory_path: Path, countries: list[str] | None = None) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path)
    usable = inventory[inventory["usable"].astype(str).str.lower().eq("true")].copy()
    usable = usable[usable["is_region"].astype(str).str.lower().eq("false")].copy()
    usable["cf_location"] = usable["tong_country"].map(COUNTRY_TO_CF_LOCATION)
    missing = usable[usable["cf_location"].isna()]["tong_country"].tolist()
    if missing:
        raise RuntimeError(f"Missing Cloudflare location mapping for: {missing}")

    if countries:
        wanted = {x.upper() for x in countries} | {x.replace("_", " ").title() for x in countries}
        usable = usable[
            usable["cf_location"].isin({x.upper() for x in countries})
            | usable["tong_country"].isin(wanted)
            | usable["tong_country"].str.lower().isin({x.replace("_", " ").lower() for x in countries})
        ].copy()

    usable["country_slug"] = usable["tong_country"].map(country_slug)
    cols = [
        "tong_country",
        "country_slug",
        "cf_location",
        "irena_country",
        "pv_capacity_mw",
        "wind_capacity_mw",
        "solar_cf_file",
        "wind_cf_file",
    ]
    return usable[cols].sort_values("tong_country").reset_index(drop=True)


def request_json_backoff(
    session: requests.Session,
    token: str,
    path: str,
    params: dict[str, Any],
    retries: int = 8,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        response = None
        data: dict[str, Any] | None = None
        try:
            response = session.get(f"{BASE}/{path}", headers=headers, params=params, timeout=90)
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)

        if data and data.get("success"):
            return data["result"]

        status = response.status_code if response is not None else None
        if data is not None:
            last_error = f"status={status}, errors={data.get('errors')}"
        if status == 429:
            retry_after = response.headers.get("Retry-After") if response is not None else None
            if retry_after and retry_after.isdigit():
                wait_seconds = min(float(retry_after), 180.0)
            else:
                wait_seconds = min(20.0 * attempt, 180.0)
            print(f"  RATE_LIMIT {path}; sleeping {wait_seconds:.1f}s before retry {attempt}/{retries}", flush=True)
            time.sleep(wait_seconds)
            continue

        time.sleep(min(2.0 * attempt, 30.0))

    raise RuntimeError(f"Cloudflare Radar request failed for {path}: {last_error}")


def fetch_series_resumable(
    session: requests.Session,
    token: str,
    *,
    name: str,
    path: str,
    base_params: dict[str, Any],
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
    raw_dir: Path,
    sleep_seconds: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    raw_series_dir = raw_dir / safe_name(name)
    raw_series_dir.mkdir(parents=True, exist_ok=True)
    for i, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        params = {
            "dateStart": iso(chunk_start),
            "dateEnd": iso(chunk_end),
            "aggInterval": "1h",
            "format": "json",
            **base_params,
        }
        raw_path = raw_series_dir / f"{stamp(chunk_start)}_{stamp(chunk_end)}.json"
        if raw_path.exists():
            result = json.loads(raw_path.read_text())
            source = "cache"
        else:
            result = request_json_backoff(session, token, path, params)
            raw_path.write_text(json.dumps(result, indent=2))
            source = "api"
            time.sleep(sleep_seconds)
        frame = result_to_frame(result, chunk_start, chunk_end)
        frames.append(frame)
        print(f"  OK {name:<34} chunk {i:02d}/{len(ranges)} rows={len(frame)} source={source}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp_utc", "chunk_start_utc"]).drop_duplicates("timestamp_utc", keep="last")
    return out.reset_index(drop=True)


def attach_country_metadata(df: pd.DataFrame, country: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out.insert(1, "cf_location", country["cf_location"])
    out.insert(2, "tong_country", country["tong_country"])
    out.insert(3, "country_slug", country["country_slug"])
    out.insert(4, "irena_country", country["irena_country"])
    out.insert(5, "pv_capacity_mw", country["pv_capacity_mw"])
    out.insert(6, "wind_capacity_mw", country["wind_capacity_mw"])
    return out


def validate_coverage(df: pd.DataFrame, countries: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    rows = []
    for _, country in countries.iterrows():
        sub = df[df["cf_location"] == country["cf_location"]]
        observed = pd.DatetimeIndex(pd.to_datetime(sub["timestamp_utc"], utc=True).dropna().unique()).sort_values()
        missing = expected.difference(observed)
        rows.append(
            {
                "tong_country": country["tong_country"],
                "country_slug": country["country_slug"],
                "cf_location": country["cf_location"],
                "rows": int(len(sub)),
                "expected_hours": int(len(expected)),
                "missing_hours": int(len(missing)),
                "first_timestamp_utc": "" if observed.empty else observed.min().isoformat(),
                "last_timestamp_utc": "" if observed.empty else observed.max().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("Set CLOUDFLARE_API_TOKEN before running this script.")

    countries = load_country_list(args.inventory, args.countries)
    start = utc_timestamp(args.start)
    end = utc_timestamp(args.end)
    ranges = chunk_ranges(start, end, args.chunk_days)

    out_dir: Path = args.out_dir
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cloudflare Radar country hourly year fetch: CF42 ===")
    print(f"range: {iso(start)} -> {iso(end)}")
    print(f"countries: {len(countries)}")
    print(f"chunks: {len(ranges)} x up to {args.chunk_days} days")
    print(f"out: {out_dir}")
    print(countries[["tong_country", "cf_location"]].to_string(index=False))

    session = requests.Session()
    total_frames: list[pd.DataFrame] = []
    bot_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for _, country in countries.iterrows():
        code = country["cf_location"]
        name = country["tong_country"]
        print(f"\n--- {name} ({code}) ---", flush=True)
        try:
            total = fetch_series_resumable(
                session,
                token,
                name=f"{code}_total_volume",
                path="http/timeseries",
                base_params={"location": code},
                ranges=ranges,
                raw_dir=raw_dir,
                sleep_seconds=args.sleep,
            )
            total = rename_existing(total, {"values": "country_total_volume_norm"})
            total_frames.append(attach_country_metadata(total, country))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL total {name} ({code}): {exc}", flush=True)
            failures.append({"tong_country": name, "cf_location": code, "series": "total_volume", "error": repr(exc)})
            continue

        if args.skip_bot_class:
            continue
        try:
            bot = fetch_series_resumable(
                session,
                token,
                name=f"{code}_bot_class",
                path="http/timeseries_groups/bot_class",
                base_params={"location": code},
                ranges=ranges,
                raw_dir=raw_dir,
                sleep_seconds=args.sleep,
            )
            bot = rename_existing(bot, {"human": "human_share_pct", "bot": "bot_share_pct"})
            bot_frames.append(attach_country_metadata(bot, country))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL bot_class {name} ({code}): {exc}", flush=True)
            failures.append({"tong_country": name, "cf_location": code, "series": "bot_class", "error": repr(exc)})

    if not total_frames:
        raise RuntimeError("No country total-volume data was downloaded.")

    country_total = pd.concat(total_frames, ignore_index=True)
    country_total = country_total.sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)
    write_csv(country_total, out_dir / "cf42_country_total_volume_hourly.csv.gz")

    aligned = country_total.copy()
    if bot_frames:
        country_bot = pd.concat(bot_frames, ignore_index=True)
        country_bot = country_bot.sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)
        write_csv(country_bot, out_dir / "cf42_country_bot_class_hourly.csv.gz")
        aligned = country_total.merge(
            country_bot[
                [
                    "timestamp_utc",
                    "cf_location",
                    "human_share_pct",
                    "bot_share_pct",
                    "normalization",
                    "agg_interval",
                ]
            ].rename(columns={"normalization": "bot_share_normalization", "agg_interval": "bot_share_agg_interval"}),
            on=["timestamp_utc", "cf_location"],
            how="left",
        )
        aligned["human_volume_proxy"] = (
            aligned["country_total_volume_norm"] * aligned["human_share_pct"] / 100.0
        )
        aligned["bot_volume_proxy"] = aligned["country_total_volume_norm"] * aligned["bot_share_pct"] / 100.0

    aligned = aligned.sort_values(["cf_location", "timestamp_utc"]).reset_index(drop=True)
    write_csv(aligned, out_dir / "cf42_country_cloudflare_hourly_year_aligned.csv.gz")

    countries.to_csv(out_dir / "cf42_country_list.csv", index=False)
    coverage = validate_coverage(aligned, countries, start, end)
    coverage.to_csv(out_dir / "cf42_country_coverage.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "cf42_fetch_failures.csv", index=False)

    summary = {
        "start_utc": iso(start),
        "end_utc": iso(end),
        "chunk_days": args.chunk_days,
        "chunks": len(ranges),
        "requested_countries": int(len(countries)),
        "total_volume_rows": int(len(country_total)),
        "aligned_rows": int(len(aligned)),
        "countries_with_total_volume": int(country_total["cf_location"].nunique()),
        "countries_with_bot_class": int(aligned.dropna(subset=["human_share_pct"])["cf_location"].nunique())
        if "human_share_pct" in aligned
        else 0,
        "countries_with_missing_hours": int((coverage["missing_hours"] > 0).sum()),
        "failures": failures,
        "caveat": (
            "country_total_volume_norm is a Cloudflare Radar MIN0_MAX index normalized within each "
            "download chunk, not raw request counts. human_volume_proxy and bot_volume_proxy are "
            "derived proxy indices, not observed request counts."
        ),
    }
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Cloudflare Radar Country Hourly Year Data: CF42\n\n"
        f"- Range: `{summary['start_utc']}` to `{summary['end_utc']}`\n"
        f"- Countries requested: `{summary['requested_countries']}`\n"
        f"- Countries with total volume: `{summary['countries_with_total_volume']}`\n"
        f"- Countries with human/bot split: `{summary['countries_with_bot_class']}`\n"
        f"- Hourly rows in aligned file: `{summary['aligned_rows']:,}`\n"
        f"- Chunking: `{summary['chunks']}` chunks, up to `{summary['chunk_days']}` days each\n\n"
        "## Files\n\n"
        "- `cf42_country_cloudflare_hourly_year_aligned.csv.gz`: main country-hour table.\n"
        "- `cf42_country_total_volume_hourly.csv.gz`: country total-volume MIN0_MAX index.\n"
        "- `cf42_country_bot_class_hourly.csv.gz`: country human/bot percentage shares.\n"
        "- `cf42_country_list.csv`: mapping from Tong CF countries to Cloudflare location codes.\n"
        "- `cf42_country_coverage.csv`: row counts and missing-hour checks by country.\n"
        "- `fetch_summary.json`: machine-readable download summary.\n"
        "- `raw/`: cached original JSON responses by endpoint, country, and chunk.\n\n"
        "## Important Caveat\n\n"
        "`country_total_volume_norm` is Cloudflare Radar's `MIN0_MAX` normalized index. "
        "Because the API is queried in monthly chunks, it is best used for within-country, "
        "within-chunk hourly patterns and residual analyses, not as raw request counts. "
        "`human_volume_proxy = country_total_volume_norm * human_share_pct / 100` is also "
        "a proxy index rather than observed human request volume.\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
