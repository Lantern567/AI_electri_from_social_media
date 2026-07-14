#!/usr/bin/env python3
"""
Fetch one year of hourly Cloudflare Radar HTTP time series.

Cloudflare Radar currently returns hourly data reliably for roughly monthly
windows, so this script downloads the requested period in chunks and merges the
results. The API token is read from CLOUDFLARE_API_TOKEN and is never written to
disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE = "https://api.cloudflare.com/client/v4/radar"
DEFAULT_OUT_DIR = Path(
    "/data2/AI_electri_from_social_media/cloudflare_radar/"
    "hourly_year_2025-05-02_2026-05-01"
)
DEFAULT_START = "2025-05-02T00:00:00Z"
DEFAULT_END = "2026-05-01T23:00:00Z"
DEFAULT_COUNTRIES = ["US", "IN", "BR", "DE", "GB", "JP", "FR", "ID", "NL", "AU", "KR", "SG", "CN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--countries", nargs="*", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Fetch and merge global series only; skip per-country total and bot-class files.",
    )
    parser.add_argument(
        "--skip-country-bot",
        action="store_true",
        help="Only fetch per-country total volume, not per-country human/bot share.",
    )
    return parser.parse_args()


def utc_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + pd.Timedelta(days=chunk_days) - pd.Timedelta(hours=1))
        ranges.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(hours=1)
    return ranges


def stamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def request_json(
    session: requests.Session,
    token: str,
    path: str,
    params: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(f"{BASE}/{path}", headers=headers, params=params, timeout=90)
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            data = None
        if data and data.get("success"):
            return data["result"]
        if data is not None:
            last_error = f"status={response.status_code}, errors={data.get('errors')}"
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"Cloudflare Radar request failed for {path}: {last_error}")


def result_to_frame(result: dict[str, Any], chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
    serie = result.get("serie_0", result)
    timestamps = serie.get("timestamps", [])
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(timestamps, utc=True)})
    for key, values in serie.items():
        if key == "timestamps":
            continue
        df[key] = pd.to_numeric(values, errors="coerce")

    meta = result.get("meta", {})
    df["chunk_start_utc"] = iso(chunk_start)
    df["chunk_end_utc"] = iso(chunk_end)
    df["agg_interval"] = meta.get("aggInterval")
    df["normalization"] = meta.get("normalization")
    return df


def fetch_series(
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
        else:
            result = request_json(session, token, path, params)
            raw_path.write_text(json.dumps(result, indent=2))
            time.sleep(sleep_seconds)
        frame = result_to_frame(result, chunk_start, chunk_end)
        frames.append(frame)
        print(f"  OK {name:<28} chunk {i:02d}/{len(ranges)} rows={len(frame)}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp_utc", "chunk_start_utc"]).drop_duplicates("timestamp_utc", keep="last")
    return out.reset_index(drop=True)


def rename_existing(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("Set CLOUDFLARE_API_TOKEN before running this script.")

    out_dir: Path = args.out_dir
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = utc_timestamp(args.start)
    end = utc_timestamp(args.end)
    ranges = chunk_ranges(start, end, args.chunk_days)

    print("=== Cloudflare Radar hourly year fetch ===")
    print(f"range: {iso(start)} -> {iso(end)}")
    print(f"chunks: {len(ranges)} x up to {args.chunk_days} days")
    print(f"out: {out_dir}")

    session = requests.Session()
    global_total = fetch_series(
        session,
        token,
        name="global_total_volume",
        path="http/timeseries",
        base_params={},
        ranges=ranges,
        raw_dir=raw_dir,
        sleep_seconds=args.sleep,
    )
    global_total = rename_existing(global_total, {"values": "total_volume_norm"})
    write_csv(global_total, out_dir / "global_total_volume_hourly.csv.gz")

    global_bot = fetch_series(
        session,
        token,
        name="global_bot_class",
        path="http/timeseries_groups/bot_class",
        base_params={},
        ranges=ranges,
        raw_dir=raw_dir,
        sleep_seconds=args.sleep,
    )
    global_bot = rename_existing(global_bot, {"human": "human_share_pct", "bot": "bot_share_pct"})
    write_csv(global_bot, out_dir / "global_bot_class_hourly.csv.gz")

    global_device = fetch_series(
        session,
        token,
        name="global_device_type",
        path="http/timeseries_groups/device_type",
        base_params={},
        ranges=ranges,
        raw_dir=raw_dir,
        sleep_seconds=args.sleep,
    )
    write_csv(global_device, out_dir / "global_device_type_hourly.csv.gz")

    global_http_version = fetch_series(
        session,
        token,
        name="global_http_version",
        path="http/timeseries_groups/http_version",
        base_params={},
        ranges=ranges,
        raw_dir=raw_dir,
        sleep_seconds=args.sleep,
    )
    write_csv(global_http_version, out_dir / "global_http_version_hourly.csv.gz")

    location_share = fetch_series(
        session,
        token,
        name="location_share_top20",
        path="http/timeseries_groups/location",
        base_params={"limit": 20},
        ranges=ranges,
        raw_dir=raw_dir,
        sleep_seconds=args.sleep,
    )
    write_csv(location_share, out_dir / "location_share_top20_hourly.csv.gz")

    global_aligned = global_total[
        ["timestamp_utc", "total_volume_norm", "chunk_start_utc", "chunk_end_utc", "agg_interval", "normalization"]
    ].merge(
        global_bot[
            [
                "timestamp_utc",
                "human_share_pct",
                "bot_share_pct",
                "normalization",
                "agg_interval",
            ]
        ].rename(columns={"normalization": "bot_share_normalization", "agg_interval": "bot_share_agg_interval"}),
        on="timestamp_utc",
        how="left",
    )
    global_aligned["human_volume_proxy"] = global_aligned["total_volume_norm"] * global_aligned["human_share_pct"] / 100.0
    global_aligned["bot_volume_proxy"] = global_aligned["total_volume_norm"] * global_aligned["bot_share_pct"] / 100.0
    write_csv(global_aligned, out_dir / "global_cloudflare_hourly_year_aligned.csv.gz")

    country_rows = 0
    if not args.global_only and args.countries:
        country_total_frames: list[pd.DataFrame] = []
        country_bot_frames: list[pd.DataFrame] = []
        for country in args.countries:
            country_total = fetch_series(
                session,
                token,
                name=f"country_{country}_total_volume",
                path="http/timeseries",
                base_params={"location": country},
                ranges=ranges,
                raw_dir=raw_dir,
                sleep_seconds=args.sleep,
            )
            country_total = rename_existing(country_total, {"values": "country_total_volume_norm"})
            country_total.insert(1, "country", country)
            country_total_frames.append(country_total)

            if not args.skip_country_bot:
                country_bot = fetch_series(
                    session,
                    token,
                    name=f"country_{country}_bot_class",
                    path="http/timeseries_groups/bot_class",
                    base_params={"location": country},
                    ranges=ranges,
                    raw_dir=raw_dir,
                    sleep_seconds=args.sleep,
                )
                country_bot = rename_existing(country_bot, {"human": "human_share_pct", "bot": "bot_share_pct"})
                country_bot.insert(1, "country", country)
                country_bot_frames.append(country_bot)

        country_total_all = pd.concat(country_total_frames, ignore_index=True)
        country_rows = int(len(country_total_all))
        write_csv(country_total_all, out_dir / "country_total_volume_hourly.csv.gz")

        if country_bot_frames:
            country_bot_all = pd.concat(country_bot_frames, ignore_index=True)
            write_csv(country_bot_all, out_dir / "country_bot_class_hourly.csv.gz")
            country_aligned = country_total_all.merge(
                country_bot_all[
                    [
                        "timestamp_utc",
                        "country",
                        "human_share_pct",
                        "bot_share_pct",
                        "normalization",
                        "agg_interval",
                    ]
                ].rename(columns={"normalization": "bot_share_normalization", "agg_interval": "bot_share_agg_interval"}),
                on=["timestamp_utc", "country"],
                how="left",
            )
            country_aligned["human_volume_proxy"] = (
                country_aligned["country_total_volume_norm"] * country_aligned["human_share_pct"] / 100.0
            )
            country_aligned["bot_volume_proxy"] = (
                country_aligned["country_total_volume_norm"] * country_aligned["bot_share_pct"] / 100.0
            )
            write_csv(country_aligned, out_dir / "country_cloudflare_hourly_year_aligned.csv.gz")

    summary = {
        "start_utc": iso(start),
        "end_utc": iso(end),
        "chunk_days": args.chunk_days,
        "chunks": len(ranges),
        "global_rows": int(len(global_aligned)),
        "country_rows": country_rows,
        "countries": [] if args.global_only else args.countries,
        "caveat": (
            "total_volume_norm and country_total_volume_norm are Cloudflare Radar MIN0_MAX "
            "indices normalized within each downloaded chunk, not raw request counts."
        ),
    }
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "README.md").write_text(
        "# Cloudflare Radar Hourly Year Data\n\n"
        f"- Range: `{summary['start_utc']}` to `{summary['end_utc']}`\n"
        f"- Chunking: `{summary['chunks']}` chunks, up to `{summary['chunk_days']}` days each\n"
        f"- Global hourly rows: `{summary['global_rows']}`\n"
        f"- Country hourly rows: `{summary['country_rows']}`\n"
        f"- Countries: `{', '.join(args.countries)}`\n\n"
        "## Important Caveat\n\n"
        "`total_volume_norm` and `country_total_volume_norm` are Cloudflare Radar "
        "`MIN0_MAX` indices. Because the API could not return a full year in one "
        "hourly request, the total-volume indices are normalized within monthly "
        "download chunks. They are useful for within-chunk shape and diurnal "
        "analysis, but should not be interpreted as globally comparable raw request "
        "counts across the entire year.\n\n"
        "The human/bot, device, HTTP version, and location files are percentage "
        "shares and are easier to compare across chunks.\n\n"
        "## Main Files\n\n"
        "- `global_cloudflare_hourly_year_aligned.csv.gz`\n"
        "- `global_total_volume_hourly.csv.gz`\n"
        "- `global_bot_class_hourly.csv.gz`\n"
        "- `global_device_type_hourly.csv.gz`\n"
        "- `global_http_version_hourly.csv.gz`\n"
        "- `location_share_top20_hourly.csv.gz`\n"
        "- `country_total_volume_hourly.csv.gz`\n"
        "- `country_bot_class_hourly.csv.gz`\n"
        "- `country_cloudflare_hourly_year_aligned.csv.gz`\n"
        "- `raw/`: original JSON responses by endpoint and chunk\n"
    )

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
