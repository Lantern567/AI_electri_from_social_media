#!/usr/bin/env python3
"""Stage 2: fetch hourly ERA5 reanalysis weather for each representative site
from the Open-Meteo archive API (free, no token, global).

For every unique site coordinate we pull the 8760-hour window that matches the
Cloudflare demand panel (2025-05-02..2026-05-01 UTC):
  shortwave_radiation (GHI, W/m2)  -> PV capacity factor
  wind_speed_100m (m/s, hub height) -> wind capacity factor
  wind_speed_10m, temperature_2m    -> wind shear check / PV cell-temp derate

Each coordinate is cached as an .npz so the run is fully resumable. The time
axis is identical across coordinates (the panel hours) and saved once.

Input : data/global_renewable_sites/representative_sites.csv
Output: data/global_renewable_sites/weather_cache/w_<lat>_<lon>.npz  (per coord)
        data/global_renewable_sites/weather_time_index.npy           (shared hours)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WORKERS = 4
_time_lock = threading.Lock()

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites"
SITES = SITE_DIR / "representative_sites.csv"
CACHE = SITE_DIR / "weather_cache"
TIME_INDEX = SITE_DIR / "weather_time_index.npy"

START, END = "2025-05-02", "2026-05-01"
N_HOURS = 8760
API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "shortwave_radiation,temperature_2m,wind_speed_100m,wind_speed_10m"


def coord_id(lat: float, lon: float) -> str:
    return f"{lat:.3f}_{lon:.3f}".replace("-", "m")


def fetch_one(lat: float, lon: float, retries: int = 4) -> dict | None:
    params = dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                  hourly=HOURLY, timezone="UTC", windspeed_unit="ms",
                  cell_selection="nearest")
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=(15, 120))
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            print(f"  HTTP {r.status_code} @ {lat},{lon}: {r.text[:120]}")
            time.sleep(2 * (attempt + 1))
        except Exception as exc:  # network hiccup
            print(f"  retry {attempt+1} @ {lat},{lon}: {type(exc).__name__}")
            time.sleep(3 * (attempt + 1))
    return None


def fetch_and_save(lat: float, lon: float) -> str:
    out = CACHE / f"w_{coord_id(lat, lon)}.npz"
    if out.exists():
        return "skip"
    j = fetch_one(lat, lon)
    if j is None:
        return "fail"
    h = j["hourly"]
    t = h["time"]
    def arr(key):
        return np.asarray([np.nan if v is None else v for v in h[key]], dtype="float32")
    np.savez_compressed(out, ghi=arr("shortwave_radiation"), t2m=arr("temperature_2m"),
                        ws100=arr("wind_speed_100m"), ws10=arr("wind_speed_10m"))
    if not TIME_INDEX.exists():
        with _time_lock:
            if not TIME_INDEX.exists():
                np.save(TIME_INDEX, np.asarray(t))
    if len(t) != N_HOURS:
        print(f"  WARN {lat},{lon} returned {len(t)} hours (expected {N_HOURS})")
    return "ok"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(SITES)
    coords = sites[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    todo = [(float(r["lat"]), float(r["lon"])) for _, r in coords.iterrows()
            if not (CACHE / f"w_{coord_id(float(r['lat']), float(r['lon']))}.npz").exists()]
    print(f"coords total={len(coords)} already_cached={len(coords)-len(todo)} to_fetch={len(todo)} workers={WORKERS}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_and_save, lat, lon): (lat, lon) for lat, lon in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "fail":
                fail += 1
                print(f"FAILED {futs[fut]}")
            if (n % 40) == 0:
                print(f"  progress {n}/{len(todo)} ok={ok} fail={fail}")

    total = len(list(CACHE.glob("w_*.npz")))
    print(f"DONE. fetched_ok={ok} failed={fail} total_cached={total}/{len(coords)}")


if __name__ == "__main__":
    main()
