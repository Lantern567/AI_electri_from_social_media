#!/usr/bin/env python3
"""Stage 1: turn the WRI Global Power Plant Database into a bounded set of
capacity-weighted representative renewable SITES for global supply curves.

We bucket stations into a 3-degree grid and, per technology, rank (country, cell)
units by capacity and keep the top units covering 90% of GLOBAL capacity; we then
guarantee every renewable country keeps at least its single largest cell (breadth).
Each kept unit becomes one representative site at the capacity-weighted centroid.
This yields ~455 global sites (vs 16k raw plants) so per-site hourly reanalysis
(NASA POWER) is feasible, while preserving ~91% of global installed capacity, full
country breadth, and the within-country geographic distribution of capacity.

Input : data/global_renewable_sites/raw/global_power_plant_database.csv
Output: data/global_renewable_sites/representative_sites.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "collective_attention_research_plan" / "data" / "global_renewable_sites"
GPPD = SITE_DIR / "raw" / "global_power_plant_database.csv"
OUT = SITE_DIR / "representative_sites.csv"

GRID_DEG = 3.0
CAP_COVERAGE = 0.90  # keep (country,cell) units covering this share of GLOBAL capacity per tech


def iso3_to_iso2() -> dict[str, str]:
    """ISO3 -> ISO2 map via geonamescache (covers all countries)."""
    import geonamescache

    gc = geonamescache.GeonamesCache()
    out = {}
    for _, info in gc.get_countries().items():
        iso2 = info.get("iso")
        iso3 = info.get("iso3")
        if iso2 and iso3:
            out[iso3] = iso2
    # WRI uses a few non-standard / older codes; patch the important ones.
    out.setdefault("ROM", "RO")
    out.setdefault("KOS", "XK")
    return out


def main() -> None:
    df = pd.read_csv(GPPD, low_memory=False)
    sw = df[df["primary_fuel"].isin(["Solar", "Wind"])].copy()
    sw = sw.dropna(subset=["latitude", "longitude", "capacity_mw"])
    sw = sw[sw["capacity_mw"] > 0]

    i3_i2 = iso3_to_iso2()
    sw["iso2"] = sw["country"].map(i3_i2)
    sw["cell"] = (np.floor(sw["latitude"] / GRID_DEG).astype(int).astype(str) + "_"
                  + np.floor(sw["longitude"] / GRID_DEG).astype(int).astype(str))

    rows = []
    for tech in ["Solar", "Wind"]:
        t = sw[sw["primary_fuel"] == tech]
        unit_cap = t.groupby(["country", "cell"])["capacity_mw"].sum().sort_values(ascending=False)
        cum = unit_cap.cumsum() / unit_cap.sum()
        keep = set(unit_cap.index[: int((cum <= CAP_COVERAGE).sum()) + 1])
        # breadth guarantee: every country keeps at least its single largest cell
        for iso3, sub in unit_cap.groupby(level=0):
            if not any(k[0] == iso3 for k in keep):
                keep.add(sub.index[0])
        for (iso3, cell) in keep:
            tc = t[(t["country"] == iso3) & (t["cell"] == cell)]
            w = tc["capacity_mw"].to_numpy()
            rows.append({
                "site_id": f"{iso3}_{tech[:1]}_{cell}",
                "iso3": iso3,
                "iso2": tc["iso2"].iloc[0],
                "country_long": tc["country_long"].iloc[0],
                "tech": "PV" if tech == "Solar" else "WIND",
                "lat": round(float(np.average(tc["latitude"], weights=w)), 3),
                "lon": round(float(np.average(tc["longitude"], weights=w)), 3),
                "wri_capacity_mw": float(w.sum()),
                "n_plants": int(len(tc)),
                "cell": cell,
            })

    sites = pd.DataFrame(rows).sort_values(["tech", "wri_capacity_mw"], ascending=[True, False])
    sites.to_csv(OUT, index=False)

    # provenance / coverage summary
    print(f"representative sites: {len(sites)}  ->  {OUT}")
    for tech in ["PV", "WIND"]:
        s = sites[sites["tech"] == tech]
        raw_cap = sw[sw["primary_fuel"] == ("Solar" if tech == "PV" else "Wind")]["capacity_mw"].sum()
        print(f"  {tech}: {len(s)} sites, {s['iso3'].nunique()} countries, "
              f"{s['wri_capacity_mw'].sum()/1000:.0f} GW kept / {raw_cap/1000:.0f} GW raw "
              f"({100*s['wri_capacity_mw'].sum()/raw_cap:.0f}% capacity coverage)")
    n_missing = sites["iso2"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} sites missing ISO2 ({sorted(sites[sites['iso2'].isna()]['iso3'].unique())})")


if __name__ == "__main__":
    main()
