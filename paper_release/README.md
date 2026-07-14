# Paper release

This directory is the reproducibility package for **Earth’s rotation and human rhythms cap hourly wind-and-solar matching for interactive computing**.

## Contents

- `data/main_and_derived/`: figure source data and frozen derived datasets.
- `data/station_expanded/`: station-expanded routing, waveform and robustness outputs.
- `figures/`: frozen PNG and PDF outputs for main and supplementary figures.
- `scripts/`: analysis, validation, dispatch and plotting scripts.
- `manuscript/`: matching main-text and Supplementary Information snapshots.
- `REPRODUCIBILITY_INDEX.md`: result-to-script/input/output crosswalk.

## Recommended run order

Run from the repository root after `uv sync`.

1. Build supply and demand datasets with `analyze_cfe_global_sites.py` and `analyze_cfe_global_sites_stations.py`.
2. Run the `analyze_cfe_r2_*` domestic-mismatch and routing analyses.
3. Run the `analyze_cfe_r3_*`, `analyze_cfe_m1_*`, and `analyze_cfe_s17_*` sensitivity analyses.
4. Run `storage_dispatch_lp.py`, `run_dispatch_cost_full.py`, and `run_r3_latency_sensitivity.py`.
5. Regenerate figures with the corresponding `plot_*.py` scripts.

Committed CSV/JSON files and figures are frozen submission outputs. API-backed acquisition may not reproduce byte-for-byte if providers revise history; use committed source data for exact figure reproduction.

`scripts/fullsystem_cost_params.py` is the shared frozen parameter module used by the full-system cost and dispatch scripts.

## Provenance note

Supplementary Table S0 records parameter, adopted value, data source and Supplementary section. Script, input and frozen-output provenance is maintained in `REPRODUCIBILITY_INDEX.md`, rather than represented as columns in Table S0.
