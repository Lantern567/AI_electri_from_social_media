# Paper release

This directory is the reproducibility package for **Chasing wind and sun narrows but does not close the data-centre clean-power gap**.

## Contents

- `data/main_and_derived/`: figure source data and frozen derived datasets.
- `data/station_expanded/`: station-expanded routing, waveform and robustness outputs.
- `figures/`: frozen PNG and PDF outputs for main and supplementary figures.
- `scripts/`: complete acquisition, preprocessing, fitting, analysis, validation, dispatch and plotting code used by the paper.
- `manuscript/`: authoritative submission DOCX files plus readable Markdown snapshots.
- `REPRODUCIBILITY_INDEX.md`: result-to-script/input/output crosswalk.
- `SUBMISSION_MANIFEST.md`: SHA-256 checksums for the authoritative submission files and Supplementary Data.
- `data/Supplementary_Data_1_country_level_domestic_mismatch.csv`: the exact machine-readable country-level table supplied with the submission.
- `data/Supplementary_Data_1_README.md`: column definitions and integrity hash for Supplementary Data 1.

## Two reproducibility levels

The repository supports two distinct workflows.

1. **Exact inspection of the submitted results.** Use the committed CSV/JSON files and frozen figures. These files are the immutable inputs and outputs used for the submitted version and do not require reacquiring provider data.
2. **Full reconstruction from public sources.** Run the acquisition and analysis pipeline below. This requires network access, the third-party source data and API credentials described in the manuscript, and the same local data-directory layout expected by the acquisition scripts. Provider revisions can prevent byte-for-byte reproduction of downloaded raw data.

## Full reconstruction order

Run from the repository root after `uv sync`.

1. Acquire demand and weather inputs with the `fetch_*.py` scripts.
2. Build renewable-site, capacity-factor and demand inputs with the `build_*.py` and `compute_country_burst_ratios.py` scripts.
3. Fit the AI workload parameters with the `fit_*.py` scripts.
4. Build supply and demand analysis datasets with `analyze_cfe_global_sites.py` and `analyze_cfe_global_sites_stations.py`.
5. Run the `analyze_cfe_r2_*` domestic-mismatch and routing analyses.
6. Run the `analyze_cfe_r3_*`, `analyze_cfe_m1_*`, and `analyze_cfe_s17_*` sensitivity analyses.
7. Run the full-system cost and storage workflow with `layer_fullsystem_cost.py`, `storage_dispatch_lp.py`, `run_dispatch_cost_full.py`, and the related decomposition/sensitivity scripts.
8. Regenerate figures with the corresponding `plot_*.py` scripts.

Committed CSV/JSON files and figures are frozen submission outputs. API-backed acquisition may not reproduce byte-for-byte if providers revise history; use committed source data for exact result inspection.

`scripts/fullsystem_cost_params.py` is the shared frozen parameter module used by the full-system cost and dispatch scripts.

See `CODE_INVENTORY.md` for the complete stage-by-stage code inventory. Computational scripts are retained even when their raw third-party inputs cannot be redistributed.

## Provenance note

Supplementary Table S0 records parameter, adopted value, data source and Supplementary section. Script, input and frozen-output provenance is maintained in `REPRODUCIBILITY_INDEX.md`, rather than represented as columns in Table S0.

## Version and licences

The submitted snapshot is tagged `v1.1.0-submission`. Code is licensed under MIT. Newly generated derived data are licensed under CC BY 4.0 as described in the repository-level `DATA_LICENSE.md`; third-party inputs retain their original terms.
