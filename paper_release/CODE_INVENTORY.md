# Computational code inventory

All files below are in `scripts/`. Together they cover the paper's computational chain from acquisition to frozen figures.

## 1. Data acquisition

- `fetch_cloudflare_radar_hourly_year.py`
- `fetch_cloudflare_radar_country_hourly_year_cf42.py`
- `fetch_cloudflare_radar_global_demand.py`
- `fetch_openmeteo_site_weather.py`

## 2. Input construction and empirical fitting

- `build_global_renewable_sites.py`
- `build_global_supply_cf.py`
- `compute_country_burst_ratios.py`
- `fit_ai_inference_growth_curves.py`
- `fit_azure_llm_inference_2024.py`
- `fit_burstgpt_workload_shape.py`
- `fit_burstgpt_workload_shape_v2.py`
- `fit_lambda_triangulation.py`

## 3. Core CFE calculation and routing

- `analyze_cfe_geographic_portfolio_ai.py`
- `analyze_cfe_global_sites.py`
- `analyze_cfe_global_sites_stations.py`
- `analyze_cfe_r2_latency_routable.py`
- `analyze_cfe_r2_spatiotemporal.py`
- `analyze_cfe_r2_st_flows.py`
- `analyze_cfe_r2_st_flows_waveform.py`
- `analyze_cfe_r2_waveform_routing.py`
- `analyze_cfe_r3_compute_migration.py`
- `analyze_cfe_r3_latency_routable.py`
- `analyze_cfe_r3_waveform_cost.py`

## 4. Cost, storage and decomposition

- `fullsystem_cost_params.py`
- `layer_fullsystem_cost.py`
- `decompose_fullsystem_portfolio.py`
- `storage_dispatch_lp.py`
- `run_dispatch_cost_full.py`
- `recompute_fig3c_storage.py`
- `compute_geodiff_scope.py`
- `recompute_geodiff_pipeline.py`
- `analyze_tx_shared_bounds.py`

## 5. Robustness, validation and diagnostic calculations

- `analyze_cfe_m1_flatbaseload.py`
- `analyze_cfe_objective_robustness.py`
- `analyze_cfe_r2_allocation_sensitivity.py`
- `analyze_cfe_r3_cost_sensitivity.py`
- `analyze_cfe_s17_computeintensity.py`
- `run_r3_latency_sensitivity.py`
- `run_r3_sensitivity_sweep.py`
- `validate_supply_cf_measured.py`
- `build_si_validation_tables.py`
- `dump_l3_diurnal.py`
- `probe_intracountry_dispatch.py`
- `probe_radius_scope.py`

## 6. Figure generation

All `plot_*.py` files in the directory are retained, including the main-figure, Supplementary Figure, routing, dispatch and geographic-decomposition plots.

Raw third-party datasets governed by provider licences are not duplicated in this repository. Their acquisition scripts, public source locations and committed derived inputs are retained.
