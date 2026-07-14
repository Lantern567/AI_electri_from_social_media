# Reproducibility index

Paths are relative to `paper_release/`.

The complete stage-by-stage script list is provided in `CODE_INVENTORY.md`.

| Paper item | Analysis / fitting script | Principal data | Frozen output |
|---|---|---|---|
| Fig. 1, domestic mismatch | `scripts/analyze_cfe_global_sites_stations.py`; `scripts/plot_cfe_geographic_portfolio_ai.py` | `data/main_and_derived/`; `data/station_expanded/` | `figures/fig1_national_mismatch.{png,pdf}` |
| Fig. 2, latency and routable share | `scripts/analyze_cfe_r2_latency_routable.py`; `scripts/plot_r2_latency_fig2.py` | `data/station_expanded/` | `figures/fig2_latency_full.{png,pdf}` |
| Fig. 3, routing and pairings | `scripts/analyze_cfe_r2_st_flows.py`; `scripts/analyze_cfe_r2_st_flows_waveform.py`; `scripts/plot_r2_pairing.py`; `scripts/plot_r3_split_figs.py` | `data/station_expanded/` | `figures/fig3_pairing.{png,pdf}`; `figures/fig3_ai_mechanism.{png,pdf}` |
| Fig. 4, phase-firming cost | `scripts/analyze_cfe_r3_waveform_cost.py`; `scripts/analyze_cfe_r3_cost_sensitivity.py`; `scripts/plot_r3_latency_fig3.py` | `data/main_and_derived/` | `figures/fig4_routing_cost.{png,pdf}` |
| Fig. 5, storage dispatch | `scripts/storage_dispatch_lp.py`; `scripts/run_dispatch_cost_full.py`; `scripts/plot_fig5_dispatch.py` | `data/main_and_derived/` | dispatch CSV files and frozen figure outputs |
| Table S22, allocation sensitivity | `scripts/analyze_cfe_r2_allocation_sensitivity.py` | station-expanded routing inputs | `data/station_expanded/r2_allocation_sensitivity.csv`; provenance JSON |
| Objective robustness | `scripts/analyze_cfe_objective_robustness.py` | station-expanded routing inputs | `data/station_expanded/r2_objective_robustness.csv`; provenance JSON |
| AI inference-share trajectory | `scripts/plot_figS1_lambda_trajectory.py` | SI S0–S1 parameters | `figures/figS1_lambda_trajectory.{png,pdf}` |
| BurstGPT fit and burst ratio | `scripts/plot_figS2_burstgpt_diurnal.py`; `scripts/plot_figS3_burst_ratio_dist.py` | public-trace-derived aggregates | corresponding `figS2` and `figS3` outputs |
| AI workload operators | `scripts/plot_figS_ai_operators.py`; `scripts/analyze_cfe_s17_computeintensity.py`; `scripts/analyze_cfe_m1_flatbaseload.py` | committed scenario tables | `figures/figS_ai_operators_a.{png,pdf}`; `figures/figS_ai_operators_b.{png,pdf}` |
| Supply validation | `scripts/validate_supply_cf_measured.py` | measured/reference series | `data/station_expanded/supply_validation_measured.csv` |

The two data directories collectively form the redistributable Supplementary Data 1 bundle. Third-party raw traces and API responses restricted by provider terms are not redistributed.

Full-system cost scripts share the frozen parameter definitions in `scripts/fullsystem_cost_params.py`.
