# Reproducibility index

Paths are relative to `paper_release/`.

The complete stage-by-stage script list is provided in `CODE_INVENTORY.md`.

| Paper item | Analysis / fitting script | Principal data | Frozen output |
|---|---|---|---|
| Fig. 1, domestic mismatch | `scripts/analyze_cfe_global_sites_stations.py`; `scripts/plot_cfe_geographic_portfolio_ai.py` | `data/Supplementary_Data_1_country_level_domestic_mismatch.csv`; `data/main_and_derived/`; `data/station_expanded/` | `figures/fig1_national_mismatch.{png,pdf}` |
| Fig. 2, latency and routable share | `scripts/analyze_cfe_r2_latency_routable.py`; `scripts/plot_r2_latency_fig2.py` | `data/station_expanded/` | `figures/fig2_latency_full.{png,pdf}` |
| Fig. 3, routing pairs and flows | `scripts/analyze_cfe_r2_st_flows.py`; `scripts/analyze_cfe_r2_st_flows_waveform.py`; `scripts/plot_r2_pairing.py` | `data/station_expanded/` | `figures/fig3_pairing.{png,pdf}` |
| Fig. 4, AI-shape and routing mechanism | `scripts/analyze_cfe_r3_waveform_cost.py`; `scripts/plot_r3_latency_fig3.py`; `scripts/plot_r3_split_figs.py` | `data/main_and_derived/r3_waveform_cost_table.csv`; `data/main_and_derived/r1_diurnal_profiles.csv` | `figures/fig3_ai_mechanism.{png,pdf}` and associated frozen panels |
| Fig. 5, routing cost and storage dispatch | `scripts/analyze_cfe_r3_cost_sensitivity.py`; `scripts/storage_dispatch_lp.py`; `scripts/run_dispatch_cost_full.py`; `scripts/plot_fig5_dispatch.py` | `data/main_and_derived/` | `figures/fig4_routing_cost.{png,pdf}`; dispatch CSV files and frozen figure outputs |
| S15, finite shared-surplus clearing | `scripts/analyze_cfe_r2_allocation_sensitivity.py` | station-expanded routing inputs | `data/station_expanded/r2_allocation_sensitivity.csv`; provenance JSON |
| S9.4 / Table S18, interannual robustness | `scripts/analyze_cfe_interannual_robustness.py` | public ERA5 inputs; submitted country-level data | `data/station_expanded/interannual_matching_gap.csv` |
| S14/S15, objective robustness | `scripts/analyze_cfe_objective_robustness.py` | station-expanded routing inputs | `data/station_expanded/r2_objective_robustness.csv`; provenance JSON |
| S15.2/S24, longitude and supply-dictionary counterfactuals | `scripts/analyze_cfe_r2_longitude_counterfactual.py` | station-expanded demand and supply shapes | printed certificate values reproduced in S15.2/S24 |
| S17, compute-intensity and flat-baseload robustness | `scripts/analyze_cfe_s17_computeintensity.py`; `scripts/analyze_cfe_m1_flatbaseload.py` | committed scenario tables | committed sensitivity CSV files |
| AI inference-share trajectory | `scripts/plot_figS1_lambda_trajectory.py` | SI S0–S1 parameters | `figures/figS1_lambda_trajectory.{png,pdf}` |
| BurstGPT fit and burst ratio | `scripts/plot_figS2_burstgpt_diurnal.py`; `scripts/plot_figS3_burst_ratio_dist.py` | public-trace-derived aggregates | corresponding `figS2` and `figS3` outputs |
| AI workload operators | `scripts/plot_figS_ai_operators.py`; `scripts/analyze_cfe_s17_computeintensity.py`; `scripts/analyze_cfe_m1_flatbaseload.py` | committed scenario tables | `figures/figS_ai_operators_a.{png,pdf}`; `figures/figS_ai_operators_b.{png,pdf}` |
| Supply validation | `scripts/validate_supply_cf_measured.py` | measured/reference series | `data/station_expanded/supply_validation_measured.csv` |

The file `data/Supplementary_Data_1_country_level_domestic_mismatch.csv` is the exact Supplementary Data 1 file supplied with the manuscript. The remaining committed data directories contain figure-source and robustness outputs. Third-party raw traces and API responses restricted by provider terms are not redistributed.

Full-system cost scripts share the frozen parameter definitions in `scripts/fullsystem_cost_params.py`.
