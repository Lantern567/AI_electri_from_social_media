# Supplementary Data 1

File: `Supplementary_Data_1_country_level_domestic_mismatch.csv`

The file contains one row for each of the 104 sampled countries. Percentages use the equal-energy domestic accounting described in the main-text Methods and Supplementary Section S8. Local hours are integer clock hours from 0 to 23.

| Column | Definition |
|---|---|
| `country_iso2` | ISO 3166-1 alpha-2 country code |
| `region` | Analysis region used in the figures |
| `continent` | Continent grouping |
| `matching_gap_U_percent` | Annual share of normalized demand not covered by contemporaneous domestic wind-plus-solar supply, in percent |
| `excess_share_percent` | Annual share of normalized supply occurring above contemporaneous demand, in percent |
| `max_hourly_uncovered_x_mean` | Maximum hourly uncovered demand divided by annual mean demand |
| `continuous_gap_segment_count` | Number of continuous positive-uncovered-demand segments in the hourly year |
| `median_continuous_gap_h` | Median continuous-gap duration, hours |
| `p95_continuous_gap_h` | 95th-percentile continuous-gap duration, hours |
| `max_continuous_gap_h` | Maximum continuous-gap duration, hours |
| `peak_demand_local_hour` | Local hour of the peak mean intraday demand profile |
| `peak_supply_local_hour` | Local hour of the peak mean intraday domestic-supply profile |
| `demand_supply_phase_lag_h` | Signed demand-peak minus supply-peak phase lag, hours, using the study's local-hour convention |
| `demand_diurnal_spectral_concentration` | Share of demand-series spectral power concentrated at the diurnal frequency |
| `supply_diurnal_spectral_concentration` | Share of supply-series spectral power concentrated at the diurnal frequency |

Integrity: SHA-256 `1A0426348507FAE13453E1F44B0DC0164A43AC135753D794274BBD827FE5EC71`.
