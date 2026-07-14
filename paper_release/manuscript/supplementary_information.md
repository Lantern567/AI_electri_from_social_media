# Supplementary Information

This document is the Supplementary Information for the main paper *Earth’s rotation and human rhythms cap hourly wind-and-solar matching for interactive computing*. Throughout, we adopt the finalized framing of the main text — the geographic lever routes **compute load across longitudes over the internet** to regions that currently hold a clean surplus (supply is fixed to the home country, with each country still using its own wind and solar), along two axes—latency tolerance $\tau$ (milliseconds, gating the reachable range by round-trip time) and routable fraction $\varphi$ (the share of load that can be served off-site); the full electricity-supply cost comprises generation, firming storage and ancillary services, and excludes inter-regional transmission. Sections S0–S8 document the empirical fits, sensitivity analyses and cross-validation behind the AI workload parameters used in Result 3 (Methods 4.5); Sections S9–S14 document the supply-side and routing mechanisms underpinning Results 1–2 — the construction and validation of capacity factors, the mathematical formulation of home-country portfolio optimization, the round-trip-time model validated against measured Azure latency, the country sample and cloud-region availability, the storage cost conversion, and the robustness of the supply and routing assumptions.

---

## Supplementary Section S0 — Overview of AI Workload Parameters

In Result 3, every AI workload parameter is determined from public traces and forecasting evidence rather than by manual tuning. Table S0 indexes each parameter to its adopted value, its data source, and the supplementary section that provides the full method, confidence intervals, cross-validation and sensitivity.

**Table S0. AI workload parameters, adopted values and data sources.**

| Parameter (operator) | Adopted value | Data source | Suppl. section |
|---|---|---|---|
| $\lambda_{\mathrm{inf}}(t)$ — AI inference share (drives the P2/P3 shape operators); $\lambda_{\mathrm{train}}(t)$ — training, flat baseload | $\lambda_{\mathrm{inf}}$ 0.130 / 0.308 / 0.560; $\lambda_{\mathrm{train}}$ 0.070 / 0.132 / 0.140 (2025 / 2030 / 2050) | Energy-based $\lambda_{\mathrm{AI}}$: Gartner, IEA-4E/EDNA, IEA (Energy and AI 2025); inference/training split: Google/Patterson 2022, IEA-4E/EDNA, EPRI | S1 |
| $\mu$, $\sigma$ — P2 sharpening of the intraday peak | 18.0 h, 3.8 h | BurstGPT v2.0 (10.1 M requests / 335 days) | S2 |
| 24-hour shape — P2_emp (no Gaussian assumption) | empirical token-share vector | BurstGPT v2.0 | S2.8 |
| $\gamma_{\mathrm{op}}$ — P3 burst amplification | 4.84 = (9.31 − 1) / 1.716 | BurstGPT hourly-aggregated P99/mean (9.31) ÷ Cloudflare $d_{\mathrm{burst}}$/mean (1.716, 104 countries) | S2, S3 |
| $\alpha_{\mathrm{batch}}$ / $\alpha_{\mathrm{shaped}}$ — Mix weights | 0.25 / 0.75 (flat batch processing / smoothed intraday-shaped inference; event bursts are isolated in P3 and not folded into Mix) | Carbon Responder (Xing et al. 2023, Meta), adapted to LLM serving | S5 |
| Independent cross-validation | $\sigma$ confirmed; $\gamma$ shown to depend on service type | Azure LLM Inference 2024 (44 M requests) | S4 |

The main results in Result 3 use the central values. Section S7 perturbs each parameter within its empirical range and confirms that the cost and gap spread from home-country ($\tau$ = 50 ms, $\varphi$ = 0, no routing) to global ($\tau$ = 500 ms, $\varphi$ = 100%, fully routable) is almost invariant to the AI shape parameters; a concise old-versus-revised comparison of these parameters is given in Section S6.

---


## Supplementary Section S1 — Triangulated energy-basis calibration of the AI demand share and the inference–training split

### S1.1 Motivation and rebasing

Result 3 reshapes a share $\lambda$ of total data centre electricity demand using AI-workload operators (Methods 4.5). A consistent $\lambda$ must satisfy two requirements that earlier single-parameter treatments failed to meet.

First, a unified measurement basis. Published AI-share figures conflate a *capacity / power-demand* basis (GW/GW, e.g. McKinsey, Goldman) with an *energy* basis (TWh/TWh, e.g. Gartner, IEA). Capacity shares are systematically higher than energy shares, because AI and accelerated servers run at higher and steadier utilization than conventional servers, so the two cannot be merged with equal weight. We therefore base $\lambda$ on a single energy (TWh/TWh) basis, retaining capacity-basis sources only for the 2050 ceiling.

Second, inference and training are physically distinct loads. The shape operators P2 (sharpening) and P3 (bursting) impose an intraday, evening-peaked, event-bursty profile—an *inference-serving* phenomenon. At fleet scale, training is a sustained, near-flat, schedulable baseload (and, within the routing framework, also the load most readily relocated and served off-site). Reshaping the entire AI-related share with intraday and bursty operators would fold the flat training baseload into the evening peak, inflating the cost of the peaked scenarios. We therefore split $\lambda$ into an inference component that drives the shape operators and a training component that enters as a flat baseload

$$ \lambda_{\mathrm{inf}}(t)=\lambda_{\mathrm{AI}}(t)\,f_{\mathrm{inf}}(t),\qquad \lambda_{\mathrm{train}}(t)=\lambda_{\mathrm{AI}}(t)\,\bigl(1-f_{\mathrm{inf}}(t)\bigr) \tag{S1} $$

where $\lambda_{\mathrm{AI}}$ is the total AI-related energy share and $f_{\mathrm{inf}}$ is the energy-basis inference fraction within AI. Only $\lambda_{\mathrm{inf}}$ is reshaped by P2/P3; $\lambda_{\mathrm{train}}$ enters each operator as a flat baseload term (Methods 4.5). The previous single AI-related set (0.231 / 0.402 / 0.699 for 2025 / 2030 / 2050) corresponds almost exactly to the total $\lambda_{\mathrm{AI}}$ here (0.20 / 0.44 / 0.70), confirming that it was originally an AI-related share rather than an inference share.

### S1.2 Sources for the energy-basis AI demand share

**Table S1. Energy-basis estimates of the AI demand share.**

| Year | Source | Definition (energy basis) | $\lambda_{\mathrm{low}}$ | $\lambda_{\mathrm{mid}}$ | $\lambda_{\mathrm{high}}$ |
|------|--------|---------------------------|-------|-------|--------|
| 2025 | Gartner (November 2025) | AI-optimized servers / total data centre electricity (93/448 TWh) | 0.183 | 0.208 | 0.233 |
| 2030 | Gartner (November 2025) | AI-optimized servers / total data centre electricity (432/980 TWh) | 0.416 | 0.441 | 0.466 |
| 2025 | IEA-4E / EDNA (2025) | AI-related share of data centre electricity (5–15 % in 2023, growing at ~30 %/yr) | 0.120 | 0.169 | 0.210 |
| 2030 | IEA-4E / EDNA (2025) | AI-related share of data centre electricity | 0.350 | 0.425 | 0.500 |

The two independent energy-basis sources agree closely for 2030 (Gartner 44.1 %, EDNA 35–50 %). The IEA *Energy and AI* report (2025) provides a plausibility check on the denominator — total data centre electricity rises from 415 TWh (2024) to about 945 TWh (2030), with accelerated servers accounting for nearly half of the net increment. Capacity-basis sources are excluded from $\lambda_{\mathrm{AI}}$ — McKinsey puts AI at ~70 % of data centre power demand (GW) in 2030, and Goldman at 39 % of data centre power—both higher than the energy share, and used only as a reference for the 2050 ceiling (S1.5). The energy-basis set (the equal-weight mean of the two sources) is $\lambda_{\mathrm{AI},\mathrm{mid}}$ = 0.19 (2025) and 0.43 (2030), rounded to the operating values $\lambda_{\mathrm{AI}}$ = 0.20 (2025) and 0.44 (2030).

### S1.3 Inference fraction

$f_{\mathrm{inf}}$ is the energy-basis inference share of AI electricity (numerator = inference electricity; denominator = AI = training + inference). Only measured-fleet or review figures are used; the 80–90 % inference numbers circulating elsewhere are compute/FLOP or revenue splits (Schneider Electric, NVIDIA), not measured-fleet energy, and are therefore excluded.

**Table S2. Measured inference fraction of AI electricity.**

| Source | Inference / training (energy) | $f_{\mathrm{inf}}$ |
|--------|-------------------------------|-------|
| Google / Patterson et al. 2022 (measured fleet) | 60 % / 40 % | 0.60 |
| IEA-4E / EDNA 2025 review | 60–70 % / 20–40 % (+<10 % model development) | 0.60–0.70 |
| EPRI | ~60 % / ~30 % (normalized 0.667) | ~0.67 |

The <10 % model-development overhead is folded into training (conservative with respect to deferrability), so that $\lambda_{\mathrm{inf}}$ + $\lambda_{\mathrm{train}}$ = $\lambda_{\mathrm{AI}}$ holds exactly. As deployment scales, the inference share rises — training is broadly amortized and one-off, whereas inference recurs throughout a model's operating life and query volumes keep growing. We adopt $f_{\mathrm{inf}}$ = 0.65 (2025), 0.70 (2030) and 0.80 (2050).

### S1.4 Decomposition into the Result 3 anchors

**Table S3. Decomposition of $\lambda$ into inference and training anchors.**

| Year | $\lambda_{\mathrm{AI}}$ (mid) | $f_{\mathrm{inf}}$ | $\lambda_{\mathrm{inf}}$ (→ P2/P3) | $\lambda_{\mathrm{train}}$ (→ flat) |
|------|-----------|-------|---------------------|----------------------|
| 2025 | 0.20 | 0.65 | 0.130 | 0.070 |
| 2030 | 0.44 | 0.70 | 0.308 | 0.132 |
| 2050 | 0.70 | 0.80 | 0.560 | 0.140 |

TWh plausibility check on the central values — 2030 inference ≈ 0.308 × 945 ≈ 291 TWh; 2030 training ≈ 0.132 × 945 ≈ 125 TWh. The 2030 inference range (low/high) is [0.21, 0.375] (from $\lambda_{\mathrm{AI}}$ [0.35, 0.50] × $f_{\mathrm{inf}}$ [0.60, 0.75]), used in the sensitivity sweep of S7.

### S1.5 Logistic extrapolation and the 2050 ceiling

The anchoring data extend only to 2030, so the saturation level is an explicit prior. The total AI energy share tends to 0.70 in 2050 (range 0.60–0.80) — conventional and cloud workloads keep growing, so AI cannot approach 100 % of data centre energy, but over decades AI grows faster than they do (McKinsey puts AI at ~70 % of data centre GW in 2030, lower once converted to an energy basis; ABI puts AI at ~64 % of data centre energy in 2035 and still rising). $f_{\mathrm{inf}}$ rises to 0.80 (range 0.70–0.85). The logistic fit for $\lambda_{\mathrm{AI}}$ is

$$ \lambda_{\mathrm{AI}}(t)=0.02+\frac{0.70-0.02}{1+\exp\!\bigl(-k\,(t-t_0)\bigr)} \tag{S2} $$

with k ≈ 0.31 and $t_0$ ≈ 2028.6, giving $\lambda_{\mathrm{AI},2050}$ = 0.699 ≈ the ceiling. Combined with $f_{\mathrm{inf},2050}$ = 0.80, this yields $\lambda_{\mathrm{inf},2050}$ = 0.560 and $\lambda_{\mathrm{train},2050}$ = 0.140.

![](figures_globalsites/figS1_lambda_trajectory.png)

**Figure S1. Triangulated energy-basis AI demand share and its inference/training split, 2024–2050.** **(a)** The AI share of data-centre electricity on an energy basis, $\lambda_{\mathrm{AI}}$, follows a logistic trajectory (floor 0.02, ceiling 0.70, $k$ = 0.31, $t_0$ = 2028.6; solid line) anchored to the ensemble mid-points $\lambda_{\mathrm{AI}}$ = 0.20 (2025), 0.44 (2030) and 0.70 (2050) (filled circles); vertical bars give the source ranges (2030 [0.35, 0.50]; 2050 [0.60, 0.80]), and the individual Gartner and IEA-4E/EDNA estimates are shown for 2025 and 2030. **(b)** Decomposition of $\lambda_{\mathrm{AI}}$ into the inference share $\lambda_{\mathrm{inf}}$ (which drives the P2/P3 intraday shape operators) and the flat training baseload $\lambda_{\mathrm{train}}$, as stacked areas summing to $\lambda_{\mathrm{AI}}$; the dashed line is the inference fraction $f_{\mathrm{inf}}$ (right axis), rising from 0.65 to 0.80. The anchor values are $\lambda_{\mathrm{inf}}$ = 0.130 / 0.308 / 0.560 and $\lambda_{\mathrm{train}}$ = 0.070 / 0.132 / 0.140 for 2025 / 2030 / 2050 (Table S3).

---


## Supplementary Section S2 — Empirical workload-shape fitting from BurstGPT v2

### S2.1 Data

We use the public BurstGPT v2.0 trace (Wang et al. 2024, arXiv — 2401.17644). The cleaned (failure-free) release contains three trace windows.

**Table S4. Cleaned BurstGPT v2.0 trace windows.**

| Trace window | Requests (cleaned) | Time span | Notes |
|------|--------------------|------|-------|
| Window 1 | 1,404,294 | ~61 days | First trace window |
| Window 2 | 3,784,213 | ~63 days | Second window |
| Window 3 | 4,956,058 | ~111 days | Third (longest) window |
| Total | 10,144,565 | ~335 days | Pooled fit |

Timestamps are given in seconds from 00:00 on day 1 and are calibrated to the user's local time zone (per the BurstGPT v2.0 documentation), so the hour-of-day directly indexes the user's local hour.

### S2.2 Methods

For each weighting w ∈ {request count, total tokens, response tokens}, we (1) aggregate w by hour-of-day = (timestamp // 3600) mod 24 and normalize to sum-to-one shares, yielding the intraday share profile; (2) fit a Gaussian by nonlinear least squares,

$$ g(h)=\mathrm{base}+\mathrm{amp}\cdot\exp\!\left(-\frac{(h-\mu)^2}{2\sigma^2}\right) \tag{S3} $$

and (3) compute burst quantiles on 1-hour binned sums, reporting the mean-relative P50, P90, P95, P99, P999 and maximum.

### S2.3 Pooled fit

**Table S5. Pooled Gaussian fits by mass weighting.**

| Weighting | Peak hour $\mu$ | $\sigma$ (h) | FWHM (h) |
|-----------|-------------|-------|----------|
| Request count | 18.31 | 3.99 | 9.40 |
| Total tokens (primary) | 18.05 | 3.79 | 8.93 |
| Response tokens (compute proxy) | 18.12 | 3.98 | 9.37 |

The three weightings give virtually identical peaks (within 0.3 h of each other) and $\sigma$ values within 0.2 h, indicating that the intraday structure is robust to the choice of mass proxy.

### S2.4 Bootstrap 95% confidence intervals

We resample 5-minute token-volume bins with replacement (500 iterations), refit the Gaussian for each resample, and report the 2.5 / 50 / 97.5 percentiles. Only the total-token weighting (primary) is shown.

**Table S6. Bootstrap 95% confidence intervals (total-token weighting).**

| Parameter | Point estimate | 95% CI (2.5–97.5 percentile) |
|-----------|---------------:|-------------------------:|
| Peak hour $\mu$ | 18.05 | [17.75, 18.39] |
| $\sigma$ (h) | 3.79 | [3.53, 4.05] |

The confidence intervals are narrow — the intraday AI-inference peak is pinned to within ±0.3 h and $\sigma$ to within ±0.3 h.

![](figures_globalsites/figS2_burstgpt_diurnal.png)

**Figure S2. Empirical intraday inference shape from the BurstGPT v2 trace, its Gaussian fit and bootstrap confidence interval.** **(a)** The normalized hour-of-day token share pooled over the BurstGPT v2 trace (10.1 M requests; grey bars) is bimodal, peaking at 0.064 at hour 17 with a secondary lobe at hours 19–20. The fitted Gaussian (token weighting, peak $\mu$ = 18.0 h, $\sigma$ = 3.8 h; purple line) and the 95% bootstrap confidence band (500 resamples; shaded) are overlaid, with the dashed line marking $\mu$ = 18.0 h. **(b)** Stationarity of the fit across the three trace slices in ($\mu$, $\sigma$) space (orange circles, with each slice's hourly P99/mean labelled) against the pooled token fit (teal diamond; P99/mean = 9.31) and the 95% bootstrap confidence rectangle ($\mu$ ∈ [17.75, 18.39], $\sigma$ ∈ [3.53, 4.05]). The peak hour and width are stable across slices; only the burst ratio varies, motivating the $\gamma$ sensitivity range used in S7.

### S2.5 Temporal stationarity

To test whether the intraday shape drifts over the ~11-month trace, we refit each window slice separately.

**Table S7. Stationarity of the intraday fit across trace slices.**

| Slice | n requests | Duration | Peak hour | $\sigma$ (h) | Hourly P99/mean |
|-------|------------|----------|-----------|-------|-----------------|
| 1 | 1.40 M | 61 d | 17.69 | 4.20 | 10.63 |
| 2 | 3.78 M | 63 d | 17.07 | 4.20 | 5.07 |
| 3 | 4.96 M | 111 d | 18.38 | 3.26 | 10.70 |

Peak hour and $\sigma$ are stable across slices (peak ∈ [17.1, 18.4], $\sigma$ ∈ [3.3, 4.2]). The only parameter with appreciable variation is the hourly P99/mean ratio (slice 2 is about 2× quieter in bursts than slices 1 and 3). We take the pooled estimate as the central value and fold this within-trace variability into the $\gamma$ sensitivity range in S7.

### S2.6 Derivation of the burst-amplification operator

The BurstGPT hourly P99 / mean ratio (pooled, token) is 9.31, with a per-slice range of [5.07, 10.70]. It is mapped through Section S3 to the P3 operator $\gamma$.

### S2.7 Slice-2 burst-ratio outlier

The hourly P99/mean for slice 2 (5.07) is roughly half that of slice 1 (10.63) and slice 3 (10.70). Investigation shows that this is not a parameter-instability issue in the Gaussian fit—the peak hour and $\sigma$ are stable (17.07 and 4.20 for slice 2 versus 17.69 / 4.20 for slice 1 and 18.38 / 3.26 for slice 3)—but rather a difference in traffic composition. The calendar window covered by slice 2 contains fewer extreme viral spikes in the trace, with a more uniform day-to-day variance. We treat slice 2 as the low-burst end of the empirical within-trace range and slices 1 and 3 as the high-burst end, with the pooled estimate ($\gamma_{\mathrm{emp}}$ = 9.31) lying between them. This within-trace variability is precisely what motivates the choices $\gamma$ ∈ {3.0, 4.84, 6.0} in S7—it spans both the stylized prior and an upper bound consistent with the high-burst slices.

### S2.8 Empirical-shape operator versus Gaussian-sharpening operator

The Gaussian obtained from the BurstGPT fit ($\mu$ = 18.0, $\sigma$ = 3.8), although convenient, smooths over a real bimodal structure in the empirical 24-hour profile — the token share peaks at hour 17 (0.064) and again at hours 19–20 (0.058–0.059), with a slight midday dip, and retains substantial mass (> 0.04) throughout hours 12–23. To remove the Gaussian assumption from the chain, we introduce the empirical-shape operator P2_emp

$$ d_c^{(P2_{\mathrm{emp}})}(t)=(1-\lambda_{\mathrm{inf}}-\lambda_{\mathrm{train}})\,d_c(t)+\lambda_{\mathrm{inf}}\,s^{\mathrm{emp}}(h(t))\,\langle d_c\rangle+\lambda_{\mathrm{train}}\,\langle d_c\rangle \tag{S4} $$

where $s^{\mathrm{emp}}$ is the pooled BurstGPT v2 24-hour share vector, normalized to mean = 1 (a parameter-free assumption); only the inference share $\lambda_{\mathrm{inf}}$ takes the empirical shape, while the training share $\lambda_{\mathrm{train}}$ is a flat baseload (S1). P2_emp is one of the four deferred operators shown in Fig. S4 (main-text Fig. 4 and Fig. 5 use only the realistic Mix as the representative case; the remaining four operators—batch-plus-training, consumer chat, measured chat and viral spike—together with the AI-free baseline Today as a reference block, are all deferred to Fig. S4).

**Table S8. Empirical-shape operator versus Gaussian-sharpening operator in 2030 (national-level Result 3 over 104 countries, $\tau$×$\varphi$ basis, full electricity-supply cost).**

| Operator | Domestic uncovered | Domestic cost (USD/MWh) | Global uncovered | Global cost (USD/MWh) |
|---|---:|---:|---:|---:|
| P2 sharpened (Gaussian) | 34.5 % | 126.6 | 14.3 % | 114.4 |
| P2_emp (BurstGPT raw 24-hour) | 31.5 % | 120.7 | 13.2 % | 108.5 |
| $\Delta$ (Gaussian relative to empirical) | +3.0 pp | +5.9 | +1.1 pp | +5.9 |

Domestic is the $\tau$=50 ms, $\varphi$=0 (no routing) corner, and global is the $\tau$=500 ms, $\varphi$=100% (fully routable) corner. The Gaussian abstraction overstates the global-tier full electricity-supply cost by about 5% (114.4 vs 108.5 USD/MWh), with the most affected firming-storage component overstated by about 19% (37.6 vs 31.6), because it concentrates the inference mass on a single sharp peak (peak height ≈ 1.7 × mean), whereas the empirical shape is broader and multi-peaked, allowing more AI load to overlap with available renewable supply. P2_emp is methodologically cleaner and is the recommended reference; we report both because (i) the Gaussian P2 is widely understood and easy to perturb (CI bounds, alternative $\mu$, $\sigma$) and (ii) showing both quantifies the bias introduced by this parametric simplification. Under the rebasing (only $\lambda_{\mathrm{inf}}$ is reshaped, $\lambda_{\mathrm{train}}$ flat), the six AI shapes at the global tier rank in ascending full electricity-supply cost as batch-plus-training (107) < measured chat (108.5) < Mix (112.3) < viral spike (112.7) < AI-free baseline (114.1) < consumer chat (114.4); the six differ by only about 7 USD/MWh, far smaller than the domestic-to-global routing drop (about 12 USD/MWh)—the burst spike, because it concentrates only in the top-1% hours and is flatter the rest of the time, is in fact more readily absorbed by routing and storage than consumer chat, which sustainedly raises the evening peak.

---


## Supplementary Section S3 — Converting the burst-amplification operator from an empirical hourly burst to an additive amplification

### S3.1 Derivation

The P3 burst-amplification operator in the main text is

$$ d^{(P_3)}(t)=(1-\lambda_{\mathrm{train}})\,d(t)+\lambda_{\mathrm{train}}\,\langle d_c\rangle+\mathbb{1}\!\left[d(t)\ge Q_{0.99}(d)\right]\,d(t)\,\lambda_{\mathrm{inf}}\,\gamma_{\mathrm{op}} \tag{S5} $$

That is, bursts (an inference phenomenon) scale with the inference share $\lambda_{\mathrm{inf}}$, whereas the training share $\lambda_{\mathrm{train}}$ is flattened into a baseload (S1). At the top-1% hours, the burst excess injected by AI equals d(t) · $\lambda_{\mathrm{inf}}$ · $\gamma_{\mathrm{op}}$. The BurstGPT trace yields the empirical burst ratio of the AI workload's own time series, $\gamma_{\mathrm{emp}}$ ≡ hourly P99 / hourly mean. To express the operator parameter $\gamma_{\mathrm{op}}$ in terms of these quantities, we equate the excess — the inference mass at a burst hour should equal $\gamma_{\mathrm{emp}}$ times its own mean. Under equal-energy alignment (mean inference load = $\lambda_{\mathrm{inf}}$ · $\mathrm{mean}_d$), this gives

$$ \gamma_{\mathrm{op}}=\frac{\gamma_{\mathrm{emp}}-1}{d_{\mathrm{burst}}/\mathrm{mean}_d} \tag{S6} $$

where $d_{\mathrm{burst}}$ / $\mathrm{mean}_d$ is the peak-to-mean ratio of human traffic at the top-1%.

### S3.2 Country-level burst ratios from Cloudflare

We computed $d_{\mathrm{burst}}$ / mean individually for each of the 104 demand countries from the Cloudflare Radar hourly human-traffic proxy (2025-05-02 to 2026-05-01).

**Table S9. Cross-country distribution of the human-traffic peak-to-mean ratio.**

| Statistic | $d_{\mathrm{burst}}$ / mean |
|-----------|---------------:|
| Minimum | 1.353 |
| 25th percentile | 1.632 |
| Median (recommended) | 1.716 |
| Mean | 1.777 |
| 75th percentile | 1.885 |
| Maximum | 2.950 |

The cross-country median of 1.716 replaces the heuristic constant of 1.75 used in earlier versions of the analysis.

### S3.3 Recommended burst-amplification coefficient

Combining S2.6 ($\gamma_{\mathrm{emp}}$ = 9.31) with S3.2 (median $d_{\mathrm{burst}}$/mean = 1.72) yields $\gamma_{\mathrm{op}}$ = (9.31 − 1) / 1.72 = 4.84, with a country-level IQR-derived range of [4.41, 5.09]. The old value $\gamma$ = 3 (used in manuscript v1) corresponds to $\gamma_{\mathrm{emp}}$ = (3 · 1.72) + 1 = 6.16, below the BurstGPT-fitted value of 9.31 — that is, $\gamma$ = 3 markedly underestimates the empirical burst structure.

![](figures_globalsites/figS3_burst_ratio_dist.png)

**Figure S3. Cross-country distribution of the human-traffic peak-to-mean ratio and the derived burst-amplification coefficient.** **(a)** Distribution across the 104 demand countries of the hourly human-traffic peak-to-mean ratio $d_{\mathrm{burst}}/\mathrm{mean}$ (Cloudflare Radar, 2025-05-02 to 2026-05-01); the cross-country median is 1.716 (orange line), the interquartile range [1.632, 1.885] is shaded, and the full range spans [1.353, 2.950]. This median replaces the heuristic constant 1.75 used in earlier versions of the analysis. **(b)** Mapping to the P3 burst-amplification coefficient $\gamma_{\mathrm{op}} = (\gamma_{\mathrm{emp}}-1)/(d_{\mathrm{burst}}/\mathrm{mean})$ with $\gamma_{\mathrm{emp}}$ = 9.31 (pooled BurstGPT hourly P99/mean) — the recommended default is $\gamma_{\mathrm{op}}$ = 4.84 at the median ratio (circle), with the IQR-derived band [4.41, 5.09] shaded.

---

## Supplementary Section S4 — Independent cross-validation on the Azure LLM Inference Trace 2024

### S4.1 Dataset

The Azure LLM Inference Dataset 2024 is an independent enterprise-grade LLM-serving trace, comprising two service types and collected from 10–19 May 2024.

**Table S10. Azure LLM Inference 2024 service traces.**

| Service | Requests | Span | UTC start | UTC end |
|---------|---------:|-----:|-----------|---------|
| conv (conversational) | 27,303,835 | 7.0 days | 2024-05-12 00:00 | 2024-05-19 00:00 |
| code (code assistant) | 16,803,695 | 7.0 days | 2024-05-10 00:00 | 2024-05-17 00:00 |
| Total | 44,107,530 | | | |

Each record contains a timestamp, an input (context) token count, and a generated token count; timestamps are absolute UTC date-times (unlike BurstGPT, no local-timezone calibration was applied).

### S4.2 Fitting and cross-validation

**Table S11. Diurnal fits and burst ratios for BurstGPT and Azure.**

| Trace | Service | Peak (h) | $\sigma$ (h) | Hourly P99/mean |
|-------|---------|---------:|------:|-----------------:|
| BurstGPT v2 merged | (all, local timezone) | 18.05 | 3.79 | 9.31 |
| Azure LLM 2024 | conv (UTC) | 13.90 (counts) / 10.6 (tokens, noisy) | 3.43 (counts) | 1.53 |
| Azure LLM 2024 | code (UTC) | 17.92 | 3.24 | 3.10 |
| Azure LLM 2024 | merged (tokens) | 17.22 (UTC) | 3.17 | — |

Three findings are noteworthy. First, $\sigma$ is consistent across the two datasets (BurstGPT 3.79 h, Azure merged 3.17 h, both within the [3.0, 4.0] h range), corroborating the roughly 9 h FWHM diurnal peak of AI inference. Second, the interpretation of the peak hour varies with the timezone reference — Azure's 17 UTC peak corresponds to European evening (19:00 local) or US-Eastern midday (13:00 EST), which, when the user population is weighted toward Europe and the United States, is broadly consistent with BurstGPT's 18 (local timezone) peak. Third, $\gamma_{\mathrm{emp}}$ varies by service type — BurstGPT consumer chat (9.31) ≫ Azure enterprise conv (1.53) > Azure code (3.10) — consumer-facing chat, subject to viral and news spikes, is far burstier than enterprise inference services. Result 3 covers this range through the S7 sensitivity over $\gamma$ ∈ {3.0, 4.84, 6.0}. The independent Azure trace corroborates the BurstGPT diurnal shape but reveals that burst intensity depends on service composition, validating both the empirical $\gamma$ default (BurstGPT, the consumer-facing worst case) and the necessity of the composite Mix operator (Supplementary Section S5).

---


## Supplementary Section S5 — Composite-workload mixing operator

### S5.1 Motivation

Real AI inference is not a single workload type. It spans batch and offline work (embeddings, content moderation, evaluation, fine-tuning jobs), which is latency-tolerant and naturally flat over 24 hours; and interactive and event-driven services (chat, copilots, voice, together with the viral and news spikes layered on top), which follow a BurstGPT-shaped diurnal distribution. The single-operator runs P1, P2 and P3 in the main text each isolate the effect of one shape in isolation. The composite Mix operator combines a flat batch share with a follow-the-sun share carrying a smooth BurstGPT evening shape. Top-1% burst amplification is deliberately not folded into the main-report Mix — concentrating all event-driven load into a single peak hour would inject the extra energy directly into the post-sunset gap window and would raise the central case on account of a single hour; event-driven bursts are therefore isolated in the separate P3 operator, whose contribution is reported on its own.

### S5.2 Mix weights

**Table S12. Composite Mix operator weights.**

| Component | Weight $\alpha$ | Basis |
|-----------|---------:|---------------|
| Batch (flat) | 0.25 | The OpenAI Batch API (50% discount, 24 h SLA) implies that scheduled inference is roughly ~20-30% of serving capacity; EcoServe reports that offline/batch makes up a large share of generative-AI inference |
| Follow-the-sun (BurstGPT shape) | 0.75 | Combines interactive services (chat, code copilots, RAG; about 0.65, the highest-revenue products that drive most inference traffic) with event-driven services (viral, news and release spikes; about 0.10, a tail phenomenon reasonably estimated at 5-15% of total serving). Both are represented by the same smooth BurstGPT diurnal shape; top-1% burst amplification is not applied here but isolated in the separate P3 operator |

The defaults sum to 1.0. The Azure LLM trace (S4) supports treating most serving as diurnally shaped — its enterprise conv ($\gamma$ ≈ 1.5) is closer to flat-shape serving, whereas consumer-facing chat (BurstGPT $\gamma$ ≈ 9.3) is strongly diurnally concentrated. The main-report Mix therefore carries a smooth diurnal shape without a single-hour spike; the burst tail is quantified separately by P3.

### S5.3 Operator definition

$$ d^{(P_{\mathrm{mix}})}(t)=(1-\lambda_{\mathrm{inf}}-\lambda_{\mathrm{train}})\,d(t)+\lambda_{\mathrm{train}}\langle d\rangle+\lambda_{\mathrm{inf}}\Bigl[\alpha_{\mathrm{batch}}\langle d\rangle+\alpha_{\mathrm{shaped}}\,w_{\mathrm{emp}}(h(t))\langle d\rangle\Bigr] \tag{S7} $$

where $w_{\mathrm{emp}}$(h) is the Gaussian fitted to BurstGPT v2 ($\mu$ = 18.0, $\sigma$ = 3.8, normalized to mean = 1), $\alpha_{\mathrm{batch}}$ = 0.25 and $\alpha_{\mathrm{shaped}}$ = 0.75. The composite AI mix acts on the inference share $\lambda_{\mathrm{inf}}$; the training share $\lambda_{\mathrm{train}}$ enters as a flat baseload (S1). Note that the $\alpha_{\mathrm{batch}}$ term (flat, scheduled inference) is conceptually distinct from $\lambda_{\mathrm{train}}$ (model training), even though both are flat. The main-report Mix contains no top-1% amplification term; each component preserves the mean of the inference share, so the operator adds no net energy and injects no single-hour spike into the evening gap window. Event-driven bursts are quantified separately by the P3 operator ($\gamma_{\mathrm{op}}$ = 4.84; Supplementary Section S3).

### S5.4 Sensitivity

With event bursts isolated in P3, the only remaining degree of freedom in Mix is the flat batch share (the rest follows the smooth diurnal shape). We report the residual uncovered share and the full electricity-supply cost for the 2030 global tier ($\tau$ = 500 ms, $\varphi$ = 100%) under three plausible batch shares, taken from the single-factor sweep in S7.

**Table S13. Sensitivity of the 2030 global-tier residual to the batch (flat) share.**

| Batch (flat) share | Global-tier uncovered share | Global-tier full electricity-supply cost (USD/MWh) |
|--------------------------------------|---------------------------:|-------------------------------:|
| 0.15 (sharper, more diurnally shaped) | 14.0% | 112.9 |
| 0.25 (default) | 13.9% | 112.3 |
| 0.35 (flatter, more batch) | 13.7% | 111.5 |

A flatter batch share is slightly cheaper, but the range is only about ±1 USD/MWh around the default, confirming that the Mix split has no material impact on the main results (full sweep in S7).

### S5.5 Presentation of the Mix operator in the main text

Mix is the only operator shown as a representative in the main text — it runs through all panels of Figure 4 (mechanism) and Figure 5 (cost)—in Figure 4, (a) the three-layer intraday demand decomposition, (b) the zero-sum gap panel over clean supply, (c) the six-shape domestic vs global uncovered-share dumbbell plot and (d) the uncovered-share heatmap of routable fraction × latency tolerance; in Figure 5, (a) the waterfall in which cost steps down as latency is relaxed (the global tier being the lowest step), (b) the rose diagram of full electricity-supply cost across 36 scenarios, (c) the ridgeline of year-by-year cost distributions for the domestic and global tiers and (d) the world map of country-by-country full electricity-supply cost. The main-text narrative cites Mix as the realistic expected case that lies between the two extremes—batch and training (flattest, cheapest) and consumer chat (sharpest evening peak, most expensive).

### S5.6 The four non-main-figure AI operators

Main-text Figures 4 and 5 use only the realistic mix (Mix) as the representative. The four deferred operators—together with the AI-free baseline (Today) as a reference block—are shown here in the same $\tau$×$\varphi$ framing and the same colour scheme as main-text Figures 4 and 5, to allow direct comparison with the main-text figures.

![](figures_globalsites/figS_ai_operators_a.png)

![](figures_globalsites/figS_ai_operators_b.png)

**Figure S4. The residual gap and cost are insensitive to AI demand shape — every shape collapses along the same domestic→global routing lever.** The operators comprise the AI-free baseline (P0, labelled Today in the figure) plus four deferred operators—batch and training (P1, flat/dispatchable), measured chat (P2_emp, empirical bimodal), consumer chat (P2, evening Gaussian peak) and viral event spikes (P3, top-1% amplification). **(a)-(e)** One block per operator — above, the intraday demand curve for that shape (2025/2030/2050 overlaid on the AI-free grey dotted line, equal-energy normalized, mean = 1); below, the 2030 heatmap of uncovered demand share over latency tolerance (columns, 50→500 ms) × routable fraction (rows, 0→100%) (cell colour and printed value encode the same share, %; layout as in main-text Figs 2a and 4d). **(f)** Cost view — dumbbell plot of the median full electricity-supply cost across 104 countries (USD/MWh, 2030) for the six shapes (including Mix) under the domestically reachable tier (50 ms, no routing) and the globally reachable tier (500 ms, fully routable). For every operator, the collapse of uncovered share and cost along the latency/routability axes far exceeds the differences produced by the shape itself — the six shapes differ by only about 7 USD/MWh within each tier, whereas the domestic-to-global routing drop is about 30; the cost ranking of the shapes is given in S2.8.

---

## Supplementary Section S6 — Summary table of revised parameters

In the table below, the revised values are those used in the published model; the old values are given for comparison only. $\lambda$ is now energy-based and is split into inference (which drives the shape operators) and training (a flat baseload); $\gamma_{\mathrm{op}}$ adopts the BurstGPT pooled P99/mean (9.31), so that all AI-workload parameters are traceable to a single, well-documented source.

**Table S14. Old vs revised AI-workload parameters.**

| Parameter | Old value | Revised value (this work) | Source |
|-----------|-------------:|---------------------------:|--------|
| $\lambda_{\mathrm{inf}}$,{2025} | 0.231 (AI-related aggregate, overall) | 0.130 [low 0.090, high 0.175] | $\lambda_{\mathrm{AI}}$ 0.20 × $f_{\mathrm{inf}}$ 0.65 (S1) |
| $\lambda_{\mathrm{inf}}$,{2030} | 0.402 (AI-related aggregate, overall) | 0.308 [low 0.210, high 0.375] | $\lambda_{\mathrm{AI}}$ 0.44 × $f_{\mathrm{inf}}$ 0.70 (S1) |
| $\lambda_{\mathrm{inf}}$,{2050} | 0.699 (AI-related aggregate, overall) | 0.560 | $\lambda_{\mathrm{AI}}$ 0.70 × $f_{\mathrm{inf}}$ 0.80 (S1) |
| $\lambda_{\mathrm{train}}$,{2025/2030/2050} | none (folded into $\lambda$) | 0.070 / 0.132 / 0.140 | $\lambda_{\mathrm{AI}}$ × (1 − $f_{\mathrm{inf}}$); flat baseload (S1) |
| P2 peak $\mu$ | 13.0 h (stylized) | 18.0 h [17.75, 18.39] | BurstGPT v2 pooled fit (S2) |
| P2 $\sigma$ | 3.0 h (stylized) | 3.8 h [3.53, 4.05] | BurstGPT v2 pooled fit (S2) |
| P3 $\gamma_{\mathrm{op}}$ | 3.0 (stylized) | 4.84 [4.41, 5.09] | BurstGPT v2 pooled P99/mean (9.31) ÷ country-by-country $d_{\mathrm{burst}}$/mean (1.716) (S2, S3) |
| $d_{\mathrm{burst}}$ / mean | 1.75 (heuristic) | 1.716 [1.63, 1.89] | Cloudflare 104-country median (S3) |
| Mix weights (batch / diurnally shaped) | none (new) | 0.25 / 0.75 | Literature-anchored AI service mix; event bursts isolated in P3 (S5) |
| Cross-validation | none (BurstGPT only) | Azure LLM 2024 (44M requests) confirms $\sigma$ and frames $\gamma$ by service type | S4 |

---


## Supplementary Section S7 — Single-factor sensitivity sweep of the 2030 main results

### S7.1 Design

For the representative year 2030, we perturb each AI-workload parameter within its empirical uncertainty interval (holding the others at their central values) and re-solve the country-level Result 3 for 104 countries at the two corner cases — domestic ($\tau$=50 ms, $\varphi$=0, no routing) and global ($\tau$=500 ms, $\varphi$=100%, fully routable). The reported quantities are the 104-country median full electricity-supply cost (LCOE, USD/MWh) and the residual uncovered share (%). Each parameter is applied to its most sensitive operator — $\lambda_{\mathrm{inf}}$ and $\gamma$ act on the P3 burst, ($\mu$,$\sigma$) on the P2 sharpening, and the batch share on the Mix (script `run_r3_latency_sensitivity.py`).

### S7.2 Sensitivity table

We evaluate the following variants on the full 104-country, country-level Result 3 portfolio.

**Table S15. Perturbation variants and their empirical sources.**

| Parameter | Variant | Source |
|-----------|----------|--------|
| $\lambda_{\mathrm{inf}}$,{2030} | low 0.21 / mid 0.308 / high 0.375 | energy-basis inference interval (S1) |
| P2 ($\mu$, $\sigma$) | CI low (17.75, 3.53) / mid (18.0, 3.79) / CI high (18.39, 4.05) | BurstGPT bootstrap 95% CI (S2) |
| P3 $\gamma$ | 3.0 stylized / 4.84 default / 6.0 upper bound | covers stylized prior + high-burst slice (S2–S3) |
| Mix batch share ($\alpha_{\mathrm{batch}}$) | 0.15 / 0.25 default / 0.35 (remainder = smooth daytime shape) | literature mixing bounds (S5) |

**Table S16. Domestic and global full electricity-supply cost (USD/MWh) and global-tier uncovered share (%) under single-factor perturbations for 2030, 104-country median.**

| Parameter | Variant | Domestic cost | Global cost | Global uncovered | Global cost spread |
|-----------|---------|------:|------:|------:|-------------:|
| $\lambda_{\mathrm{inf}}$,{2030} (applied to P3 burst) | low (0.21) | 123.3 | 113.6 | 14.0 % | — |
| | mid (0.308) | 123.9 | 112.7 | 14.3 % | central |
| | high (0.375) | 124.1 | 112.2 | 14.5 % | ±0.8 % |
| P2 ($\mu$, $\sigma$) bootstrap CI (applied to P2 sharpening) | CI low (17.75, 3.53) | 126.7 | 114.5 | 14.4 % | — |
| | CI mid (18.0, 3.79) | 126.6 | 114.4 | 14.4 % | central |
| | CI high (18.39, 4.05) | 126.8 | 114.5 | 14.6 % | ±0.1 % |
| P3 $\gamma$ (applied to P3 burst) | stylized (3.0) | 123.4 | 113.6 | 13.9 % | — |
| | default (4.84) | 123.9 | 112.7 | 14.3 % | central |
| | upper bound (6.0) | 124.2 | 112.3 | 14.5 % | ±0.8 % |
| Mix batch share (applied to Mix) | default (0.25) | 124.9 | 112.3 | 13.9 % | central |
| | flatter (0.35) | 124.2 | 111.5 | 13.7 % | — |
| | sharper (0.15) | 126.2 | 112.9 | 14.0 % | ±0.5 % |

### S7.3 Key observations

Every sensitivity interval preserves the domestic-to-global drop in cost and gap — across all 12 variants the domestic full electricity-supply cost stays within [123.3, 126.8] and the global cost within [111.5, 114.5] USD/MWh, with the global uncovered share within [13.7, 14.6] %—the roughly 12 USD/MWh (about −19 percentage points) routing drop from domestic to global is almost invariant to any AI parameter, with the latency tolerance × routable fraction dual-axis lever overwhelmingly dominant. Each AI parameter moves global cost little (spread ≤ ±0.8 %) — the $\lambda_{\mathrm{inf}}$ interval and P3 $\gamma$ are slightly larger (about ±0.8 %), the Mix batch share next (about ±0.5 %), and the P2 CI negligible (±0.1 %); because the central $\lambda_{\mathrm{inf}}$,{2030} = 0.308 is the main-result value, the $\lambda_{\mathrm{inf}}$-mid row coincides exactly with the $\gamma$-default row that is likewise applied to P3 (domestic 123.9, global 112.7), confirming internal consistency between the sweep and the main results. The P2 bootstrap CI is already narrow enough to be ignored at the main-result level — across the entire 95 % CI the global cost barely moves (114.4–114.5). Finally, the bias introduced by the parameterized Gaussian abstraction in P2 exceeds the within-Gaussian parameter uncertainty (S2.8 — global-tier P2_emp 108.5 vs Gaussian P2 114.4, ≈ +5 %), which supports reporting both, using P2_emp as the methodologically cleaner reference while retaining the Gaussian P2 for its analytical tractability under perturbation.

### S7.4 Factors not covered

A full multi-factor Monte Carlo (jointly sampling $\lambda$, $\mu$, $\sigma$, $\gamma$ and the mixing weights) would yield a tighter probabilistic envelope but requires about 500 Result 3 re-runs. The single-factor sweep above gives an upper bound on parameter-induced uncertainty, because inter-parameter correlations would partially cancel.

---

## Supplementary Section S8 — Country-level domestic mismatch table

Under the equal-energy domestic accounting of Result 1 (Methods 4.1–4.2), Table S17 reports for each of the 104 sampled countries — the annual hourly wind-and-solar matching gap $U_c$ (i.e. the annual uncovered-demand share; its identities with the 24/7 CFE score, temporal self-sufficiency, load match index and overlap coefficient are given in S21), the peak local hours of demand and supply and their phase lag $\Delta h_c$, continuous-gap-segment statistics (median, 95th percentile, maximum segment length, hours), the maximum hourly uncovered intensity relative to mean demand, and the intraday spectral concentration of demand and supply. The complete country-by-country table is provided as Table S17.

---


## Supplementary Section S9 — Supply-side capacity-factor construction and validation

### S9.1 Construction

The supply side is built from real wind and solar PV power stations rather than a single national representative capacity factor (Methods 4.4). From the WRI Global Power Plant Database we took all 16,009 solar PV and wind units (latitude, longitude, installed capacity) and aggregated them on a 3° grid into 455 capacity-weighted representative stations, retaining about 91 % of global installed capacity and covering every country on record. The three demand countries with no domestic stations on record (Indonesia, Libya, Paraguay) each received a national-centroid pairing (one solar PV, one wind), for 461 candidate stations in total. For each station we drew hourly shortwave irradiance, 2 m air temperature and 100 m wind speed from the Open-Meteo ERA5 archive for 2025-05-02 to 2026-05-01, aligned with the demand window.

Solar PV capacity factors follow the first-order PVWatts/GSEE model,

$$ \mathrm{CF}_{\mathrm{PV}}=\mathrm{clip}\!\left(\frac{G}{1000}\bigl[1+\gamma\,(T_{\mathrm{cell}}-25)\bigr],\,0,\,1\right) \tag{S8} $$

$$ T_{\mathrm{cell}}=T_{2\mathrm{m}}+\frac{\mathrm{NOCT}-20}{800}\,G \tag{S9} $$

with NOCT = 45 °C and $\gamma$ = −0.0035 $^{\circ}\mathrm{C}^{-1}$. Wind capacity factors use a smoothed IEC single-turbine power curve (cut-in 3.5, rated 12, cut-out 25 m/s), with the wind-speed distribution Gaussian-smoothed to approximate multi-turbine aggregation; because reanalysis systematically underestimates output at high-quality wind sites (Davidson & Millstein 2022), a ×1.2 bias correction is applied to the wind capacity factor. National solar PV and wind capacity factors are the domestic capacity-weighted means of the respective station factors; cross-national weighting uses IRENA grid-connected solar PV / wind installed capacity.

### S9.2 Validation

Three lines of evidence delimit the realism of the constructed factors. First, equal-energy normalization makes the results insensitive to level — demand and supply are each rescaled to the same annual total (Methods 4.1), so any residual mismatch reflects only differences in temporal shape, not the absolute level of the capacity factors. Sweeping the wind ×1.2 correction between 1.0 and 1.4 changes only the absolute level of the wind capacity factor, not its temporal shape, so the portfolio results are essentially unchanged (Methods 4.6). Second, coverage is not a binding choice — lowering the capacity coverage of the 3° grid representative stations from the default ~91 % to ~80 % or raising it to ~96 % shifts the global-reach median uncovered share by less than 0.5 pp (Methods 4.6). Third, the methodology is well established — the reanalysis-to-capacity-factor chain (PVWatts/GSEE solar PV model, IEC wind power curve on ERA5) is the same validation approach used by Pfenninger & Staffell (2016) and Tong et al. (2021), and the resulting national mean capacity factors fall within the published ranges for utility-scale solar PV (~10–25 %) and onshore wind (~20–45 %) reported by IRENA and the IEA.

We further validate the modelled capacity factors directly against measured hourly generation, at the level of timing rather than annual mean. Two countries with open, hourly measured generation and representing distinct regimes were chosen — Germany (SMARD / Bundesnetzagentur; continental, solar and onshore wind) and Great Britain (Elexon / BMRS; maritime, offshore-wind-dominated) — and over the paper's own window (2 May 2025 to 1 May 2026) their measured hourly solar/wind generation was aligned hour-by-hour with our modelled national capacity factors (each normalised to unit mean, so that only timing is compared), yielding the hourly correlation, diurnal-profile correlation, peak-hour offset and seasonal correlation of Table S18. For solar, the hourly correlation is 0.94 in both countries, the diurnal-profile correlation 0.96, the peak-hour offset ≤ 1 h and the seasonal correlation 0.99; for wind, the hourly correlation reaches 0.94 in offshore-dominated Great Britain and is 0.68 in onshore, complex-terrain Germany (a known limitation of reanalysis for onshore-wind sub-daily variability), while its diurnal and seasonal structure still correlate at 0.91 and 0.95. The phase structure on which this study depends — sunset timing, diurnal shape and seasonality — is thus directly supported by measured generation; wind–solar complementarity and continuous-deficit duration are built on these validated timing features rather than on annual-mean energy.

**Table S18. Timing validation of the modelled capacity factors against measured hourly generation (2 May 2025 – 1 May 2026, shapes normalised to unit mean).**

| Country · tech | Measured source | Hourly r | Diurnal r | Peak-hour offset | Seasonal r |
|---|---|---|---|---|---|
| Germany · solar PV | SMARD | 0.94 | 0.96 | +1 h | 0.99 |
| Germany · wind | SMARD | 0.68 | 0.91 | +2 h | 0.95 |
| Great Britain · solar PV | Elexon | 0.94 | 0.96 | +1 h | 0.99 |
| Great Britain · wind | Elexon | 0.94 | 0.93 | −1 h | 0.98 |

---

## Supplementary Section S10 — Cross-national supply portfolio optimization

The domestic portfolio is a non-negative least-squares (NNLS) complementarity problem (Methods 4.3; Sterl et al. 2020, Tong et al. 2021). Supply is fixed domestic — for demand country c, only the station–technology units of c itself each contribute one basis function — its hourly capacity factor normalized to unit mean — stacked into a matrix $\Phi_c$. The mixed supply curve $\tilde{s}_c$ = $\Phi_c$ $w_c$ is fitted to hourly demand through

$$ \min_{w_c\ge 0}\ \lVert \Phi_c\,w_c-d_c\rVert_2^2 \tag{S10} $$

solved by non-negative least squares. After solving, the weights are rescaled to equal annual energy, the residual uncovered share follows the definition of $U_c$ (Methods 4.2), and $s_c$ is replaced by the mixed curve $\tilde{s}_c$. Because the objective is a squared deviation while the reported metric is the one-sided uncovered energy, we cross-check with a linear program that minimizes uncovered energy directly under the same equal-energy constraint. The minimum-uncovered domestic mix solves

$$ \min_{w_c\ge 0,\ u\ge 0}\ \sum_t u_t \quad\text{s.t.}\quad u_t\ge d_c(t)-(\Phi_c w_c)(t),\qquad \sum_t (\Phi_c w_c)(t)=\sum_t d_c(t) \tag{S10a} $$

where $u_t$ is the per-hour uncovered slack and $\sum_t u_t$ the total uncovered energy. The global routed floor uses a two-step analog — the residual of (S10a), $g_c(t)=\max(d_c-\Phi_c w_c,0)$, has its routable part $r_c(t)=\min(\varphi\,d_c(t),g_c(t))$ maximally absorbed by an equal-energy non-negative mix of reachable partners' clean shapes $\hat s_i$,

$$ \max_{w\ge 0,\ 0\le \mathrm{cov}\le r_c}\ \sum_t \mathrm{cov}_t \quad\text{s.t.}\quad \mathrm{cov}_t\le \sum_{i\in\mathcal R_c(\tau)} w_i\,\hat s_i(t),\qquad \sum_t\sum_i w_i\,\hat s_i(t)=\sum_t r_c(t). \tag{S10b} $$

Both are standard linear programs. The domestic baseline falls from 41.8% to 40.8% and the global routed floor from 12.7% to 12.4%, each only about 1 percentage point lower (per-country median difference < 0.2 pp), so the reported residuals are robust to the objective and are conservative upper bounds on the geometric floor. $\Phi_c$ contains only the stations of c and aggregates nothing across borders, so the portfolio step involves no cross-border transmission or its efficiency losses; the geographic lever is provided by cross-longitude routing of the compute load (rather than by supply aggregation), as set out in the two-pass procedure and round-trip-time gating below (S11).

Two basis sets are used. Result 1–2 use a domestic station-level basis (461 representative stations assigned to their home country, each country using only its own stations), letting the optimizer select the specific stations most complementary to domestic demand. Result 3 (the AI-workload-shape scenario) uses a country × technology capacity-weighted basis (105 countries × {solar PV, wind} = 210 basis functions) — re-solving the station-level basis over the full 6 operators × multiple years × 6 latitude tolerances × 6 routable fractions × 104 countries (about 4 × $10^{4}$ NNLS problems of size 8,760 × N) is computationally infeasible. Cross-checking with the station-level basis at the domestic and global corners of $\tau$×$\varphi$ reproduces the same lever ordering and same-order-of-magnitude costs, so the conclusions of Result 3 do not depend on this basis choice.

The compute load is routed and coupled to the portfolio through a monotonicity-preserving two-pass procedure (Methods 4.4) — the domestic portfolio is first solved on the unrouted demand to obtain the residual gap; the routable part of that gap is then, within its network round-trip-time budget (RTT ≤ $\tau$), covered within the same hour by matching its timing to a static non-negative mix of reachable receiving countries' clean-generation shapes (equal-energy-normalised, magnitude divided out), the covered amount being the elementwise minimum of the two; the domestic portfolio is then re-solved on the post-routing effective demand. The two-pass procedure guarantees that enabling routing never increases the residual.

---

## Supplementary Section S11 — Round-trip-time model and validation against measured Azure latencies

### S11.1 Model

Cross-national round-trip time is built from the great-circle distance d(c, i) between capacity-weighted national centroids (Methods 4.4; Luo & Yang distance model)

$$ L=\frac{d(c,i)}{200\ \mathrm{km\,ms^{-1}}}+20\ \mathrm{ms},\qquad \mathrm{RTT}=L\times 2\times 1.4 \tag{S11} $$

where 200 km/ms is the effective speed of light in fibre (~2/3 c), 20 ms is routing and protocol overhead, the factor 2 is for the round trip and the factor 1.4 for WAN inflation. For the 31-country core sample, measured Microsoft Azure inter-region P50 latencies are used directly (nearest-region mapping); the distance model gives consistent estimates and extends the RTT to all 104 demand countries.

### S11.2 Validation

We validate the distance model against measured Azure inter-region P50 latencies (50 regions), covering every country pair mapped to an Azure region (17 mappable countries, 136 pairs). The agreement is strong, and the model is deliberately conservative — Pearson correlation r (modelled, measured) = 0.865; median absolute error 39 ms; median signed error (modelled − measured) +30 ms, i.e. the distance model overestimates latency; modelled range [65, 322] ms compared with measured [11, 336] ms.

**Table S19. Modelled and measured Azure round-trip times for representative country pairs.**

| Country pair | Modelled RTT (ms) | Measured Azure RTT (ms) | $\Delta$ (ms) |
|---|---:|---:|---:|
| US–GB | 156 | 78 | +78 |
| US–JP | 194 | 164 | +30 |
| US–BR | 155 | 118 | +37 |
| US–IN | 243 | 234 | +8 |
| DE–FR | 66 | 12 | +54 |
| GB–DE | 68 | 17 | +51 |
| JP–KR | 69 | 29 | +40 |
| JP–AU | 151 | 104 | +48 |
| BR–DE | 185 | 195 | −10 |
| AU–US | 263 | 198 | +64 |

The positive bias is largest on the short-distance, intra-regional country pairs that Azure carries over a dedicated backbone (e.g. DE–FR, GB–DE) and smallest on the long-distance intercontinental pairs (US–IN +8 ms, BR–DE −10 ms). Because the model never underestimates latency on the country pairs that matter most for routing, the latency tolerance $\tau$ is harder to satisfy under the model than in reality, so the routing feasibility of the compute load — and the gains from relaxing the latency tolerance — are a conservative lower bound rather than overstated.

---


## Supplementary Section S12 — Country sample, routing reach and cloud-region availability

The analysis sample comprises 104 demand countries that simultaneously have a Cloudflare human-traffic curve and station-level renewable supply, distributed across four regional blocks.

**Table S20. Country sample by geographic block.**

| Geographic block | No. of countries |
|---|---:|
| Americas | 24 |
| Asia | 30 |
| Europe–Africa–Middle East | 47 |
| Oceania | 3 |
| Total | 104 |

Routing reach is set continuously by the latency tolerance $\tau$ through a round-trip-time gate (rather than through nested discrete supply pools) — the receiving countries reachable by sending country c under a latency budget $\tau$ are all receivers with RTT(c, r) ≤ $\tau$ (RTT model in S11), and this set expands monotonically as $\tau$ is relaxed from 50 ms (real-time local only) to 500 ms (cross-hemisphere global reach). The regional blocks (Americas / Europe–Africa–Middle East / Asia / Oceania; see Table S20) are used only to group results for presentation and do not constitute a reachability constraint. Supply is fixed domestically and what is routed is compute load rather than electricity, so HVDC corridors, transmission efficiency and distance ceilings are not involved.

Cloud-region availability imposes a threshold on the routable fraction — a country can receive routed compute only if it hosts at least one large hyperscale cloud provider (AWS / Azure / GCP) region as of 2025; countries without one can act only as sending sources. Within the documented 31-country core, the 20 receivers are AU, BR, CA, CL, DE, FR, GB, ID, IN, IT, JP, KR, MX, MY, NZ, PL, SE, TH, US, ZA (excluded — AR, CO, PE, PY, DZ, GH, LY, MA, NG, TN, UA); across the full 104-country sample about 40 countries meet this condition (i.e. the candidate set for the roughly 39 clean-surplus receiving hubs noted in the main text), spanning North America, Europe, East Asia, the Gulf, Australia–New Zealand, Brazil, Chile, Mexico, India, Southeast Asia, South Africa, Türkiye and Egypt. The complete country-by-country mismatch and grouping table is provided as Table S17.

---

## Supplementary Section S13 — Storage cost conversion and duration-dependent storage economics

### S13.1 Short-duration conversion

The residual gap is converted into firming storage cost through the NREL battery learning curve (Methods 4.5). For a data centre with a 100 MW average load, the annual uncovered energy is $E_c$ = $U_c$ · 8,760 · 100 MWh, and the annualized storage cost is

$$ C^{\mathrm{year}}=\frac{E_c}{\rho}\cdot\frac{\mathrm{capex}^{\mathrm{year}}\cdot 1000\cdot(\mathrm{CRF}+\beta_{\mathrm{OM}})}{365} \tag{S12} $$

where the round-trip efficiency is $\rho$ = 0.85, the capital recovery factor is CRF = r(1+r)ⁿ / ((1+r)ⁿ − 1) (discount rate r = 0.07, lifetime n = 15 yr), and the fixed-O&M fraction is $\beta_{\mathrm{OM}}$ = 0.025. Eq. (S12) is only an illustrative fixed-duration lithium-ion conversion; the firming-storage power and energy capacities in the main text and Figure 5, and their cost, are in fact set endogenously by the hourly storage-dispatch optimisation (Methods), which draws on the Li/LDES power costs, energy costs, round-trip efficiencies (Li 0.86, LDES 0.60) and lifetimes given in this section and S13.2, reported as unit cost per megawatt-hour delivered (USD/MWh).

The mid-case lithium-ion overnight capex (USD/kWh) and its three-case range are as follows.

**Table S21. Lithium-ion overnight capex range (USD/kWh).**

| Year | Conservative (high) | Central (mid) | Optimistic (low) |
|---|---:|---:|---:|
| 2025 | 415 | 332 | 236 |
| 2030 | 382 | 289.5 | 194 |
| 2035 | 349 | 247 | 152 |
| 2050 | 333 | 184 | 111 |

### S13.2 Duration-dependent technology selection

For the forward-looking costs (Figure 5c,d), the choice and capacity of lithium-ion versus long-duration energy storage (LDES) are set jointly by the hourly storage-dispatch optimisation (Methods), rather than fixed at 4 h lithium-ion or sized by gap percentile. To illustrate why the two cross economically with gap duration, the levelized storage cost (LCOS) of serving a segment of power P and energy E, taking the cheaper technology, is

$$ \mathrm{LCOS}=\min_{\mathrm{tech}\in\{\mathrm{Li},\,\mathrm{LDES}\}}\ \frac{(\mathrm{CRF}_{\mathrm{tech}}+\mathrm{FOM})\,(p_{\mathrm{tech}}P+e_{\mathrm{tech}}E)\cdot 10^3}{E^{\mathrm{ann}}\,\eta_{\mathrm{tech}}} \tag{S13} $$

namely short-duration lithium-ion (power cost p ≈ 280 USD/kW, $\eta$ = 0.86, 15 yr) and long-duration LDES (p ≈ 1000 USD/kW, $\eta$ = 0.60, 20 yr, with a much lower energy cost). The per-kWh capacity cost of lithium-ion rises with duration whereas that of LDES is amortized down with duration, the two crossing at ≈ 10 h (consistent with the DOE ≥10 h = long-duration definition); the discount rate is 7 % and FOM is 2.5 %/yr. The 2050 LDES energy cost is 60 / 25 / 12 USD/kWh (conservative / central / optimistic), anchored to Sepulveda et al. (transformational design space \$1–20/kWh), the DOE Long Duration Storage Shot (≥10 h systems reaching \$0.05/kWh·cycle by 2030), and the public targets of Form Energy and the LDES Council. Because this convention assigns long gaps to long-duration storage, the costs in Figure 5c,d are slightly higher than the fixed 4 h lithium-ion daily-cycling value, and the storage-technology range is wider than the AI-shape range—the uncertainty in forward-looking cost is dominated by the storage-technology pathway.

---

## Supplementary Section S14 — Robustness of supply and routing assumptions

Section S7 quantified the AI-workload parameter uncertainty for Result 3; this section reports the supply-side and routing-side robustness underpinning Results 1–2, all directional single-factor checks (Methods 4.6).

- **(a) The routing model contains no transmission assumptions** — the geographic lever moves compute load rather than electricity, riding on the existing internet, with supply fixed domestically (each country still uses its own wind and solar), so it involves no assumptions—HVDC corridor inventories, transmission efficiency or line costs—that would need to be perturbed; the only physical threshold for routing is the network round-trip time (RTT) (see (d)).
- **(b) Wind ×1.2 bias correction ∈ [1.0, 1.4]** — this changes only the level of the wind capacity factor, not its temporal shape; after equal-energy normalization the portfolio result is essentially unchanged.
- **(c) Representative-station capacity coverage 80 % / 91 % / 96 %** — the median uncovered share in the global-reach tier varies by less than 0.5 pp.
- **(d) RTT-model parameters (200 km/ms, 20 ms) perturbed by ±20 %** — cross-continental RTT estimates vary within about 100–300 ms, mainly affecting whether the 200 ms tier can be achieved across hemispheres; the effect on the uncovered share of the global tier under full routing is below 0.5 pp, and the conclusion that latency benefits saturate at a few hundred milliseconds is unchanged.
- **(e) The routable fraction $\varphi$ is one axis directly swept in the main results** — $\varphi$ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} is given tier by tier as a primary axis in main-text Fig. 2a and Fig. 4d and is not a fixed assumption; its near-linear effect on gap and cost is itself the main result (under global reach, the gap falls from about 35% at 20% routing to 12.7% at full routability).
- **(f) Geopolitical feasibility (boundary clause)** — this framework imposes no political exclusion matrix. The dominant receiving hubs are mostly Western or neutral economies (Mexico, Chile, Australia, Japan, South Africa, Spain, United States, Brazil), the share from any single restricted source is small, and political constraints are expected to have a mild effect on the ordering of the routing lever; a formal political-feasibility constraint is left to future work (see the Result 2 limitations in the main text).


## Supplementary Section S15 — Boundaries of deliverability and statistical convention

The routing grid in Result 2 of the main text performs timing (waveform) matching — under equal-energy normalisation it asks only whether the shape of reachable partners' clean generation can cover each country's shortfall in time, magnitude being divided out at every step, so no single share of surplus is reused across deficit countries. The resulting median residual of 12.7% under full routability is a timing lower bound on phase alignability. Turning this timing potential into delivered clean power further requires the absolute quantity of clean surplus, receiving-side compute capacity, trans-oceanic bandwidth and electricity-market clearing—an absolute-quantity physical settlement that must be modelled separately once each country's absolute capacity and cross-border transmission are connected, and that lies outside the scope of this study. Its direction is unambiguous — imposing any deliverability constraint can only raise, not lower, the residual, so the main-text figure is a timing lower bound rather than an attainable delivered value.

A separate note on statistical convention — all medians in the main text weight countries equally. Weighting instead by each country's absolute installed capacity (solar PV and wind MW), the weighted mean of the domestic uncovered share—on the same simple equal-energy-mix basis as the 40.1% headline median—is about 27%; the median and the equal-weighted mean are themselves almost identical (39.7%), so this drop comes entirely from weighting rather than any median-vs-mean effect. Large-capacity countries (such as the United States at 16.8%, and Germany, the United Kingdom and Spain) both have lower mismatch and a large absolute surplus, so the global burden under the energy-weighted convention is smaller than the median country suggests (its policy implication is discussed in the main-text limitations). Reproduction code is provided together with the public repository (see the Code Availability statement).

### S15.1 Directional ledger for the 12.7% residual

The 12.7% residual is the answer to a strictly defined problem—interactive load, same-hour execution, wind-and-solar supply, phase only—and not an unconditional one-sided statistical bound. The simplifying assumptions fall into three classes — **scope definitions** (which fix the question being asked, so relaxing them turns to a different question), **in-scope un-modelled factors** (which, within the question asked, bias the realizable residual away from 12.7% in one consistent direction—upward), and **modelling approximations** (which have a direction and cut both ways). Table S22 lists them item by item.

**Table S22. Directional ledger for the 12.7% residual (φ = 1, τ = 500 ms, waveform basis).**

| Simplifying assumption | Class | Direction on 12.7% | Magnitude / evidence |
|---|---|---|---|
| Same-hour execution (interactive load cannot defer across hours) | Scope definition | Defines the load class; deferrable training/batch is a different, lower-residual problem | Methods |
| Supply is domestic wind-and-solar only, excluding hydro/nuclear/geothermal firm clean | Scope definition | Isolates wind-solar geometry; adding firm clean would lower the residual but changes the supply question | — |
| Quantity/deliverability placed out of scope (phase only) | Scope definition | Asks about timing alignment, not power delivery | This section |
| Cross-country surplus contention, receiving-side absolute capacity and trans-oceanic bandwidth | In-scope, un-modelled | Raises the realizable residual | Direction is unambiguous (can only raise it); magnitude depends on the allocation caliber and capacity assumptions (see analyze_cfe_r2_allocation_sensitivity.py) |
| Demand shaped by request count, not weighted by per-request compute | In-scope, un-modelled | Raises the realizable residual | κ = +1 gives 12.7% → 15.5% (Table S25) |
| Only countries hosting a hyperscale cloud region may receive | In-scope, un-modelled | Raises the realizable residual (fewer receivers) | — |
| Round-trip time via the P50 distance model | Modelling approximation | Conservative on the low side (routability understated) | Median overestimate of +30 ms on long-distance pairs (Table S19) |
| Tail latency (P95/P99) and session-state migration un-modelled | Modelling approximation | Opposes the row above; not quantified | — |

**Conclusion.** 12.7% is not an unconditional one-sided bound. Within the strictly defined problem, the leading un-modelled factors (contention, compute weighting, cloud-region gating) all push the realizable residual upward, so 12.7% is an optimistic lower bound within scope; the factors that would lower it lie in relaxing the scope (allowing cross-hour deferral, adding firm clean supply) and answer a different load class or supply mix rather than tightening a bound on this residual.


## Supplementary Section S16 — Robustness of full electricity-supply cost to reliability target, annual cycling and storage-cost scenario

The full electricity-supply cost in the main text (domestic generation, firming storage and ancillary services, excluding transmission) is determined by hourly storage-dispatch optimisation — for each country a linear program co-optimises the generation overbuild and the power and energy capacities of lithium-ion (Li) and long-duration energy storage (LDES) and their hourly charge/discharge, satisfying the hourly energy balance under round-trip efficiency, state-of-charge (SOC) recursion, annual cycling and curtailment, with unserved load priced at the value of lost load (VOLL) (Methods). Firming-storage capacity is endogenous to this optimisation, so there is no free sizing-percentile parameter—the earlier heuristic that sized storage to the 95th percentile of contiguous gap segments, and its P90–P100 sensitivity, are superseded by this dispatch model. This section tests the robustness of the cost to three key choices of the dispatch model.

**Reliability target and annual cycling.** Under the 2030 mixed-AI shape, relaxing the energy-reliability target from 100% to 99.9% and 99%, and toggling the annual SOC-cycling constraint, leaves the full electricity-supply cost of representative countries essentially unchanged (Table S23, differences < 1%) — at the standard value VOLL = 10,000 USD/MWh, building extra storage to cover the tail gaps remains cheaper than shedding load, so the model serves ≈100% throughout the 99%–100% range; nor is the cost driven by cross-seasonal storage. The headline therefore does not depend on the precise reliability threshold or cycling constraint.

**Table S23. Robustness of the full electricity-supply cost to the reliability target and annual SOC cycling (representative countries, 2030 local tier, USD/MWh).**

| Country | 100% · cyclic | 99.9% · cyclic | 99% · cyclic | 99.9% · non-cyclic |
|---|---:|---:|---:|---:|
| United States | 112 | 112 | 112 | 112 |
| Germany | 142 | 142 | 142 | 142 |
| Palestine | 483 | 483 | 483 | 483 |

**Storage-cost scenario.** The year-by-year storage capex follows conservative / central / optimistic scenarios (S13, spanning roughly ±50% in Li and LDES energy cost); storage is the component that drives the cost's variation with reach, so the cost scenario is the dominant uncertainty band, yet the cost advantage of the global tier over the local tier holds consistently across the three scenarios. Reproduction code is provided together with the public repository (see the Code Availability statement).


## Supplementary Section S17 — Scope of the demand proxy, compute-intensity weighting, and robustness to the flat-baseload share

### S17.1 Scope

The demand curves in this study are taken from country-level human web traffic in Cloudflare Radar, which measures human digital demand—that is, user-facing, network-latency-constrained interactive/inference service load that can be served from a remote location. Flat training and batch baseload (model training, cache and CDN hits) has no diurnal rhythm and is carried separately by the flat training share $\lambda_{\mathrm{train}}$ in the AI workload operator (Methods 4.5), so it does not enter the diurnal shape.

### S17.2 Real-trace corroboration of the shape

The diurnal shape (evening peak) of the human-traffic proxy agrees independently with two real large-language-model inference service loads — BurstGPT v2 (Supplementary Section S2) and the Azure LLM Inference Trace 2024 (44 million requests, Supplementary Section S4); the diurnal power of both, converted from request rate, exhibits an evening peak. For the interactive/inference service tier that is of true interest here, the phase of the proxy is therefore corroborated by real service loads rather than assumed.

### S17.3 Robustness to the flat-baseload share

To delineate what the headline figures would be if real data centre load were flatter than human traffic, we mix each country's demand with an entropy-preserving flat component, $d_\beta = (1-\beta)\,d_{\mathrm{human}} + \beta\,\langle d_{\mathrm{human}}\rangle$, where $\beta$ is the flat-baseload share, and recompute the Result 1 country-level equal-energy mismatch and the global routing residual ($\varphi$ = 1, $\tau$ = 500 ms, waveform). Table S24 reports the median across 104 countries.

Key point — flattening the load does not reduce the mismatch but slightly raises it (Result 1 rises from 40.1% to 41.5%, and routing residual falls marginally from 12.7% to 12.0%). The reason is that the mismatch is constrained by the roughly half-day window in which solar PV output is zero at night, not by the shape of the evening demand peak; flattening the load merely shifts demand from the evening peak to the equally sunless deep night, leaving the total uncovered share almost unchanged. The headline figures are therefore robust to whether real data centre load is flatter than human traffic and conservative with respect to admixing flat baseload.

**Table S24. Robustness of the headline metrics to the flat-baseload share $\beta$ (median across 104 countries, %).**

| Flat-baseload share $\beta$ | Result 1 country mismatch | Global routing residual ($\varphi$=1, $\tau$=500) |
|---|---:|---:|
| 0 (pure human traffic) | 40.1 | 12.7 |
| 0.25 | 40.8 | 12.1 |
| 0.50 | 41.5 | 12.0 |

### S17.4 Robustness to compute-intensity weighting

The request-to-electricity mapping above assumes that the diurnal composition of request types is broadly stationary. If that assumption were broken—if high-compute-intensity requests (large-language-model conversation, code, and image/video generation) were more concentrated in the interactive evening peak than low-compute-intensity requests (search, static pages, and CDN hits)—then reweighting by per-request energy would sharpen the evening demand peak; the opposite composition would flatten it. We characterise this directly with a single compute-intensity concentration exponent $\kappa$ — modelling per-request energy as $e(t)\propto (R(t)/\langle R\rangle)^{\kappa}$, the energy-weighted demand shape is $d_\kappa(t)\propto R(t)^{1+\kappa}$ (rescaled to each country's original annual total). $\kappa=0$ is the request-count headline; $\kappa>0$ concentrates high-compute requests at the evening peak (sharper); $\kappa<0$ is low-compute/CDN-dominated (flatter). For each $\kappa$ we recompute the national equal-energy mismatch (R1) and the global routing residual (R2; $\varphi=1$, $\tau=500$ ms, waveform match); Table S25 gives the 104-country median.

The national mismatch is robust to compute intensity—R1 stays within 40.1–41.5% across $\kappa\in[-0.5,+1.0]$—while the routing residual rises monotonically with compute intensity (12.0% → 12.7% → 15.5%), because pushing demand further into the solar-absent evening only lowers phase-alignability. Hence if the true load is more compute-heavy than the raw request count, the headline mismatch is essentially unchanged and the routing residual can only be higher; the main-text figure (12.7%) is therefore on the conservative (low) side of the residual, and the conclusion is robust—indeed conservative—to building the demand shape from request counts rather than from compute.

**Table S25. Robustness of the headline metrics to the compute-intensity concentration exponent $\kappa$ (104-country median, %).**

| Compute intensity $\kappa$ | R1 national mismatch | Routing residual ($\varphi=1$, $\tau=500$) |
|---|---:|---:|
| −0.50 (low-compute / flatter) | 41.2 | 12.0 |
| −0.25 | 40.6 | 12.2 |
| 0.00 (request count, headline) | 40.1 | 12.7 |
| +0.50 | 40.4 | 14.0 |
| +1.00 (high-compute / sharper) | 41.5 | 15.5 |

Reproduction code (analyze_cfe_s17_computeintensity.py) is provided together with the public repository (see the Code Availability statement).

Reproduction code is provided together with the public repository (see the Code Availability statement).

---

## Supplementary Section S18 — Scenario anchoring of the latency tolerance τ — measured service-level targets of real workloads and the compute-to-network-budget conversion

### S18.1 The conversion

The main text and Sections S11 and S12 define the latency tolerance τ as the **network round-trip time** a single request can tolerate, which gates the reachable receiving countries (τ = 50 ms local only; 100–150 ms reaches about 3,000–6,700 km; 200–300 ms near-global; ≥ 500 ms global). Which tier a real workload falls in is not set directly by its **end-to-end** service-level target (SLA), but by the round-trip budget a single request actually leaves for the network once on-server compute/generation is subtracted

$$ \text{network budget} = \text{end-to-end SLA} - \text{on-server compute time}. \tag{S14} $$

The conversion polarizes workloads into two ends. Voice assistants, inline code completion and real-time retrieval-augmented generation already have end-to-end SLAs within one second, and their compute (the ASR + LLM + TTS pipeline, model serving, retrieval + prefill) already consumes most of it, leaving only tens of milliseconds of network budget after subtraction—too little to cross a border, so the load is absorbed on-site. Conversely, image and video generation and batch inference take seconds to tens of minutes or even 24 hours of compute per request, far exceeding the ceiling of inter-continental round-trip time (about 340 ms; see S11), so their network budget is effectively unbounded and they can be routed to any longitude. Whether a load can be routed across longitudes is therefore set chiefly by whether it can be served off-site (the routable share φ) rather than by network latency—consistent with the main-text finding that what limits the gap's descent is not network latency but how much load can be served off-site. Table S26 anchors each τ tier to concrete workloads using published measured service-level targets and compute times.

**Table S26. Measured service-level targets, compute times and assigned latency-tolerance tiers of real workloads.**

| Workload scenario | Measured end-to-end SLA | Compute/generation time | Network round-trip budget | τ tier | Source |
|---|---|---|---|---|---|
| Inline code completion (Google internal IDE) | < 100 ms end-to-end | ~40 ms model serving | ~60 ms | 50 ms local | [S18-7] |
| Code completion (GitHub Copilot production) | < 200 ms mean | not itemized | < 200 ms (unsplit) | 50 ms local | [S18-6] |
| Real-time voice assistant (measured pipeline) | < 1 s (acceptable threshold) | 0.94 s (ASR 0.05 + LLM TTFT 0.11 + generation 0.67 + TTS 0.29) | ~60 ms | 50 ms local | [S18-9]; turn-taking [S18-10] |
| Real-time retrieval-augmented generation (low-latency schedule) | TTFT ~0.03 s | retrieval > 80% | ~0 | 50 ms local | [S18-5] |
| Interactive chat (tight SLO) | TTFT 0.25 s / TPOT 0.1 s | prefill ~0.1 s | ~150 ms | 100–150 ms regional | [S18-1], [S18-2] |
| Web search (tolerated added delay) | +200–400 ms → −0.29 to −0.59% daily searches | not a compute term | ~200–400 ms (engagement cost) | 100–150 ms regional | [S18-11], [S18-12] |
| Interactive chat (loose QoE target) | TTFT target 1.3 s | sub-second prefill | ~1 s | 200–300 ms near-global | [S18-3] |
| Throughput-optimized retrieval-augmented generation | TTFT ~2.47 s | ~1.54 s | ~0.9 s | 200–300 ms near-global | [S18-5] |
| Server-side image generation (accelerated diffusion, H100, 1024²) | tolerance ~1 s | 0.1–1.1 s | ~0.9 s to unbounded | 200–300 ms near-global | [S18-13], [S18-12] |
| On-device / slow image generation | tolerance ~10 s | up to 12 s | unbounded | 500 ms+ cross-hemisphere global reach | [S18-14], [S18-12] |
| Video generation | no interactive SLA | 32 s–~30 min | unbounded | 500 ms+ cross-hemisphere global reach | [S18-15], [S18-16] |
| Batch inference (up to 55% of serving capacity is offline) | 24 h | seconds, negligible | unbounded | 500 ms+ cross-hemisphere global reach | [S18-17], [S18-18], [S18-19] |

Measured inter-continental round-trip times as a reachability reference — New York–London about 59–80 ms; Los Angeles–Tokyo about 100–115 ms; United States–Australia about 140–200 ms; Tokyo–London about 226 ms; the global antipodal ceiling is about 340 ms (a long-haul-link threshold of 57 ms per 5,700 km at 2/3 c propagation, [S18-20]; modelled values in S11, Table S19).

### S18.2 Strength and reasoning notes

Most cells of Table S26 are given directly by public sources, but the following are reasoned rather than directly quoted and are flagged conservatively — (i) the ~0.1 s prefill compute for the interactive-chat tight SLO tier is derived by dividing 512 tokens by the ~5,000 tokens/s prefill throughput (the two figures come separately from [S18-1] and [S18-3]), and the prefill throughput is a hardware-specific measured value, not a universal constant; (ii) GitHub Copilot reports only a < 200 ms mean end-to-end figure without splitting compute from network, so its local-tier assignment is reasoned; (iii) the 200–400 ms for web search is a tolerated **added** delay / engagement cost, not a compute time or a hard SLA, so mapping it to the regional tier is an interpretation; (iv) the unbounded network budgets for image/video generation and batch inference are qualitative judgements (per-request compute far exceeds the ~340 ms inter-continental round-trip ceiling), not exact subtractions; (v) every τ-tier assignment applies the reach thresholds of the main text and Sections S11–S12 (50 / 100–150 / 200–300 / ≥ 500 ms) rather than re-deriving them from the cited latency sources. These flags do not change the directional two-ended conclusion.

### S18.3 Sources

[S18-1] DistServe: disaggregating prefill and decoding for goodput-optimized large language model serving. OSDI 2024. arXiv:2401.09670. (OPT-13B/ShareGPT: TTFT SLO 0.25 s, TPOT SLO 0.1 s; a 512-token prefill makes an A100 near compute-bound.)
[S18-2] Sarathi-Serve: taming the throughput-latency tradeoff in LLM inference. OSDI 2024. arXiv:2403.02310. (Strict P99 TBT: Mistral-7B 0.1 s, Yi-34B 0.2 s, LLaMA2-70B/Falcon-180B 1.0 s.)
[S18-3] Andes: defining and enhancing quality-of-experience in LLM-based text streaming services. arXiv:2404.16283. (TTFT target 1.3 s, per Google web-page-loading guidance; reading/listening 4.8/3.3 tokens/s; prefill throughput ~5,000 tokens/s.)
[S18-5] RAGO: systematic performance optimization for retrieval-augmented generation serving. ISCA 2025. arXiv:2503.14649. (Min-TTFT 0.03 s, Max-QPS 2.47 s (baseline 1.54 s); hyperscale retrieval > 80% of latency.)
[S18-6] How GitHub Copilot serves completions (InfoQ presentation, infoq.com/presentations/github-copilot). (< 200 ms mean response; > 400 million completion requests/day.)
[S18-7] ML-enhanced code completion improves developer productivity. Google Research blog (2022). (< 100 ms end-to-end; ~40 ms median model serving.)
[S18-8] Full line code completion: bringing AI to desktop IDEs (JetBrains FLCC). arXiv:2405.08704. (On-device 150 ms mean time-to-show.)
[S18-9] Voice agent architecture. LiveKit docs (end-to-end < 1 s; speech-to-speech < 500 ms); measured pipeline arXiv:2508.04721 (total 0.94 s with stage breakdown; < 1 s acceptable threshold).
[S18-10] Universals and cultural variation in turn-taking in conversation. PNAS 106, 10587–10592 (2009). doi:10.1073/pnas.0903616106. (Turn-taking overall mode 0 ms, mean offset +208 ms.)
[S18-11] Speed matters for Google web search. Google (2009). (Added 200 ms → −0.29%, 400 ms → −0.59% daily searches/user.)
[S18-12] Response time limits. Nielsen Norman Group. (0.1 / 1 / 10 s usability limits.)
[S18-13] SANA-Sprint: one-step diffusion with continuous-time consistency distillation. arXiv:2503.09641. (H100, 1024² image 0.1 s vs FLUX-schnell 1.1 s.)
[S18-14] Speed is all you need: on-device acceleration of large diffusion models. arXiv:2304.11267. (Samsung S23 Ultra, 512², 20 steps, < 12 s.)
[S18-15] AdaCache: adaptive caching for faster video generation with diffusion transformers. arXiv:2411.02397. (Single A100: Open-Sora 720p-2s 419.6 s, Latte 32.45 s.)
[S18-16] On-device Sora. arXiv:2502.04363. (Server denoising < 1 min; unoptimized on-device about 29.5 min.)
[S18-17] Batch API guide. OpenAI (developers.openai.com/api/docs/guides/batch). (24 h completion window; 50% discount.)
[S18-18] Message Batches API. Anthropic (claude.com/blog/message-batches-api). (24 h; 50% discount.)
[S18-19] EcoServe. arXiv:2502.05043. (Offline inference 24 h SLO; offline up to 55% of serving capacity, average 21%/45%.)
[S18-20] Inter-continental round-trip-time measurements: EXA Express (New York–London 58.95 ms), Southern Cross (Australia–US West Coast 140.44 ms), GeoCables/RIPE Atlas (New York–London 70–80 ms, Los Angeles–Tokyo 100–115 ms, New York–Sydney 170–200 ms), WonderNetwork (Tokyo–London about 226 ms); long-haul-link threshold 5,700 km ↔ 57 ms (2/3 c), arXiv:2303.02514.

---

## Supplementary Section S19 — Positioning relative to prior work

The table below compares this study with the closest prior work along six dimensions, highlighting our increment — for the first time it brings together globally, in local phase, human digital activity, station-level wind and solar output and measured network latency, and quantifies the phase-alignability (the geometric residual) of same-hour cross-longitude migration.

**Table S27. Dimensional comparison of this study with the closest prior work.**

| Study | Actual network-latency gating | Global country coverage (104 countries) | Local-phase human demand curve | Station-level wind/solar supply | Routable-fraction axis | Phase-alignability / geometric residual |
|---|---|---|---|---|---|---|
| Carbon-aware temporal scheduling (ref3,4) | — | — | — | — | — (time axis only) | — |
| Single-operator spatiotemporal migration (ref5) | — | — | — | partial | — | — |
| Latency–geography framework (ref6, Luo & Yang) | stylized (RTT distance) | — | — | — | partial | — |
| Global wind–solar interconnection (ref22, Jiang; moves electricity, not compute) | not applicable | ✓ | — | partial | — | — |
| This study | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Supplementary Section S20 — Independent external corroboration (E3)

Two findings of an independent industry analysis of data-centre load growth (ref32, E3) support, from outside our framework, the two pillars of this study — the diurnal shape of interactive demand, and the firm-capacity residual that renewables leave behind.

### S20.1 Diurnal-shape corroboration

E3 independently anticipates the demand-shape premise of our study. Its analysis notes that data centres have historically been high-load-factor, near-flat facilities dominated by baseload and training, but that as user-facing utilisation (inference) overtakes training, the daily load curve comes to be shaped by human use — a business-oriented tool peaks in working hours, whereas a personal tool develops evening peaks resembling residential load. Figure S5 places E3's modelled scenarios alongside our empirical interactive curve. The direction agrees — E3's personal-use scenario peaks in the evening, coincident with our interactive-demand peak — while the amplitude contrast makes the case for isolating the interactive slice — E3's total-load curve stays within about ±5% of flat because baseload dilutes the signal, whereas our interactive-only curve (the Cloudflare median) swings from about 0.3 to 1.4 times its daily mean. The comparison is qualitative — E3's daily curves are modelled illustrations of total U.S. data-centre load, not measured data, so they corroborate the direction of our premise and the rationale for the flat-baseload separation (S17.3) rather than validating our hourly curve, for which the empirical anchors are the BurstGPT and Azure traces (S2, S4).

![](figures_globalsites/figS_e3_datacenter_shape.png)

**Figure S5. Independent, qualitative corroboration of the diurnal-shape premise (E3).** E3's modelled daily shapes of total U.S. data-centre load (digitised from Figure 5 of ref32; each curve integrates to one over 24 h) are compared with our empirical 104-country median interactive-demand share by local hour, all on a daily-mean = 1 basis. **(a)** E3's two usage scenarios — a business tool peaks in the afternoon (about 14 h), a personal tool in the evening (about 19 h, coincident with our interactive peak at 20 h, dashed line); the y-axis is zoomed because total-load variation is small. **(b)** Amplitude contrast on a common axis — E3's total-load curve stays within about ±5% of flat because baseload and training dilute the interactive signal, whereas our interactive slice alone swings from about 0.3 to 1.4 times the daily mean. The comparison is directional (E3's curves are modelled illustrations of total load, not measured hourly data), supporting the shape premise and the flat-baseload separation, not reproducing our curve.

### S20.2 The firming residual

An independent grid-planning assessment of U.S. data-centre load growth (ref32, E3) reaches the same conclusion as our full-system cost result (Result 3) from a different starting point and with a different method. Using an effective-load-carrying-capability (ELCC) framework, that study finds that meeting 100% of the incremental data-centre energy demand with renewables would require about 115 GW of nameplate wind and solar by 2030, yet these deliver only about 23 GW of effective (reliability) capacity — roughly one-fifth of nameplate — leaving a residual firm-capacity gap of about 16 GW that must still be met by storage or other firm resources; the effective contribution of solar paired with short-duration storage itself declines over the same horizon (its ELCC falling from about 0.5 in 2024 to about 0.35 in 2030).

The implication is the same as ours — building enough renewables to supply all of the energy does not close the timing gap. A substantial firming residual persists because renewable output does not arrive at the hours of peak need — the mismatch is temporal, not volumetric. This is precisely the mechanism behind our finding that cross-longitude routing is cheaper almost entirely through the firming storage it avoids rather than through cheaper generation, and behind the paper's central claim that capacity expansion inside a single grid cannot close a phase-set mismatch. Here that claim is corroborated by an independent, U.S. grid-planning ELCC calculation rather than by our hourly-alignment framework.

The comparison is directional rather than a reproduction of our figures — E3 is U.S.-specific and reports a capacity credit (reliability contribution at peak) rather than our hourly uncovered-energy share, and it addresses total data-centre load rather than the interactive slice. It therefore supports the mechanism and the direction of our cost result, not our specific residual percentages.

---

## Supplementary Section S21 — Identities linking the wind-and-solar matching gap U to established metrics

The main text uses the hourly wind-and-solar matching gap \(U_c\) (Eq. 1) as its matching metric. This section proves that, under equal-energy normalisation, \(U_c\) is mathematically identical to four established metrics drawn respectively from corporate procurement, distributed energy, net-zero-energy buildings and statistics, so the phase-alignability floor reported here can be cross-checked against those established conventions rather than being a bespoke measure.

Let a country's hourly demand \(d(t)\ge 0\) and clean supply \(s(t)\ge 0\) each be normalised to the same annual total \(\sum_t d=\sum_t s=E\), and write \(U=\sum_t\max(d-s,0)/E\).

**Proposition 1 (self-sufficiency).** \(U=1-\mathrm{SS}\), where the temporal self-sufficiency rate \(\mathrm{SS}=\sum_t\min(d,s)/\sum_t d\) (Luthander et al. 2015). *Proof.* Pointwise \(d=\min(d,s)+\max(d-s,0)\); summing over \(t\) and dividing by \(E\) gives \(1=\sum_t\min(d,s)/E+U\). This step needs only \(\sum_t d=E\). ∎

**Proposition 2 (24/7 CFE matching score).** \(1-U\) equals, hour by hour, the 24/7 carbon-free energy matching score \(\mathrm{CFE}=\sum_t\min(d,s)/\sum_t d\) (Google 2020; Riepin & Brown 2024). *Proof.* Google's convention caps single-hour carbon-free supply at that hour's demand and does not carry surplus forward — mathematically \(\min(d,s)\); by Proposition 1, \(\mathrm{CFE}=\sum_t\min(d,s)/E=1-U\). ∎

**Proposition 3 (load match index).** \(1-U\) equals the hourly load match index of net-zero-energy buildings, \(f_{\mathrm{load}}=\sum_t\min(d,s)/\sum_t d\) (Sartori et al. 2012). With the annual energy balanced (their net-zero line, i.e. our \(\sum_t d=\sum_t s\)), that literature reads the hourly load match index as pure day–night phase mismatch, isomorphic to our convention.

**Proposition 4 (total variation / overlap).** Let \(p=d/E,\ q=s/E\) be two discrete probability distributions. Then \(U=\tfrac12\sum_t|d-s|/E=\mathrm{TV}(p,q)\) and \(1-U=\sum_t\min(p,q)=\mathrm{OVL}(p,q)\), so \(U\) is the total-variation distance between the two normalised shape distributions and \(1-U\) their overlap coefficient (Weitzman 1970). *Proof.* Write \(M_d=\sum_t\max(d-s,0)\), \(M_s=\sum_t\max(s-d,0)\). From \(\sum_t d=\sum_t s=E\), \(M_d=M_s\); since \(|d-s|=\max(d-s,0)+\max(s-d,0)\), \(\tfrac12\sum_t|d-s|=\tfrac12(M_d+M_s)=M_d=UE\). ∎

**Equal energy is indispensable.** Propositions 2 and 4 rely on \(M_d=M_s\) (i.e. \(\sum d=\sum s\)). Dropping equal-energy normalisation gives a counterexample — for \(d=(2,0),\ s=(0,0)\), \(U=\sum\max(d-s,0)/\sum d=2/2=1\), whereas \(\tfrac12\sum|d-s|/\sum d=(2+0)/(2\cdot2)=0.5\neq U\). This shows that our equal-energy normalisation is precisely what removes the volume shortfall dimension so that \(U\) measures phase mismatch alone; the un-normalised real CFE score also carries a structural energy shortfall.

Under equal-energy normalisation, \(U=1-\mathrm{SS}=1-\mathrm{CFE}=1-f_{\mathrm{load}}=\mathrm{TV}=1-\mathrm{OVL}\) forms a closed chain of equivalences (Table S28). The original definition of \(U_c\) is in S8, its optimisation forms (NNLS and L1-LP) in S10, and the phase-alignability context of the 12.7% residual in S15.

**Table S28. Metric definitions and equivalences.**

| Metric | Definition | Relation to U | Source |
|---|---|---|---|
| 24/7 CFE matching score | Σ min(d,s)/Σ d (hourly-capped) | 1−U | Google 2020; Riepin & Brown 2024 |
| Temporal self-sufficiency SS | Σ min(d,s)/Σ d | 1−U | Luthander et al. 2015 |
| Load match index f_load | Σ min(d,s)/Σ d (annual-balanced) | 1−U | Sartori et al. 2012 |
| Overlap coefficient OVL | Σ min(p,q) | 1−U | Weitzman 1970 |
| Total-variation distance TV | \(\tfrac12\sum_t|p-q|\) | U | — |

The script reproducing these identities and the counterexample is provided with the public repository (see Code Availability).

---

## Supplementary Section S22 — Converting the phase-firming premium Π to absolute levelized cost

The cost side of the main text uses the dimensionless phase-firming premium \(\Pi_c=(\mathrm{LCOE}^{\mathrm{firm}}-\mathrm{LCOE}^{\mathrm{gen}})/\mathrm{LCOE}^{\mathrm{gen}}\) (Eq. 5) as its metric. This section gives the year-by-year conversion between \(\Pi\) and its absolute basis — the full electricity-supply cost per MWh of digital demand (generation + firming storage + ancillary services) — so readers can recover the absolute figures and cross-check them. The absolute cost is determined by hourly storage-dispatch optimisation (firming-storage power and energy capacities set endogenously, not percentile-sized; see S13 and Methods), with anchors — 2030 global firm ≈ 184, on-site ≈ 220; storage component ≈ 60 / 77 USD/MWh. Table S29 gives the 104-country year-by-year medians under the expected mixed-AI shape for the on-site (τ = 50 ms, φ = 0) and global-reach (τ = 500 ms, φ = 1) tiers (USD/MWh; Π and σ in per cent).

**Table S29. Year-by-year conversion between the phase-firming premium Π and absolute levelized cost (expected mixed-AI shape; hourly storage-dispatch basis).**

| Tier | Year | Generation | Firming storage | Ancillary | Firm total | Π | σ |
|---|---|---|---|---|---|---|---|
| On-site (τ = 50 ms, φ = 0) | 2025 | 170 | 91 | 6 | 265 | 53% | 35% |
| | 2030 | 135 | 77 | 6 | 220 | 61% | 38% |
| | 2035 | 122 | 73 | 6 | 202 | 63% | 39% |
| | 2040 | 111 | 68 | 6 | 184 | 66% | 40% |
| | 2045 | 102 | 62 | 6 | 170 | 66% | 40% |
| | 2050 | 92 | 56 | 6 | 156 | 68% | 40% |
| Global reach (τ = 500 ms, φ = 1) | 2025 | 144 | 68 | 6 | 220 | 47% | 32% |
| | 2030 | 119 | 60 | 5 | 184 | 55% | 36% |
| | 2035 | 107 | 56 | 5 | 168 | 57% | 36% |
| | 2040 | 96 | 52 | 5 | 152 | 59% | 37% |
| | 2045 | 89 | 48 | 5 | 141 | 59% | 37% |
| | 2050 | 82 | 44 | 5 | 129 | 62% | 38% |

**Trend direction — absolute cost falls, relative premium rises.** Table S29 exposes a fact revealed by normalisation and hidden by the absolute convention — the firm full-system cost falls year by year as generation and storage cheapen (global 220→129, on-site 265→156), yet the phase-firming premium Π rises year by year (global 47%→62%, on-site 53%→68%). The mechanism is in the denominator — the wind-and-solar generation built (with firming overbuild) falls faster over 2025–2050 (global 144→82 USD/MWh), whereas the firming storage that fills the phase mismatch declines far more slowly (global 68→44), so the weight of storage relative to generation — i.e. Π — rises rather than falls. In other words, as clean generation grows ever cheaper, this mismatch tax takes an ever-larger share of the clean electricity price, consistent with the main text's finding that the phase constraint does not recede with technological progress but grows in relative economic weight. \(\sigma=\Pi/(1+\Pi)\) gives the equivalent reading on a common [0,1) scale. Year-by-year Π is the country-level median of (firm−gen)/gen, consistent with Fig. 5c. Firming-storage capacity is set endogenously by the hourly dispatch optimisation (not percentile sizing), so there is no P90–P100 sizing-percentile sensitivity; the cost is robust to the reliability target (99%–100%) and to annual SOC cycling.

The script reproducing this table is provided with the public repository (see Code Availability).

## Supplementary Section S23 — Literature basis and derivation of the two dimensionless measures

This section reviews the design motivation, definitions, derivation and literature lineage of the pair of dimensionless metrics carried throughout the paper—the hourly wind-and-solar matching gap \(U\) and the phase-firming premium \(\Pi\)—showing that they are not devised by this study but inherit established conventions from corporate procurement, energy systems, buildings and statistics. The formal proofs of the identities between \(U\) and established metrics, together with the de-normalisation counterexample, are given in S21 (Table S28); the year-by-year conversion between \(\Pi\) and absolute levelized cost is given in S22 (Table S29). This section does not repeat those derivations but sets out the sources and the overall logic.

**Design motivation and shared premise.** The ideal measure would account, country by country, for the absolute electricity use of digital load and its carbon-free supply gap. But user-facing interactive and inference load is spread across the globe; its true draw is not metered country by country and hour by hour, and the absolute energy of compute can only be triangulated within wide bounds (see S1), so the complete absolute demand curve cannot be obtained at global scale. What can be robustly observed is waveform rather than magnitude — the intraday rhythm of human digital activity is captured by Cloudflare network request traffic (see S17), and that of clean output by station-level capacity factors (see S9). We therefore normalise each country's demand \(d_c(t)\) and supply \(s_c(t)\) curves to the same annual total \(\sum_t d_c=\sum_t s_c=E_c\), stripping out the unobtainable absolute magnitude and retaining only the phase relation of the two waveforms. This normalisation turns the intrinsic data limitation into a methodological demarcation — every metric measures only temporal phase mismatch and admits no structural energy shortfall; the counterexample showing this premise cannot be dropped is given in S21.

**Definition and lineage of the wind-and-solar matching gap \(U\).** The hourly wind-and-solar matching gap is defined as the share of demand exceeding contemporaneous supply under equal-energy normalisation,

\[ U_c = \frac{\sum_t \max\big(d_c(t)-s_c(t),\,0\big)}{\sum_t d_c(t)} \in [0,1]. \]

Its authoritative anchor is the 24/7 carbon-free energy (CFE) standard — its complement \(1-U_c\) equals hour by hour the 24/7 CFE matching score—the share of annual demand met by carbon-free supply capped at each hour's demand with surplus not carried forward. This standard was introduced by Google's 24/7 CFE initiative (Google 2020) and established through system-level assessment (Riepin & Brown 2024), and is now in wide use across corporate clean-power procurement and policy. Under the equal-energy premise, the same \(1-U_c\) also equals hour by hour the temporal self-sufficiency rate in the distributed-energy literature (Luthander et al. 2015) and the load match index of net-zero-energy buildings (Sartori et al. 2012), while \(U_c\) itself equals the total-variation distance between the two normalised shape distributions and its complement the overlap coefficient in statistics (Weitzman 1970). The four identities form a closed chain \(U=1-\mathrm{SS}=1-\mathrm{CFE}=1-f_{\mathrm{load}}=\mathrm{TV}=1-\mathrm{OVL}\), letting the same value be cross-checked by these four established conventions; the term-by-term proofs of Propositions 1–4, the closed chain and the de-normalisation counterexample are given in S21.

**Definition and lineage of the phase-firming premium \(\Pi\).** To quantify the firming cost of the same residual phase mismatch, the phase-firming premium is defined as the relative increment, over pure generation cost, of firming the residual phase gap with domestic storage,

\[ \Pi_c = \frac{\mathrm{LCOE}^{\mathrm{firm}}_c - \mathrm{LCOE}^{\mathrm{gen}}_c}{\mathrm{LCOE}^{\mathrm{gen}}_c} \in [0,\infty), \]

where the firm full-system cost per MWh comprises generation, firming storage and ancillary services (construction in S13, S16). This definition is the standard practice of the integration-cost (system-LCOE) framework for variable renewables — Ueckerdt et al. (2013) introduced system LCOE to add the integration cost of variable renewables on top of bare generation cost; Hirth et al. (2015) provided the economic framework for integration costs; and Joskow (2011) showed that comparing intermittent with dispatchable sources by bare levelized cost alone systematically understates the former—precisely why this paper reports the cost with \(\Pi\) rather than bare LCOE. \(\Pi_c\) is dimensionless and zeroed at perfect phase alignment (the generation floor, \(\Pi=0\)); to lie on the same \([0,1)\) scale as \(U_c\) it can be written equivalently as the phase-chasing cost share \(\sigma_c=\Pi_c/(1+\Pi_c)\). The premium implied here falls within the range reported by IRENA (2026) for the reliable 24/7 delivery of comparable wind and solar plus storage; the year-by-year conversion of \(\Pi\) to absolute USD levelized cost, and the reversal whereby absolute cost falls while the relative premium rises, are given in S22.

**Relationship between the two.** \(U\) and \(\Pi\) are two readings of the same physical quantity—the phase mismatch between demand and clean supply at the hourly scale. The better the phase alignment, the lower \(U\), the fewer continuous gap segments firming storage must fill, and the closer \(\Pi\) falls to zero; the phase-unalignable residual turns, through firming storage and ancillary services, into the premium. The paper therefore takes no absolute figure of energy, capacity or money as its endpoint, working throughout on this pair of observable, cross-checkable metrics rooted in the established literature.

**Concept–reference map.**

| Concept | Relation to metric | Source (main-text reference number) |
|---|---|---|
| 24/7 CFE matching score | \(=1-U\) | Google 2020 (8); Riepin & Brown 2024 (9) |
| Temporal self-sufficiency rate (PV self-consumption) | \(=1-U\) | Luthander et al. 2015 (10) |
| Net-zero-energy-building load match index | \(=1-U\) | Sartori et al. 2012 (11) |
| Total-variation distance / overlap coefficient | \(U=\mathrm{TV}\), \(1-U=\mathrm{OVL}\) | Weitzman 1970 (12) |
| System LCOE / integration cost | \(\Pi=\) integration cost / generation cost | Ueckerdt et al. 2013 (13); Hirth et al. 2015 (14) |
| Bare LCOE understates intermittent sources | Rationale for \(\Pi\) | Joskow 2011 (15) |
| Economics of reliable wind-and-solar delivery | Established calibration of \(\Pi\)'s magnitude | IRENA 2026 (16) |

**References cited in this section.**

- Google. *24/7 by 2030 — Realizing a Carbon-Free Future* (Google, 2020).
- Riepin, I. & Brown, T. On the means, costs, and system-level impacts of 24/7 carbon-free energy procurement. *Energy Strategy Reviews* 54, 101488 (2024).
- Luthander, R., Widén, J., Nilsson, D. & Palm, J. Photovoltaic self-consumption in buildings — a review. *Applied Energy* 142, 80–94 (2015).
- Sartori, I., Napolitano, A. & Voss, K. Net zero energy buildings — a consistent definition framework. *Energy and Buildings* 48, 220–232 (2012).
- Weitzman, M. S. *Measures of Overlap of Income Distributions of White and Negro Families in the United States*. Technical Paper No. 22 (U.S. Bureau of the Census, 1970).
- Ueckerdt, F., Hirth, L., Luderer, G. & Edenhofer, O. System LCOE — what are the costs of variable renewables? *Energy* 63, 61–75 (2013).
- Hirth, L., Ueckerdt, F. & Edenhofer, O. Integration costs revisited — an economic framework for wind and solar variability. *Renewable Energy* 74, 925–939 (2015).
- Joskow, P. L. Comparing the costs of intermittent and dispatchable electricity generating technologies. *American Economic Review* 101, 238–241 (2011).
- International Renewable Energy Agency. *24/7 Renewables — The Economics of Firm Solar and Wind* (IRENA, 2026).

