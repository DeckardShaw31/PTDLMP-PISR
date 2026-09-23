# Scientific Evaluation Report: Empirical Assessment of PTDLMP-PISR on Amazon Last-Mile Cohort

**Date:** 2026-09-23  
**Evaluation Scope:** Empirical evaluation on the official Amazon Last Mile Routing Research Challenge dataset (`code/data/raw/`, AWS Open Data, CC BY-NC 4.0). Includes:
1. **Explicit-Deadline Audit**: Complete audit of official customer time windows (`time_window.end_time_utc`), disclosing dataset sparsity and single-class limitations.
2. **RQ1 Chronological Prediction Pipeline**: Predeclared Logistic Regression with validation Platt calibration, evaluated under the primary 4-hour dispatch service threshold and sensitivity thresholds $H \in \{3.0, 4.0, 5.0, 6.0\}$ hours.
3. **RQ2 & RQ3 Routing Benchmark**: 288 factorial runs across held-out test routes under primary threshold ($H = 4.0$ hours), Clarke-Wright robustness runs, 480 random-seed runs, and multi-threshold sensitivity suite ($H \in \{3, 4, 5, 6\}$).
4. **Statistical Inference**: Cell-level paired bootstrap tests, route-clustered testing ($N=3$), Holm–Bonferroni multiplicity correction, and prospective power estimation.

**Associated Artifacts:**
- Raw factorial runs: `code/outputs/routing_benchmark_runs.csv` (288 runs)
- Random seed runs: `code/outputs/random_seed_runs.csv` (480 seed-level runs)
- Threshold sensitivity runs: `code/outputs/threshold_sensitivity_runs.csv` (288 runs across $H \in \{3, 4, 5, 6\}$)
- Summary statistics: `code/outputs/benchmark_summary.json`
- Clarke-Wright robustness runs: `code/outputs/robustness_clark_wright_runs.csv` (72 runs)
- RQ1 prediction metrics: `code/outputs/rq1_prediction_results.json`
- RQ1 test predictions: `code/outputs/rq1_test_predictions.csv` (534 test stops)
- Generated Figures: `figures/fig2_policy_comparison.png`, `figures/fig3_route_before_after.png`

---

## Executive Summary & Scientific Findings

### 1. Explicit-Deadline Audit (Empirical Limitation)
- **Cohort Reality**: Across the 13 official routes (2,051 total stops, 2,038 customer drop-offs), only **128 stops (6.3%)** have explicit customer delivery time windows (`time_window.end_time_utc`).
- **Chronological Breakdown**:
  - Training Block: 95 stops (0 late deliveries, 0.0% prevalence).
  - Validation Block: 20 stops (2 late deliveries, 10.0% prevalence).
  - Held-Out Test Block: 13 stops (0 late deliveries, 0.0% prevalence).
- **Single-Class Test Limitation**: With 0 positive lateness labels in the test block, discrimination metrics (ROC-AUC, PR-AUC) are uninformative (0.500 / 0.000).
- **Zero-Tardiness Routing Baseline**: Evaluating baseline Nearest Neighbor dispatch on the 3 held-out test routes using only explicit deadlines yields exactly **0.00 minutes of baseline tardiness**. With no active tardiness to eliminate, zero forward relocations are accepted across all policies ($\Delta TT = 0, \Delta NL = 0$).
- **Methodological Remedy**: We explicitly disclose this sparsity as a dataset limitation. To conduct a meaningful scientific evaluation, the primary study investigates a **derived four-hour dispatch service threshold** ($\tau_i(H) = t_{\text{departure}} + H$), representing a carrier dispatch service objective rather than an observed customer guarantee.

### 2. RQ1 (Predictability of Dispatch Service-Threshold Lateness)
- **Predeclared Primary Architecture**: Logistic Regression with post-hoc Platt calibration fitted exclusively on the chronological validation block.
- **Primary Scenario ($H = 4.0$ Hours, 534 Test Stops)**:
  - Base Test Prevalence: **53.93%** (288 late stops out of 534).
  - Logistic Regression (Primary): **Test ROC-AUC = 0.595**, **Test PR-AUC = 0.596**, **Test Brier Score = 0.2404**, **Test Log Loss = 0.6704**, **Test ECE = 0.0095**.
  - Prevalence Baseline: Test ROC-AUC = 0.500, PR-AUC = 0.539, Brier = 0.2514, ECE = 0.0542.
  - Random Forest: Test ROC-AUC = 0.547, PR-AUC = 0.610, Brier = 0.2445, ECE = 0.0203.
- **Threshold Sensitivity ($H \in \{3, 4, 5, 6\}$ Hours)**:
  - $H = 3.0$h: Test Prev = 66.5%, ROC-AUC = 0.613, PR-AUC = 0.752, Brier = 0.2096, ECE = 0.0492.
  - $H = 4.0$h: Test Prev = 53.9%, ROC-AUC = 0.595, PR-AUC = 0.596, Brier = 0.2404, ECE = 0.0095.
  - $H = 5.0$h: Test Prev = 42.5%, ROC-AUC = 0.469, PR-AUC = 0.428, Brier = 0.2433, ECE = 0.0456.
  - $H = 6.0$h: Test Prev = 27.3%, ROC-AUC = 0.518, PR-AUC = 0.287, Brier = 0.2005, ECE = 0.0553.
  - Test lateness prevalence decreases monotonically from 66.5% to 27.3%, confirming operational realism across diverse operational horizons.

### 3. RQ2 (Policy Targeting Comparison under Primary Scenario)
Evaluated across **288 factorial runs** on 3 held-out test routes ($B \in \{5\%, 10\%, 20\%, 30\%\}$, $\delta \in \{0\%, 2\%, 5\%, 10\%\}$):

| Policy | Targeting Score Formula | Avg $\Delta TT$ (min) | Avg TT Red (%) | Avg $\Delta NL$ | Avg Dist Inc (%) | Feasibility Rate | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| **RA (Proposed)** | $p_i \cdot g_i$ | **+1,417.91 min** | **12.1%** | **+3.65** | +2.05% | **100.0%** | 0.255s |
| **AB (Actionability)** | $g_i$ | **+1,434.54 min** | **12.2%** | **+3.52** | +2.11% | **100.0%** | 0.272s |
| **RB (Risk-Based)** | $p_i$ | **+367.63 min** | **2.9%** | **+0.83** | +1.93% | **100.0%** | 0.241s |
| **Slack-Based** | $\tau_i - c_i(R^0)$ | **+403.50 min** | **4.0%** | **+0.40** | +1.48% | **100.0%** | 0.379s |
| **Threshold-Based** | $\tau_i$ | **+272.05 min** | **2.2%** | **+0.73** | +2.28% | **100.0%** | 0.180s |
| **Uniform Random** | 10 Random Seeds | **+299.49 min** | **2.5%** | **+0.72** | +1.95% | **100.0%** | 0.213s |

- **Actionability Dominance**: Actionability-informed policies (RA and AB) achieve $>1,400$ minutes of tardiness reduction, outperforming pure risk (RB), slack, threshold, and random targeting by more than $3\times$.
- **RA vs. AB Comparison**: Difference is small ($-16.62$ min, 1.1% relative) and statistically indistinguishable ($p = 0.4825$). This failure to reject the null is not proof of equivalence; a provisional power calculation indicates that detecting a subtle difference requires $N \approx 34$ independent routes.

### 4. RQ3 (Operational Improvement vs. Routing Effort)
- **Proposition 1 Monotonicity**: Verified in 100% of runs ($\min \Delta TT = 0.0000$ min, exactly 0 violations).
- **Distance Constraint Compliance**: Cumulative distance constraint $\Delta D \le \delta$ was respected on 100% of runs.
- **Phenomenological Decoupling**: In 11 runs, $\Delta NL < 0$ occurred while $\Delta TT > 0$, demonstrating that continuous tardiness optimization decouples from binary threshold crossing.
- **Computational Efficiency**: Precomputing actionability scores requires an average of $1.31$~seconds per route, while executing selective forward relocation (SFR) averages $0.26$~seconds per route (combined decision latency of $\sim 1.57$~seconds), confirming operational viability for morning dispatch staging.

---

## Statistical Hypothesis Testing (Paired Differences: $\Delta TT(\text{RA}) - \Delta TT(\text{Competitor})$)

### Cell-Level vs. Route-Clustered Statistical Inference

| Comparison | Mean Diff (min) | 95% Bootstrap CI | Cell-Level $p$ ($N=48$) | Route $t$-stat ($N=3$) | Unadj $p$ | Holm-Bonferroni Adj $p$ |
|---|---:|:---:|---:|---:|---:|---:|
| **RA vs. AB** | **-16.62 min** | [-49.92, +18.20] | 0.3626 | -0.855 | 0.4825 | 0.4825 |
| **RA vs. RB** | **+1,050.28 min** | [+932.39, +1,179.75] | $< 0.0001$ | +8.816 | 0.0126 | 0.0631 |
| **RA vs. Slack** | **+1,014.41 min** | [+846.26, +1,179.74] | $< 0.0001$ | +3.151 | 0.0877 | 0.1754 |
| **RA vs. Threshold** | **+1,145.87 min** | [+1,020.43, +1,264.07] | $< 0.0001$ | +6.795 | 0.0210 | 0.0839 |
| **RA vs. Random** | **+1,118.42 min** | [+987.39, +1,253.40] | $< 0.0001$ | +6.235 | 0.0248 | 0.0839 |

---

## Threshold Sensitivity Benchmark ($H \in \{3, 4, 5, 6\}$ Hours)

| Threshold $H$ | Policy | Base $TT$ (min) | Base $NL$ | $\Delta TT$ (min) | TT Red (%) | $\Delta NL$ | $\Delta D$ (%) | Accepted Moves |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **$H = 3.0$h** | **RA** | 17,804.6 | 112.3 | **+1,804.2** | 9.9% | +2.33 | +2.83% | 19.0 |
| | **AB** | 17,804.6 | 112.3 | +1,767.3 | 9.8% | +1.67 | +2.25% | 17.7 |
| | **RB** | 17,804.6 | 112.3 | +386.0 | 2.0% | +0.67 | +1.94% | 7.2 |
| | **Slack** | 17,804.6 | 112.3 | +395.8 | 2.5% | -0.25 | +1.59% | 11.9 |
| | **Random** | 17,804.6 | 112.3 | +387.1 | 2.3% | +0.25 | +2.09% | 6.9 |
| **$H = 4.0$h** | **RA** | 11,700.8 | 90.0 | **+1,693.7** | 14.3% | +4.42 | +2.02% | 18.9 |
| | **AB** | 11,700.8 | 90.0 | +1,665.5 | 14.0% | +4.17 | +2.20% | 17.9 |
| | **RB** | 11,700.8 | 90.0 | +386.4 | 3.0% | +1.00 | +2.39% | 7.3 |
| | **Slack** | 11,700.8 | 90.0 | +386.6 | 3.9% | +0.33 | +1.59% | 11.9 |
| | **Random** | 11,700.8 | 90.0 | +357.3 | 3.3% | +1.08 | +2.00% | 7.0 |
| **$H = 5.0$h** | **RA** | 6,903.9 | 69.3 | +1,269.8 | 17.8% | +5.92 | +2.17% | 17.8 |
| | **AB** | 6,903.9 | 69.3 | **+1,301.2** | 18.6% | +6.25 | +2.15% | 17.9 |
| | **RB** | 6,903.9 | 69.3 | +159.8 | 2.4% | +1.00 | +2.37% | 7.0 |
| | **Slack** | 6,903.9 | 69.3 | +309.1 | 5.6% | +1.50 | +1.56% | 11.7 |
| | **Random** | 6,903.9 | 69.3 | +268.7 | 4.4% | +1.75 | +2.11% | 6.9 |
| **$H = 6.0$h** | **RA** | 3,495.0 | 45.7 | **+1,009.1** | 28.0% | +6.17 | +1.96% | 17.9 |
| | **AB** | 3,495.0 | 45.7 | +1,003.7 | 29.1% | +6.67 | +2.00% | 16.9 |
| | **RB** | 3,495.0 | 45.7 | +106.8 | 3.9% | +0.33 | +2.19% | 7.0 |
| | **Slack** | 3,495.0 | 45.7 | +223.0 | 9.7% | +1.25 | +1.38% | 11.4 |
| | **Random** | 3,495.0 | 45.7 | +177.6 | 6.1% | +1.25 | +2.06% | 6.7 |

---

## Clarke-Wright Savings Baseline Robustness

Under Clarke-Wright baseline tours ($B = 0.20$, $\delta = 0.05$):

| Policy | Avg $\Delta TT$ (min) | Avg TT Reduction (%) | Avg $\Delta NL$ | Avg Distance Increase (%) |
|---|---:|---:|---:|---:|
| **RA** | **+1,761.54 min** | **13.1%** | **+6.00** | +4.75% |
| **AB** | **+1,931.79 min** | **14.1%** | **+7.00** | +4.67% |
| **RB** | **+577.97 min** | **4.2%** | **+1.33** | +4.19% |
| **Slack** | **+938.52 min** | **6.0%** | **+2.67** | +4.59% |

---

## Invariant & Phenomenological Verification

| Invariant / Check | Theoretical Expectation | Observed Value | Result |
|---|---|---:|:---:|
| **Proposition 1 Monotonicity** | $\min \Delta TT \ge 0$ | **0.0000 min** | **VERIFIED (0 violations)** |
| **Distance Tolerance Bound** | $\Delta D \le \delta$ | $\le \delta$ on all runs | **VERIFIED** |
| **Route Feasibility Rate** | 100.0% open tours preserved | **100.0%** (288/288) | **VERIFIED** |
| **Phenomenological Decoupling** | Allowed by Proposition 1 remark | **11 runs** ($\Delta NL < 0, \Delta TT > 0$) | **VERIFIED** |
| **Zero Baseline Tardiness Runs** | 0 runs under derived threshold | **0 of 288 (0.0%)** | **VERIFIED** |
