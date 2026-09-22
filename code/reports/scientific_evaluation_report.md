# Scientific Evaluation Report: Exploratory Empirical Assessment of PTDLMP-PISR on Amazon Challenge Cohort

**Date:** 2026-09-22  
**Evaluation Scope:** Exploratory evaluation on a 13-route cohort of the official Amazon Last Mile Routing Research Challenge dataset (`code/data/raw/`), chronological ML prediction pipeline (RQ1), and held-out route resequencing benchmark with cell-level and route-clustered statistical inference (RQ2).  
**Associated Artifacts:**
- Raw factorial runs: `code/outputs/routing_benchmark_runs.csv` (288 runs across held-out Amazon routes)
- Summary statistics: `code/outputs/benchmark_summary.json`
- Clarke-Wright robustness runs: `code/outputs/robustness_clark_wright_runs.csv`
- RQ1 prediction metrics: `code/outputs/rq1_prediction_results.json`
- Dataset Checksums: `code/data/raw/` (verified via SHA-256)

---

## Executive Summary & Scientific Verdict

This report presents an empirical evaluation of the **Prediction-Informed Selective Route Resequencing (PISR)** framework executed on a 13-route exploratory convenience cohort from the **Amazon Last Mile Routing Research Challenge dataset** (AWS Open Data, licensed under CC BY-NC 4.0). Adhering to strict chronological splitting, feature leakage guards, and route-clustered statistical testing:

1. **RQ1 (Predictability of Promised Delivery Lateness Proxy)**:
   - Evaluated on **2,038 drop-off delivery stops** (2,051 total stops including depots) across 12 calendar dates from **7 station codes** (`DBO2`, `DCH4`, `DLA7`, `DLA8`, `DLA9`, `DSE4`, `DSE5`).
   - *Label Definition*: The public Amazon Challenge dataset does not provide observed customer-level delivery completion timestamps. Consequently, ground-truth delivery lateness is strictly defined as a **route-propagated promised-time violation proxy** computed by propagating the driver's actual sequence through the official historical average travel-time matrix and planned service durations. Base lateness prevalence is 50.34%.
   - *Model Selection on Validation Split*: Based strictly on **validation set criteria** (chronological split), **Logistic Regression** is the superior model, achieving a validation Brier score of **0.2387** (vs. 0.2461 for Random Forest and 0.2505 for Prevalence Baseline), validation log loss of **0.6640** (vs. 0.6848), validation ECE of **0.0159** (vs. 0.1293), and validation ROC-AUC of **0.5327** (vs. 0.4920). Consequently, calibrated probabilities from Logistic Regression were selected to populate $p_i$ in the routing benchmark. On the held-out test cohort, Logistic Regression achieves ROC-AUC = 0.595, PR-AUC = 0.596, and Brier score = 0.2404.
2. **RQ2 (Routing Effectiveness & Policy Comparison)**:
   - Evaluated across **288 factorial runs** on 3 held-out test routes (`DBO2`, `DSE4`, and `DSE5`) spanning budgets $B \in \{5\%, 10\%, 20\%, 30\%\}$ and distance tolerances $\delta \in \{0\%, 2\%, 5\%, 10\%\}$. All distance metrics $\Delta D$ represent a Haversine straight-line distance proxy. The Random baseline is evaluated across 10 distinct random seeds (seeds 42 to 51) per parameter cell.
   - **Proposition 1 Monotonicity**: Verified in 100% of runs ($\min \Delta TT \geq 0.0000$ min, exactly 0 violations). Resequencing never degraded route performance.
   - **Hard Feasibility**: 100% of final routes passed the independent validator (no subtours, no duplicated/dropped customers, depot departure preserved).
3. **Core Hypothesis Resolution & Defensible Statistical Verdict**:
   - **Defensible Conclusion**: Actionability dominance ($g_i = \max \Delta TT$) drives simulated tardiness reductions; pure risk without actionability ($RB$) fails. Current evidence does not show that multiplying actionability by predicted risk ($RA$) improves outcomes over actionability alone ($AB$). None of the route-clustered comparisons remain statistically significant after Holm–Bonferroni correction because only three independent test routes are available ($N=3$).
   - **Route-Clustered Statistical Inference ($N=3$)**:
     - **RA vs. RB (Pure Risk)**: Mean diff = **+1,050.28 min**, route-level 95% bootstrap CI = **[+830.29, +1,239.51] min**, unadjusted $p = 0.0126$, Holm-Bonferroni adjusted $p = 0.0631$. Pure risk targeting selects stops that are geographically inaccessible, squandering limited relocation capacity.
     - **RA vs. Slack**: Mean diff = **+1,014.41 min**, route-level 95% bootstrap CI = **[+394.52, +1,475.30] min**, unadjusted $p = 0.0877$, Holm-Bonferroni adjusted $p = 0.1754$.
     - **RA vs. Deadline**: Mean diff = **+1,145.87 min**, route-level 95% bootstrap CI = **[+834.88, +1,414.37] min**, unadjusted $p = 0.0210$, Holm-Bonferroni adjusted $p = 0.0839$.
     - **RA vs. Random**: Mean diff = **+1,118.42 min**, route-level 95% bootstrap CI = **[+759.68, +1,299.85] min**, unadjusted $p = 0.0248$, Holm-Bonferroni adjusted $p = 0.0839$.
     - **RA vs. AB (Pure Actionability)**: Mean diff = **-16.62 min**, route-level 95% bootstrap CI = **[-55.48, +3.95] min**, unadjusted $p = 0.4825$, Holm-Bonferroni adjusted $p = 0.4825$.
   - **Sample Size & Prospective Power**: Establishing formal statistical confirmation under route clustering requires expanding the held-out route sample based on a formal prospective power analysis ($N \approx 34$ for RA vs AB, $N \approx 5$ for RA vs RB).
   - **Scientific Insight**: Actionability score $g_i$ captures virtually all single-vehicle schedule headroom. Weighting by predicted risk ($p_i \cdot g_i$) provides a sensible tie-breaker without degrading performance, but does not yield a statistically significant advantage over pure $g_i$ alone in this single-vehicle setting.

---

## 1. RQ1: Chronological ML Prediction Pipeline Results

### Experimental Setup
- **Dataset**: Official Amazon Challenge cohort (`code/data/raw/`): 13 routes, 2,051 total stops (2,038 drop-off stops), 3,129 packages across 12 calendar dates from 7 distribution centers (`DBO2`, `DCH4`, `DLA7`, `DLA8`, `DLA9`, `DSE4`, `DSE5`).
- **Target Label Proxy**: Route-propagated promised-time violation proxy derived from actual driver sequence and historical average travel times.
- **Leakage Guard**: Enforced before model fitting. Only ex-ante features available at vehicle departure time $t \le \text{departure\_time}$ were utilized:
  - Geographic: distance to depot (Haversine proxy), 2 km local stop density.
  - Operational: planned service seconds, package volume ($\text{cm}^3$), package count at stop.
  - Route context: total route stops, total planned route service minutes, departure hour, day of week.
  - Deadline slack: time until promised delivery deadline ($\tau_i - \text{departure\_time}$).
- **Chronological Split**:
  - **Train**: 1,173 stops across earliest 7 dates (60%).
  - **Validation (Model Selection & Calibration)**: 331 stops across next 2 dates (20%).
  - **Test (Held-out Evaluation)**: 534 stops across latest 3 dates (20%).

### Model Selection on Validation Split

| Model | Validation ROC-AUC | Validation PR-AUC | Validation Brier Score | Validation Log Loss | Validation ECE |
|---|---:|---:|---:|---:|---:|
| **Prevalence Baseline** | 0.500 | 0.511 | 0.2505 | 0.6942 | 0.0255 |
| **Logistic Regression (Selected)** | **0.533** | 0.515 | **0.2387** | **0.6640** | **0.0159** |
| **Random Forest** | 0.492 | **0.538** | 0.2461 | 0.6848 | 0.1293 |

*Decision*: Logistic Regression demonstrates superior probability calibration (ECE = 0.0159 vs. 0.1293), lower Brier score, and lower log loss on validation data, making it the mathematically preferred model for probability estimation.

### Test Cohort Performance (Out-of-Sample)

| Model | Test ROC-AUC | Test PR-AUC | Test Brier Score | Test Log Loss | Test ECE |
|---|---:|---:|---:|---:|---:|
| **Prevalence Baseline** | 0.500 | 0.539 | 0.2514 | 0.6959 | 0.0542 |
| **Logistic Regression** | **0.595** | 0.596 | **0.2404** | **0.6704** | **0.0095** |
| **Random Forest** | 0.547 | **0.610** | 0.2445 | 0.6819 | 0.0203 |

---

## 2. RQ2: Empirical Routing Benchmark on Held-Out Amazon Routes

### Factorial Evaluation Matrix
- **Held-Out Test Routes**: 3 independent routes spanning 407 drop-off delivery stops across test stations `DBO2`, `DSE4`, and `DSE5`.
- **Factors**:
  - Baseline route: Nearest Neighbor ($R^0$).
  - Targeting policies: RA, AB, RB, Slack, Deadline, Random (evaluated over 10 random seeds: 42 to 51).
  - Intervention budget: $B \in \{0.05, 0.10, 0.20, 0.30\}$.
  - Distance tolerance: $\delta \in \{0.00, 0.02, 0.05, 0.10\}$.
- **Total Factorial Runs**: 288 runs (3 routes $\times$ 4 budgets $\times$ 4 tolerances $\times$ 6 policies).

### Policy Performance Summary

| Policy | Targeting Formula | Avg $\Delta TT$ (min) | Avg TT Red (%) | Avg $\Delta NL$ | Avg Dist Inc (%) | Avg Relocations | Feasibility Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| **RA** | $p_i \cdot g_i$ | **+1,417.91 min** | **12.1%** | **+3.65** | +2.05% | 17.85 | **100.0%** |
| **AB** | $g_i$ | **+1,434.54 min** | **12.2%** | **+3.52** | +2.11% | 17.29 | **100.0%** |
| **RB** | $p_i$ | **+367.63 min** | **2.9%** | **+0.83** | +1.93% | 7.42 | **100.0%** |
| **Slack** | $\tau_i - c_i(R^0)$ | **+403.50 min** | **4.0%** | **+0.40** | +1.48% | 11.02 | **100.0%** |
| **Deadline** | $\tau_i$ | **+272.05 min** | **2.2%** | **+0.73** | +2.28% | 6.31 | **100.0%** |
| **Random** | Uniform Random (10 seeds) | **+299.49 min** | **2.5%** | **+0.72** | +1.95% | 5.99 | **100.0%** |

---

## 3. Statistical Inference: Cell-Level vs. Route-Clustered Hypothesis Tests

$$\Delta TT(\text{RA}) - \Delta TT(\text{Competitor})$$

| Comparison | Mean Diff (min) | Route-Level 95% Bootstrap CI | Cell-Level 95% Bootstrap CI | Route-Clustered $t$ ($N=3$) | Route-Clustered $p$ (unadjusted) | Holm-Bonferroni Adj. $p$ | Significant at $\alpha=0.05$? |
|---|---:|:---:|:---:|---:|---:|---:|:---:|
| **RA vs. AB** | **-16.62 min** | [-55.48, +3.95] | [-49.92, +18.20] | -0.855 | 0.4825 | 0.4825 | **NO** |
| **RA vs. RB** | **+1,050.28 min** | [+830.29, +1,239.51] | [+932.39, +1,179.75] | +8.816 | 0.0126 | 0.0631 | **Marginal** (unadjusted) |
| **RA vs. Slack** | **+1,014.41 min** | [+394.52, +1,475.30] | [+846.26, +1,179.74] | +3.151 | 0.0877 | 0.1754 | **NO** |
| **RA vs. Deadline** | **+1,145.87 min** | [+834.88, +1,414.37] | [+1,020.43, +1,264.07] | +6.795 | 0.0210 | 0.0839 | **Marginal** (unadjusted) |
| **RA vs. Random** | **+1,118.42 min** | [+759.68, +1,299.85] | [+987.39, +1,253.40] | +6.235 | 0.0248 | 0.0839 | **Marginal** (unadjusted) |

### Key Methodological Takeaways:
1. **Dangers of Pseudo-Replication**: Cell-level pooling treats repeated runs on the same route as independent, artificially inflating statistical power ($p < 0.0001$). Route clustering correctly identifies the independent experimental unit ($N=3$), reflecting the true sampling uncertainty.
2. **Substantive vs. Statistical Significance**: Although the magnitude of tardiness reduction for RA over RB (+1,050 min) and Deadline (+1,146 min) is operationally dramatic, statistical confirmation after family-wise error rate control requires expanding the held-out route sample based on a formal prospective power analysis.
3. **Equivalence of RA and AB**: Both at the cell level ($p = 0.3626$) and route-clustered level ($p = 0.4825$), RA and AB are statistically indistinguishable. Actionability $g_i$ carries the primary optimization signal.

---

## 4. Invariant & Phenomenological Audit

| Audit Item | Observed Value | Expected Theoretical Behavior | Status |
|---|---:|---:|:---:|
| **Minimum $\Delta TT$ across all runs** | **0.0000 min** | $\Delta TT \ge 0$ (Proposition 1) | **VERIFIED (0 violations)** |
| **Route Feasibility Rate** | **100.0%** (288/288) | 100% | **VERIFIED** |
| **Occurrences of $\Delta NL < 0$ while $\Delta TT > 0$** | **11 runs** | Allowed by Proposition 1 remark | **OBSERVED & DOCUMENTED** |
| **Runs starting with 0 baseline tardiness** | **0 of 288 (0.0%)** | Realistic operational stress | **CONFIRMED** |

### Clarification on $\Delta NL < 0$:
In 11 factorial runs, advancing a customer with large tardiness saved substantial delay minutes ($\Delta TT > 0$), but pushed downstream customers slightly past their deadline window, increasing total late deliveries by 1 ($\Delta NL = -1$). This empirically illustrates the distinction between the smooth continuous objective (total tardiness) and the non-smooth step objective (lateness count).

---

## 5. Robustness Analysis: Clarke-Wright Savings Baseline

Under Clarke-Wright baseline tours ($B = 0.20$, $\delta = 0.05$):

| Policy | Avg $\Delta TT$ (min) | Avg TT Reduction (%) | Avg $\Delta NL$ | Avg Distance Increase (%) |
|---|---:|---:|---:|---:|
| **RA** | **+1,761.54 min** | **13.1%** | **+6.00** | +4.75% |
| **AB** | **+1,931.79 min** | **14.1%** | **+7.00** | +4.67% |
| **RB** | **+577.97 min** | **4.2%** | **+1.33** | +4.19% |
| **Slack** | **+938.52 min** | **6.0%** | **+2.67** | +4.59% |

---

## 6. Recommendations for Manuscript Revision (`main.tex`)

1. **Avoid Overstating Statistical Claims**: Explicitly state that while RA achieves large operational reductions over RB (+1,050 min) and Slack (+1,014 min), route-clustered testing on $N=3$ test routes yields marginal unadjusted significance ($p \approx 0.01 - 0.08$) that does not survive Holm-Bonferroni correction ($p \approx 0.06 - 0.17$).
2. **Report Label Construction Transparently**: Disclose that customer arrival times in the public Amazon Challenge are route-propagated proxies based on historical travel-time matrices.
3. **Emphasize Actionability ($g_i$)**: Frame actionability as the indispensable spatial filter that prevents pure ML models from selecting futile moves.
4. **Include Dataset Attribution & License**: Add formal citation of Merchán et al. (2021) and the CC BY-NC 4.0 license notice.
