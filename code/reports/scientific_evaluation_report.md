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

This report presents an empirical evaluation of the **Prediction-Informed Selective Route Resequencing (PISR)** framework executed on a 13-route representative cohort from the **Amazon Last Mile Routing Research Challenge dataset** (AWS Open Data, licensed under CC BY-NC 4.0). Adhering to strict chronological splitting, feature leakage guards, and route-clustered statistical testing:

1. **RQ1 (Predictability of Promised Delivery Lateness Proxy)**:
   - Evaluated on **2,038 drop-off delivery stops** (2,051 total stops including depots) across 12 calendar dates from **7 station codes** (`DBO2`, `DCH4`, `DLA7`, `DLA8`, `DLA9`, `DSE4`, `DSE5`).
   - *Label Definition*: The public Amazon Challenge dataset does not provide observed customer-level delivery completion timestamps. Consequently, ground-truth delivery lateness is constructed as a **route-propagated time-window violation proxy** computed by propagating the driver's actual sequence through the official historical average travel-time matrix and planned service durations. Base lateness prevalence is 50.34%.
   - *Model Selection*: Based strictly on **validation set criteria** (chronological split), **Logistic Regression** is the superior model, achieving a validation Brier score of **0.2387** (vs. 0.2461 for Random Forest), validation log loss of **0.6640** (vs. 0.6848), validation ECE of **0.0159** (vs. 0.1293), and validation ROC-AUC of **0.5327** (vs. 0.4920). On the held-out test cohort, Logistic Regression achieves ROC-AUC = 0.595, PR-AUC = 0.596, and Brier score = 0.2404.
2. **RQ2 (Routing Effectiveness & Policy Comparison)**:
   - Evaluated across **288 factorial runs** on 3 held-out test routes (`DBO2`, `DSE4`, and `DSE5`) spanning budgets $B \in \{5\%, 10\%, 20\%, 30\%\}$ and distance tolerances $\delta \in \{0\%, 2\%, 5\%, 10\%\}$.
   - **Proposition 1 Monotonicity**: Verified in 100% of runs ($\min \Delta TT \geq 0.0000$ min, exactly 0 violations). Resequencing never degraded route performance.
   - **Hard Feasibility**: 100% of final routes passed the independent validator (no subtours, no duplicated/dropped customers, depot departure preserved).
3. **Core Hypothesis Resolution & Statistical Inference**:
   - **Cell-Level vs. Route-Clustered Tests**: Cell-level tests ($N=48$ cells) yield nominal $p < 0.0001$ for comparisons against heuristics. However, treating repeated $(B, \delta)$ cells as independent introduces pseudo-replication. When clustering at the route level ($N=3$ independent held-out routes):
     - **RA vs. RB (Pure Risk)**: Mean diff = **+1,158.64 min**, unadjusted $p = 0.0227$. Pure risk targeting selects stops that are geographically inaccessible, squandering limited relocation capacity.
     - **RA vs. Slack**: Mean diff = **+1,049.59 min**, unadjusted $p = 0.0851$.
     - **RA vs. Deadline**: Mean diff = **+1,181.05 min**, unadjusted $p = 0.0208$.
     - **RA vs. Random**: Mean diff = **+1,144.12 min**, unadjusted $p = 0.0270$.
     - **RA vs. AB (Pure Actionability)**: Mean diff = **+8.69 min**, unadjusted $p = 0.1730$.
   - **Multiple Testing Correction**: Under Holm-Bonferroni correction across the 5 policy comparisons, none of the differences reach formal significance at $\alpha = 0.05$ (adjusted $p \in [0.0908, 0.3460]$) due to the limited degrees of freedom ($N=3$ routes). While the directional effect sizes are large (>1,000 minutes saved), establishing formal statistical significance under route clustering requires expanding the held-out sample to $N \ge 20$ independent routes.
   - **Scientific Insight**: Actionability score $g_i$ captures virtually all single-vehicle schedule headroom. Weighting by predicted risk ($p_i \cdot g_i$) provides a sensible tie-breaker without degrading performance, but does not yield a statistically significant advantage over pure $g_i$ alone in this single-vehicle setting.

---

## 1. RQ1: Chronological ML Prediction Pipeline Results

### Experimental Setup
- **Dataset**: Official Amazon Challenge cohort (`code/data/raw/`): 13 routes, 2,051 total stops (2,038 drop-off stops), 3,129 packages across 12 calendar dates from 7 distribution centers (`DBO2`, `DCH4`, `DLA7`, `DLA8`, `DLA9`, `DSE4`, `DSE5`).
- **Target Label Proxy**: Route-propagated time-window violation indicator derived from actual driver sequence and historical average travel times.
- **Leakage Guard**: Enforced before model fitting. Only ex-ante features available at vehicle departure time $t \le \text{departure\_time}$ were utilized:
  - Geographic: distance to depot, 2 km local stop density.
  - Operational: planned service seconds, package volume ($\text{cm}^3$), package count at stop.
  - Route context: total route stops, total planned route service minutes, departure hour, day of week.
  - Deadline slack: time until promised delivery deadline ($\tau_i - \text{departure\_time}$).
- **Chronological Split**:
  - **Train**: 1,223 stops across earliest 6 dates (60%).
  - **Validation (Model Selection & Calibration)**: 408 stops across next 3 dates (20%).
  - **Test (Held-out Evaluation)**: 407 stops across latest 3 dates (20%).

### Model Selection on Validation Split

| Model | Validation ROC-AUC | Validation PR-AUC | Validation Brier Score | Validation Log Loss | Validation ECE |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression (Selected)** | **0.533** | 0.515 | **0.2387** | **0.6640** | **0.0159** |
| **Random Forest** | 0.492 | **0.538** | 0.2461 | 0.6848 | 0.1293 |

*Decision*: Logistic Regression demonstrates superior probability calibration (ECE = 0.0159 vs. 0.1293), lower Brier score, and lower log loss on validation data, making it the mathematically preferred model for probability estimation.

### Test Cohort Performance (Out-of-Sample)

| Model | Test ROC-AUC | Test PR-AUC | Test Brier Score | Test Log Loss | Test ECE |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.595** | 0.596 | **0.2404** | **0.6704** | **0.0095** |
| **Random Forest** | 0.547 | **0.610** | 0.2445 | 0.6819 | 0.0203 |

---

## 2. RQ2: Empirical Routing Benchmark on Held-Out Amazon Routes

### Factorial Evaluation Matrix
- **Held-Out Test Routes**: 3 independent routes spanning 407 drop-off delivery stops across test stations `DBO2`, `DSE4`, and `DSE5`.
- **Factors**:
  - Baseline route: Nearest Neighbor ($R^0$).
  - Targeting policies: RA, AB, RB, Slack, Deadline, Random.
  - Intervention budget: $B \in \{0.05, 0.10, 0.20, 0.30\}$.
  - Distance tolerance: $\delta \in \{0.00, 0.02, 0.05, 0.10\}$.
- **Total Factorial Runs**: 288 runs (3 routes $\times$ 4 budgets $\times$ 4 tolerances $\times$ 6 policies).

### Policy Performance Summary

| Policy | Targeting Formula | Avg $\Delta TT$ (min) | Avg TT Red (%) | Avg $\Delta NL$ | Avg Dist Inc (%) | Avg Relocations | Feasibility Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| **RA** | $p_i \cdot g_i$ | **+1,453.09 min** | **12.41%** | **+3.62** | +2.04% | 17.17 | **100.0%** |
| **AB** | $g_i$ | **+1,444.40 min** | **12.33%** | **+3.58** | +2.13% | 17.35 | **100.0%** |
| **RB** | $p_i$ | **+294.45 min** | **2.40%** | **+0.60** | +1.90% | 7.25 | **100.0%** |
| **Slack** | $\tau_i - c_i(R^0)$ | **+403.50 min** | **3.96%** | **+0.40** | +1.48% | 11.02 | **100.0%** |
| **Deadline** | $\tau_i$ | **+272.05 min** | **2.17%** | **+0.73** | +2.28% | 6.31 | **100.0%** |
| **Random** | Uniform Random | **+308.98 min** | **2.62%** | **+0.78** | +1.88% | 6.10 | **100.0%** |

---

## 3. Statistical Inference: Cell-Level vs. Route-Clustered Hypothesis Tests

$$\Delta TT(\text{RA}) - \Delta TT(\text{Competitor})$$

| Comparison | Mean Diff (min) | 95% Confidence Interval | Cell-Level $p$ ($N=48$) | Route-Clustered $t$ ($N=3$) | Route-Clustered $p$ (unadjusted) | Holm-Bonferroni Adj. $p$ | Significant at $\alpha=0.05$? |
|---|---:|:---:|---:|---:|---:|---:|:---:|
| **RA vs. AB** | **+8.69 min** | [-3.06, +21.93] | 0.1768 | +2.080 | 0.1730 | 0.3460 | **NO** |
| **RA vs. RB** | **+1,158.64 min** | [+1,043.49, +1,270.88] | < 0.0001 | +6.526 | 0.0227 | 0.0908 | **Marginal** (unadjusted) |
| **RA vs. Slack** | **+1,049.59 min** | [+880.44, +1,216.10] | < 0.0001 | +3.205 | 0.0851 | 0.0908 | **NO** |
| **RA vs. Deadline** | **+1,181.05 min** | [+1,064.95, +1,302.08] | < 0.0001 | +6.825 | 0.0208 | 0.1040 | **Marginal** (unadjusted) |
| **RA vs. Random** | **+1,144.12 min** | [+1,022.29, +1,272.24] | < 0.0001 | +5.967 | 0.0270 | 0.0908 | **Marginal** (unadjusted) |

### Key Methodological Takeaways:
1. **Dangers of Pseudo-Replication**: Cell-level pooling treats repeated runs on the same route as independent, artificially inflating statistical power ($p < 0.0001$). Route clustering correctly identifies the independent experimental unit ($N=3$), reflecting the true sampling uncertainty.
2. **Substantive vs. Statistical Significance**: Although the magnitude of tardiness reduction for RA over RB (+1,158 min) and Deadline (+1,181 min) is operationally dramatic, statistical confirmation after family-wise error rate control requires expanding the route sample ($N \ge 20$).
3. **Equivalence of RA and AB**: Both at the cell level ($p = 0.1768$) and route-clustered level ($p = 0.1730$), RA and AB are statistically indistinguishable. Actionability $g_i$ carries the primary optimization signal.

---

## 4. Invariant & Phenomenological Audit

| Audit Item | Observed Value | Expected Theoretical Behavior | Status |
|---|---:|---:|:---:|
| **Minimum $\Delta TT$ across all runs** | **0.0000 min** | $\Delta TT \ge 0$ (Proposition 1) | **VERIFIED (0 violations)** |
| **Route Feasibility Rate** | **100.0%** (288/288) | 100% | **VERIFIED** |
| **Occurrences of $\Delta NL < 0$ while $\Delta TT > 0$** | **9 runs** | Allowed by Proposition 1 remark | **OBSERVED & DOCUMENTED** |
| **Runs starting with 0 baseline tardiness** | **0 of 288 (0.0%)** | Realistic operational stress | **CONFIRMED** |

### Clarification on $\Delta NL < 0$:
In 9 factorial runs, advancing a customer with large tardiness saved substantial delay minutes ($\Delta TT > 0$), but pushed downstream customers slightly past their deadline window, increasing total late deliveries by 1 ($\Delta NL = -1$). This empirically illustrates the distinction between the smooth continuous objective (total tardiness) and the non-smooth step objective (lateness count).

---

## 5. Robustness Analysis: Clarke-Wright Savings Baseline

Under Clarke-Wright baseline tours ($B = 0.20$, $\delta = 0.05$):

| Policy | Avg $\Delta TT$ (min) | Avg TT Reduction (%) | Avg $\Delta NL$ | Avg Distance Increase (%) |
|---|---:|---:|---:|---:|
| **RA** | **+1,965.55 min** | **14.32%** | **+7.00** | +4.64% |
| **AB** | **+1,931.79 min** | **14.11%** | **+7.00** | +4.68% |
| **RB** | **+473.48 min** | **3.29%** | **+0.67** | +3.86% |
| **Slack** | **+938.52 min** | **6.03%** | **+2.67** | +4.59% |

---

## 6. Recommendations for Manuscript Revision (`main.tex`)

1. **Avoid Overstating Statistical Claims**: Explicitly state that while RA achieves large operational reductions over RB (+1,158 min) and Slack (+1,050 min), route-clustered testing on $N=3$ test routes yields marginal unadjusted significance ($p \approx 0.02 - 0.08$) that does not survive Holm-Bonferroni correction.
2. **Report Label Construction Transparently**: Disclose that customer arrival times in the public Amazon Challenge are route-propagated proxies based on historical travel-time matrices.
3. **Emphasize Actionability ($g_i$)**: Frame actionability as the indispensable spatial filter that prevents pure ML models from selecting futile moves.
4. **Include Dataset Attribution & License**: Add formal citation of Merchán et al. (2021) and the CC BY-NC 4.0 license notice.
