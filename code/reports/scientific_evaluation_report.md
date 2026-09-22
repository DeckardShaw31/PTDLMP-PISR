# Scientific Evaluation Report: Empirical Assessment of PTDLMP-PISR

**Date:** 2026-09-22  
**Evaluation Scope:** Official Amazon Challenge adapter, chronological ML prediction pipeline (RQ1), and large-scale held-out route resequencing benchmark with paired statistical inference (RQ2).  
**Associated Artifacts:**
- Raw factorial runs: `code/outputs/routing_benchmark_runs.csv` (576 runs)
- Summary statistics: `code/outputs/benchmark_summary.json`
- Clarke-Wright robustness runs: `code/outputs/robustness_clark_wright_runs.csv`
- RQ1 prediction metrics: `code/outputs/rq1_prediction_results.json`

---

## Executive Summary & Scientific Verdict

This report provides the first complete, rigorous empirical evaluation of the **Prediction-Informed Selective Route Resequencing (PISR)** framework. Following the strict data contract, chronological splitting, and leakage guard requirements specified in `design.md`:

1. **RQ1 (Predictability of Promised Delivery Lateness)**:
   - Evaluated on 626 stops across 10 dates with an 18.05% baseline lateness prevalence.
   - An ex-ante Random Forest model achieved an out-of-sample **ROC-AUC of 0.816**, **PR-AUC of 0.458**, and a calibrated **Brier score of 0.1467** (ECE = 0.0983) under strict chronological separation (Train: 60%, Val: 20%, Test: 20%).
2. **RQ2 (Routing Effectiveness & Policy Comparison)**:
   - Evaluated across 576 factorial runs on held-out test routes spanning budgets $B \in \{5\%, 10\%, 20\%, 30\%\}$ and distance tolerances $\delta \in \{0\%, 2\%, 5\%, 10\%\}$.
   - **Proposition 1 Monotonicity**: Verified in 100% of runs ($\min \Delta TT \geq 0.0000$ min, 0 violations).
   - **Hard Feasibility**: 100% of final routes passed the independent validator.
3. **Core Hypothesis Resolution: Does RA Outperform AB, RB, and Heuristics?**
   - **RA vs. RB (Pure Risk)**: RA achieves a **+10.60 min** higher tardiness reduction ($p < 0.0001$, paired t-test; $p < 0.0001$, Wilcoxon). **RA significantly outperforms RB.** Pure risk targeting frequently targets geographically trapped stops that cannot be relocated without violating distance limits.
   - **RA vs. Slack / Deadline**: RA outperforms operational Slack by **+1.74 min** ($p = 0.0001$) and earliest Deadline by **+9.40 min** ($p < 0.0001$).
   - **RA vs. Random**: RA outperforms Random targeting by **+15.04 min** ($p < 0.0001$).
   - **RA vs. AB (Pure Actionability)**: RA does **NOT statistically significantly outperform AB** (mean difference: $-0.155$ min, 95% CI: $[-0.413, +0.020]$, paired $t = -1.330$, $p = 0.1867$; Wilcoxon $p = 0.5000$).
   - **Scientific Insight**: Actionability score $g_i$ measures the exact schedule tardiness savings realizable by an admissible forward move. In a single-vehicle resequencing context, $g_i$ captures the overwhelming majority of the optimization headroom. Multiplying by predicted risk $p_i$ slightly shifts prioritization among equally actionable stops, but does not provide a statistically superior aggregate advantage over pure $g_i$.

---

## 1. RQ1: Chronological ML Prediction Pipeline Results

### Experimental Setup
- **Dataset**: 30 routes, 626 customer dropoff stops across 10 calendar dates.
- **Leakage Guard**: Enforced before training. Only ex-ante features available at vehicle departure time $t \le \text{decision\_time}$ were utilized:
  - Geographic: distance to depot, 2 km local stop density.
  - Operational: planned service seconds, package volume ($\text{cm}^3$), package count at stop.
  - Route context: total route stops, total planned route service minutes, departure hour, day of week.
  - Deadline slack: time until promised delivery deadline ($\tau_i - \text{departure\_time}$).
- **Chronological Split**:
  - **Train**: 380 stops across earliest 6 dates (60%).
  - **Validation (Calibration)**: 114 stops across next 2 dates (20%).
  - **Test (Held-out)**: 132 stops across latest 2 dates (20%).
- **Probability Calibration**: Platt scaling (logistic calibration) fitted strictly on validation split probabilities to optimize Brier score.

### Test Cohort Performance

| Model | ROC-AUC | PR-AUC | Brier Score | Log Loss | Expected Calibration Error (ECE) |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.841** | **0.484** | **0.1317** | **0.3943** | **0.0852** |
| **Random Forest** | **0.816** | **0.458** | **0.1467** | **0.4423** | **0.0983** |

*Conclusion for RQ1*: Delivery lateness risk is genuinely predictable from ex-ante features prior to departure without data leakage, yielding well-calibrated probabilities suitable for scaling actionability scores.

---

## 2. RQ2: Empirical Routing Benchmark on Held-Out Routes

### Factorial Evaluation Matrix
- **Routes Evaluated**: 6 held-out test routes (132 stops) across test dates 2018-07-09 and 2018-07-10.
- **Factors**:
  - Baseline route: Nearest Neighbor ($R^0$).
  - Targeting policies: RA, AB, RB, Slack, Deadline, Random (evaluated over 10 distinct random seeds).
  - Intervention budget: $B \in \{0.05, 0.10, 0.20, 0.30\}$.
  - Distance tolerance: $\delta \in \{0.00, 0.02, 0.05, 0.10\}$.
- **Total Factorial Runs**: 576 runs.

### Policy Performance Summary

| Policy | Targeting Formula | Avg $\Delta TT$ (min) | Avg TT Red (%) | Avg $\Delta NL$ | Avg Dist Inc (%) | Avg Relocations | Feasibility Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| **RA** | $p_i \cdot g_i$ | **+21.41 min** | **25.0%** | **+0.24** | +3.73% | 1.14 | **100.0%** |
| **AB** | $g_i$ | **+21.57 min** | **25.1%** | **+0.24** | +3.73% | 1.17 | **100.0%** |
| **RB** | $p_i$ | **+10.81 min** | **12.3%** | **+0.02** | +1.57% | 0.54 | **100.0%** |
| **Slack** | $\tau_i - c_i(R^0)$ | **+19.67 min** | **24.1%** | **+0.23** | +3.05% | 0.78 | **100.0%** |
| **Deadline** | $\tau_i$ | **+12.01 min** | **9.2%** | **+0.00** | +0.61% | 0.44 | **100.0%** |
| **Random** | Uniform (10 seeds) | **+6.37 min** | **7.9%** | **+0.05** | +1.20% | 0.42 | **100.0%** |

---

## 3. Paired Statistical Hypothesis Tests

To definitively test whether RA statistically outperforms competing policies, paired difference tests were performed across all corresponding $(route, \delta, B)$ experimental conditions:

$$\Delta TT(\text{RA}) - \Delta TT(\text{Competitor})$$

| Comparison | Mean Difference (min) | 95% Bootstrap Confidence Interval | Paired $t$-statistic | $p$-value ($t$-test) | $p$-value (Wilcoxon) | Statistically Significant? |
|---|---:|:---:|---:|---:|---:|:---:|
| **RA vs. AB** | **-0.155 min** | **[-0.413, +0.020]** | **-1.330** | **0.1867** | **0.5000** | **NO** ($p > 0.05$) |
| **RA vs. RB** | **+10.599 min** | **[+6.909, +14.887]** | **+5.132** | **< 0.0001** | **< 0.0001** | **YES** |
| **RA vs. Slack** | **+1.738 min** | **[+0.951, +2.620]** | **+4.001** | **0.0001** | **< 0.0001** | **YES** |
| **RA vs. Deadline** | **+9.404 min** | **[+6.376, +13.020]** | **+5.546** | **< 0.0001** | **< 0.0001** | **YES** |
| **RA vs. Random** | **+15.036 min** | **[+11.429, +19.241]** | **+7.545** | **< 0.0001** | **< 0.0001** | **YES** |

### Key Scientific Takeaways:
1. **The Fallacy of Pure Risk Targeting (RB)**: RB achieves less than half the tardiness reduction of RA (+10.8 min vs. +21.4 min). Machine learning models without spatial/routing awareness select stops that cannot be relocated due to distance limits, wasting scarce intervention capacity ($B$).
2. **Actionability Sufficiency**: AB and RA are statistically equivalent ($p = 0.1867$). When actionability $g_i$ is computed via full schedule propagation across all admissible forward moves, $g_i$ already encodes both deadline urgency and routing detour feasibility. Multiplying by $p_i$ does not harm performance, but it does not produce a significant advantage over $g_i$ alone in this single-vehicle context.

---

## 4. Invariant & Phenomenological Audit

| Audit Item | Observed Value | Expected Theoretical Behavior | Status |
|---|---:|---:|:---:|
| **Minimum $\Delta TT$ across all runs** | **0.0000 min** | $\Delta TT \ge 0$ (Proposition 1) | **VERIFIED (0 violations)** |
| **Route Feasibility Rate** | **100.0%** (576/576) | 100% | **VERIFIED** |
| **Occurrences of $\Delta NL < 0$ while $\Delta TT > 0$** | **32 runs** | Allowed by Proposition 1 remark | **OBSERVED & DOCUMENTED** |
| **Runs starting with 0 baseline tardiness** | **192 of 576 (33.3%)** | Expected on unconstrained routes | **TRANSPARENTLY REPORTED** |

### Clarification on $\Delta NL < 0$:
In 32 factorial runs, advancing a high-tardiness customer saved substantial total minutes of delay ($\Delta TT > 0$), but pushed one downstream customer slightly past their deadline window, temporarily increasing total late deliveries by 1 ($\Delta NL = -1$). This confirms the theoretical remark in Section 3 of the manuscript and validates the non-myopic nature of the schedule propagation engine.

---

## 5. Robustness Analysis: Clarke-Wright Savings Baseline

To ensure findings are not an artifact of Nearest Neighbor construction, all test routes were also evaluated using the **Clarke-Wright Savings heuristic** as $R^0$ ($B = 0.20$, $\delta = 0.05$):

| Policy | Avg $\Delta TT$ (min) | Avg TT Reduction (%) | Avg $\Delta NL$ | Avg Distance Increase (%) |
|---|---:|---:|---:|---:|
| **RA** | **+105.83 min** | **52.3%** | **+1.33** | +1.92% |
| **AB** | **+107.64 min** | **52.8%** | **+1.67** | +2.03% |
| **RB** | **+64.67 min** | **31.8%** | **+1.00** | +0.49% |
| **Slack** | **+105.14 min** | **52.4%** | **+1.50** | +1.98% |

*Finding*: Under Clarke-Wright baseline tours, PISR achieves even greater impact, eliminating over **105 minutes of tardiness (52% reduction)** while extending route distance by less than 2.0%. The parity between RA and AB remains consistent across baseline algorithms.

---

## 6. Guidance for Manuscript Revision & Defense

Based on this evidence, the following updates are recommended for `PTDLMP manuscript_09122026.tex`:

1. **Retract the claim that RA is categorically superior to AB**: State accurately that while RA significantly outperforms pure risk targeting (RB, $p < 0.0001$), operational slack ($p = 0.0001$), and random targeting ($p < 0.0001$), it achieves parity with pure actionability (AB, $p = 0.187$).
2. **Highlight the Value of Actionability ($g_i$)**: Frame $g_i$ as the indispensable routing filter that rescues predictive machine learning from selecting futile, geographically impossible interventions.
3. **Include the Paired Confidence Interval Table**: Replace preliminary synthetic numbers with Table 3 of this report, showcasing the 576 held-out challenge runs and paired statistical significance.
4. **Document the 32 Cases of $\Delta NL < 0$**: Present these 32 cases as empirical evidence supporting the theoretical distinction between minimizing total tardiness (smooth objective) and minimizing customer lateness count (step-function objective).
