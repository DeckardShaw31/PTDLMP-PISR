# PISR Software Verification & Workability Report (Synthetic Benchmark)

**Audit & Verification Date:** 2026-09-22  
**Evaluation Type:** Synthetic Software Verification & Mathematical Invariant Audit  
**Document Context:** Evaluation of PTDLMP-PISR routing kernel based on `design.md` and `PTDLMP manuscript_09122026.tex`.  
**Overall Verdict:** **CONDITIONAL WORKABILITY**. The core routing algorithm and mathematical invariants are verified in software; empirical claims of policy superiority (RA > AB) and real-world effectiveness remain unproven until tested on official challenge data with trained ML predictions.

---

## 1. Executive Summary & Verdict

### What Is Verified:
1. **Mathematical Invariant Holds (Proposition 1)**: Across all unit tests and 180 synthetic factorial runs, Selective Forward Relocation (SFR) strictly satisfied $TT(R') \leq TT(R^0)$ in 100% of accepted moves. Total route tardiness never worsened.
2. **Cumulative Distance Enforcement**: The distance tolerance constraint $D(R') \leq (1 + \delta) D(R^0) + \epsilon$ was preserved deterministically by the independent hard validator across all trials.
3. **Downstream Delay Propagation**: The schedule propagation engine correctly accounts for downstream impacts: advancing stop $i$ evaluates subsequent delays inflicted on stops $k > i$.
4. **Software Resilience**: The hardened route validator rejects malformed routes (unknown nodes, NaNs/infinities, missing deadlines, missing matrix edges) with clean `feasible=False` outcomes rather than unhandled crashes.

### What Remains Conditional & Unproven:
1. **No Demonstrated Superiority of RA over AB**: In the preliminary synthetic simulations, RA ($p_i \cdot g_i$) and AB ($g_i$) produced **identical results across every tested scenario** because synthetic $p_i$ values were collinear with deadline tightness and actionability. The manuscript's central hypothesis that RA outperforms AB has **not yet been demonstrated**.
2. **Synthetic Fixtures Only**: The initial 180 runs evaluated four handcrafted synthetic routes, not the official Amazon Last Mile Routing Challenge dataset.
3. **No Trained ML Model**: Values for $p_i$ were heuristic proxies, not out-of-sample calibrated probabilities from an ex-ante ML classifier.
4. **Limited Tardiness Impact**: The headline "100% tardiness reduction" represented the elimination of just 2.1 minutes of tardiness on a single 14-stop toy instance (`Arterial_Corridor`).
5. **Zero-Tardiness Inactivity**: 45 of the 180 runs (25%) began on an instance with zero baseline tardiness (`Urban_Two_Cluster`), where no relocation could occur by definition.

---

## 2. Transparent Audit of Preliminary Simulation Limitations

To maintain scientific integrity, the following overstatements from earlier preliminary drafts are formally retracted and documented:

| Item | Preliminary Claim | Empirical Reality & Audit Finding |
|---|---|---|
| **Data Source** | "Amazon benchmark evaluation" | Four handcrafted synthetic fixtures (12–14 stops). The official Amazon Challenge JSON dataset was not loaded. |
| **Prediction $p_i$** | "Predictive risk $p_i$" | Handcrafted heuristic based directly on deadline tightness; no ML training, validation split, or calibration was performed. |
| **RA Superiority** | "RA outperformed RB and AB" | **Untrue in tested runs**. RA and AB had identical numerical scores, candidate selections, and tardiness reductions in all 180 runs. |
| **$\Delta NL < 0$ Occurrence** | "Observed in tight routes" | No run in the 180 factorial runs produced $\Delta NL < 0$. (Software unit test 3 proves it is mathematically possible, but it did not occur in the reported experiment). |
| **100% Reduction Scope** | "Substantial reductions up to 100%" | Removing 2.1 minutes of tardiness on a 14-stop synthetic route. On other instances, reduction was 0.0% to 12.4%. |
| **Zero-Tardiness Runs** | Unreported in aggregate statistics | 45 of 180 runs evaluated an instance starting with 0 tardiness, skewing aggregate percentage metrics. |
| **Untested Parameters** | Claims regarding $\delta < 3\%$, $\delta = 15\%$, and $B > 30\%$ | The tested grid only covered $B \in \{0.10, 0.20, 0.30\}$ and $\delta \in \{0.05, 0.10, 0.20\}$. Values outside this range were extrapolations. |

---

## 3. Synthetic Benchmark Results (Reproducible Audit)

The table below reports the exact reproducible numbers from the 180 synthetic factorial runs ($4 \text{ instances} \times 3 \text{ budgets} \times 3 \text{ tolerances} \times 5 \text{ policies}$):

### Factorial Aggregate Metrics

| Policy | Targeting Score Formula | Avg $\Delta TT$ (min) | Avg TT Reduction (%) | Avg Late Avoided ($\Delta NL$) | Avg Distance Increase (%) | Candidate Relocation Rate (%) |
|---|---|---:|---:|---:|---:|---:|
| **RA** | $p_i \cdot g_i$ | +1.13 min | 27.1% | +0.25 | +4.08% | 25.0% |
| **AB** | $g_i$ | +1.13 min | 27.1% | +0.25 | +4.08% | 25.0% |
| **RB** | $p_i$ | +0.95 min | 26.4% | +0.25 | +3.61% | 19.4% |
| **Slack** | $\tau_i - c_i(R^0)$ | +0.75 min | 18.0% | +0.17 | +2.72% | 11.1% |
| **Random** | Uniform Random | +0.48 min | 19.4% | +0.19 | +2.32% | 9.3% |

*Critical Observation*: RA and AB are completely identical in this synthetic experiment. To demonstrate whether RA is superior to AB, $p_i$ must be generated by a trained ML model that incorporates non-geographic features (package volume, historical courier delivery profile, service duration) that decouple risk from geometric actionability.

### Primary Setting Breakdown ($B = 20\%$, $\delta = 10\%$)

| Instance | Baseline TT | Base NL | Policy | Final TT | $\Delta TT$ (%) | Final NL | $\Delta NL$ | $\Delta D$ (%) | Accepted / Admitted |
|---|---:|---:|---|---:|---:|---:|---:|---:|:---:|
| **Amazon_AMZ_001** (12 stops) | 2.9 min | 1 | **RA / AB / RB / Slack / Random** | 2.9 min | **0.0%** | 1 | +0 | +0.00% | 0/2 |
| **Loop_Circuit** (12 stops) | 25.8 min | 2 | **RA / AB / RB / Slack** | 22.6 min | **-12.4%** | 2 | +0 | +8.47% | 1/2 |
| | | | **Random** | 25.8 min | **0.0%** | 2 | +0 | +0.00% | 0/2 |
| **Urban_Two_Cluster** (14 stops) | 0.0 min | 0 | **All Policies** | 0.0 min | **0.0%** | 0 | +0 | +0.00% | 0/2 |
| **Arterial_Corridor** (14 stops) | 2.1 min | 1 | **All Policies** | 0.0 min | **-100.0%** | 0 | +1 | +6.19% | 1/2 |

---

## 4. Software Verification & Hardened Validator Status

All 17 automated unit tests pass (`python -m pytest code/tests -v`):

```
code/tests/test_schedule_and_invariants.py::test_hand_calculated_3_customer_route PASSED
code/tests/test_schedule_and_invariants.py::test_downstream_delay_evaluated_globally PASSED
code/tests/test_schedule_and_invariants.py::test_tt_decreases_while_nl_increases PASSED
code/tests/test_schedule_and_invariants.py::test_distance_limit_violation_rejection PASSED
code/tests/test_schedule_and_invariants.py::test_lexicographic_tie_breaking PASSED
code/tests/test_schedule_and_invariants.py::test_validator_detects_violations PASSED
code/tests/test_schedule_and_invariants.py::test_budget_zero PASSED
code/tests/test_schedule_and_invariants.py::test_proposition_1_invariant PASSED
code/tests/test_schedule_and_invariants.py::test_cumulative_distance_tolerance_invariant PASSED
code/tests/test_schedule_and_invariants.py::test_amazon_sample_instance_end_to_end PASSED
code/tests/test_schedule_and_invariants.py::test_no_admissible_move_leaves_route_unchanged PASSED
code/tests/test_schedule_and_invariants.py::test_asymmetric_travel_times_respected PASSED
code/tests/test_schedule_and_invariants.py::test_validator_rejects_unknown_nodes_without_crashing PASSED
code/tests/test_schedule_and_invariants.py::test_validator_rejects_nan_and_inf PASSED
code/tests/test_schedule_and_invariants.py::test_validator_rejects_missing_promised_times PASSED
code/tests/test_schedule_and_invariants.py::test_validator_rejects_missing_matrix_edges PASSED
code/tests/test_schedule_and_invariants.py::test_slack_vs_deadline_ranking_divergence PASSED
```

### Hardened Validator Checks (Conforming to Section 7.1 of `design.md`):
1. **Unknown / Unassigned Nodes**: Route stops not present in `instance.stops` trigger immediate rejection without `KeyError` or crashes.
2. **Coordinate & Duration Integrity**: Out-of-bounds, infinite, or `NaN` values in `lat`, `lng`, or `service_seconds` are rejected.
3. **Promised Delivery Deadlines**: Routes containing customer dropoffs with `promised_time=None` are rejected when `require_promised_times=True`.
4. **Matrix Edge Completeness**: Any sequential edge $(u, v)$ missing from `travel_times` or `distances` is rejected when `require_complete_matrices=True`.
5. **Decoupled Operational Slack**: True operational slack ($\tau_i - c_i(R^0)$) is implemented and verified to diverge from raw deadline ($\tau_i$).

---

## 5. Roadmap to Full Scientific Evidence

To elevate this codebase from software verification to rigorous peer-reviewed scientific evidence, the following four milestones are currently in progress:

1. **Official Amazon Last Mile Adapter**: Full parser for the nested official schema (`routes.json`, `package_data.json`, `travel_times.json`, `actual_sequences.json`), with multi-package stop aggregation and UTC time-window handling.
2. **RQ1 Chronological ML Prediction Pipeline**: Chronological 60/20/20 date split, ex-ante features under strict leakage guard ($t \le \text{decision\_time}$), and probability calibration (Platt/isotonic) optimizing Brier score.
3. **RQ2 Held-Out Routing Evaluation**: Execution of PISR across a large cohort of held-out real routes with multiple random seeds, Clarke-Wright baseline comparisons, and return-to-depot sensitivity checks.
4. **Statistical Hypothesis Testing**: Paired Wilcoxon signed-rank and paired t-tests, 95% bootstrap confidence intervals for $\Delta TT$ and $\Delta NL$, establishing whether RA achieves statistically significant superiority over AB and RB.

---

## 6. Honest Recommendation for Your Advisor / Teacher

> **What to tell your teacher:**
> "The core algorithmic kernel of PISR (Algorithm 1) and Proposition 1 are mathematically correct and verified in code. Resequencing with schedule propagation successfully prevents downstream disruptions and enforces distance limits.
> 
> However, our initial 180 simulation runs were on synthetic fixtures with heuristic probabilities, where RA and AB yielded identical results. Therefore, **we do not yet have scientific proof that RA outperforms AB, nor is the work ready for publication submission**.
> 
> We are actively implementing the official challenge adapter, training a genuine calibrated ML model on chronological splits, and testing on held-out routes to establish empirical validity."