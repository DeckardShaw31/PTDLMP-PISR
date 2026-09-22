# Prediction-Informed Selective Route Resequencing (PISR) for Promised-Time Delivery Lateness Mitigation

This repository contains the official codebase, experimental pipeline, reproduction scripts, and manuscript for the study:
**"Prediction-Informed Selective Route Resequencing for Promised-Time Delivery Lateness Mitigation under Limited Intervention Capacity" (PTDLMP-PISR)**.

---

## 📌 Overview

Last-mile delivery operations face stringent promised delivery time windows. While machine-learning delay models can estimate customer-level lateness risk, risk identification alone does not prescribe operational routing actions.

The **Prediction-Informed Selective Route Resequencing (PISR)** framework bridges ex-ante lateness prediction with selective route resequencing under constrained intervention capacity:
1. **Ex-Ante Lateness Risk Estimation ($p_i$)**: Estimates lateness probability prior to route departure ($t \le \text{departure\_time}$) using strictly leakage-free features.
2. **Actionability Assessment ($g_i$)**: Evaluates potential baseline tardiness reduction subject to a cumulative route-distance tolerance ($\delta$).
3. **Targeting Prioritization**: Prioritizes customers using Risk-Based ($RB$), Actionability-Based ($AB$), combined Risk-Actionability ($RA$: $RAS_i = p_i \cdot g_i$), Slack, Deadline, or Random targeting.
4. **Selective Forward Relocation (SFR)**: Algorithm 1 sequentially moves selected customers forward subject to strict schedule monotonicity ($TT(R') < TT(R^c)$).
5. **Counterfactual Route Outcome Evaluation**: Propagates arrivals and completions for the whole route to quantify avoided lateness ($\Delta NL$), reduced tardiness ($\Delta TT$), and routing distance change ($\Delta D$).

---

## 🗂 Repository Structure

```text
PTDLMP-PISR/
├── PTDLMP manuscript_09122026.tex   # LaTeX manuscript source (clean, publication-ready)
├── design.md                         # Detailed system design, mathematical invariants, and audit protocol
├── LICENSE                           # MIT License with CC BY-NC 4.0 Dataset Notice
├── .gitignore                        # Git exclusion rules
├── figures/                          # Publication-grade figures
│   ├── fig1_framework.png            # PISR framework architecture
│   ├── fig2_policy_comparison.png    # Quantitative policy comparison bar charts
│   └── fig3_route_before_after.png   # Amazon Challenge route resequencing map
├── code/
│   ├── data/
│   │   ├── raw/                      # Official Amazon Challenge dataset (AWS Open Data)
│   │   │   ├── route_data.json       # 13 routes, 2,051 total stops (2,038 drop-off)
│   │   │   ├── package_data.json     # 3,129 packages with dimensions and service times
│   │   │   ├── travel_times.json     # 37,000+ directed travel-time edges
│   │   │   └── actual_sequences.json # Official driver delivery sequences
│   │   └── challenge/                # Secondary benchmark instances
│   ├── src/
│   │   ├── data/
│   │   │   ├── schemas.py            # Canonical dataclasses (Stop, RouteInstance, etc.)
│   │   │   ├── amazon_adapter.py     # Official Amazon Challenge format adapter
│   │   │   └── amazon_loader.py      # Format parser and multi-package aggregator
│   │   ├── features/
│   │   │   ├── build_features.py     # Ex-ante feature extraction and ground truth labeling
│   │   │   └── leakage_guard.py      # Strict leakage assertion and chronological splitting
│   │   ├── prediction/
│   │   │   ├── calibrate.py          # Platt probability calibration
│   │   │   ├── evaluate.py           # Multi-metric prediction evaluation (Brier, ECE, AUC)
│   │   │   └── train.py              # Chronological ML training pipeline
│   │   ├── routing/
│   │   │   ├── distance.py           # Haversine distance and travel-time lookups
│   │   │   ├── schedule.py           # Route-time propagation engine (Eqs. 1–4)
│   │   │   ├── baseline.py           # Deterministic Nearest Neighbor and Clarke-Wright baselines
│   │   │   ├── validator.py          # Independent 10-point hard route feasibility validator
│   │   │   ├── actionability.py      # Actionability scores g_i, policy ranking, budget selection
│   │   │   └── sfr.py                # Algorithm 1: Selective Forward Relocation
│   │   └── evaluation/
│   │       ├── statistics.py         # Cell-level & route-clustered paired hypothesis tests with Holm-Bonferroni
│   │       └── tables.py             # Markdown and LaTeX publication table formatters
│   ├── tests/
│   │   ├── test_amazon_adapter.py    # Official Amazon adapter tests
│   │   ├── test_prediction_pipeline.py # Leakage guard and prediction pipeline tests
│   │   └── test_schedule_and_invariants.py # Mathematical invariant tests (Proposition 1)
│   ├── experiments/
│   │   ├── run_rq1_prediction.py     # RQ1 chronological prediction experiment
│   │   ├── run_routing_benchmark.py  # RQ2 factorial routing benchmark
│   │   ├── verify_workability.py     # Rapid invariant verification
│   │   └── generate_figures.py       # High-resolution figure generator
│   ├── outputs/                      # Experimental run logs, metrics, and summaries
│   │   ├── rq1_prediction_results.json
│   │   ├── routing_benchmark_runs.csv
│   │   ├── robustness_clark_wright_runs.csv
│   │   └── benchmark_summary.json
│   └── reports/
│       ├── scientific_evaluation_report.md # Comprehensive empirical benchmark report
│       └── workability_assessment.md       # Software verification & audit
└── report/
    └── thesis_provenance_audit.md    # Academic provenance and methodological comparison audit
```

---

## 🔒 Dataset Integrity & Checksums

The dataset in `code/data/raw/` is derived directly from the AWS Open Data repository for the Amazon Last-Mile Routing Challenge:

| File Name | File Size | SHA-256 Checksum |
|---|---:|---|
| `actual_sequences.json` | 21,937 bytes | `174625ccf3e0aed722f1129feee558ee5672372a3757d475072f7e04c300c2db` |
| `package_data.json` | 717,877 bytes | `27fc132c7f9de5498301844e27e0d57eba7898bd2f2a247c51742d5b87400c95` |
| `route_data.json` | 178,703 bytes | `5b6292cf03ed6abc1a8e37a61405c05e14751dc2820a0856f04fd14563f362a3` |
| `travel_times.json` | 4,310,705 bytes | `3a96c295d70f1eb372c89126e77e69f60d7bb1b9d38b2a913777d9341d560ab6` |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Requirements
- Python 3.10+
- Dependencies: `pip install numpy pandas scipy scikit-learn pytest matplotlib`

### 2. Run Automated Invariant Unit Tests
All 23 mathematical invariant and system tests can be executed via:
```bash
python -m pytest code/tests -v
```

### 3. Run RQ1: Chronological ML Prediction Pipeline
Trains ex-ante models under strict chronological date splits (60% Train, 20% Val, 20% Test) with Platt probability calibration:
```bash
python code/experiments/run_rq1_prediction.py
```
Outputs are written to `code/outputs/rq1_prediction_results.json`.

### 4. Run RQ2: Routing Resequencing Benchmark
Executes 288 factorial runs on held-out Amazon routes across 4 budgets ($B \in \{5\%, 10\%, 20\%, 30\%\}$) and 4 distance tolerances ($\delta \in \{0\%, 2\%, 5\%, 10\%\}$):
```bash
python code/experiments/run_routing_benchmark.py
```
This generates `routing_benchmark_runs.csv`, `robustness_clark_wright_runs.csv`, and `benchmark_summary.json`.

---

## 📊 Summary of Empirical Results (Amazon Challenge Exploratory Cohort)

### Policy Comparison (Held-out Routes, 288 Factorial Runs)

| Policy | Targeting Formula | Avg $\Delta TT$ (min) | Avg TT Red (%) | Avg $\Delta NL$ (late avoided) | Avg Dist Inc (%) | Feasibility Rate |
|---|---|---:|---:|---:|---:|---:|
| **RA** | $p_i \cdot g_i$ | **+1,417.91 min** | **12.1%** | **+3.65** | +2.05% | **100.0%** |
| **AB** | $g_i$ | **+1,434.54 min** | **12.2%** | **+3.52** | +2.11% | **100.0%** |
| **RB** | $p_i$ | **+367.63 min** | **2.9%** | **+0.83** | +1.93% | **100.0%** |
| **Slack** | $\tau_i - c_i(R^0)$ | **+403.50 min** | **4.0%** | **+0.40** | +1.48% | **100.0%** |
| **Deadline** | $\tau_i$ | **+272.05 min** | **2.2%** | **+0.73** | +2.28% | **100.0%** |
| **Random** | Uniform Random (10 seeds) | **+299.49 min** | **2.5%** | **+0.72** | +1.95% | **100.0%** |

### Statistical Inference: Cell-Level vs. Route-Clustered Tests

$$\Delta TT(\text{RA}) - \Delta TT(\text{Competitor})$$

| Comparison | Mean Diff (min) | Route-Level 95% Bootstrap CI | Cell-Level $p$ ($N=48$) | Route-Clustered $p$ ($N=3$) | Holm-Bonferroni Adj. $p$ |
|---|---:|:---:|---:|---:|---:|
| **RA vs. AB** | **-16.62 min** | [-55.48, +3.95] | $0.3626$ | $0.4825$ | $0.4825$ |
| **RA vs. RB** | **+1,050.28 min** | [+830.29, +1,239.51] | $< 0.0001$ | $0.0126$ | $0.0631$ |
| **RA vs. Slack** | **+1,014.41 min** | [+394.52, +1,475.30] | $< 0.0001$ | $0.0877$ | $0.1754$ |
| **RA vs. Deadline** | **+1,145.87 min** | [+834.88, +1,414.37] | $< 0.0001$ | $0.0210$ | $0.0839$ |
| **RA vs. Random** | **+1,118.42 min** | [+759.68, +1,299.85] | $< 0.0001$ | $0.0248$ | $0.0839$ |

*Defensible Scientific Conclusion*: RA shows large operational improvements over risk-only and conventional targeting (averaging >1,000 minutes of saved tardiness), but none of the route-clustered comparisons remain statistically significant after Holm–Bonferroni correction because only three independent test routes are available ($N=3$). Formal confirmation of statistical significance under route clustering requires expanding the held-out route sample based on a prospective power analysis before writing definitive results into `main.tex`.

---

## 📜 License & Attribution Notice

- **Software Code**: Licensed under the [MIT License](LICENSE).
- **Amazon Last-Mile Routing Challenge Dataset**:
  The dataset utilized in this research is provided by Amazon.com, Inc. and its affiliates under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
  
  **Citation**:
  > Merchán, D., Pachon, J., Arora, J., et al. (2021). *2021 Amazon Last-Mile Routing Research Challenge Dataset*. Amazon.com, Inc. or its affiliates. Available via AWS Open Data: https://registry.opendata.aws/amazon-last-mile-challenges/
