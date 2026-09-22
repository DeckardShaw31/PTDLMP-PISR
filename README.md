# Prediction-Informed Selective Route Resequencing (PISR) for Promised-Time Delivery Lateness Mitigation

This repository contains the official codebase, experimental pipeline, reproduction scripts, and manuscript for the study:
**"Prediction-Informed Selective Route Resequencing for Promised-Time Delivery Lateness Mitigation under Limited Intervention Capacity" (PTDLMP-PISR)**.

---

## 📌 Overview

Last-mile delivery operations face stringent promised delivery time windows. While machine-learning delay models can estimate customer-level lateness risk, identification alone does not prescribe operational routing actions.

The **Prediction-Informed Selective Route Resequencing (PISR)** framework connects ex-ante lateness prediction with selective route resequencing under constrained intervention capacity:
1. **Ex-Ante Lateness Risk Estimation ($p_i$)**: Estimates lateness probability prior to route execution using leakage-free information.
2. **Actionability Assessment ($g_i$)**: Evaluates potential baseline tardiness reduction subject to a cumulative route-distance tolerance ($\delta$).
3. **Targeting Prioritization**: Prioritizes customers using Risk-Based ($RB$), Actionability-Based ($AB$), or combined Risk-Actionability ($RA$: $RAS_i = p_i \cdot g_i$).
4. **Selective Forward Relocation (SFR)**: Algorithm 1 sequentially moves selected customers forward subject to strict tardiness reduction ($TT(R') < TT(R^c)$).
5. **Counterfactual Route Outcome Evaluation**: Propagates arrivals and completions for the whole route to quantify avoided lateness ($\Delta NL$), reduced tardiness ($\Delta TT$), and routing distance change ($\Delta D$).

---

## 🗂 Repository Structure

```text
PTDLMP-PISR/
├── PTDLMP manuscript_09122026.tex   # LaTeX manuscript source (clean, publication-ready)
├── design.md                         # Detailed system design, invariants, and audit protocol
├── LICENSE                           # MIT License
├── .gitignore                        # Git exclusion rules
├── figures/                          # Publication-grade figures
│   ├── fig1_framework.png            # PISR framework architecture
│   ├── fig2_policy_comparison.png    # Quantitative policy comparison bar charts
│   └── fig3_route_before_after.png   # Amazon Challenge route resequencing map
├── code/
│   ├── data/                         # Sample route and order datasets (Amazon Challenge, Cainiao)
│   ├── src/
│   │   ├── data/
│   │   │   ├── schemas.py            # Canonical dataclasses (Stop, RouteInstance, etc.)
│   │   │   └── amazon_loader.py      # Amazon Last-Mile Challenge format parser
│   │   └── routing/
│   │       ├── distance.py           # Haversine distance and travel times
│   │       ├── schedule.py           # Sequential route-time propagation (Eqs. 1–4)
│   │       ├── baseline.py           # Deterministic Nearest Neighbor baseline (Eq. 6)
│   │       ├── validator.py          # Independent 10-point hard route feasibility validator
│   │       ├── actionability.py      # Actionability scores g_i, policy ranking, budget selection
│   │       └── sfr.py                # Algorithm 1: Selective Forward Relocation
│   ├── tests/
│   │   └── test_schedule_and_invariants.py # 12 automated mathematical invariant tests
│   ├── experiments/
│   │   ├── verify_workability.py     # 180-run factorial verification simulation
│   │   └── generate_figures.py       # High-resolution figure generation script
│   └── reports/
│       └── workability_assessment.md # Comprehensive workability assessment report
└── report/
    └── thesis_provenance_audit.md    # Academic provenance and methodological comparison audit
```

---

## 🚀 Quickstart & Verification

### 1. Requirements
- Python 3.10+
- Dependencies: `pytest`, `matplotlib`, `numpy`

### 2. Run Automated Invariant Unit Tests
All 12 mathematical invariants (Proposition 1, cumulative distance bound, tie-breaking, hand-calculated 3-customer baseline, downstream delays) can be run via:
```bash
python -m pytest code/tests -v
```

### 3. Run Factorial Workability Simulation
Executes a 180-run factorial grid across 4 benchmark instances, 5 policies (`RA`, `AB`, `RB`, `Slack`, `Random`), 3 budgets ($B$), and 3 distance tolerances ($\delta$):
```bash
python code/experiments/verify_workability.py
```
This generates the summary report in `code/reports/workability_assessment.md`.

### 4. Generate Publication Figures
To re-generate the high-resolution charts in `figures/`:
```bash
python code/experiments/generate_figures.py
```

---

## 📊 Summary of Experimental Results

| Policy | Targeting Score | Avg $\Delta TT$ (Tardiness Saved) | Avg TT Reduction (%) | Avg Late Avoided ($\Delta NL$) | Distance Increase ($\Delta D$) | Relocation Acceptance Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **RA (Proposed)** | $p_i \cdot g_i$ | **+1.13 min** | **27.1%** | **+0.25** | **+4.08%** | **25.0%** |
| **AB (Actionability)** | $g_i$ | **+1.13 min** | **27.1%** | **+0.25** | **+4.08%** | **25.0%** |
| **RB (Pure Risk)** | $p_i$ | **+0.95 min** | **26.4%** | **+0.25** | **+3.61%** | **19.4%** |
| **Slack (Heuristic)** | $\tau_i - a_i$ | **+0.75 min** | **18.0%** | **+0.17** | **+2.72%** | **11.1%** |
| **Random (Baseline)** | Uniform Random | **+0.48 min** | **19.4%** | **+0.19** | **+2.32%** | **9.3%** |

- **Proposition 1 Guarantee**: 100% of final routes generated by Algorithm 1 satisfy $TT(R') \leq TT(R^0)$.
- **Intervention Efficiency**: $RA$ targeting achieves a **25.0% acceptance rate** (>2.5× higher than Random), saving up to 100% of route tardiness on congested commercial routes.

---

## 📜 License and Terms of Use

- **Source Code**: All software code in the `code/` folder is licensed under the [MIT License](LICENSE).
- **Manuscript & Documentation**: The text, methodology, and LaTeX manuscript files are Copyright © 2026 Authors. All rights reserved.
- **Datasets**: The Amazon Last Mile Routing Challenge, Xe Dù Ho Chi Minh City, and Cainiao LaDe datasets are the property of their respective creators and are used under their respective research terms.
