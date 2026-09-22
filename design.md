# PISR coding and experiment design

## 1. Purpose of this document

This document turns the framework in `main.tex` into an implementable and auditable research system. The system must answer four separate questions:

1. **Is the input data suitable?** Does it contain a defensible promised time, an outcome or outcome proxy, route groups, locations, travel information, and an ex-ante feature set?
2. **Is a generated route feasible?** Does it serve every assigned stop exactly once, respect the route model, remain within the distance tolerance, and have a reproducible propagated schedule?
3. **Is the intervention useful?** Does it reduce total tardiness and/or late deliveries at an acceptable routing cost?
4. **Is the evidence scientifically valid?** Are prediction, tuning, route construction, and testing separated chronologically and free from post-outcome leakage?

These questions must not be collapsed into one vague “good route” decision. A route can be mechanically feasible but operationally worse, and an intervention can look successful while relying on leaked information.

## 2. Immediate recommendation

Do **not** combine the three listed datasets row-by-row. They describe different operations, use different identifiers, and do not share customers, couriers, dates, or coordinate systems. Treat them as candidate data sources or separate studies.

The recommended order is:

1. **Prototype and primary route experiment: Amazon Last Mile Routing Research Challenge data.** It has route, stop, package, time-window, service-time, actual-sequence, and pairwise travel-time structures. It is the closest match to the route-resequencing component.
2. **Secondary promised-lateness prediction study: Xe dù Ho Chi Minh City data.** It has `expectedDeliveryTime` and `deliveredAt`, so the lateness label is direct. However, it represents orders with sender and receiver locations and does not directly provide the delivery-only route/travel-time structure assumed in `main.tex`.
3. **Optional scale/robustness study: LaDe-D.** It has large courier delivery-event data, but its published delivery table does not expose an explicit promised-delivery deadline. It cannot support “promised-time lateness” without a clearly justified SLA or deadline proxy.

### Important manuscript decision

Amazon does not provide a directly observed delivery-completion timestamp for each package in the documented challenge files. Arrival/completion times can instead be reconstructed from departure time, actual sequence, pairwise travel times, and planned service times. If Amazon is the primary dataset, the manuscript must call the target a **route-propagated time-window violation proxy**, not claim that it is a directly observed promised-time outcome.

If the teacher requires a directly observed promised-time label, use Xe dù for RQ1 and explicitly present the Amazon experiment as a separate route-simulation study. Do not claim a single end-to-end empirical pipeline across the two datasets unless a valid transfer-learning design is added and defended.

## 3. Dataset suitability audit

Before model development, create a machine-generated `reports/data_audit.md` and `outputs/audit_summary.csv`. The audit should determine whether each dataset passes the minimum contract below.

### 3.1 Minimum data contract

Every order/stop used by the end-to-end PISR experiment needs:

| Canonical field | Meaning | Required |
|---|---|---:|
| `route_id` | Stable courier-day or vehicle-trip group | Yes |
| `customer_id` | Unique stop within the route | Yes |
| `route_date` | Date used for chronological splitting | Yes |
| `decision_time` | Time at which features are assumed available | Yes |
| `promised_time` | Latest acceptable service time | Yes |
| `actual_or_proxy_time` | Outcome used only to form historical labels | Training/evaluation only |
| `latitude`, `longitude` | Delivery-stop coordinates | Yes |
| `depot_id` or depot coordinates | Common route origin | Yes for the current model |
| `service_seconds` | Stop service duration | Yes; defensible imputation allowed |
| `travel_seconds(i,j)` | Directed travel time between route nodes | Yes; estimated values must be documented |
| `distance(i,j)` | Directed/undirected route distance | Yes |
| `courier_id` | Courier identifier | Desirable |
| ex-ante package/order attributes | Prediction features | Yes |

The audit must fail a dataset for the end-to-end study when a required field is absent and no scientifically defensible construction is available.

### 3.2 Candidate-dataset mapping

#### Amazon

- `route_id`: route identifier.
- `route_date`: `date_YYYY_MM_DD`.
- route origin: stop with type `Station`.
- customers: stops with type `Dropoff`.
- coordinates: stop `lat`, `lng`.
- promised time: package `time_window.end_time_utc`.
- service duration: sum package `planned_service_time_seconds` at a stop.
- travel time: `travel_times.json`.
- distance: calculate Haversine distance between stop coordinates unless a road-distance matrix is later supplied.
- historical sequence: `actual_sequences.json`; use only to construct historical outcome proxies and validation benchmarks, never as an ex-ante feature or baseline route.
- package-to-stop aggregation: when multiple packages share a stop, use the earliest package deadline as the stop deadline for the strict primary analysis. Also report a package-level sensitivity analysis.

Audit questions:

- What fraction of packages and stops have no time window?
- Does every route have exactly one station?
- Is the directed travel-time matrix complete for all nodes?
- Are all time-window timestamps aligned with the route date and departure time?
- How often is the derived actual-sequence arrival later than the time-window end?
- Does using the earliest deadline at a multi-package stop produce enough positive labels for prediction?

#### Xe dù Ho Chi Minh City

- direct label: `deliveredAt > expectedDeliveryTime`.
- date: derive from `createdAt` or the decision epoch agreed with the teacher.
- grouping candidate: `shipper` plus local calendar date.
- delivery coordinates: `receiverLat`, `receiverLng`.
- package/order features: weight, service type, creation time, geographic features, and strictly historical shipper statistics.

Major mismatch: every record also has a sender location, indicating a pickup-and-delivery operation. The present PISR model assumes all packages are already loaded at one station and only delivery stops are resequenced. Ignoring sender stops could create impossible routes. Therefore, this dataset passes RQ1 readily but passes the route experiment only if inspection shows a defensible common-depot/delivery-only subset or the mathematical model is extended to preserve pickup-before-delivery precedence.

#### LaDe-D

- route group candidate: `courier_id` plus service date.
- delivery location: `lng`, `lat`.
- outcome time: `delivery_time`.
- available event information includes acceptance and GPS event times/locations.

Major mismatch: the published delivery schema does not contain an explicit promised time. Do not silently treat acceptance time plus an arbitrary duration as a promise. A derived SLA is acceptable only as a separately named sensitivity experiment, with the SLA estimated using training data only and agreed with the teacher before coding.

### 3.3 Dataset go/no-go rule

A dataset is approved for the end-to-end experiment only if all of these are true:

- at least 95% of retained rows have valid route, time, and coordinate fields;
- each retained route has a defined start node;
- every retained customer has a promised time or an explicitly approved proxy;
- travel time and service duration can be propagated for every route edge;
- a chronological split is possible;
- positive and negative lateness cases exist in every model-development period;
- the route definition matches the manuscript assumptions;
- no post-execution field is needed to construct prediction features or the baseline route.

The 95% threshold is a proposed engineering threshold, not a fact from the datasets. Report exclusions rather than hiding them.

## 4. Proposed system architecture

```text
Raw dataset
    |
    v
Dataset-specific adapter -----> data audit and exclusion report
    |
    v
Canonical order/stop tables + route matrices
    |
    +------> chronological splitter
    |             |
    |             v
    |       feature builder ---> calibrated risk model p_i
    |
    v
Ex-ante baseline route R0 (nearest neighbor / Clarke-Wright)
    |
    v
Schedule propagation ---> baseline NL, TT, and actionability g_i
    |                                      |
    +----------------------+---------------+
                           v
                RB / AB / RA candidate ranking
                           |
                           v
                 Selective Forward Relocation
                           |
                           v
                 route feasibility validator
                           |
                           v
             counterfactual outcomes and statistics
```

### 4.1 Suggested repository layout

```text
configs/
  experiment.yaml
data/
  raw/                 # not committed when license/size forbids it
  interim/
  processed/
outputs/
  models/
  predictions/
  routes/
  metrics/
reports/
figures/
src/
  data/
    amazon_adapter.py
    xedu_adapter.py
    lade_adapter.py
    audit.py
    schemas.py
  features/
    build_features.py
    leakage_guard.py
  prediction/
    train.py
    calibrate.py
    evaluate.py
  routing/
    distance.py
    baseline.py
    schedule.py
    actionability.py
    sfr.py
    validate.py
  experiments/
    run_prediction.py
    run_routing.py
    run_robustness.py
  evaluation/
    operational_metrics.py
    statistics.py
    plots.py
tests/
  test_schedule.py
  test_baseline.py
  test_actionability.py
  test_sfr.py
  test_validator.py
  test_no_leakage.py
```

Keep raw-data download instructions separate from code and record dataset version, file names, sizes, and checksums in `data/manifest.csv`.

## 5. Canonical processed tables

Use explicit tables rather than passing dataset-specific columns throughout the system.

### 5.1 `stops.parquet`

One row per route stop:

```text
route_id, route_date, stop_id, courier_id, stop_type,
latitude, longitude, promised_time, service_seconds,
decision_time, observed_time, outcome_source
```

`outcome_source` must be one of `observed`, `propagated_actual_sequence`, or `derived_sla`. This prevents a proxy from being reported as an observed value.

### 5.2 `packages.parquet`

One row per package/order, including package attributes and its `stop_id`. Preserve this table even when routing is performed at stop level.

### 5.3 `travel_times.parquet` and `distances.parquet`

```text
route_id, from_stop_id, to_stop_id, travel_seconds
route_id, from_stop_id, to_stop_id, distance_km
```

Never assume symmetry unless verified. Amazon travel times are directed. Haversine distance is symmetric and should be described as a geometric routing-effort proxy, not road distance.

### 5.4 `features.parquet`

One row per prediction unit with `feature_available_at` metadata. A feature is legal only when `feature_available_at <= decision_time`.

## 6. Time and distance propagation

This is the core needed to decide whether a route is feasible and beneficial.

For route `R = [depot, i1, ..., in]`, define:

```text
arrival[i1]    = departure_time + travel(depot, i1)
completion[i1] = arrival[i1] + service[i1]
arrival[ik]    = completion[i(k-1)] + travel(i(k-1), ik)
completion[ik] = arrival[ik] + service[ik]
```

The manuscript currently calls `a_i(R)` the “route-propagated service time.” The code and paper must choose whether this means arrival or service completion and use that definition everywhere. **Recommended:** use completion time, because package hand-off is not complete at arrival. Run arrival-time lateness as a sensitivity analysis if needed.

For each stop:

```text
late_i      = int(completion_i > promised_time_i)
tardiness_i = max(0, completion_i - promised_time_i) in minutes
```

For the route:

```text
NL = sum(late_i)
TT = sum(tardiness_i)
D  = sum(distance(edge))
```

Decide before experiments whether the route returns to the depot. The present manuscript does not state this. Recommended primary definition: include depot-to-first-stop edges but no return edge because lateness intervention concerns completion of deliveries. Add a return-to-depot distance sensitivity analysis.

## 7. Route feasibility validator

Implement `validate_route(instance, route, baseline_route, delta)` as an independent module. It must run before any route is scored or saved.

### 7.1 Hard feasibility checks

The validator returns `feasible = true` only when:

1. the first node is the expected depot;
2. every assigned customer appears exactly once;
3. no unassigned or unknown node appears;
4. all required coordinates are finite and in valid latitude/longitude ranges;
5. every used edge has finite nonnegative travel time and distance;
6. every service duration is finite and nonnegative;
7. every promised time is parseable and uses the route timezone consistently;
8. propagated times are nondecreasing;
9. `D(candidate) <= (1 + delta) * D(R0) + epsilon`;
10. courier/route assignment is unchanged.

If Xe dù is modeled as pickup-and-delivery, add these hard checks:

- every pickup and delivery appears once;
- pickup precedes its corresponding delivery;
- vehicle load never becomes negative or exceeds capacity;
- the same vehicle performs both operations.

Do not claim feasibility for Xe dù by checking receiver stops alone unless a defensible delivery-only subset has been established.

### 7.2 SFR acceptance checks

A proposed relocation is accepted only when:

- it is a forward move (`new_position < current_position`);
- the complete candidate route passes the hard validator;
- candidate total tardiness is strictly below current total tardiness, using a numerical tolerance;
- the lexicographic selection is deterministic: minimum `TT`, then minimum distance, then minimum displacement, then stable stop ID as a final software tie-break.

Record the reason for every rejection, for example `distance_limit`, `no_tardiness_gain`, `missing_edge`, or `invalid_route`.

### 7.3 Validator output

Return a structured object:

```json
{
  "feasible": true,
  "violations": [],
  "distance_km": 23.41,
  "distance_limit_km": 24.08,
  "nl": 3,
  "tt_minutes": 47.2,
  "served_customers": 18
}
```

This validator, not a plotting function, is the authority for “feasible.”

## 8. Prediction design for RQ1

### 8.1 Label

Primary label:

```text
y_i = 1 if actual_or_proxy_time_i > promised_time_i else 0
```

Generate the label only after features have been frozen. Never include actual delivery time, actual sequence position, final scan status, delivery GPS event, same-day realized lateness, or any variable computed from them in `x_i`.

### 8.2 Legal feature groups

- calendar features known at decision time: weekday, month, holiday flag if an approved calendar is used;
- promised-time features: deadline hour and available slack from departure/decision time;
- location features: coordinates or training-fitted spatial clusters;
- planned workload: number of stops/packages, total planned service duration, package volume/weight;
- route geometry derived from the ex-ante customer set: depot distance and neighborhood density;
- package/service attributes;
- historical courier, station, zone, or service-type rates calculated strictly from earlier dates.

All historical aggregates need a cutoff-date implementation. For a row dated day `t`, its aggregate may use days `< t`, never other outcomes from day `t`.

### 8.3 Models

Use a small, interpretable model ladder:

1. prevalence-only baseline;
2. logistic regression;
3. random forest or histogram gradient boosting;
4. one stronger tree-boosting model only if its library is already available and reproducible.

The purpose is not to maximize the number of models. The paper needs calibrated probabilities because `p_i` is multiplied by actionability.

### 8.4 Chronological evaluation

Use dates, not random rows:

- earliest 60% of dates: training;
- next 20%: validation, tuning, and calibration;
- latest 20%: untouched test.

If the date span is short, use rolling-origin folds in development and retain the latest block as test. Fit encoders, imputers, scalers, spatial clusters, and calibration only on the appropriate training period.

### 8.5 Prediction metrics

Report:

- ROC-AUC;
- PR-AUC, especially if late deliveries are rare;
- log loss;
- Brier score;
- calibration intercept/slope or reliability plot;
- recall and precision at intervention-relevant top-budget fractions.

Accuracy alone is inadequate for an imbalanced ranking problem.

## 9. Baseline routes

### 9.1 Primary: deterministic nearest neighbor

Start at the depot and repeatedly choose the unvisited customer with minimum available distance. Resolve ties by earliest promised time and then stable stop ID. The tie rules must be fixed before testing.

### 9.2 Robustness: Clarke-Wright savings

Implement a deterministic parallel-savings version. Since each Amazon challenge route is already assigned to one vehicle, use it only to construct one feasible customer sequence for the fixed customer set; document how merged chains are converted into the final sequence.

### 9.3 Benchmark only: actual historical sequence

For Amazon, the historical sequence may be used after the main analysis as a realism benchmark. It must not be used as the ex-ante baseline or prediction feature because the manuscript explicitly forbids realized service sequence information at decision time.

## 10. Actionability and selective forward relocation

Implement the manuscript equations literally.

For each customer `i` at baseline position `k`:

1. generate all forward insertion positions `j < k`;
2. validate each `R0(i -> j)` against the distance tolerance;
3. propagate the complete route schedule;
4. compute `DeltaTT_ij = TT(R0) - TT(candidate)`;
5. set `g_i = max(0, max feasible DeltaTT_ij)`.

Freeze all `g_i` values before intervention execution. Then create four ranking policies:

- **Random:** seeded random ranking; repeat many seeds.
- **RB:** descending `p_i`.
- **AB:** descending `g_i`.
- **RA:** descending `p_i * g_i`.

An earliest-deadline-first or smallest-slack policy is also a useful non-ML comparator.

For each policy and budget, select at most `K_r(B)` customers, then run SFR sequentially in the frozen ranking order. Store a full relocation log.

### Budget edge case requiring teacher approval

Equation 13 currently uses `max(1, floor(B*n_r))`. This means even a nominal zero budget would allow one candidate and small routes may receive a much larger intervention fraction than requested. Recommended implementation:

```text
K_r(0) = 0
K_r(B) = max(1, floor(B*n_r)) for B > 0
```

Alternatively use `ceil(B*n_r)`. Choose once, update the paper, and include a sensitivity check for small routes.

## 11. Experimental matrix

Pre-register a manageable primary grid before seeing test results.

### Primary settings

- baseline: nearest neighbor;
- policy: Random, deadline/slack, RB, AB, RA;
- budget `B`: 0%, 5%, 10%, 20%;
- distance tolerance `delta`: 0%, 2%, 5%, 10%;
- lateness time: completion time;
- route distance: depot-to-last-customer without return.

### Robustness settings

- Clarke-Wright baseline;
- arrival-time rather than completion-time lateness;
- return-to-depot distance;
- alternative service-time aggregation;
- calibrated versus uncalibrated probabilities;
- routes stratified by route size, baseline lateness, city/station, and time-window density;
- multiple random-policy seeds.

Do not tune `B` or `delta` on the final test block. They are scenario parameters, not values to optimize using test outcomes.

## 12. Operational evaluation for RQ2 and RQ3

For every test route and policy-scenario pair, save:

- baseline and final sequence;
- selected and actually relocated customers;
- number of accepted relocations;
- `NL(R0)`, `NL(R')`, `DeltaNL`;
- `TT(R0)`, `TT(R')`, `DeltaTT`;
- `D(R0)`, `D(R')`, `DeltaD`;
- feasibility result and rejection reasons;
- runtime.

Also report:

- fraction of routes with at least one accepted relocation;
- fraction of selected customers actually relocated;
- late deliveries avoided per 100 considered interventions;
- tardiness minutes reduced per additional kilometer;
- routes where `DeltaNL < 0`, even though `DeltaTT >= 0`;
- distribution, median, mean, and confidence interval—not only aggregate totals.

### Statistical comparisons

Use paired route-level comparisons because every policy operates on the same test routes:

- bootstrap confidence intervals over routes for mean/median differences;
- Wilcoxon signed-rank test for paired non-normal outcomes where appropriate;
- McNemar-style paired analysis for binary route-improvement indicators if used;
- effect sizes and confidence intervals in addition to p-values;
- multiplicity correction for the limited set of confirmatory policy comparisons.

The main confirmatory comparison should be defined in advance, for example RA versus RB at `B = 10%` and `delta = 5%`.

## 13. “Good to go” decision gates

Use these gates before claiming success.

### Gate A: data validity

- canonical audit passes;
- exclusions and missingness are documented;
- timezones and time semantics are fixed;
- route model matches the chosen data.

### Gate B: software correctness

- all unit and invariant tests pass;
- all reported final routes pass the independent validator;
- repeated runs with the same seed are identical;
- tiny hand-calculated examples match code outputs.

### Gate C: predictive validity

- test predictions are genuinely out of time;
- no leakage guard violations occur;
- model beats the prevalence/rule baseline on proper probability metrics;
- calibration is acceptable or calibrated probabilities are used.

### Gate D: operational validity

- every accepted SFR move strictly reduces current `TT`;
- every final route satisfies `DeltaD <= delta` within numerical tolerance;
- customer assignments are unchanged;
- improvements are computed by full schedule propagation for all customers.

### Gate E: research usefulness

The algorithm is promising only if RA or another prediction-informed policy provides a practically meaningful paired improvement over risk-unaware comparators under a predeclared budget/tolerance. Feasibility alone is not evidence of usefulness. If RA does not outperform, report that result and analyze whether prediction quality, probability calibration, actionability construction, or limited intervention opportunity explains it.

## 14. Required tests

At minimum, code the following tests before running full data.

1. A three-customer hand example with manually computed arrivals and tardiness.
2. A relocation that reduces one customer’s tardiness but delays downstream customers; confirm full-route `TT` is used.
3. A relocation that reduces `TT` but increases `NL`; confirm the proposition is not incorrectly applied to `NL`.
4. A move with lower `TT` that violates `delta`; confirm rejection.
5. Two equal-`TT` candidates; confirm distance and displacement tie-breaking.
6. Duplicate, missing, and unknown stop IDs; confirm validator failure.
7. Missing and asymmetric travel-time edges.
8. A selected customer with no admissible move; route must remain unchanged.
9. Multiple accepted moves whose cumulative distance remains measured against `D(R0)`.
10. `B = 0`, very small routes, and single-customer routes.
11. Feature rows containing forbidden post-outcome columns; leakage guard must fail loudly.
12. Historical aggregate at date `t`; confirm it cannot see outcomes on or after `t`.

Property-style tests should additionally verify that SFR never returns `TT(R') > TT(R0)` and never returns `D(R') > (1 + delta)D(R0)`.

## 15. Figures and tables the code should produce

The final scripts should generate publication-ready artifacts rather than manually edited numbers:

- dataset flow/exclusion table;
- lateness prevalence by chronological split;
- predictive metrics table;
- calibration/reliability plot;
- precision/recall at intervention budgets;
- route example before and after SFR, annotated with deadlines and arrival/completion times;
- `DeltaTT` and `DeltaNL` versus budget for each policy;
- service improvement versus `DeltaD` Pareto plot;
- accepted-relocation and feasibility rates;
- ablation table: RB, AB, RA, random, deadline/slack;
- robustness table across baseline heuristics and time/distance definitions.

Every table cell should be generated from saved route-level result files so the manuscript can be reproduced.

## 16. Reproducibility and run interface

Use one configuration file to store:

```yaml
dataset: amazon
timezone: UTC
lateness_clock: completion
include_return_to_depot: false
baseline: nearest_neighbor
budgets: [0.0, 0.05, 0.10, 0.20]
distance_tolerances: [0.0, 0.02, 0.05, 0.10]
random_seeds: [101, 202, 303]
```

Target command flow:

```text
python -m src.data.audit --config configs/experiment.yaml
python -m src.experiments.run_prediction --config configs/experiment.yaml
python -m src.experiments.run_routing --config configs/experiment.yaml
python -m src.experiments.run_robustness --config configs/experiment.yaml
python -m pytest tests
```

Each run should write its configuration, Git commit if available, environment/package versions, random seeds, start/end time, dataset checksum, and output checksums to a run manifest.

## 17. Inspection of the “thesis we can use” folder

The folder is now present and has been audited. The detailed findings are recorded in `reports/thesis_provenance_audit.md`. The supplied work is a January 2026 bachelor thesis by Hoang Minh Anh titled *Data-Driven Decision Support for Last Mile Delivery in E-Commerce: A Case Study*. It is a conceptual predecessor to this project and uses the same Ho Chi Minh City dataset. The current manuscript should be described as an extension and methodological redesign, subject to confirmation by the teacher and the thesis author/advisor.

The audit found no substantial exact phrase match between the normalized thesis text and `main.tex`, but it identified major technical limitations in the supplied code. In particular, the earlier pipeline uses post-outcome features, actual route sequences, ground-truth delayed counts during targeting, and a fixed intervention-success rate. Its code and numerical results must not be reused as evidence for the PISR framework.

The following procedure should be retained for future revisions or additional source material.

### Audit procedure

1. Inventory every file with path, type, size, and checksum.
2. Extract PDF text into temporary files for inspection; do not modify the source thesis.
3. Record thesis title, author, institution, year, supervisor, persistent link, and license/usage terms.
4. Compare its research problem, notation, equations, algorithm steps, dataset, experimental design, figure structure, and references with `main.tex`.
5. Search for distinctive matching phrases and matching sequences of citations.
6. Build a provenance table with columns `manuscript element`, `possible thesis source`, `relationship`, and `required action`.
7. Classify relationships as `independent`, `inspired by`, `adapted`, or `reproduced`.
8. Cite all adapted concepts and clearly label reproduced/modified algorithms or figures. Do not copy prose merely by changing a few words.
9. If code exists, check its license before reuse and preserve required notices.
10. Ask the teacher to confirm the intended relationship: replication, extension, adaptation, or independent study.

Recommended output after the folder arrives: `reports/thesis_provenance_audit.md`. This is an academic-integrity safeguard, not an accusation that the manuscript is copied.

## 18. Decisions needed from the teacher before full implementation

1. Which dataset is the **primary** empirical dataset?
2. Is a route-propagated time-window violation acceptable when a directly observed completion timestamp is unavailable?
3. Is service time defined by arrival or completion?
4. Does route distance include return to depot?
5. For multi-package stops, is the stop deadline the earliest package deadline, or should evaluation remain package-level?
6. Should Xe dù be treated only as an RQ1 dataset, or should the model be extended to pickup-and-delivery routing?
7. Is a derived SLA permitted for LaDe? If yes, what operational rule defines it?
8. Which comparison is confirmatory and which analyses are exploratory?
9. What practical effect is large enough to matter: late deliveries avoided, tardiness minutes reduced, or improvement per added kilometer?
10. Is the supplied thesis intended for replication or extension, and what must be cited?

## 19. Phased implementation plan

### Phase 0 — provenance and data decision

- add the thesis folder and dataset samples;
- run the thesis provenance audit;
- run schema/missingness audits on all candidate datasets;
- select the primary dataset and freeze time/distance semantics.

**Deliverable:** signed-off dataset and assumptions note.

### Phase 1 — routing engine on synthetic data

- implement distance, propagation, metrics, validator, nearest neighbor, actionability, and SFR;
- create hand-worked unit tests;
- verify the monotonic tardiness and distance-bound invariants.

**Deliverable:** correct route engine independent of machine learning.

### Phase 2 — dataset adapter and descriptive study

- convert the primary dataset to canonical tables;
- generate exclusions, route-size distributions, time-window coverage, and baseline lateness;
- stop if the dataset fails the go/no-go rule.

**Deliverable:** reproducible processed data and audit report.

### Phase 3 — leakage-free prediction

- implement chronological features and splits;
- train, calibrate, and evaluate the model ladder;
- freeze test probabilities.

**Deliverable:** `p_i` values and RQ1 tables/figures.

### Phase 4 — policy experiment

- construct ex-ante baselines;
- compute frozen `g_i`;
- run Random, deadline/slack, RB, AB, and RA over the experiment grid;
- validate and log every route.

**Deliverable:** route-level results for RQ2 and RQ3.

### Phase 5 — robustness and manuscript integration

- run prespecified robustness checks and paired statistics;
- generate final figures/tables;
- revise `main.tex` so all empirical claims match the actual data semantics.

**Deliverable:** reproducible results package and manuscript-ready evidence.

## 20. Source notes checked for this design

- Xe dù dataset card: <https://www.kaggle.com/datasets/hunhquanglc/xe-d-ho-chi-minh-city-delivery-service-dataset>
- Amazon open-data registry: <https://registry.opendata.aws/amazon-last-mile-challenges/>
- Amazon file documentation: <https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md>
- LaDe project page: <https://cainiaoai.github.io/LaDe-website/>
- LaDe-D published schema/viewer: <https://huggingface.co/datasets/Cainiao-AI/LaDe-D>

These source notes support dataset selection and field mapping. The definitive audit must still be performed on the exact downloaded files and versions used in the study.