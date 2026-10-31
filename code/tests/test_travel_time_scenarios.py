import os
import sys
import json
import random
from datetime import datetime, date, time, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import RouteInstance, Stop
from src.routing.schedule import propagate_schedule
from src.evaluation.scenarios import (
    build_route_time_data, draw_standard_normals, scenario_travel_times, lognormal_sigma,
    encode_sequences, evaluate_sequences, matrix_fingerprint
)

BASE_DATE = date(2024, 5, 10)

def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime.combine(BASE_DATE, time(hour, minute, second))

def make_random_instance(n_customers: int = 8, seed: int = 7, full_matrix: bool = True) -> RouteInstance:
    rng = random.Random(seed)
    stops = {"DEPOT": Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")}
    for i in range(n_customers):
        c_id = f"C{i:02d}"
        stops[c_id] = Stop(
            stop_id=c_id,
            lat=rng.uniform(-0.02, 0.02),
            lng=rng.uniform(-0.02, 0.02),
            stop_type="Dropoff",
            promised_time=dt(8, 0) + timedelta(minutes=rng.randint(10, 60)),
            service_seconds=rng.choice([60.0, 90.5, 120.0, 45.25])
        )
    travel_times: Dict[Tuple[str, str], float] = {}
    for u in stops:
        for v in stops:
            if u != v:
                travel_times[(u, v)] = round(rng.uniform(60.0, 600.0), 1)
    if not full_matrix:
        travel_times.pop(("C00", "C01"))
    return RouteInstance(
        route_id="SCEN_TEST",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops=stops,
        travel_times=travel_times
    )

# ----------------------------------------------------------------------
# Scenario generator
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cv", [0.10, 0.20, 0.30])
def test_lognormal_multipliers_preserve_mean_and_match_cv(cv):
    mean = np.ones((400, 400))
    z = draw_standard_normals("ROUTE_X", 0, 400, base_seed=2026)
    mult = scenario_travel_times(mean, z, cv) / mean
    assert abs(mult.mean() - 1.0) < 0.005
    assert abs(mult.std() / mult.mean() - cv) < 0.01
    assert lognormal_sigma(cv) == pytest.approx(np.sqrt(np.log(1.0 + cv ** 2)))

def test_cv_zero_returns_mean_matrix_exactly():
    mean = np.array([[0.0, 120.5], [300.25, 0.0]])
    z = draw_standard_normals("ROUTE_X", 3, 2, base_seed=2026)
    out = scenario_travel_times(mean, z, 0.0)
    assert np.array_equal(out, mean)
    assert out is not mean

def test_scenario_draws_are_reproducible_and_scenario_specific():
    a = draw_standard_normals("ROUTE_A", 5, 30, base_seed=2026)
    # Drawing other scenarios in between must not change scenario 5
    for s in range(20):
        draw_standard_normals("ROUTE_A", s, 30, base_seed=2026)
    b = draw_standard_normals("ROUTE_A", 5, 30, base_seed=2026)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, draw_standard_normals("ROUTE_A", 6, 30, base_seed=2026))
    assert not np.array_equal(a, draw_standard_normals("ROUTE_B", 5, 30, base_seed=2026))
    assert not np.array_equal(a, draw_standard_normals("ROUTE_A", 5, 30, base_seed=2027))
    mean = np.full((30, 30), 100.0)
    assert matrix_fingerprint(scenario_travel_times(mean, a, 0.2)) == matrix_fingerprint(scenario_travel_times(mean, b, 0.2))

def test_zero_mean_and_missing_edges_are_never_imputed():
    mean = np.array([
        [0.0, 0.0, 200.0],
        [150.0, 0.0, np.nan],
        [90.0, 60.0, 0.0]
    ])
    z = draw_standard_normals("ROUTE_X", 0, 3, base_seed=2026)
    out = scenario_travel_times(mean, z, 0.3)
    assert out[0, 1] == 0.0
    assert np.isnan(out[1, 2])
    positive = np.isfinite(mean) & (mean > 0)
    assert np.all(np.isfinite(out[positive])) and np.all(out[positive] > 0)

# ----------------------------------------------------------------------
# Strict evaluator
# ----------------------------------------------------------------------
def test_build_route_time_data_requires_official_matrix():
    inst = make_random_instance()
    inst.travel_times = None
    with pytest.raises(ValueError, match="Haversine"):
        build_route_time_data(inst)

def test_evaluator_rejects_missing_edge_instead_of_haversine_fallback():
    inst = make_random_instance(full_matrix=False)
    route = ["DEPOT"] + [f"C{i:02d}" for i in range(8)]
    # The deterministic engine silently falls back to Haversine for the missing edge ...
    propagate_schedule(inst, route)
    # ... the scenario evaluator must refuse instead
    data = build_route_time_data(inst)
    with pytest.raises(ValueError, match="C00 -> C01"):
        evaluate_sequences(data, data.mean_travel_seconds, encode_sequences(data, [route]))

def test_evaluator_matches_propagate_schedule_on_mean_and_scenario_matrices():
    inst = make_random_instance(n_customers=9, seed=11)
    data = build_route_time_data(inst)
    customers = [f"C{i:02d}" for i in range(9)]
    rng = random.Random(3)
    routes = []
    for _ in range(25):
        perm = list(customers)
        rng.shuffle(perm)
        routes.append(["DEPOT"] + perm)
    encoded = encode_sequences(data, routes)

    z = draw_standard_normals(inst.route_id, 0, len(data.stop_ids), base_seed=2026)
    for cv in [0.0, 0.25]:
        matrix = scenario_travel_times(data.mean_travel_seconds, z, cv)
        tt, nl = evaluate_sequences(data, matrix, encoded)

        scen_inst = RouteInstance(
            route_id=inst.route_id, depot_id=inst.depot_id, departure_time=inst.departure_time, stops=inst.stops,
            travel_times={(u, v): float(matrix[data.index[u], data.index[v]]) for u in data.stop_ids for v in data.stop_ids if u != v}
        )
        for k, route in enumerate(routes):
            ref = propagate_schedule(scen_inst, route)
            assert tt[k] == pytest.approx(ref.total_tardiness, abs=1e-4)
            assert nl[k] == ref.total_lateness

def test_encode_sequences_rejects_invalid_frozen_routes():
    inst = make_random_instance(n_customers=4)
    data = build_route_time_data(inst)
    good = ["DEPOT", "C00", "C01", "C02", "C03"]
    encode_sequences(data, [good, ["DEPOT", "C03", "C02", "C01", "C00"]])
    with pytest.raises(ValueError):
        encode_sequences(data, [good, ["C00", "DEPOT", "C01", "C02", "C03"]])
    with pytest.raises(ValueError):
        encode_sequences(data, [good, ["DEPOT", "C00", "C00", "C02", "C03"]])

# ----------------------------------------------------------------------
# End-to-end: plan -> freeze -> evaluate on an official held-out route
# ----------------------------------------------------------------------
def test_frozen_scenario_pipeline_on_official_route(tmp_path, monkeypatch):
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        pytest.skip(f"Official raw dataset not found in '{raw_dir}'")

    import experiments.run_travel_time_scenarios as tts

    out_dir = str(tmp_path / "scenario_out")
    tts.run_travel_time_scenarios(
        raw_dir=raw_dir, output_dir=out_dir, stage="plan",
        route_ids=["RouteID_693060a6-88bb-4324-9e9c-925d5240263c"],
        budgets=[0.20], deltas=[0.05], policies=["RA", "AB", "Random"], random_seeds=[42, 43]
    )

    # The evaluation stage must not plan again: any call to the planning kernel fails the test
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning kernel called during scenario evaluation")
    for name in ["selective_forward_relocation", "compute_actionability_scores", "rank_candidates",
                 "select_intervention_set", "train_and_evaluate_pipeline", "inject_calibrated_risks_into_instances"]:
        monkeypatch.setattr(tts, name, forbidden)

    summary = tts.run_travel_time_scenarios(
        raw_dir=raw_dir, output_dir=out_dir, stage="evaluate", n_scenarios=3, cv_levels=[0.0, 0.2]
    )

    assert summary["all_checks_passed"], json.dumps(summary["checks"], indent=1, default=str)
    repro = summary["checks"]["deterministic_benchmark_reproduced"]
    assert repro["frozen_plans_compared"] == repro["frozen_plans_total"] == 4

    scen = pd.read_csv(os.path.join(out_dir, "scenario_level_runs.csv.gz"))
    assert len(scen) == 1 * 2 * 3 * 3  # route x CV x scenario x policy (Random averaged over seeds)
    cv0 = scen[scen["cv"] == 0.0]
    assert cv0.groupby("policy")["delta_tt"].nunique().max() == 1  # all CV = 0 scenarios are identical
    # Policies on the same scenario share R0 outcomes (same matrix)
    assert scen.groupby(["cv", "scenario_id"])["base_tt"].nunique().max() == 1
    assert os.path.exists(os.path.join(out_dir, "travel_time_scenario_pilot.md"))

def test_exactly_met_deadline_is_on_time_despite_float_rounding():
    # 0.1 s + 0.2 s sums to 0.30000000000000004 s in floating point, but the datetime schedule meets 0.3 s exactly
    stops = {
        "DEPOT": Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station"),
        "A": Stop(stop_id="A", lat=0.0, lng=0.001, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=0.0),
        "B": Stop(stop_id="B", lat=0.0, lng=0.002, stop_type="Dropoff",
                  promised_time=dt(8, 0) + timedelta(seconds=0.3), service_seconds=0.0)
    }
    inst = RouteInstance(route_id="TIE", depot_id="DEPOT", departure_time=dt(8, 0), stops=stops,
                         travel_times={("DEPOT", "A"): 0.1, ("A", "B"): 0.2, ("DEPOT", "B"): 0.3,
                                       ("A", "DEPOT"): 0.1, ("B", "A"): 0.2, ("B", "DEPOT"): 0.3})
    route = ["DEPOT", "A", "B"]
    data = build_route_time_data(inst)
    tt, nl = evaluate_sequences(data, data.mean_travel_seconds, encode_sequences(data, [route]))
    sched = propagate_schedule(inst, route)
    assert sched.total_lateness == 0
    assert nl[0] == 0 and tt[0] == 0.0
