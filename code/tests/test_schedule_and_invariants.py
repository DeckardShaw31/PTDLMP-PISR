import pytest
from datetime import datetime, date, time
from typing import Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import RouteInstance, Stop
from src.routing.distance import haversine_distance, compute_route_distance
from src.routing.schedule import propagate_schedule
from src.routing.validator import validate_route
from src.routing.baseline import build_nearest_neighbor_baseline
from src.routing.actionability import compute_actionability_scores, rank_candidates, select_intervention_set
from src.routing.sfr import selective_forward_relocation

BASE_DATE = date(2024, 5, 10)

def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime.combine(BASE_DATE, time(hour, minute, second))

# ----------------------------------------------------------------------
# Test 1: Hand-calculated 3-customer route
# ----------------------------------------------------------------------
def test_hand_calculated_3_customer_route():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 30), service_seconds=300)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 40), service_seconds=300)
    c3 = Stop(stop_id="C3", lat=0.03, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 35), service_seconds=300)

    # Fixed travel times: 600s between consecutive stops
    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 600.0,
        ("C1", "C2"): 600.0,
        ("C2", "C3"): 600.0,
        ("DEPOT", "C3"): 600.0,
        ("C3", "C1"): 600.0,
        ("C1", "C3"): 600.0,
        ("C3", "C2"): 600.0,
        ("DEPOT", "C2"): 600.0,
    }
    # Symmetric Euclidean-like distances
    distances: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1.0, ("C1", "DEPOT"): 1.0,
        ("C1", "C2"): 1.0, ("C2", "C1"): 1.0,
        ("C2", "C3"): 1.0, ("C3", "C2"): 1.0,
        ("DEPOT", "C3"): 1.5, ("C3", "DEPOT"): 1.5,
        ("C3", "C1"): 1.2, ("C1", "C3"): 1.2,
        ("DEPOT", "C2"): 1.8, ("C2", "DEPOT"): 1.8,
    }

    inst = RouteInstance(
        route_id="HAND_01",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2, "C3": c3},
        travel_times=travel_times,
        distances=distances
    )

    route = ["DEPOT", "C1", "C2", "C3"]
    sched = propagate_schedule(inst, route)

    # C1: arr 08:10, comp 08:15 <= 08:30 -> Tardiness = 0
    assert sched.arrival_times["C1"] == dt(8, 10)
    assert sched.completion_times["C1"] == dt(8, 15)
    assert sched.tardiness_minutes["C1"] == 0.0
    assert sched.lateness["C1"] == 0

    # C2: arr 08:25, comp 08:30 <= 08:40 -> Tardiness = 0
    assert sched.arrival_times["C2"] == dt(8, 25)
    assert sched.completion_times["C2"] == dt(8, 30)
    assert sched.tardiness_minutes["C2"] == 0.0

    # C3: arr 08:40, comp 08:45 > 08:35 -> Tardiness = 10 min
    assert sched.arrival_times["C3"] == dt(8, 40)
    assert sched.completion_times["C3"] == dt(8, 45)
    assert sched.tardiness_minutes["C3"] == 10.0
    assert sched.lateness["C3"] == 1

    assert sched.total_lateness == 1
    assert sched.total_tardiness == 10.0
    assert sched.total_distance == 3.0

# ----------------------------------------------------------------------
# Test 2: Downstream delay check & route-wide TT evaluation
# ----------------------------------------------------------------------
def test_downstream_delay_evaluated_globally():
    # If C3 is moved to position 1, C1 and C2 are pushed later
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 25), service_seconds=300)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 40), service_seconds=300)
    c3 = Stop(stop_id="C3", lat=0.03, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 30), service_seconds=300)

    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 600.0, ("C1", "C2"): 600.0, ("C2", "C3"): 600.0,
        ("DEPOT", "C3"): 600.0, ("C3", "C1"): 600.0,
    }
    distances: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1.0, ("C1", "C2"): 1.0, ("C2", "C3"): 1.0,
        ("DEPOT", "C3"): 1.0, ("C3", "C1"): 1.0,
    }
    inst = RouteInstance(
        route_id="TEST_DOWNSTREAM",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2, "C3": c3},
        travel_times=travel_times,
        distances=distances
    )

    r_base = ["DEPOT", "C1", "C2", "C3"]
    sched_base = propagate_schedule(inst, r_base)
    # Under base: C1 comp 08:15 <= 08:25 (0); C2 comp 08:30 <= 08:40 (0); C3 comp 08:45 > 08:30 (15 min) -> TT=15
    assert sched_base.total_tardiness == 15.0

    # Moving C3 to pos 1: ["DEPOT", "C3", "C1", "C2"]
    # C3 comp 08:15 <= 08:30 (0 min)
    # C1 comp 08:30 > 08:25 (5 min tardy!)
    # C2 comp 08:45 > 08:40 (5 min tardy!)
    # Total TT = 0 + 5 + 5 = 10 min < 15 min.
    r_cand = ["DEPOT", "C3", "C1", "C2"]
    sched_cand = propagate_schedule(inst, r_cand)
    assert sched_cand.total_tardiness == 10.0
    # Downstream customers C1 and C2 suffered delays, but global TT still improved by 5 min.
    assert sched_cand.total_tardiness < sched_base.total_tardiness

# ----------------------------------------------------------------------
# Test 3: Total tardiness strictly decreases while NL increases (Delta TT > 0, Delta NL < 0)
# ----------------------------------------------------------------------
def test_tt_decreases_while_nl_increases():
    """
    Validates the critical manuscript observation (Proposition 1 remark):
    Monotonicity holds for TT, but not necessarily for NL!
    """
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    # C1 deadline 08:30, C2 deadline 08:35, C3 deadline 07:45 (severely late at 08:45)
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 30), service_seconds=300)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 35), service_seconds=300)
    c3 = Stop(stop_id="C3", lat=0.03, lng=0.0, stop_type="Dropoff", promised_time=dt(7, 45), service_seconds=300)

    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 600.0, ("C1", "C2"): 600.0, ("C2", "C3"): 600.0,
        ("DEPOT", "C3"): 600.0, ("C3", "C1"): 600.0,
    }
    distances: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1.0, ("C1", "C2"): 1.0, ("C2", "C3"): 1.0,
        ("DEPOT", "C3"): 1.0, ("C3", "C1"): 1.0,
    }
    inst = RouteInstance(
        route_id="TEST_NL_DIFF",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2, "C3": c3},
        travel_times=travel_times,
        distances=distances
    )

    r_base = ["DEPOT", "C1", "C2", "C3"]
    sched_base = propagate_schedule(inst, r_base)
    # Base: C1 comp 08:15 (0 min); C2 comp 08:30 (0 min); C3 comp 08:45 (60 min tardy).
    # NL = 1, TT = 60.0
    assert sched_base.total_lateness == 1
    assert sched_base.total_tardiness == 60.0

    # Relocate C3 -> pos 1: ["DEPOT", "C3", "C1", "C2"]
    # C3 comp 08:15 (30 min tardy, late)
    # C1 comp 08:30 <= 08:30 (0 min tardy, on time)
    # C2 comp 08:45 > 08:35 (10 min tardy, late!)
    # NL = 2 (C3, C2), TT = 30 + 10 = 40.0
    r_cand = ["DEPOT", "C3", "C1", "C2"]
    sched_cand = propagate_schedule(inst, r_cand)
    assert sched_cand.total_lateness == 2
    assert sched_cand.total_tardiness == 40.0

    delta_tt = sched_base.total_tardiness - sched_cand.total_tardiness
    delta_nl = sched_base.total_lateness - sched_cand.total_lateness

    # Delta TT is positive (improved), but Delta NL is negative (more late stops)!
    assert delta_tt == 20.0
    assert delta_nl == -1

# ----------------------------------------------------------------------
# Test 4: Distance tolerance violation causes rejection
# ----------------------------------------------------------------------
def test_distance_limit_violation_rejection():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 30), service_seconds=300)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 15), service_seconds=300)

    # Base distance: DEPOT -> C1 = 1.0, C1 -> C2 = 1.0 (Total = 2.0)
    # If C2 moved to pos 1: DEPOT -> C2 = 5.0, C2 -> C1 = 5.0 (Total = 10.0, 400% increase!)
    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 600.0, ("C1", "C2"): 600.0,
        ("DEPOT", "C2"): 300.0, ("C2", "C1"): 300.0
    }
    distances: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1.0, ("C1", "C2"): 1.0,
        ("DEPOT", "C2"): 5.0, ("C2", "C1"): 5.0
    }
    inst = RouteInstance(
        route_id="TEST_DIST_LIMIT",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2},
        travel_times=travel_times,
        distances=distances
    )

    baseline = ["DEPOT", "C1", "C2"]
    # With delta = 0.05 (5% distance allowance), max distance allowed is 2.1 km
    final_route, logs = selective_forward_relocation(inst, baseline, candidate_set=["C2"], delta=0.05)

    # Must be rejected due to distance constraint
    assert final_route == baseline
    assert len(logs) == 1
    assert logs[0].accepted is False
    assert "distance_limit" in logs[0].rejection_reason

# ----------------------------------------------------------------------
# Test 5: Lexicographic tie-breaking: min TT, then min D, then min displacement
# ----------------------------------------------------------------------
def test_lexicographic_tie_breaking():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(10, 0), service_seconds=100)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(10, 0), service_seconds=100)
    c3 = Stop(stop_id="C3", lat=0.03, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 20), service_seconds=100)

    # Let C3 at pos 3 have two forward positions: pos 1 and pos 2.
    # Both give TT = 0, but pos 2 has smaller route distance than pos 1.
    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 100.0, ("C1", "C2"): 100.0, ("C2", "C3"): 100.0,
        ("DEPOT", "C3"): 100.0, ("C3", "C1"): 100.0, ("C1", "C3"): 100.0, ("C3", "C2"): 100.0
    }
    distances: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1.0, ("C1", "C2"): 1.0, ("C2", "C3"): 1.0,
        ("DEPOT", "C3"): 1.5, ("C3", "C1"): 1.5,  # Route [DEPOT, C3, C1, C2] distance = 1.5 + 1.5 + 1.0 = 4.0
        ("C1", "C3"): 1.1, ("C3", "C2"): 1.1,      # Route [DEPOT, C1, C3, C2] distance = 1.0 + 1.1 + 1.1 = 3.2
    }
    inst = RouteInstance(
        route_id="TEST_TIE_BREAK",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2, "C3": c3},
        travel_times=travel_times,
        distances=distances
    )

    baseline = ["DEPOT", "C1", "C2", "C3"]
    # Base TT: C3 finishes at 8:00 + 3*100 + 3*100 = 8:10 <= 8:20 (0 min).
    # Wait, let's make baseline C3 late: promise 8:08 -> Base TT = 2 min.
    c3.promised_time = dt(8, 8)
    # Both pos 1 and pos 2 will finish C3 earlier (comp <= 8:04), so both give TT = 0!
    final_route, logs = selective_forward_relocation(inst, baseline, candidate_set=["C3"], delta=1.0)

    assert logs[0].accepted is True
    # pos 2 gives distance 3.2 km, while pos 1 gives distance 4.0 km.
    # Tie-break on minimum distance must pick pos 2: ["DEPOT", "C1", "C3", "C2"]!
    assert final_route == ["DEPOT", "C1", "C3", "C2"]
    assert logs[0].attempted_position == 2

# ----------------------------------------------------------------------
# Test 6: Validator failure on invalid routes
# ----------------------------------------------------------------------
def test_validator_detects_violations():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=10.0, lng=20.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    c2 = Stop(stop_id="C2", lat=10.1, lng=20.1, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)

    inst = RouteInstance(
        route_id="TEST_VAL",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2}
    )

    # Missing C2
    res_missing = validate_route(inst, ["DEPOT", "C1"])
    assert res_missing.feasible is False
    assert any("Missing" in v for v in res_missing.violations)

    # Duplicate C1
    res_dup = validate_route(inst, ["DEPOT", "C1", "C1"])
    assert res_dup.feasible is False
    assert any("Duplicate" in v for v in res_dup.violations)

    # Not starting with depot
    res_no_depot = validate_route(inst, ["C1", "C2"])
    assert res_no_depot.feasible is False
    assert any("must start with depot" in v for v in res_no_depot.violations)

# ----------------------------------------------------------------------
# Test 7: Budget B = 0 produces 0 interventions
# ----------------------------------------------------------------------
def test_budget_zero():
    ranked = [("C1", 10.0), ("C2", 8.0), ("C3", 5.0)]
    selected = select_intervention_set(ranked, budget_b=0.0, num_customers=3)
    assert selected == []

# ----------------------------------------------------------------------
# Test 8: Proposition 1 Invariant Test (TT(R') <= TT(R0))
# ----------------------------------------------------------------------
def test_proposition_1_invariant():
    """
    Proposition 1 guarantee: Under any arbitrary candidate set and parameters,
    SFR strictly guarantees TT(R') <= TT(R0).
    """
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    stops = {"DEPOT": depot}
    for i in range(1, 6):
        c_id = f"C{i}"
        stops[c_id] = Stop(
            stop_id=c_id,
            lat=0.01 * i,
            lng=0.005 * i,
            stop_type="Dropoff",
            promised_time=dt(8, 15 + i * 5),
            service_seconds=180
        )

    inst = RouteInstance(
        route_id="TEST_PROP_1",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops=stops
    )
    baseline = build_nearest_neighbor_baseline(inst)
    sched_base = propagate_schedule(inst, baseline)

    candidates = ["C5", "C4", "C3", "C2", "C1"]
    final_route, logs = selective_forward_relocation(inst, baseline, candidates, delta=0.10)
    sched_final = propagate_schedule(inst, final_route)

    # Monotonicity of TT: TT(R') <= TT(R0)
    assert sched_final.total_tardiness <= sched_base.total_tardiness + 1e-6
    if any(log.accepted for log in logs):
        assert sched_final.total_tardiness < sched_base.total_tardiness

# ----------------------------------------------------------------------
# Test 9: Cumulative distance tolerance Invariant (D(R') <= (1+delta)*D(R0))
# ----------------------------------------------------------------------
def test_cumulative_distance_tolerance_invariant():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    stops = {"DEPOT": depot}
    for i in range(1, 6):
        c_id = f"C{i}"
        stops[c_id] = Stop(
            stop_id=c_id,
            lat=0.01 * (i % 2),
            lng=0.01 * i,
            stop_type="Dropoff",
            promised_time=dt(8, 10 + i * 2),
            service_seconds=120
        )

    inst = RouteInstance(
        route_id="TEST_DIST_INV",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops=stops
    )
    baseline = build_nearest_neighbor_baseline(inst)
    base_dist = compute_route_distance(inst, baseline)

    delta = 0.05
    candidates = ["C5", "C4", "C3", "C2"]
    final_route, logs = selective_forward_relocation(inst, baseline, candidates, delta=delta)
    final_dist = compute_route_distance(inst, final_route)

    # D(R') must not exceed (1 + delta) * D(R0) + eps
    assert final_dist <= (1.0 + delta) * base_dist + 1e-6

# ----------------------------------------------------------------------
# Test 10: Amazon sample route loader and end-to-end PISR pipeline
# ----------------------------------------------------------------------
def test_amazon_sample_instance_end_to_end():
    from src.data.amazon_loader import load_amazon_sample_instance

    routes_path = os.path.join(os.path.dirname(__file__), "..", "data", "amazon_sample_routes.json")
    pkgs_path = os.path.join(os.path.dirname(__file__), "..", "data", "amazon_sample_packages.json")

    inst = load_amazon_sample_instance(routes_path, pkgs_path)
    assert inst.route_id == "RouteID_AMZ_001"
    assert inst.depot_id == "DEPOT"
    assert len(inst.stops) == 13  # DEPOT + 12 Dropoffs

    # 1. Build baseline route (Nearest Neighbor)
    baseline = build_nearest_neighbor_baseline(inst)
    assert baseline[0] == "DEPOT"
    assert len(baseline) == 13

    # Check baseline schedule
    base_sched = propagate_schedule(inst, baseline)
    assert base_sched.total_lateness >= 0
    assert base_sched.total_tardiness >= 0.0

    # 2. Compute actionability
    g_scores, _ = compute_actionability_scores(inst, baseline, delta=0.10)
    assert len(g_scores) == 12

    # 3. Policy ranking & selection (RA with B = 0.20)
    ranked = rank_candidates(inst, baseline, policy="RA", g_scores=g_scores)
    selected = select_intervention_set(ranked, budget_b=0.20, num_customers=12)
    assert len(selected) <= 3

    # 4. Selective Forward Relocation (SFR)
    final_route, logs = selective_forward_relocation(inst, baseline, selected, delta=0.10)

    # 5. Independent validation
    val_res = validate_route(inst, final_route, baseline_route=baseline, delta=0.10)
    assert val_res.feasible is True
    assert val_res.violations == []

    # 6. Verify Proposition 1
    final_sched = propagate_schedule(inst, final_route)
    assert final_sched.total_tardiness <= base_sched.total_tardiness + 1e-6

# ----------------------------------------------------------------------
# Test 11: Customer with no admissible forward move leaves route unchanged
# ----------------------------------------------------------------------
def test_no_admissible_move_leaves_route_unchanged():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)

    inst = RouteInstance(
        route_id="TEST_NO_MOVE",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2}
    )

    baseline = ["DEPOT", "C1", "C2"]
    # Delta = 0.0, and already no tardiness to gain
    final_route, logs = selective_forward_relocation(inst, baseline, candidate_set=["C2"], delta=0.0)
    assert final_route == baseline
    assert len(logs) == 1
    assert logs[0].accepted is False

# ----------------------------------------------------------------------
# Test 12: Asymmetric travel time edges respected
# ----------------------------------------------------------------------
def test_asymmetric_travel_times_respected():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", service_seconds=0)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", service_seconds=0)

    # Asymmetric: C1 -> C2 is fast (100s), but C2 -> C1 is slow (900s)
    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 100.0,
        ("C1", "C2"): 100.0,
        ("DEPOT", "C2"): 100.0,
        ("C2", "C1"): 900.0,
    }
    inst = RouteInstance(
        route_id="TEST_ASYM",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2},
        travel_times=travel_times
    )

    # Route 1: [DEPOT, C1, C2] -> arrival at C2 = 8:00 + 100 + 100 = 8:03:20 (200s)
    sched1 = propagate_schedule(inst, ["DEPOT", "C1", "C2"])
    assert (sched1.arrival_times["C2"] - dt(8, 0)).total_seconds() == 200.0

    # Route 2: [DEPOT, C2, C1] -> arrival at C1 = 8:00 + 100 + 900 = 8:16:40 (1000s)
    sched2 = propagate_schedule(inst, ["DEPOT", "C2", "C1"])
    assert (sched2.arrival_times["C1"] - dt(8, 0)).total_seconds() == 1000.0

# ----------------------------------------------------------------------
# Test 13: Validator rejects unknown nodes without unhandled exception
# ----------------------------------------------------------------------
def test_validator_rejects_unknown_nodes_without_crashing():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    inst = RouteInstance(
        route_id="TEST_UNKNOWN",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1}
    )

    # Route contains completely unknown node "GHOST_NODE"
    cand_route = ["DEPOT", "C1", "GHOST_NODE"]
    # Must NOT raise KeyError or unhandled exception
    res = validate_route(inst, cand_route)
    assert res.feasible is False
    assert any("Unknown or unassigned" in v for v in res.violations)
    assert "GHOST_NODE" in str(res.violations)

# ----------------------------------------------------------------------
# Test 14: Validator rejects NaN and Inf coordinates and service times
# ----------------------------------------------------------------------
def test_validator_rejects_nan_and_inf():
    import math
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=float('nan'), lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    inst_nan = RouteInstance(
        route_id="TEST_NAN",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1}
    )
    res_nan = validate_route(inst_nan, ["DEPOT", "C1"])
    assert res_nan.feasible is False
    assert any("NaN or Inf coordinates" in v for v in res_nan.violations)

    c2 = Stop(stop_id="C2", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=float('inf'))
    inst_inf = RouteInstance(
        route_id="TEST_INF",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C2": c2}
    )
    res_inf = validate_route(inst_inf, ["DEPOT", "C2"])
    assert res_inf.feasible is False
    assert any("NaN or Inf service duration" in v for v in res_inf.violations)

# ----------------------------------------------------------------------
# Test 15: Validator rejects missing promised delivery times
# ----------------------------------------------------------------------
def test_validator_rejects_missing_promised_times():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=None, service_seconds=100)
    inst = RouteInstance(
        route_id="TEST_NO_DEADLINE",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1}
    )
    res = validate_route(inst, ["DEPOT", "C1"], require_promised_times=True)
    assert res.feasible is False
    assert any("missing promised delivery time" in v for v in res.violations)

# ----------------------------------------------------------------------
# Test 16: Validator rejects missing travel matrix edges
# ----------------------------------------------------------------------
def test_validator_rejects_missing_matrix_edges():
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(9, 0), service_seconds=100)
    
    # Missing the edge ("C1", "C2")
    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 100.0,
        ("DEPOT", "C2"): 150.0
    }
    inst = RouteInstance(
        route_id="TEST_MISSING_EDGE",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2},
        travel_times=travel_times
    )
    res = validate_route(inst, ["DEPOT", "C1", "C2"], require_complete_matrices=True)
    assert res.feasible is False
    assert any("Missing directed travel-time edge: (C1, C2)" in v for v in res.violations)

# ----------------------------------------------------------------------
# Test 17: Slack vs Deadline ranking divergence
# ----------------------------------------------------------------------
def test_slack_vs_deadline_ranking_divergence():
    """
    Demonstrates that operational slack (tau_i - c_i(R0)) decouples from
    raw deadline (tau_i) when cumulative service and travel times differ.
    C1: promised 08:30, completion 08:25 -> Slack = +5 min (least urgent)
    C2: promised 08:35, completion 08:45 -> Slack = -10 min (most urgent, tardy!)
    Deadline policy ranks C1 first (08:30 < 08:35).
    Slack policy ranks C2 first (-10 min slack < +5 min slack).
    """
    depot = Stop(stop_id="DEPOT", lat=0.0, lng=0.0, stop_type="Station")
    c1 = Stop(stop_id="C1", lat=0.01, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 30), service_seconds=300)
    c2 = Stop(stop_id="C2", lat=0.02, lng=0.0, stop_type="Dropoff", promised_time=dt(8, 35), service_seconds=300)

    travel_times: Dict[Tuple[str, str], float] = {
        ("DEPOT", "C1"): 1200.0,  # 20 min -> arrive 08:20, complete 08:25 <= 08:30 (Slack = +5 min)
        ("C1", "C2"): 900.0       # 15 min -> arrive 08:40, complete 08:45 > 08:35 (Slack = -10 min)
    }
    inst = RouteInstance(
        route_id="TEST_SLACK_DIFF",
        depot_id="DEPOT",
        departure_time=dt(8, 0),
        stops={"DEPOT": depot, "C1": c1, "C2": c2},
        travel_times=travel_times
    )

    baseline = ["DEPOT", "C1", "C2"]
    g_scores = {"C1": 0.0, "C2": 10.0}

    # Deadline ranking: earliest promised time first -> C1 (08:30) before C2 (08:35)
    deadline_ranked = rank_candidates(inst, baseline, policy="DEADLINE", g_scores=g_scores)
    assert [c[0] for c in deadline_ranked] == ["C1", "C2"]

    # Slack ranking: smallest/most negative slack first -> C2 (-600s) before C1 (+300s)
    slack_ranked = rank_candidates(inst, baseline, policy="SLACK", g_scores=g_scores)
    assert [c[0] for c in slack_ranked] == ["C2", "C1"]


