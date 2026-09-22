import os
import sys
import random
import pytest
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import RouteInstance, Stop
from src.routing.distance import (
    compute_route_distance,
    compute_relocation_distance_delta,
    get_edge_distance
)

def create_synthetic_instance(num_stops: int = 15, seed: int = 42):
    rng = random.Random(seed)
    stops = {
        "DEPOT": Stop("DEPOT", lat=34.0, lng=-118.0, stop_type="Station")
    }
    for i in range(1, num_stops):
        sid = f"CUST_{i:02d}"
        stops[sid] = Stop(
            sid,
            lat=34.0 + rng.uniform(-0.15, 0.15),
            lng=-118.0 + rng.uniform(-0.15, 0.15),
            stop_type="Dropoff"
        )
    inst = RouteInstance(
        route_id=f"SYNTH_{seed}",
        depot_id="DEPOT",
        departure_time=datetime(2021, 1, 1, 8, 0),
        stops=stops
    )
    route = ["DEPOT"] + [f"CUST_{i:02d}" for i in range(1, num_stops)]
    return inst, route

def test_distance_delta_random_routes_property():
    """
    Property test: verifies exact mathematical equivalence (|Delta D_local - Delta D_full| < 1e-9)
    between O(1) compute_relocation_distance_delta and O(N) compute_route_distance
    across random instances, adjacent moves, non-adjacent moves, last-stop relocations,
    and both return-to-depot configurations.
    """
    total_evaluated = 0
    seeds = [42, 123, 999]

    for seed in seeds:
        for num_stops in [6, 12, 20]:
            rng = random.Random(seed)
            stops = {
                "DEPOT": Stop("DEPOT", lat=34.0, lng=-118.0, stop_type="Station")
            }
            for i in range(1, num_stops):
                sid = f"CUST_{i:02d}"
                stops[sid] = Stop(
                    sid,
                    lat=34.0 + rng.uniform(-0.2, 0.2),
                    lng=-118.0 + rng.uniform(-0.2, 0.2),
                    stop_type="Dropoff"
                )
            inst = RouteInstance(
                route_id=f"SYNTH_{seed}_{num_stops}",
                depot_id="DEPOT",
                departure_time=datetime(2021, 1, 1, 8, 0),
                stops=stops
            )
            route = ["DEPOT"] + [f"CUST_{i:02d}" for i in range(1, num_stops)]

            for return_to_depot in [False, True]:
                base_dist = compute_route_distance(inst, route, return_to_depot=return_to_depot)

                for k in range(1, len(route)):
                    for j in range(1, k):
                        # 1. Local O(1) delta
                        delta_local = compute_relocation_distance_delta(
                            inst, route, k=k, j=j, return_to_depot=return_to_depot
                        )

                        # 2. Full route distance re-computation
                        cand_route = list(route)
                        val = cand_route.pop(k)
                        cand_route.insert(j, val)
                        cand_dist = compute_route_distance(inst, cand_route, return_to_depot=return_to_depot)
                        delta_full = cand_dist - base_dist

                        # 3. Assert exact equivalence within floating point tolerance
                        assert abs(delta_local - delta_full) < 1e-9, (
                            f"Mismatch for seed={seed}, n={num_stops}, return_to_depot={return_to_depot}, "
                            f"k={k} ({route[k]}), j={j} ({route[j]}): "
                            f"local={delta_local:.8f}, full={delta_full:.8f}"
                        )
                        total_evaluated += 1

    assert total_evaluated > 1000, f"Expected >1000 moves tested, got {total_evaluated}"

def test_distance_delta_specific_scenarios():
    """
    Explicitly tests targeted structural edge cases:
      1. Adjacent moves (j == k - 1)
      2. Non-adjacent moves (j < k - 1)
      3. Last-stop relocation (k == len(route) - 1)
      4. Forward relocation to position 1 (immediately after depot)
      5. Both return_to_depot=False and return_to_depot=True
    """
    stops = {
        "DEPOT": Stop("DEPOT", lat=0.0, lng=0.0, stop_type="Station"),
        "S1": Stop("S1", lat=0.01, lng=0.02, stop_type="Dropoff"),
        "S2": Stop("S2", lat=0.05, lng=0.01, stop_type="Dropoff"),
        "S3": Stop("S3", lat=0.03, lng=0.06, stop_type="Dropoff"),
        "S4": Stop("S4", lat=0.08, lng=0.04, stop_type="Dropoff"),
        "S5": Stop("S5", lat=0.02, lng=0.09, stop_type="Dropoff"),
    }
    inst = RouteInstance("TEST_SPECIFIC", "DEPOT", datetime(2021, 1, 1, 8, 0), stops)
    route = ["DEPOT", "S1", "S2", "S3", "S4", "S5"]
    n = len(route)

    for ret in [False, True]:
        base_dist = compute_route_distance(inst, route, return_to_depot=ret)

        # Case 1: Adjacent move (S3 moved to position of S2)
        k_adj, j_adj = 3, 2
        d_adj = compute_relocation_distance_delta(inst, route, k_adj, j_adj, return_to_depot=ret)
        r_cand = list(route); r_cand.insert(j_adj, r_cand.pop(k_adj))
        assert abs(d_adj - (compute_route_distance(inst, r_cand, return_to_depot=ret) - base_dist)) < 1e-10

        # Case 2: Non-adjacent move (S4 moved to position 1)
        k_nonadj, j_nonadj = 4, 1
        d_nonadj = compute_relocation_distance_delta(inst, route, k_nonadj, j_nonadj, return_to_depot=ret)
        r_cand = list(route); r_cand.insert(j_nonadj, r_cand.pop(k_nonadj))
        assert abs(d_nonadj - (compute_route_distance(inst, r_cand, return_to_depot=ret) - base_dist)) < 1e-10

        # Case 3: Last-stop relocation (S5 at k=n-1 moved to j=1)
        k_last, j_first = n - 1, 1
        d_last = compute_relocation_distance_delta(inst, route, k_last, j_first, return_to_depot=ret)
        r_cand = list(route); r_cand.insert(j_first, r_cand.pop(k_last))
        assert abs(d_last - (compute_route_distance(inst, r_cand, return_to_depot=ret) - base_dist)) < 1e-10

def test_distance_delta_invalid_indices_raise():
    stops = {
        "DEPOT": Stop("DEPOT", lat=0.0, lng=0.0, stop_type="Station"),
        "S1": Stop("S1", lat=0.01, lng=0.02, stop_type="Dropoff"),
        "S2": Stop("S2", lat=0.05, lng=0.01, stop_type="Dropoff"),
    }
    inst = RouteInstance("TEST_INVALID", "DEPOT", datetime(2021, 1, 1, 8, 0), stops)
    route = ["DEPOT", "S1", "S2"]

    # j >= k
    with pytest.raises(ValueError, match="Invalid relocation indices"):
        compute_relocation_distance_delta(inst, route, k=1, j=1)

    # j < 1 (relocating before depot)
    with pytest.raises(ValueError, match="Invalid relocation indices"):
        compute_relocation_distance_delta(inst, route, k=2, j=0)

    # k >= n
    with pytest.raises(ValueError, match="Invalid relocation indices"):
        compute_relocation_distance_delta(inst, route, k=3, j=1)
