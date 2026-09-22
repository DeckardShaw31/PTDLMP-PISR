import math
from typing import List, Optional
from ..data.schemas import RouteInstance, ValidationResult
from .schedule import propagate_schedule
from .distance import compute_route_distance

def validate_route(
    instance: RouteInstance,
    candidate_route: List[str],
    baseline_route: Optional[List[str]] = None,
    delta: float = 0.0,
    epsilon: float = 1e-6,
    use_completion_time: bool = True,
    return_to_depot: bool = False,
    require_promised_times: bool = True,
    require_complete_matrices: bool = True
) -> ValidationResult:
    """
    Independent hard route feasibility validator implementing Section 7.1 of design.md.
    Returns ValidationResult with feasible=True/False and an exhaustive list of violations.
    Guarantees no unhandled exceptions/crashes on unknown nodes, NaNs, infs, or missing data.
    """
    violations = []

    # 1. First node must be depot
    if not candidate_route or candidate_route[0] != instance.depot_id:
        violations.append(
            f"Route must start with depot '{instance.depot_id}', got '{candidate_route[0] if candidate_route else 'empty'}'"
        )

    # 2. Check existence of all candidate stops in instance.stops
    unknown_nodes = [s for s in candidate_route if s not in instance.stops]
    if unknown_nodes:
        violations.append(f"Unknown or unassigned nodes present in route: {sorted(list(set(unknown_nodes)))[:5]}")

    expected_customers = {
        stop_id for stop_id, stop in instance.stops.items()
        if stop.stop_type == "Dropoff" or stop_id != instance.depot_id
    }
    actual_customers = candidate_route[1:] if len(candidate_route) > 1 else []

    # 3. Every assigned customer appears exactly once (permutation check)
    customer_set = set(actual_customers)
    if len(actual_customers) != len(customer_set):
        duplicates = [s for s in customer_set if actual_customers.count(s) > 1]
        violations.append(f"Duplicate customer stops found in candidate route: {duplicates[:5]}")

    missing = expected_customers - customer_set
    if missing:
        violations.append(f"Missing {len(missing)} assigned customer stops: {sorted(list(missing))[:5]}")

    # 4. If any nodes are unknown, distance/schedule propagation is unsafe to run
    if unknown_nodes:
        return ValidationResult(
            feasible=False,
            violations=violations,
            distance_km=float('inf'),
            distance_limit_km=float('inf'),
            nl=0,
            tt_minutes=float('inf'),
            served_customers=len(actual_customers)
        )

    # 5. Coordinate integrity (finite, valid lat/lng)
    for stop_id in candidate_route:
        s = instance.stops[stop_id]
        if math.isnan(s.lat) or math.isnan(s.lng) or math.isinf(s.lat) or math.isinf(s.lng):
            violations.append(f"Stop {stop_id} has NaN or Inf coordinates: ({s.lat}, {s.lng})")
        elif not (-90.0 <= s.lat <= 90.0 and -180.0 <= s.lng <= 180.0):
            violations.append(f"Stop {stop_id} has out-of-bounds coordinates: ({s.lat}, {s.lng})")

    # 6. Service duration integrity
    for stop_id in actual_customers:
        s = instance.stops[stop_id]
        if math.isnan(s.service_seconds) or math.isinf(s.service_seconds):
            violations.append(f"Stop {stop_id} has NaN or Inf service duration: {s.service_seconds}")
        elif s.service_seconds < 0:
            violations.append(f"Stop {stop_id} has negative service duration: {s.service_seconds}")

    # 7. Promised time existence (Section 3.1 & 7.1)
    if require_promised_times:
        missing_deadlines = [s for s in actual_customers if instance.stops[s].promised_time is None]
        if missing_deadlines:
            violations.append(f"Customer stops missing promised delivery time: {missing_deadlines[:5]}")

    # 8. Matrix edge completeness (check precomputed travel times & distances if provided)
    if require_complete_matrices:
        for i in range(len(candidate_route) - 1):
            u, v = candidate_route[i], candidate_route[i + 1]
            if instance.distances is not None and (u, v) not in instance.distances:
                violations.append(f"Missing directed distance edge: ({u}, {v})")
            if instance.travel_times is not None and (u, v) not in instance.travel_times:
                violations.append(f"Missing directed travel-time edge: ({u}, {v})")

    # 9. Cumulative distance constraint check: D(R') <= (1 + delta) * D(R0) + eps
    try:
        cand_dist = compute_route_distance(instance, candidate_route, return_to_depot=return_to_depot)
    except Exception as e:
        violations.append(f"Distance calculation failed: {str(e)}")
        cand_dist = float('inf')

    dist_limit = float('inf')
    if baseline_route is not None:
        try:
            base_dist = compute_route_distance(instance, baseline_route, return_to_depot=return_to_depot)
            dist_limit = (1.0 + delta) * base_dist + epsilon
            if cand_dist > dist_limit:
                violations.append(
                    f"Distance limit violated: candidate {cand_dist:.4f} km > limit {dist_limit:.4f} km "
                    f"(base {base_dist:.4f} km, delta={delta})"
                )
        except Exception as e:
            violations.append(f"Baseline distance calculation failed: {str(e)}")

    # 10. Monotonic non-decreasing schedule propagation
    sched = None
    try:
        sched = propagate_schedule(instance, candidate_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
        times = [sched.arrival_times[s] for s in candidate_route]
        for i in range(1, len(times)):
            if times[i] < times[i - 1]:
                violations.append(f"Propagated arrival times are decreasing between stop {candidate_route[i-1]} and {candidate_route[i]}")
                break
    except Exception as e:
        violations.append(f"Schedule propagation failed with exception: {str(e)}")

    is_feasible = (len(violations) == 0)
    return ValidationResult(
        feasible=is_feasible,
        violations=violations,
        distance_km=cand_dist,
        distance_limit_km=dist_limit,
        nl=sched.total_lateness if sched else 0,
        tt_minutes=sched.total_tardiness if sched else 0.0,
        served_customers=len(actual_customers)
    )
