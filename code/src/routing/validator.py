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
    return_to_depot: bool = False
) -> ValidationResult:
    """
    Independent hard route feasibility validator implementing Section 7.1 of design.md.
    Returns ValidationResult with feasible=True/False and a list of violations.
    """
    violations = []

    # 1. First node must be depot
    if not candidate_route or candidate_route[0] != instance.depot_id:
        violations.append(f"Route must start with depot '{instance.depot_id}', got '{candidate_route[0] if candidate_route else 'empty'}'")

    # Customer set
    expected_customers = {
        stop_id for stop_id, stop in instance.stops.items()
        if stop.stop_type == "Dropoff" or stop_id != instance.depot_id
    }
    actual_customers = candidate_route[1:] if len(candidate_route) > 1 else []

    # 2. Every assigned customer appears exactly once
    customer_set = set(actual_customers)
    if len(actual_customers) != len(customer_set):
        violations.append(f"Duplicate customer stops found in candidate route: {len(actual_customers)} items but {len(customer_set)} unique")

    missing = expected_customers - customer_set
    if missing:
        violations.append(f"Missing {len(missing)} assigned customer stops: {sorted(list(missing))[:5]}")

    # 3. No unassigned or unknown node appears
    unknown = customer_set - expected_customers
    if unknown:
        violations.append(f"Unknown customer stops found in candidate route: {sorted(list(unknown))[:5]}")

    # 4. Valid coordinates
    for stop_id in candidate_route:
        if stop_id in instance.stops:
            s = instance.stops[stop_id]
            if not (-90.0 <= s.lat <= 90.0 and -180.0 <= s.lng <= 180.0):
                violations.append(f"Stop {stop_id} has invalid coordinates: ({s.lat}, {s.lng})")
            if math.isnan(s.lat) or math.isnan(s.lng):
                violations.append(f"Stop {stop_id} has NaN coordinates")

    # 5. Feasible service duration
    for stop_id in actual_customers:
        if stop_id in instance.stops:
            s = instance.stops[stop_id]
            if s.service_seconds < 0 or math.isnan(s.service_seconds):
                violations.append(f"Stop {stop_id} has negative or NaN service duration: {s.service_seconds}")

    # Distance tolerance check (Check 9)
    cand_dist = compute_route_distance(instance, candidate_route, return_to_depot=return_to_depot)
    dist_limit = float('inf')
    if baseline_route is not None:
        base_dist = compute_route_distance(instance, baseline_route, return_to_depot=return_to_depot)
        dist_limit = (1.0 + delta) * base_dist + epsilon
        if cand_dist > dist_limit:
            violations.append(f"Distance limit violated: candidate {cand_dist:.4f} km > limit {dist_limit:.4f} km (base {base_dist:.4f} km, delta={delta})")

    # 8. Time propagation non-decreasing
    sched = None
    if not violations or (len(violations) == 1 and "Distance limit" in violations[0]):
        try:
            sched = propagate_schedule(instance, candidate_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
            # Check times are non-decreasing
            times = [sched.arrival_times[s] for s in candidate_route]
            for i in range(1, len(times)):
                if times[i] < times[i - 1]:
                    violations.append(f"Propagated arrival times are decreasing between stop {candidate_route[i-1]} and {candidate_route[i]}")
                    break
        except Exception as e:
            violations.append(f"Time propagation failed with error: {str(e)}")

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
