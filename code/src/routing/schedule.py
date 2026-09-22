from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ..data.schemas import RouteInstance, ScheduleResult
from .distance import get_edge_travel_time, compute_route_distance

def propagate_schedule(
    instance: RouteInstance,
    route: List[str],
    use_completion_time: bool = True,
    return_to_depot: bool = False
) -> ScheduleResult:
    """
    Propagates route times sequentially from departure at depot.
    Computes arrival, completion, lateness L_i(R) in {0, 1},
    tardiness T_i(R) in minutes, NL(R), TT(R), and total distance D(R).
    
    route: List of stop IDs where route[0] is the depot.
    """
    arrival_times: Dict[str, datetime] = {}
    completion_times: Dict[str, datetime] = {}
    lateness: Dict[str, int] = {}
    tardiness_minutes: Dict[str, float] = {}

    current_time = instance.departure_time
    total_lateness = 0
    total_tardiness = 0.0

    if not route:
        return ScheduleResult(
            route=[],
            arrival_times={},
            completion_times={},
            lateness={},
            tardiness_minutes={},
            total_lateness=0,
            total_tardiness=0.0,
            total_distance=0.0
        )

    # Depot departure
    depot_id = route[0]
    arrival_times[depot_id] = current_time
    completion_times[depot_id] = current_time

    # Propagate through customer stops
    for idx in range(1, len(route)):
        prev_stop = route[idx - 1]
        curr_stop = route[idx]
        stop_obj = instance.stops[curr_stop]

        travel_secs = get_edge_travel_time(instance, prev_stop, curr_stop)
        arrival = current_time + timedelta(seconds=travel_secs)
        completion = arrival + timedelta(seconds=stop_obj.service_seconds)

        arrival_times[curr_stop] = arrival
        completion_times[curr_stop] = completion
        current_time = completion

        eval_time = completion if use_completion_time else arrival

        if stop_obj.promised_time is not None:
            if eval_time > stop_obj.promised_time:
                is_late = 1
                diff_seconds = (eval_time - stop_obj.promised_time).total_seconds()
                tard_min = diff_seconds / 60.0
            else:
                is_late = 0
                tard_min = 0.0
        else:
            is_late = 0
            tard_min = 0.0

        lateness[curr_stop] = is_late
        tardiness_minutes[curr_stop] = tard_min
        total_lateness += is_late
        total_tardiness += tard_min

    total_dist = compute_route_distance(instance, route, return_to_depot=return_to_depot)

    return ScheduleResult(
        route=list(route),
        arrival_times=arrival_times,
        completion_times=completion_times,
        lateness=lateness,
        tardiness_minutes=tardiness_minutes,
        total_lateness=total_lateness,
        total_tardiness=total_tardiness,
        total_distance=total_dist
    )
