import os
import math
from datetime import datetime, date, time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ..data.schemas import RouteInstance, Stop
from ..routing.distance import haversine_distance
from ..routing.schedule import propagate_schedule

LEGAL_EX_ANTE_FEATURES = [
    "dist_from_depot_km",
    "local_density_2km",
    "planned_service_seconds",
    "package_volume_cm3",
    "num_packages",
    "time_to_deadline_minutes",
    "departure_hour",
    "day_of_week",
    "route_total_stops",
    "route_total_service_min"
]

FORBIDDEN_LEAKAGE_SUBSTRINGS = [
    "actual", "realized", "observed", "future", "post_",
    "delivery_time", "completion_time", "arrival_time", "sequence"
]

def assert_no_data_leakage(feature_names: List[str]) -> None:
    """
    Enforces strict ex-ante feature contract (Section 8.1 & 8.2 of design.md).
    Raises ValueError if any feature column matches forbidden post-outcome patterns.
    """
    for f in feature_names:
        lower_f = f.lower()
        for forbidden in FORBIDDEN_LEAKAGE_SUBSTRINGS:
            if forbidden in lower_f:
                raise ValueError(
                    f"Data Leakage Violation: Feature '{f}' contains forbidden token '{forbidden}'. "
                    f"Post-decision or outcome-derived features are strictly prohibited."
                )

def extract_ex_ante_features_for_route(
    instance: RouteInstance
) -> pd.DataFrame:
    """
    Extracts strictly ex-ante features for all customer stops on a given route.
    Guarantees no post-departure outcome leakage.
    """
    depot_id = instance.depot_id
    depot = instance.stops[depot_id]
    dep_time = instance.departure_time
    dep_hour = dep_time.hour + dep_time.minute / 60.0
    day_of_week = dep_time.weekday()

    customers = [
        s_id for s_id, s in instance.stops.items()
        if s.stop_type == "Dropoff" or s_id != depot_id
    ]

    total_route_stops = len(customers)
    total_service_min = sum(instance.stops[c].service_seconds for c in customers) / 60.0

    rows = []
    for c_id in customers:
        stop = instance.stops[c_id]
        
        # 1. Geographic distance from depot (Haversine proxy)
        dist_depot = haversine_distance(depot.lat, depot.lng, stop.lat, stop.lng)

        # 2. Local neighborhood density (stops within 2.0 km)
        density_2km = 0
        for other_id in customers:
            if other_id != c_id:
                other_stop = instance.stops[other_id]
                d = haversine_distance(stop.lat, stop.lng, other_stop.lat, other_stop.lng)
                if d <= 2.0:
                    density_2km += 1

        # 3. Time until deadline
        if stop.promised_time is not None:
            time_to_deadline_min = (stop.promised_time - dep_time).total_seconds() / 60.0
        else:
            time_to_deadline_min = 240.0  # Default 4 hours if unspecified

        num_pkgs = stop.custom_data.get("num_packages", 1) if stop.custom_data else 1

        row = {
            "route_id": instance.route_id,
            "route_date": instance.route_date or dep_time.strftime("%Y-%m-%d"),
            "customer_id": c_id,
            "dist_from_depot_km": dist_depot,
            "local_density_2km": density_2km,
            "planned_service_seconds": stop.service_seconds,
            "package_volume_cm3": stop.package_volume_cm3,
            "num_packages": num_pkgs,
            "time_to_deadline_minutes": time_to_deadline_min,
            "departure_hour": dep_hour,
            "day_of_week": day_of_week,
            "route_total_stops": total_route_stops,
            "route_total_service_min": total_service_min
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def compute_ground_truth_labels(
    instance: RouteInstance,
    use_actual_sequence_if_available: bool = True,
    strict_sequence_audit: bool = True
) -> Dict[str, int]:
    """
    Computes ground-truth binary lateness labels as a route-propagated promised-time violation proxy.
    Propagates the historical actual sequence with planned service times and historical travel times.
    If strict_sequence_audit is True, asserts that the actual sequence is complete.
    """
    depot_id = instance.depot_id
    customers = [
        s_id for s_id, s in instance.stops.items()
        if s.stop_type == "Dropoff" or s_id != depot_id
    ]

    route_seq = None
    if use_actual_sequence_if_available and instance.actual_sequence:
        # Sort by actual sequence number
        sorted_stops = sorted(instance.actual_sequence.items(), key=lambda x: x[1])
        seq = [s[0] for s in sorted_stops if s[0] in instance.stops]
        if seq and seq[0] != depot_id:
            seq = [depot_id] + [s for s in seq if s != depot_id]
        if len(seq) == len(instance.stops):
            route_seq = seq
        elif strict_sequence_audit:
            raise ValueError(
                f"Route '{instance.route_id}' has incomplete actual sequence: "
                f"{len(seq)} stops in sequence vs {len(instance.stops)} total stops."
            )

    if route_seq is None:
        if strict_sequence_audit and use_actual_sequence_if_available:
            raise ValueError(
                f"Route '{instance.route_id}' has no actual sequence. "
                "Silent baseline fallback is rejected under strict sequence audit."
            )
        from ..routing.baseline import build_nearest_neighbor_baseline
        route_seq = build_nearest_neighbor_baseline(instance)

    sched = propagate_schedule(instance, route_seq)
    labels = {c: sched.lateness.get(c, 0) for c in customers}
    return labels
