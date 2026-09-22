import json
from datetime import datetime, date, time
from typing import Dict, List, Optional
from .schemas import RouteInstance, Stop

def load_amazon_sample_instance(
    routes_json_path: str,
    packages_json_path: str,
    route_id: Optional[str] = None,
    base_date: date = date(2024, 5, 10)
) -> RouteInstance:
    """
    Loads Amazon sample routes and packages JSON files into a canonical RouteInstance.
    """
    with open(routes_json_path, "r", encoding="utf-8") as f:
        routes_data = json.load(f)

    with open(packages_json_path, "r", encoding="utf-8") as f:
        packages_data = json.load(f)

    if route_id is None:
        route_id = list(routes_data.keys())[0]

    r_info = routes_data[route_id]
    pkg_info = packages_data.get(route_id, {})

    # Parse departure time
    dep_time_str = r_info.get("departure_time", "08:00:00")
    t_parts = [int(p) for p in dep_time_str.split(":")]
    dep_time = datetime.combine(base_date, time(t_parts[0], t_parts[1], t_parts[2]))

    stops_dict: Dict[str, Stop] = {}
    depot_id = None

    for s_id, s_data in r_info["stops"].items():
        s_type = s_data.get("type", "Dropoff")
        if s_type == "Station":
            depot_id = s_id

        # Package info for this stop
        p_info = pkg_info.get(s_id, {})
        service_secs = float(p_info.get("planned_service_time_seconds", 120))
        vol = float(p_info.get("package_volume_cm3", 0.0))

        # Time window
        promised_dt = None
        tw = p_info.get("time_window")
        if tw and "end" in tw and tw["end"]:
            end_parts = [int(p) for p in tw["end"].split(":")]
            promised_dt = datetime.combine(base_date, time(end_parts[0], end_parts[1], end_parts[2]))

        stop = Stop(
            stop_id=s_id,
            lat=float(s_data["lat"]),
            lng=float(s_data["lng"]),
            stop_type=s_type,
            promised_time=promised_dt,
            service_seconds=service_secs,
            package_volume_cm3=vol,
            predicted_risk_pi=0.5  # default neutral prior
        )
        stops_dict[s_id] = stop

    if depot_id is None:
        raise ValueError(f"No Station/depot stop found for route {route_id}")

    return RouteInstance(
        route_id=route_id,
        depot_id=depot_id,
        departure_time=dep_time,
        stops=stops_dict
    )
