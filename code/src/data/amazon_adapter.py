import os
import json
import math
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from .schemas import Stop, RouteInstance
from ..routing.distance import haversine_distance

def parse_iso_or_time_string(time_str: Optional[str], route_date: date) -> Optional[datetime]:
    """
    Parses a time string which may be:
      - Full timestamp: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS.fZ'
      - Time-only: 'HH:MM:SS' or 'HH:MM'
    Returns a timezone-naive UTC datetime aligned with route_date.
    """
    if not time_str or time_str.lower() in ("none", "null", ""):
        return None

    cleaned = time_str.strip().replace("T", " ").rstrip("Z")
    
    # Try parsing full datetime
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass

    # Try parsing time-only
    for fmt in ("%H:%M:%S", "%H:%M", "%H:%M:%S.%f"):
        try:
            t = datetime.strptime(cleaned, fmt).time()
            return datetime.combine(route_date, t)
        except ValueError:
            pass

    return None

def load_official_amazon_dataset(
    routes_path: str,
    packages_path: Optional[str] = None,
    travel_times_path: Optional[str] = None,
    actual_sequences_path: Optional[str] = None,
    fill_missing_travel_times_with_haversine: bool = True,
    haversine_speed_kmh: float = 25.0
) -> Dict[str, RouteInstance]:
    """
    Adapter for the official Amazon Last Mile Routing Challenge dataset schema.
    Conforms to Section 3.2 and Section 5 of design.md.

    Supports:
      1. routes.json: {route_id: {station_code, date_YYYY_MM_DD, departure_time_utc, executor_capacity_cm3, stops: {stop_id: {lat, lng, type, zone_id}}}}
      2. package_data.json: {route_id: {stop_id: {pkg_id: {time_window: {start_time_utc, end_time_utc}, planned_service_time_seconds, dimensions: {depth_cm, height_cm, width_cm}}}}}
      3. travel_times.json: {route_id: {from_stop_id: {to_stop_id: travel_seconds}}}
      4. actual_sequences.json: {route_id: {'actual': {stop_id: sequence_num}}}

    Multi-package aggregation at stop level:
      - service_seconds = sum(planned_service_time_seconds)
      - promised_time = min(valid package time_window.end_time_utc)
      - package_volume_cm3 = sum(depth * height * width)
      - predicted_risk_pi = 0.0 (populated subsequently by RQ1 prediction model)
    """
    with open(routes_path, "r", encoding="utf-8") as f:
        routes_raw = json.load(f)

    packages_raw = {}
    if packages_path and os.path.exists(packages_path):
        with open(packages_path, "r", encoding="utf-8") as f:
            packages_raw = json.load(f)

    travel_times_raw = {}
    if travel_times_path and os.path.exists(travel_times_path):
        with open(travel_times_path, "r", encoding="utf-8") as f:
            travel_times_raw = json.load(f)

    actual_sequences_raw = {}
    if actual_sequences_path and os.path.exists(actual_sequences_path):
        with open(actual_sequences_path, "r", encoding="utf-8") as f:
            actual_sequences_raw = json.load(f)

    instances: Dict[str, RouteInstance] = {}

    for route_id, r_info in routes_raw.items():
        date_str = r_info.get("date_YYYY_MM_DD", "2021-01-01")
        try:
            route_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            route_date = date(2021, 1, 1)

        dep_time_str = r_info.get("departure_time_utc", "08:00:00")
        dep_time = parse_iso_or_time_string(dep_time_str, route_date)
        if dep_time is None:
            dep_time = datetime.combine(route_date, time(8, 0, 0))

        stops_dict: Dict[str, Stop] = {}
        depot_id = None

        raw_stops = r_info.get("stops", {})
        for stop_id, s_info in raw_stops.items():
            lat = float(s_info.get("lat", 0.0))
            lng = float(s_info.get("lng", 0.0))
            stype = s_info.get("type", "Dropoff")
            zone_id = s_info.get("zone_id")

            if stype == "Station":
                depot_id = stop_id

            # Package aggregation for this stop
            stop_pkgs = packages_raw.get(route_id, {}).get(stop_id, {})
            total_service_sec = 0.0
            total_volume = 0.0
            earliest_deadline: Optional[datetime] = None
            pkg_count = 0

            if isinstance(stop_pkgs, dict):
                # Check whether it is official nested (pkg_id -> dict) or flat simplified
                is_nested = any(isinstance(v, dict) and ("time_window" in v or "planned_service_time_seconds" in v) for v in stop_pkgs.values())
                
                if is_nested:
                    pkg_count = len(stop_pkgs)
                    for pkg_id, p_data in stop_pkgs.items():
                        # Service time
                        srv = float(p_data.get("planned_service_time_seconds", 0.0) or 0.0)
                        total_service_sec += srv
                        
                        # Volume
                        dims = p_data.get("dimensions") or {}
                        d = float(dims.get("depth_cm", 0.0) or 0.0)
                        h = float(dims.get("height_cm", 0.0) or 0.0)
                        w = float(dims.get("width_cm", 0.0) or 0.0)
                        if d > 0 and h > 0 and w > 0:
                            total_volume += (d * h * w)

                        # Time window
                        tw = p_data.get("time_window") or {}
                        end_str = tw.get("end_time_utc") or tw.get("end")
                        deadline_dt = parse_iso_or_time_string(end_str, route_date)
                        if deadline_dt:
                            if earliest_deadline is None or deadline_dt < earliest_deadline:
                                earliest_deadline = deadline_dt
                else:
                    # Simplified flat format
                    total_service_sec = float(stop_pkgs.get("planned_service_time_seconds", 180.0))
                    total_volume = float(stop_pkgs.get("package_volume_cm3", 2500.0))
                    tw = stop_pkgs.get("time_window") or {}
                    end_str = tw.get("end_time_utc") or tw.get("end")
                    earliest_deadline = parse_iso_or_time_string(end_str, route_date)
                    pkg_count = 1

            stop_obj = Stop(
                stop_id=stop_id,
                lat=lat,
                lng=lng,
                stop_type=stype,
                promised_time=earliest_deadline,
                service_seconds=total_service_sec if stype != "Station" else 0.0,
                package_volume_cm3=total_volume,
                predicted_risk_pi=0.0,
                custom_data={
                    "zone_id": zone_id,
                    "num_packages": pkg_count,
                    "station_code": r_info.get("station_code")
                }
            )
            stops_dict[stop_id] = stop_obj

        if depot_id is None:
            # Fallback: create or find station
            for sid, sobj in stops_dict.items():
                if "station" in sid.lower() or "depot" in sid.lower():
                    depot_id = sid
                    sobj.stop_type = "Station"
                    break
            if depot_id is None and stops_dict:
                depot_id = list(stops_dict.keys())[0]
                stops_dict[depot_id].stop_type = "Station"

        # Directed travel times
        tt_matrix: Dict[Tuple[str, str], float] = {}
        raw_tt = travel_times_raw.get(route_id, {})
        for u, targets in raw_tt.items():
            if u in stops_dict:
                for v, sec in targets.items():
                    if v in stops_dict:
                        tt_matrix[(u, v)] = float(sec)

        # Distance matrix (Haversine)
        dist_matrix: Dict[Tuple[str, str], float] = {}
        stop_ids = list(stops_dict.keys())
        for i in range(len(stop_ids)):
            u = stop_ids[i]
            su = stops_dict[u]
            for j in range(len(stop_ids)):
                v = stop_ids[j]
                sv = stops_dict[v]
                d_km = haversine_distance(su.lat, su.lng, sv.lat, sv.lng)
                dist_matrix[(u, v)] = d_km

                # Impute missing travel times via Haversine if requested
                if fill_missing_travel_times_with_haversine and (u, v) not in tt_matrix:
                    if u == v:
                        tt_matrix[(u, v)] = 0.0
                    else:
                        # Convert km to seconds at given average km/h
                        tt_matrix[(u, v)] = (d_km / haversine_speed_kmh) * 3600.0

        actual_seq = actual_sequences_raw.get(route_id, {}).get("actual")

        instance = RouteInstance(
            route_id=route_id,
            depot_id=depot_id or "DEPOT",
            departure_time=dep_time,
            stops=stops_dict,
            travel_times=tt_matrix if tt_matrix else None,
            distances=dist_matrix,
            route_date=date_str,
            station_code=r_info.get("station_code"),
            executor_capacity_cm3=r_info.get("executor_capacity_cm3"),
            route_score=r_info.get("route_score"),
            actual_sequence=actual_seq
        )
        instances[route_id] = instance

    return instances
