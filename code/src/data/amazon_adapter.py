import os
import json
import math
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from .schemas import Stop, RouteInstance
from ..routing.distance import haversine_distance

def parse_iso_or_time_string(time_str: Optional[Any], route_date: date) -> Optional[datetime]:
    """
    Parses a time string which may be:
      - Full timestamp: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS.fZ'
      - Time-only: 'HH:MM:SS' or 'HH:MM'
    Returns a timezone-naive UTC datetime aligned with route_date.
    """
    if time_str is None:
        return None
    if isinstance(time_str, float):
        if math.isnan(time_str):
            return None
        return None

    s = str(time_str).strip()
    if not s or s.lower() in ("none", "null", "nan", ""):
        return None

    cleaned = s.replace("T", " ").rstrip("Z")
    
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
    strict_mode: bool = True,
    fill_missing_travel_times_with_haversine: bool = False,
    haversine_speed_kmh: float = 25.0,
    default_sla_hours: Optional[float] = None
) -> Dict[str, RouteInstance]:
    """
    Adapter for the official Amazon Last Mile Routing Challenge dataset schema.
    Conforms to Section 3.2 and Section 5 of design.md.

    Strict Mode Configuration:
      - Fails explicitly on missing or corrupt files (no silent exception suppression).
      - Rejects missing travel-time matrix edges unless fill_missing_travel_times_with_haversine=True.
      - Disables synthetic four-hour SLA imputation (default_sla_hours=None in strict mode).
      - Audits actual sequence completeness and reports imputed-edge counts.
    """
    total_imputed_edges = 0
    total_missing_sla = 0

    # If routes_path is a directory, automatically discover files inside
    if os.path.isdir(routes_path):
        data_dir = routes_path
        routes_candidates = ["route_data.json", "routes.json", "routes_challenge.json"]
        packages_candidates = ["package_data.json", "packages.json", "package_data_challenge.json"]
        travel_candidates = ["travel_times.json", "travel_times_challenge.json"]
        seq_candidates = ["actual_sequences.json", "actual_sequences_challenge.json"]

        routes_path = next((os.path.join(data_dir, f) for f in routes_candidates if os.path.exists(os.path.join(data_dir, f))), None)
        if routes_path is None:
            raise FileNotFoundError(f"Could not find route data JSON file in directory '{data_dir}'")
        if packages_path is None:
            packages_path = next((os.path.join(data_dir, f) for f in packages_candidates if os.path.exists(os.path.join(data_dir, f))), None)
            if strict_mode and packages_path is None:
                raise FileNotFoundError(f"Strict mode: Required package data file not found in '{data_dir}'")
        if travel_times_path is None:
            travel_times_path = next((os.path.join(data_dir, f) for f in travel_candidates if os.path.exists(os.path.join(data_dir, f))), None)
            if strict_mode and travel_times_path is None:
                raise FileNotFoundError(f"Strict mode: Required travel times file not found in '{data_dir}'")
        if actual_sequences_path is None:
            actual_sequences_path = next((os.path.join(data_dir, f) for f in seq_candidates if os.path.exists(os.path.join(data_dir, f))), None)
            if strict_mode and actual_sequences_path is None:
                raise FileNotFoundError(f"Strict mode: Required actual sequences file not found in '{data_dir}'")

    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"Route data file does not exist: {routes_path}")

    try:
        with open(routes_path, "r", encoding="utf-8") as f:
            routes_raw = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load or parse route data file '{routes_path}': {e}") from e

    packages_raw = {}
    if packages_path:
        if not os.path.exists(packages_path):
            if strict_mode:
                raise FileNotFoundError(f"Strict mode: Packages file does not exist: {packages_path}")
        else:
            try:
                with open(packages_path, "r", encoding="utf-8") as f:
                    packages_raw = json.load(f)
            except Exception as e:
                if strict_mode:
                    raise ValueError(f"Strict mode: Failed to parse packages file '{packages_path}': {e}") from e

    travel_times_raw = {}
    if travel_times_path:
        if not os.path.exists(travel_times_path):
            if strict_mode:
                raise FileNotFoundError(f"Strict mode: Travel times file does not exist: {travel_times_path}")
        else:
            try:
                with open(travel_times_path, "r", encoding="utf-8") as f:
                    travel_times_raw = json.load(f)
            except Exception as e:
                if strict_mode:
                    raise ValueError(f"Strict mode: Failed to parse travel times file '{travel_times_path}': {e}") from e

    actual_sequences_raw = {}
    if actual_sequences_path:
        if not os.path.exists(actual_sequences_path):
            if strict_mode:
                raise FileNotFoundError(f"Strict mode: Actual sequences file does not exist: {actual_sequences_path}")
        else:
            try:
                with open(actual_sequences_path, "r", encoding="utf-8") as f:
                    actual_sequences_raw = json.load(f)
            except Exception as e:
                if strict_mode:
                    raise ValueError(f"Strict mode: Failed to parse actual sequences file '{actual_sequences_path}': {e}") from e

    instances: Dict[str, RouteInstance] = {}

    for route_id, r_info in routes_raw.items():
        date_str = r_info.get("date_YYYY_MM_DD")
        if not date_str:
            if strict_mode:
                raise ValueError(f"Strict mode: Route '{route_id}' is missing required field 'date_YYYY_MM_DD'.")
            date_str = "2021-01-01"

        try:
            route_date = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").date()
        except Exception as e:
            if strict_mode:
                raise ValueError(f"Strict mode: Route '{route_id}' has invalid date_YYYY_MM_DD '{date_str}': {e}") from e
            route_date = date(2021, 1, 1)

        dep_time_str = r_info.get("departure_time_utc")
        if not dep_time_str:
            if strict_mode:
                raise ValueError(f"Strict mode: Route '{route_id}' is missing required field 'departure_time_utc'.")
            dep_time_str = "08:00:00"

        dep_time = parse_iso_or_time_string(dep_time_str, route_date)
        if dep_time is None:
            if strict_mode:
                raise ValueError(f"Strict mode: Route '{route_id}' has unparseable departure_time_utc '{dep_time_str}'.")
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

            if earliest_deadline is None and stype != "Station":
                if default_sla_hours is not None:
                    earliest_deadline = dep_time + timedelta(hours=default_sla_hours)
                else:
                    total_missing_sla += 1

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
            if strict_mode:
                raise ValueError(f"Strict mode: Route '{route_id}' does not define a Station depot stop.")
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

        # Distance matrix (Haversine straight-line proxy)
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
                if (u, v) not in tt_matrix:
                    if u == v:
                        tt_matrix[(u, v)] = 0.0
                    elif fill_missing_travel_times_with_haversine:
                        tt_matrix[(u, v)] = (d_km / haversine_speed_kmh) * 3600.0
                        total_imputed_edges += 1
                    elif strict_mode and travel_times_raw:
                        raise ValueError(
                            f"Strict mode violation: Route {route_id} missing travel time for edge ({u} -> {v}). "
                            f"Haversine imputation is disabled in strict mode."
                        )

        actual_seq = actual_sequences_raw.get(route_id, {}).get("actual")
        if actual_seq is not None and strict_mode:
            # Audit completeness of actual sequence
            dropoff_stops = {s_id for s_id, s in stops_dict.items() if s.stop_type != "Station"}
            actual_keys = set(actual_seq.keys())
            missing_in_actual = dropoff_stops - actual_keys
            if missing_in_actual:
                raise ValueError(
                    f"Strict mode audit: Route {route_id} actual sequence is incomplete. "
                    f"Missing {len(missing_in_actual)} customer stops in actual sequence."
                )

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

    if strict_mode:
        print(f"[AmazonAdapter] Strict mode: successfully verified {len(instances)} routes. "
              f"Imputed edges: {total_imputed_edges}, Missing SLA stops: {total_missing_sla}.")

    return instances
