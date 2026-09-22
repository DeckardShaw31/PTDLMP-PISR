import os
import json
import math
import random
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Any

from ..routing.distance import haversine_distance

def generate_official_amazon_dataset(
    output_dir: str,
    num_routes: int = 30,
    start_date: str = "2018-07-01",
    num_days: int = 10,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generates a realistic multi-route, multi-day benchmark dataset strictly following
    the official Amazon Last Mile Routing Challenge schema (Section 3.2 & 5 of design.md).
    Produces 4 JSON files:
      - routes.json
      - package_data.json
      - travel_times.json
      - actual_sequences.json
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(random_seed)

    base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    dates = [base_date + timedelta(days=i) for i in range(num_days)]

    routes_data: Dict[str, Any] = {}
    packages_data: Dict[str, Any] = {}
    travel_times_data: Dict[str, Any] = {}
    actual_sequences_data: Dict[str, Any] = {}

    station_codes = ["DLA3", "DLA7", "DLA9"]
    center_coords = {
        "DLA3": (34.0522, -118.2437),  # Los Angeles Downtown
        "DLA7": (34.1478, -118.1445),  # Pasadena
        "DLA9": (33.9425, -118.4081)   # LAX / Inglewood
    }

    routes_per_day = max(1, num_routes // num_days)
    route_counter = 0

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        for r_idx in range(routes_per_day):
            route_counter += 1
            if route_counter > num_routes:
                break

            r_id = f"RouteID_{route_counter:03d}_{date_str.replace('-', '')}"
            st_code = rng.choice(station_codes)
            depot_lat, depot_lng = center_coords[st_code]

            # Departure time between 07:30 and 09:30 UTC
            dep_hour = rng.choice([7, 8, 9])
            dep_min = rng.choice([0, 15, 30, 45])
            dep_time_str = f"{dep_hour:02d}:{dep_min:02d}:00"
            dep_dt = datetime.combine(d, time(dep_hour, dep_min))

            # Number of dropoffs between 15 and 25
            num_stops = rng.randint(15, 25)

            # Generate stops with geographic structure
            stops_dict: Dict[str, Any] = {
                "DEPOT": {
                    "lat": depot_lat,
                    "lng": depot_lng,
                    "type": "Station",
                    "zone_id": None
                }
            }

            # Geographic layout archetype: cluster or corridor
            archetype = rng.choice(["cluster", "corridor", "radial"])
            for s_num in range(1, num_stops + 1):
                sid = f"Stop_{s_num:02d}"
                if archetype == "cluster":
                    # Two sub-clusters
                    cluster_center = (depot_lat + rng.choice([0.03, -0.03]), depot_lng + rng.choice([0.03, -0.03]))
                    lat = cluster_center[0] + rng.gauss(0, 0.01)
                    lng = cluster_center[1] + rng.gauss(0, 0.01)
                elif archetype == "corridor":
                    # Linear corridor
                    step = s_num * 0.005
                    lat = depot_lat + step + rng.gauss(0, 0.003)
                    lng = depot_lng + step * 0.5 + rng.gauss(0, 0.003)
                else:
                    # Radial loop
                    angle = (s_num / num_stops) * 2 * math.pi
                    rad = 0.02 + 0.02 * (s_num / num_stops)
                    lat = depot_lat + rad * math.cos(angle)
                    lng = depot_lng + rad * math.sin(angle)

                zone_char = chr(65 + (s_num % 4))
                stops_dict[sid] = {
                    "lat": round(lat, 6),
                    "lng": round(lng, 6),
                    "type": "Dropoff",
                    "zone_id": f"{zone_char}-{(s_num % 3) + 1}.1"
                }

            routes_data[r_id] = {
                "station_code": st_code,
                "date_YYYY_MM_DD": date_str,
                "departure_time_utc": dep_time_str,
                "executor_capacity_cm3": 4500000.0,
                "route_score": rng.choice(["High", "Medium"]),
                "stops": stops_dict
            }

            # Generate package data (nested official format: stop -> pkg_id -> {tw, service, dim})
            pkg_data_for_route: Dict[str, Any] = {}
            for s_num in range(1, num_stops + 1):
                sid = f"Stop_{s_num:02d}"
                num_pkgs_at_stop = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]

                # Distance from depot
                dist_km = haversine_distance(depot_lat, depot_lng, stops_dict[sid]["lat"], stops_dict[sid]["lng"])
                # Estimate travel accumulation for realistic deadline synthesis
                est_minutes = (dist_km / 25.0) * 60.0 + (s_num * 5.0)

                # Deadline tightness: 25% tight, 50% medium, 25% loose
                tightness_category = rng.choices(["tight", "medium", "loose"], weights=[0.25, 0.50, 0.25])[0]
                if tightness_category == "tight":
                    deadline_offset_min = est_minutes + rng.uniform(-10, 20)
                elif tightness_category == "medium":
                    deadline_offset_min = est_minutes + rng.uniform(30, 90)
                else:
                    deadline_offset_min = est_minutes + rng.uniform(120, 240)

                deadline_dt = dep_dt + timedelta(minutes=max(20.0, deadline_offset_min))
                deadline_str = deadline_dt.strftime("%Y-%m-%d %H:%M:%S")

                pkgs_at_stop: Dict[str, Any] = {}
                for p_i in range(1, num_pkgs_at_stop + 1):
                    pkg_id = f"Pkg_{r_id}_{sid}_{p_i}"
                    service_sec = float(rng.choice([60, 90, 120, 180]))
                    depth = round(rng.uniform(15, 40), 1)
                    height = round(rng.uniform(10, 30), 1)
                    width = round(rng.uniform(10, 30), 1)

                    # Only some packages have explicit time windows
                    has_tw = (p_i == 1) or (rng.random() < 0.3)
                    tw_dict = {
                        "start_time_utc": dep_time_str,
                        "end_time_utc": deadline_str if has_tw else None
                    }

                    pkgs_at_stop[pkg_id] = {
                        "scan_status": "DELIVERED",
                        "time_window": tw_dict,
                        "planned_service_time_seconds": service_sec,
                        "dimensions": {
                            "depth_cm": depth,
                            "height_cm": height,
                            "width_cm": width
                        }
                    }

                pkg_data_for_route[sid] = pkgs_at_stop

            packages_data[r_id] = pkg_data_for_route

            # Generate pairwise travel times with urban congestion noise
            all_node_ids = list(stops_dict.keys())
            tt_matrix: Dict[str, Dict[str, float]] = {}
            for u in all_node_ids:
                tt_matrix[u] = {}
                u_lat, u_lng = stops_dict[u]["lat"], stops_dict[u]["lng"]
                for v in all_node_ids:
                    if u == v:
                        tt_matrix[u][v] = 0.0
                    else:
                        v_lat, v_lng = stops_dict[v]["lat"], stops_dict[v]["lng"]
                        d_km = haversine_distance(u_lat, u_lng, v_lat, v_lng)
                        # Speed 20-30 km/h + asymmetric congestion noise
                        speed_kmh = rng.uniform(20.0, 30.0)
                        sec = (d_km / speed_kmh) * 3600.0 + rng.uniform(10, 45)
                        tt_matrix[u][v] = round(sec, 1)

            travel_times_data[r_id] = tt_matrix

            # Generate realistic actual historical sequence
            actual_seq_map: Dict[str, int] = {"DEPOT": 0}
            cust_nodes = [s for s in all_node_ids if s != "DEPOT"]
            # Courier visits nearby stops with some human variability
            curr = "DEPOT"
            unvis = set(cust_nodes)
            step_idx = 1
            while unvis:
                # 85% pick closest, 15% random choice
                if rng.random() < 0.85:
                    next_stop = min(unvis, key=lambda x: tt_matrix[curr][x])
                else:
                    next_stop = rng.choice(list(unvis))
                actual_seq_map[next_stop] = step_idx
                step_idx += 1
                unvis.remove(next_stop)
                curr = next_stop

            actual_sequences_data[r_id] = {"actual": actual_seq_map}

    # Write files
    routes_fp = os.path.join(output_dir, "routes_challenge.json")
    packages_fp = os.path.join(output_dir, "package_data_challenge.json")
    travel_fp = os.path.join(output_dir, "travel_times_challenge.json")
    sequences_fp = os.path.join(output_dir, "actual_sequences_challenge.json")

    with open(routes_fp, "w", encoding="utf-8") as f:
        json.dump(routes_data, f, indent=2)
    with open(packages_fp, "w", encoding="utf-8") as f:
        json.dump(packages_data, f, indent=2)
    with open(travel_fp, "w", encoding="utf-8") as f:
        json.dump(travel_times_data, f, indent=2)
    with open(sequences_fp, "w", encoding="utf-8") as f:
        json.dump(actual_sequences_data, f, indent=2)

    return {
        "routes": routes_fp,
        "packages": packages_fp,
        "travel_times": travel_fp,
        "actual_sequences": sequences_fp
    }

if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "challenge"))
    paths = generate_official_amazon_dataset(out_dir, num_routes=30, num_days=10)
    print(f"Generated official Amazon challenge dataset in {out_dir}")
