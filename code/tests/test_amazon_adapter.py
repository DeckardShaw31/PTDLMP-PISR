import os
import json
import pytest
from datetime import datetime, date, time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset, parse_iso_or_time_string

def test_parse_iso_or_time_string():
    d = date(2018, 7, 24)
    # Full ISO format
    dt1 = parse_iso_or_time_string("2018-07-24 16:30:00", d)
    assert dt1 == datetime(2018, 7, 24, 16, 30, 0)

    # Time only
    dt2 = parse_iso_or_time_string("18:45:00", d)
    assert dt2 == datetime(2018, 7, 24, 18, 45, 0)

    # None/Empty
    assert parse_iso_or_time_string(None, d) is None
    assert parse_iso_or_time_string("", d) is None

def test_load_official_amazon_dataset_multi_package_aggregation(tmp_path):
    # Create temporary official-format Amazon files
    routes_data = {
        "RouteID_OFFICIAL_1": {
            "station_code": "DLA3",
            "date_YYYY_MM_DD": "2018-07-24",
            "departure_time_utc": "08:00:00",
            "executor_capacity_cm3": 4247527.0,
            "route_score": "High",
            "stops": {
                "STATION_1": {"lat": 34.05, "lng": -118.25, "type": "Station", "zone_id": None},
                "STOP_A": {"lat": 34.06, "lng": -118.24, "type": "Dropoff", "zone_id": "Z1"},
                "STOP_B": {"lat": 34.07, "lng": -118.23, "type": "Dropoff", "zone_id": "Z2"}
            }
        }
    }

    # STOP_A has 2 packages: one with deadline 09:30, one with deadline 09:00
    # planned_service_time: 60s + 90s = 150s
    # volume: (10*10*10) + (20*10*5) = 1000 + 1000 = 2000 cm3
    packages_data = {
        "RouteID_OFFICIAL_1": {
            "STOP_A": {
                "PKG_1": {
                    "scan_status": "DELIVERED",
                    "time_window": {"start_time_utc": "2018-07-24 08:00:00", "end_time_utc": "2018-07-24 09:30:00"},
                    "planned_service_time_seconds": 60.0,
                    "dimensions": {"depth_cm": 10.0, "height_cm": 10.0, "width_cm": 10.0}
                },
                "PKG_2": {
                    "scan_status": "DELIVERED",
                    "time_window": {"start_time_utc": "2018-07-24 08:00:00", "end_time_utc": "2018-07-24 09:00:00"},
                    "planned_service_time_seconds": 90.0,
                    "dimensions": {"depth_cm": 20.0, "height_cm": 10.0, "width_cm": 5.0}
                }
            },
            "STOP_B": {
                "PKG_3": {
                    "scan_status": "DELIVERED",
                    "time_window": {"start_time_utc": "2018-07-24 08:00:00", "end_time_utc": "2018-07-24 10:00:00"},
                    "planned_service_time_seconds": 120.0,
                    "dimensions": {"depth_cm": 15.0, "height_cm": 15.0, "width_cm": 15.0}
                }
            }
        }
    }

    travel_times_data = {
        "RouteID_OFFICIAL_1": {
            "STATION_1": {"STOP_A": 300.0, "STOP_B": 600.0},
            "STOP_A": {"STATION_1": 320.0, "STOP_B": 400.0},
            "STOP_B": {"STATION_1": 650.0, "STOP_A": 410.0}
        }
    }

    routes_file = str(tmp_path / "routes.json")
    packages_file = str(tmp_path / "packages.json")
    travel_file = str(tmp_path / "travel_times.json")

    with open(routes_file, "w") as f:
        json.dump(routes_data, f)
    with open(packages_file, "w") as f:
        json.dump(packages_data, f)
    with open(travel_file, "w") as f:
        json.dump(travel_times_data, f)

    instances = load_official_amazon_dataset(routes_file, packages_file, travel_file)
    assert "RouteID_OFFICIAL_1" in instances
    inst = instances["RouteID_OFFICIAL_1"]

    assert inst.depot_id == "STATION_1"
    assert inst.station_code == "DLA3"
    assert inst.departure_time == datetime(2018, 7, 24, 8, 0, 0)
    assert len(inst.stops) == 3

    # Check STOP_A aggregations
    stop_a = inst.stops["STOP_A"]
    assert stop_a.service_seconds == 150.0  # 60 + 90
    assert stop_a.package_volume_cm3 == 2000.0  # 1000 + 1000
    assert stop_a.promised_time == datetime(2018, 7, 24, 9, 0, 0)  # min(09:30, 09:00)
    assert stop_a.custom_data["num_packages"] == 2
    assert stop_a.custom_data["zone_id"] == "Z1"

    # Check travel times
    assert inst.travel_times is not None
    assert inst.travel_times[("STATION_1", "STOP_A")] == 300.0
    assert inst.travel_times[("STOP_A", "STOP_B")] == 400.0
