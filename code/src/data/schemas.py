from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time, timedelta

@dataclass
class Stop:
    stop_id: str
    lat: float
    lng: float
    stop_type: str  # "Station" or "Dropoff"
    promised_time: Optional[datetime] = None
    service_seconds: float = 0.0
    package_volume_cm3: float = 0.0
    predicted_risk_pi: float = 0.0  # p_i estimated lateness risk
    custom_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouteInstance:
    route_id: str
    depot_id: str
    departure_time: datetime
    stops: Dict[str, Stop]  # stop_id -> Stop
    # Directed travel times in seconds: (from_stop, to_stop) -> seconds
    travel_times: Optional[Dict[Tuple[str, str], float]] = None
    # Directed distances in km: (from_stop, to_stop) -> km
    distances: Optional[Dict[Tuple[str, str], float]] = None
    # Official metadata fields
    route_date: Optional[str] = None
    station_code: Optional[str] = None
    executor_capacity_cm3: Optional[float] = None
    route_score: Optional[str] = None
    actual_sequence: Optional[Dict[str, int]] = None

@dataclass
class ScheduleResult:
    route: List[str]  # [depot_id, stop_1, stop_2, ...]
    arrival_times: Dict[str, datetime]
    completion_times: Dict[str, datetime]
    lateness: Dict[str, int]  # stop_id -> 0 or 1
    tardiness_minutes: Dict[str, float]  # stop_id -> max(0, completion - promise) in minutes
    total_lateness: int  # NL(R)
    total_tardiness: float  # TT(R) in minutes
    total_distance: float  # D(R) in km

@dataclass
class ValidationResult:
    feasible: bool
    violations: List[str] = field(default_factory=list)
    distance_km: float = 0.0
    distance_limit_km: float = 0.0
    nl: int = 0
    tt_minutes: float = 0.0
    served_customers: int = 0

@dataclass
class RelocationLogEntry:
    customer_id: str
    original_position: int
    attempted_position: Optional[int]
    accepted: bool
    rejection_reason: Optional[str]
    tt_before: float
    tt_after: float
    distance_before: float
    distance_after: float
    distance_limit: float
