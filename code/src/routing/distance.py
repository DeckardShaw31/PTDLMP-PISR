import math
from typing import List, Tuple
from ..data.schemas import RouteInstance

EARTH_RADIUS_KM = 6371.0088

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c

def get_edge_distance(instance: RouteInstance, u: str, v: str) -> float:
    """
    Returns distance between stop u and stop v in km.
    Uses precomputed matrix if available; otherwise computes Haversine distance.
    
    IMPORTANT METHODOLOGICAL NOTE:
      All distance calculations throughout PTDLMP-PISR (including Delta D and cumulative
      distance bounds) represent a Haversine (great-circle/straight-line) distance proxy.
      They do not represent turn-by-turn road network distances.
    """
    if u == v:
        return 0.0
    if instance.distances is not None and (u, v) in instance.distances:
        return instance.distances[(u, v)]
    
    stop_u = instance.stops[u]
    stop_v = instance.stops[v]
    return haversine_distance(stop_u.lat, stop_u.lng, stop_v.lat, stop_v.lng)

def get_edge_travel_time(instance: RouteInstance, u: str, v: str, default_speed_kmh: float = 25.0) -> float:
    """
    Returns travel time between stop u and stop v in seconds.
    Uses precomputed matrix if available; otherwise derives travel time from Haversine distance.
    """
    if u == v:
        return 0.0
    if instance.travel_times is not None and (u, v) in instance.travel_times:
        return instance.travel_times[(u, v)]
    
    # Distance in km / speed in km/s (Haversine-based fallback)
    dist_km = get_edge_distance(instance, u, v)
    speed_km_per_sec = default_speed_kmh / 3600.0
    return dist_km / speed_km_per_sec

def compute_route_distance(instance: RouteInstance, route: List[str], return_to_depot: bool = False) -> float:
    """
    Computes total route distance in km using the Haversine straight-line distance proxy.
    By default in PTDLMP manuscript: depot-to-last-customer without return to depot.
    """
    if len(route) <= 1:
        return 0.0
    
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += get_edge_distance(instance, route[i], route[i + 1])
    
    if return_to_depot and len(route) > 1:
        total_dist += get_edge_distance(instance, route[-1], route[0])
        
    return total_dist

def compute_relocation_distance_delta(
    instance: RouteInstance,
    route: List[str],
    k: int,
    j: int,
    return_to_depot: bool = False
) -> float:
    """
    Computes exact change in total route distance Delta D = D(R') - D(R) in O(1) time
    when moving customer at index k to forward position j (where 1 <= j < k).
    Supports:
      - Adjacent forward relocation (j == k - 1)
      - Non-adjacent forward relocation (j < k - 1)
      - Last-stop relocation (k == len(route) - 1)
      - Both return_to_depot=False and return_to_depot=True
    """
    n = len(route)
    if not (1 <= j < k < n):
        raise ValueError(f"Invalid relocation indices: j={j}, k={k} for route of length {n}")

    vk = route[k]
    v_prev_k = route[k - 1]
    v_next_k = route[k + 1] if k + 1 < n else (route[0] if return_to_depot else None)

    if j == k - 1:
        # Adjacent move: vk swapped with v_prev_k
        v_prev2 = route[k - 2]
        delta_d = (
            - get_edge_distance(instance, v_prev2, v_prev_k)
            - get_edge_distance(instance, v_prev_k, vk)
            + get_edge_distance(instance, v_prev2, vk)
            + get_edge_distance(instance, vk, v_prev_k)
        )
        if v_next_k is not None:
            delta_d += (
                - get_edge_distance(instance, vk, v_next_k)
                + get_edge_distance(instance, v_prev_k, v_next_k)
            )
    else:
        # Non-adjacent forward move: vk inserted between v_prev_j and vj
        vj = route[j]
        v_prev_j = route[j - 1]
        delta_d = (
            - get_edge_distance(instance, v_prev_j, vj)
            - get_edge_distance(instance, v_prev_k, vk)
            + get_edge_distance(instance, v_prev_j, vk)
            + get_edge_distance(instance, vk, vj)
        )
        if v_next_k is not None:
            delta_d += (
                - get_edge_distance(instance, vk, v_next_k)
                + get_edge_distance(instance, v_prev_k, v_next_k)
            )

    return delta_d

