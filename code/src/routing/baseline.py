from typing import List, Set
from datetime import datetime
from ..data.schemas import RouteInstance
from .distance import get_edge_distance

def build_nearest_neighbor_baseline(instance: RouteInstance) -> List[str]:
    """
    Constructs an ex-ante baseline route R0 using deterministic Nearest Neighbor (NN).
    Starts at depot.
    Ties are resolved deterministically by earliest promised delivery time, 
    and then alphabetical stop ID.
    """
    depot_id = instance.depot_id
    unvisited: Set[str] = {
        s_id for s_id, s in instance.stops.items()
        if s.stop_type == "Dropoff" or s_id != depot_id
    }

    route: List[str] = [depot_id]
    current_node = depot_id

    while unvisited:
        # Evaluate distance to all unvisited customers
        candidates = []
        for cand in unvisited:
            dist = get_edge_distance(instance, current_node, cand)
            cand_stop = instance.stops[cand]
            # Handle possible None in promised_time for tie-breaking
            p_time = cand_stop.promised_time if cand_stop.promised_time is not None else datetime.max
            candidates.append((dist, p_time, cand))

        # Sort candidates deterministically: min dist, then earliest promise time, then stop ID
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        next_node = candidates[0][2]
        route.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return route
