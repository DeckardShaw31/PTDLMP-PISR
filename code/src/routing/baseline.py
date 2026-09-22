from typing import List, Set, Dict
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

def build_clarke_wright_baseline(instance: RouteInstance) -> List[str]:
    """
    Constructs an alternative ex-ante baseline route R0 using deterministic Clarke-Wright Savings heuristic
    (Section 9.2 of design.md).
    Calculates savings: s_ij = d(depot, i) + d(depot, j) - d(i, j).
    Sorts pairs in descending order of savings.
    Merges routes while maintaining endpoints until a single tour is formed.
    """
    depot_id = instance.depot_id
    customers = [
        s_id for s_id, s in instance.stops.items()
        if s.stop_type == "Dropoff" or s_id != depot_id
    ]

    if not customers:
        return [depot_id]
    if len(customers) == 1:
        return [depot_id, customers[0]]

    # Compute savings for all pairs (i, j) with i != j
    savings = []
    for i_idx in range(len(customers)):
        i = customers[i_idx]
        d_depot_i = get_edge_distance(instance, depot_id, i)
        p_time_i = instance.stops[i].promised_time or datetime.max
        for j_idx in range(i_idx + 1, len(customers)):
            j = customers[j_idx]
            d_depot_j = get_edge_distance(instance, depot_id, j)
            d_ij = get_edge_distance(instance, i, j)
            s_val = d_depot_i + d_depot_j - d_ij
            p_time_j = instance.stops[j].promised_time or datetime.max
            earliest_p = min(p_time_i, p_time_j)
            savings.append((s_val, earliest_p, i, j))

    # Sort descending by savings, tie-break by earliest promised time, then stop IDs
    savings.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

    # Initialize disjoint chains: each customer in its own list
    chains: Dict[str, List[str]] = {c: [c] for c in customers}
    head_map: Dict[str, str] = {c: c for c in customers}  # head -> chain key
    tail_map: Dict[str, str] = {c: c for c in customers}  # tail -> chain key

    for s_val, _, i, j in savings:
        # Check if i and j are in different chains and both are endpoints
        i_is_head = (i in head_map)
        i_is_tail = (i in tail_map)
        j_is_head = (j in head_map)
        j_is_tail = (j in tail_map)

        if not (i_is_head or i_is_tail) or not (j_is_head or j_is_tail):
            continue

        c_i = head_map.get(i) or tail_map.get(i)
        c_j = head_map.get(j) or tail_map.get(j)
        if c_i == c_j:
            continue  # Same chain, would create cycle

        chain_i = chains[c_i]
        chain_j = chains[c_j]

        # Four merge configurations:
        # 1. tail of chain_i connects to head of chain_j: chain_i + chain_j
        if chain_i[-1] == i and chain_j[0] == j:
            merged = chain_i + chain_j
        # 2. tail of chain_j connects to head of chain_i: chain_j + chain_i
        elif chain_j[-1] == j and chain_i[0] == i:
            merged = chain_j + chain_i
        # 3. tail of chain_i connects to tail of chain_j: chain_i + reversed(chain_j)
        elif chain_i[-1] == i and chain_j[-1] == j:
            merged = chain_i + list(reversed(chain_j))
        # 4. head of chain_i connects to head of chain_j: reversed(chain_i) + chain_j
        elif chain_i[0] == i and chain_j[0] == j:
            merged = list(reversed(chain_i)) + chain_j
        else:
            continue

        # Clean old endpoints
        del head_map[chain_i[0]]
        del tail_map[chain_i[-1]]
        del head_map[chain_j[0]]
        del tail_map[chain_j[-1]]
        del chains[c_i]
        del chains[c_j]

        # Register merged chain
        new_key = merged[0]
        chains[new_key] = merged
        head_map[merged[0]] = new_key
        tail_map[merged[-1]] = new_key

        if len(chains) == 1:
            break

    # If any disconnected chains remain, connect them sequentially
    final_customers = []
    for c_list in chains.values():
        final_customers.extend(c_list)

    return [depot_id] + final_customers

