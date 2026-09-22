from typing import List, Tuple, Optional
from ..data.schemas import RouteInstance, RelocationLogEntry
from .schedule import propagate_schedule
from .distance import compute_route_distance

def selective_forward_relocation(
    instance: RouteInstance,
    baseline_route: List[str],
    candidate_set: List[str],
    delta: float = 0.05,
    epsilon: float = 1e-6,
    use_completion_time: bool = True,
    return_to_depot: bool = False
) -> Tuple[List[str], List[RelocationLogEntry]]:
    """
    Algorithm 1: Selective Forward Relocation (SFR).
    
    Sequentially attempts forward relocation for each selected candidate customer
    subject to:
      1. Forward insertion only (j < k)
      2. Cumulative distance constraint: D(R^{c, i -> j}) <= (1 + delta) * D(R0) (Eq. 15)
      3. Deterministic lexicographical selection: min TT, min D, min displacement (k - j), stable stop_id (Eq. 16)
      4. Strict tardiness reduction: TT(R^{c, i -> j*}) < TT(R^c) (Eq. 17)
    
    Guarantees Proposition 1: TT(R') <= TT(R0) and D(R') <= (1 + delta) * D(R0).
    """
    current_route = list(baseline_route)
    base_dist = compute_route_distance(instance, baseline_route, return_to_depot=return_to_depot)
    dist_limit = (1.0 + delta) * base_dist + epsilon

    current_sched = propagate_schedule(instance, current_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
    current_tt = current_sched.total_tardiness
    current_dist = current_sched.total_distance

    logs: List[RelocationLogEntry] = []

    for cust_id in candidate_set:
        if cust_id not in current_route:
            continue

        curr_pos = current_route.index(cust_id)
        if curr_pos <= 1:
            # Already at position 1 (first after depot), cannot move forward
            logs.append(RelocationLogEntry(
                customer_id=cust_id,
                original_position=curr_pos,
                attempted_position=None,
                accepted=False,
                rejection_reason="already_at_first_position",
                tt_before=current_tt,
                tt_after=current_tt,
                distance_before=current_dist,
                distance_after=current_dist,
                distance_limit=dist_limit
            ))
            continue

        # Enumerate candidate positions j < curr_pos (j >= 1)
        admissible_candidates = []
        for j in range(1, curr_pos):
            cand_route = list(current_route)
            val = cand_route.pop(curr_pos)
            cand_route.insert(j, val)

            cand_dist = compute_route_distance(instance, cand_route, return_to_depot=return_to_depot)
            # Check Eq. (15): distance tolerance relative to baseline route R0
            if cand_dist <= dist_limit:
                cand_sched = propagate_schedule(instance, cand_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
                displacement = curr_pos - j
                # Tuple for lexicographic sorting: (TT, distance, displacement, j)
                admissible_candidates.append((
                    cand_sched.total_tardiness,
                    cand_dist,
                    displacement,
                    j,
                    cand_route
                ))

        if not admissible_candidates:
            logs.append(RelocationLogEntry(
                customer_id=cust_id,
                original_position=curr_pos,
                attempted_position=None,
                accepted=False,
                rejection_reason="distance_limit_or_no_admissible_move",
                tt_before=current_tt,
                tt_after=current_tt,
                distance_before=current_dist,
                distance_after=current_dist,
                distance_limit=dist_limit
            ))
            continue

        # Lexicographic selection: Eq. (16)
        # min TT, then min D, then min displacement (k - j)
        admissible_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        best_cand = admissible_candidates[0]
        best_tt, best_cand_dist, _, best_j, best_route = best_cand

        # Strict tardiness improvement: Eq. (17)
        if best_tt < current_tt - epsilon:
            # Accept relocation
            current_route = best_route
            tt_before = current_tt
            dist_before = current_dist
            current_tt = best_tt
            current_dist = best_cand_dist

            logs.append(RelocationLogEntry(
                customer_id=cust_id,
                original_position=curr_pos,
                attempted_position=best_j,
                accepted=True,
                rejection_reason=None,
                tt_before=tt_before,
                tt_after=current_tt,
                distance_before=dist_before,
                distance_after=current_dist,
                distance_limit=dist_limit
            ))
        else:
            # Reject: does not strictly reduce current total tardiness
            logs.append(RelocationLogEntry(
                customer_id=cust_id,
                original_position=curr_pos,
                attempted_position=best_j,
                accepted=False,
                rejection_reason="no_strict_tardiness_gain",
                tt_before=current_tt,
                tt_after=current_tt,
                distance_before=current_dist,
                distance_after=current_dist,
                distance_limit=dist_limit
            ))

    return current_route, logs
