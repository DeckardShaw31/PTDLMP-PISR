import math
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ..data.schemas import RouteInstance
from .schedule import propagate_schedule
from .distance import compute_route_distance

def compute_actionability_scores(
    instance: RouteInstance,
    baseline_route: List[str],
    delta: float = 0.05,
    use_completion_time: bool = True,
    return_to_depot: bool = False
) -> Tuple[Dict[str, float], Dict[str, Optional[int]]]:
    """
    Computes actionability score g_i for all customers on the baseline route R0 (Eqs. 7-9).
    For each customer i at position k:
      - Considers earlier insertion positions j < k (j >= 1).
      - Checks cumulative distance condition: D(R0^{i -> j}) <= (1 + delta) * D(R0)
      - Computes Delta TT_ij = TT(R0) - TT(R0^{i -> j})
      - g_i = max(0, max_{j in F_i} Delta TT_ij)
    
    Returns:
      g_scores: Dict[customer_id, g_i]
      best_j: Dict[customer_id, best forward position j* in baseline evaluation]
    """
    base_sched = propagate_schedule(instance, baseline_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
    base_tt = base_sched.total_tardiness
    base_dist = base_sched.total_distance
    dist_limit = (1.0 + delta) * base_dist + 1e-6

    g_scores: Dict[str, float] = {}
    best_j_map: Dict[str, Optional[int]] = {}

    # Customers start at index 1 (index 0 is depot)
    for k in range(1, len(baseline_route)):
        cust_id = baseline_route[k]
        best_delta_tt = 0.0
        best_j = None

        # Enumerate candidate forward positions j < k (j >= 1)
        for j in range(1, k):
            cand_route = list(baseline_route)
            val = cand_route.pop(k)
            cand_route.insert(j, val)

            cand_dist = compute_route_distance(instance, cand_route, return_to_depot=return_to_depot)
            if cand_dist <= dist_limit:
                cand_sched = propagate_schedule(instance, cand_route, use_completion_time=use_completion_time, return_to_depot=return_to_depot)
                delta_tt = base_tt - cand_sched.total_tardiness
                if delta_tt > best_delta_tt:
                    best_delta_tt = delta_tt
                    best_j = j

        g_scores[cust_id] = max(0.0, best_delta_tt)
        best_j_map[cust_id] = best_j

    return g_scores, best_j_map

def rank_candidates(
    instance: RouteInstance,
    baseline_route: List[str],
    policy: str,
    g_scores: Dict[str, float],
    random_seed: Optional[int] = 42
) -> List[Tuple[str, float]]:
    """
    Ranks customers on baseline route in descending order of targeting score.
    Policies:
      - 'RA': Risk-Actionability score RAS_i = p_i * g_i (Eq. 12)
      - 'RB': Risk-Based score S_i^{RB} = p_i (Eq. 10)
      - 'AB': Actionability-Based score S_i^{AB} = g_i (Eq. 11)
      - 'Random': Seeded random ranking
      - 'Slack': Smallest slack (tau_i - arrival_or_completion)
      - 'Deadline': Earliest promised deadline tau_i
    
    Returns: List of (customer_id, score) tuples sorted in descending priority order.
    """
    customers = baseline_route[1:]
    scored_candidates = []

    if policy.upper() == "RA":
        for c in customers:
            p_i = instance.stops[c].predicted_risk_pi
            g_i = g_scores.get(c, 0.0)
            score = p_i * g_i
            # Tuples: (score, g_i, p_i, customer_id)
            scored_candidates.append((c, score, g_i, p_i))
        # Descending by score, then g_i, then p_i, then customer_id
        scored_candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        return [(x[0], x[1]) for x in scored_candidates]

    elif policy.upper() == "RB":
        for c in customers:
            p_i = instance.stops[c].predicted_risk_pi
            g_i = g_scores.get(c, 0.0)
            scored_candidates.append((c, p_i, g_i))
        scored_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(x[0], x[1]) for x in scored_candidates]

    elif policy.upper() == "AB":
        for c in customers:
            g_i = g_scores.get(c, 0.0)
            p_i = instance.stops[c].predicted_risk_pi
            scored_candidates.append((c, g_i, p_i))
        scored_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(x[0], x[1]) for x in scored_candidates]

    elif policy.upper() == "RANDOM":
        rng = random.Random(random_seed)
        shuffled = list(customers)
        rng.shuffle(shuffled)
        scored_candidates = [(c, float(len(shuffled) - idx)) for idx, c in enumerate(shuffled)]

    elif policy.upper() in ["SLACK", "DEADLINE"]:
        for c in customers:
            stop = instance.stops[c]
            # Earliest deadline gets highest score (negated timestamp)
            p_time = stop.promised_time.timestamp() if stop.promised_time else float('inf')
            scored_candidates.append((c, -p_time))
        scored_candidates.sort(key=lambda x: (-x[1], x[0]))

    else:
        raise ValueError(f"Unknown targeting policy: '{policy}'")

    return scored_candidates

def select_intervention_set(
    ranked_candidates: List[Tuple[str, float]],
    budget_b: float,
    num_customers: int
) -> List[str]:
    """
    Computes capacity K_r(B) = max(1, floor(B * n_r)) for B > 0 (and 0 for B = 0),
    and selects top K_r(B) customers from ranked candidates (Eqs. 13-14).
    """
    if budget_b <= 0.0 or num_customers <= 0:
        k_cap = 0
    else:
        k_cap = max(1, math.floor(budget_b * num_customers))

    selected = [c[0] for c in ranked_candidates[:k_cap]]
    return selected
