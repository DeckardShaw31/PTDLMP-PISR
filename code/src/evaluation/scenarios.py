import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from ..data.schemas import RouteInstance

DEADLINE_TOLERANCE_SECONDS = 1e-6

@dataclass
class RouteTimeData:
    """
    Dense, index-aligned view of one route used by travel-time scenario evaluation.
    Stops are ordered by sorted stop_id so that scenario draws do not depend on dict ordering.
    """
    route_id: str
    depot_id: str
    stop_ids: List[str]
    index: Dict[str, int]
    mean_travel_seconds: np.ndarray  # (n, n) official historical mean; NaN where the directed edge is missing
    service_seconds: np.ndarray  # (n,) planned service duration
    deadline_offset_seconds: np.ndarray  # (n,) promised_time - departure_time; +inf when no promise

def lognormal_sigma(cv: float) -> float:
    """
    Log-scale standard deviation of a lognormal multiplier with coefficient of variation CV:
    sigma = sqrt(ln(1 + CV^2)).
    """
    if cv < 0.0 or not math.isfinite(cv):
        raise ValueError(f"Coefficient of variation must be finite and non-negative, got {cv}")
    return math.sqrt(math.log1p(cv * cv))

def route_seed_key(route_id: str) -> int:
    """
    Stable 32-bit key for a route (independent of route ordering and of Python's hash randomization).
    """
    return int(hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:8], 16)

def build_route_time_data(instance: RouteInstance) -> RouteTimeData:
    """
    Extracts the official directed travel-time matrix, planned service durations and service thresholds.
    Missing directed edges stay NaN: travel times are never imputed from Haversine distances here.
    """
    if not instance.travel_times:
        raise ValueError(
            f"Route '{instance.route_id}' has no official travel-time matrix. "
            "Scenario evaluation does not impute travel times from Haversine distances."
        )

    stop_ids = sorted(instance.stops.keys())
    index = {s: i for i, s in enumerate(stop_ids)}
    n = len(stop_ids)

    mean_tt = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mean_tt, 0.0)
    for (u, v), secs in instance.travel_times.items():
        if u in index and v in index:
            mean_tt[index[u], index[v]] = float(secs)

    service = np.array([float(instance.stops[s].service_seconds) for s in stop_ids], dtype=float)
    deadlines = np.full(n, np.inf, dtype=float)
    for s in stop_ids:
        promised = instance.stops[s].promised_time
        if promised is not None:
            deadlines[index[s]] = (promised - instance.departure_time).total_seconds()

    return RouteTimeData(
        route_id=instance.route_id,
        depot_id=instance.depot_id,
        stop_ids=stop_ids,
        index=index,
        mean_travel_seconds=mean_tt,
        service_seconds=service,
        deadline_offset_seconds=deadlines
    )

def draw_standard_normals(route_id: str, scenario_id: int, n: int, base_seed: int) -> np.ndarray:
    """
    Draws Z_uv^(s) ~ N(0, 1) for every directed edge (u, v) of a route.
    The stream depends only on (base_seed, route_id, scenario_id), so:
      - every policy evaluated on the same route and scenario sees the same matrix,
      - scenario s is identical whether 20 or 100 scenarios are generated,
      - the same Z is reused across CV levels (common random numbers).
    """
    seq = np.random.SeedSequence([int(base_seed), route_seed_key(route_id), int(scenario_id)])
    return np.random.default_rng(seq).standard_normal((n, n))

def scenario_travel_times(mean_travel_seconds: np.ndarray, z: np.ndarray, cv: float) -> np.ndarray:
    """
    Mean-preserving lognormal edge multipliers:
        t_uv^(s) = mean_uv * exp(sigma * Z_uv^(s) - sigma^2 / 2),  sigma = sqrt(ln(1 + CV^2)),
    so that E[t_uv^(s)] = mean_uv. CV = 0 returns the mean matrix exactly.
    Edges with a zero mean (co-located stops) stay zero; missing edges stay NaN.
    """
    if z.shape != mean_travel_seconds.shape:
        raise ValueError(f"Shape mismatch: Z {z.shape} vs mean matrix {mean_travel_seconds.shape}")

    sigma = lognormal_sigma(cv)
    if sigma == 0.0:
        return mean_travel_seconds.copy()

    multipliers = np.exp(sigma * z - 0.5 * sigma * sigma)
    scenario = mean_travel_seconds * multipliers

    known = np.isfinite(mean_travel_seconds)
    if not np.all(np.isfinite(scenario[known])) or np.any(scenario[known] < 0.0):
        raise ValueError("Scenario travel times must be finite and non-negative on every known edge.")
    if np.any(scenario[known & (mean_travel_seconds > 0.0)] <= 0.0):
        raise ValueError("Scenario travel times must stay strictly positive on edges with a positive mean.")
    return scenario

def matrix_fingerprint(matrix: np.ndarray) -> str:
    """Short SHA-256 fingerprint of a travel-time matrix, used to audit reproducibility."""
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float64).tobytes()).hexdigest()[:16]

def encode_sequences(data: RouteTimeData, sequences: Sequence[Sequence[str]]) -> np.ndarray:
    """
    Converts frozen route sequences into an integer index array of shape (num_sequences, route_length).
    Every sequence must start at the depot and visit the same stop set.
    """
    if not sequences:
        raise ValueError("No sequences to encode.")
    length = len(sequences[0])
    encoded = np.empty((len(sequences), length), dtype=np.int64)
    expected = set(sequences[0])
    for k, seq in enumerate(sequences):
        if len(seq) != length or set(seq) != expected or len(set(seq)) != length:
            raise ValueError(f"Sequence {k} of route '{data.route_id}' is not a permutation of the baseline stop set.")
        if seq[0] != data.depot_id:
            raise ValueError(f"Sequence {k} of route '{data.route_id}' does not start at depot '{data.depot_id}'.")
        unknown = [s for s in seq if s not in data.index]
        if unknown:
            raise ValueError(f"Sequence {k} of route '{data.route_id}' contains unknown stops: {unknown[:5]}")
        encoded[k] = [data.index[s] for s in seq]
    return encoded

def evaluate_sequences(
    data: RouteTimeData,
    travel_seconds: np.ndarray,
    encoded_sequences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagates fixed sequences on one travel-time matrix (Eqs. 1-4 with completion-time evaluation,
    open route, departure at the depot). All sequences share the same matrix in a single call.

    Returns:
      total_tardiness_minutes: TT(R) per sequence
      total_lateness: NL(R) per sequence
    """
    origins = encoded_sequences[:, :-1]
    targets = encoded_sequences[:, 1:]
    legs = travel_seconds[origins, targets]
    if not np.all(np.isfinite(legs)):
        k, pos = np.argwhere(~np.isfinite(legs))[0]
        u = data.stop_ids[origins[k, pos]]
        v = data.stop_ids[targets[k, pos]]
        raise ValueError(
            f"Route '{data.route_id}' is missing directed travel-time edge ({u} -> {v}). "
            "Haversine imputation is disabled in scenario evaluation."
        )

    completion = np.cumsum(legs + data.service_seconds[targets], axis=1)
    deadlines = data.deadline_offset_seconds[targets]
    # Float cumulative sums can overshoot an exactly-met deadline by ~1e-11 s; propagate_schedule works at
    # datetime (microsecond) resolution, so differences below one microsecond count as on time.
    late = completion - deadlines > DEADLINE_TOLERANCE_SECONDS
    tardiness_min = np.where(late, (completion - deadlines) / 60.0, 0.0)
    return tardiness_min.sum(axis=1), late.sum(axis=1)
