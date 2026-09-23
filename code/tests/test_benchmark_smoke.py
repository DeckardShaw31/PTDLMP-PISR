import os
import sys
import pytest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.routing.baseline import build_nearest_neighbor_baseline
from src.routing.schedule import propagate_schedule
from src.routing.actionability import compute_actionability_scores, rank_candidates, select_intervention_set
from src.routing.sfr import selective_forward_relocation
from src.routing.validator import validate_route
from experiments.run_routing_benchmark import run_benchmark_experiments

def test_run_benchmark_experiments_smoke(tmp_path):
    """
    Directly calls run_benchmark_experiments() on official Amazon data with a reduced slice:
    1 route, 1 budget, 1 tolerance delta, 2 policies, 2 random seeds.
    Verifies that run_benchmark_experiments executes end-to-end without unhandled crashes
    and writes all required output artifacts (catching variable or parameter errors).
    """
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        pytest.skip(f"Official raw dataset not found in '{raw_dir}'")

    out_dir = str(tmp_path / "smoke_benchmark_out")
    test_route_id = "RouteID_693060a6-88bb-4324-9e9c-925d5240263c"

    summary = run_benchmark_experiments(
        raw_dir=raw_dir,
        output_dir=out_dir,
        route_ids=[test_route_id],
        budgets=[0.10],
        deltas=[0.05],
        policies=["RA", "Random"],
        random_seeds=[42, 43],
        default_sla_hours=None,
        save_outputs=True
    )

    assert summary is not None
    assert "model_selection" in summary
    assert "policy_summary" in summary
    assert "invariant_audit" in summary
    assert summary["invariant_audit"]["min_delta_tt"] >= 0.0

    # Verify generated artifacts
    assert os.path.exists(os.path.join(out_dir, "benchmark_summary.json"))
    assert os.path.exists(os.path.join(out_dir, "routing_benchmark_runs.csv"))
    assert os.path.exists(os.path.join(out_dir, "random_seed_runs.csv"))

def test_benchmark_reduced_grid_smoke():
    """
    Component-level smoke test executing the routing kernel on official data.
    Verifies actionability computation, candidate selection, relocation, schedule propagation,
    and runtime profiling execute without error across all policies.
    """
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        pytest.skip(f"Official raw dataset not found in '{raw_dir}'")

    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, default_sla_hours=None)
    assert len(instances) > 0

    # Pick single test route
    r_id, inst = list(instances.items())[0]
    num_customers = len([s for s, sobj in inst.stops.items() if sobj.stop_type == "Dropoff" or s != inst.depot_id])

    base_route = build_nearest_neighbor_baseline(inst)
    sched_base = propagate_schedule(inst, base_route)
    assert sched_base.total_distance > 0.0

    delta = 0.05
    budget = 0.10
    policies = ["RA", "AB", "RB", "Slack", "Deadline", "Random"]

    # Actionability computation
    t0 = time.perf_counter()
    g_scores, best_j_map = compute_actionability_scores(inst, base_route, delta=delta)
    act_time = time.perf_counter() - t0
    assert act_time >= 0.0
    assert len(g_scores) == num_customers

    for pol in policies:
        if pol == "Random":
            seeds = [42, 43]
            for s in seeds:
                t_pol = time.perf_counter()
                ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores, random_seed=s)
                selected = select_intervention_set(ranked, budget_b=budget, num_customers=num_customers)
                final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
                v_res = validate_route(inst, final_r, baseline_route=base_route, delta=delta, require_promised_times=False)
                sched_fin = propagate_schedule(inst, final_r)
                pol_time = time.perf_counter() - t_pol

                assert v_res.feasible is True
                assert sched_fin.total_tardiness <= sched_base.total_tardiness + 1e-6
                assert pol_time >= 0.0
        else:
            t_pol = time.perf_counter()
            ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores)
            selected = select_intervention_set(ranked, budget_b=budget, num_customers=num_customers)
            final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
            v_res = validate_route(inst, final_r, baseline_route=base_route, delta=delta, require_promised_times=False)
            sched_fin = propagate_schedule(inst, final_r)
            pol_time = time.perf_counter() - t_pol

            assert v_res.feasible is True
            assert sched_fin.total_tardiness <= sched_base.total_tardiness + 1e-6
            assert sched_fin.total_distance <= (1.0 + delta) * sched_base.total_distance + 1e-6
            assert pol_time >= 0.0
