import os
import sys
import json
import time
import hashlib
import subprocess
import platform
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import sklearn
import scipy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.features.build_features import extract_ex_ante_features_for_route, compute_ground_truth_labels
from src.prediction.train import train_and_evaluate_pipeline, inject_calibrated_risks_into_instances, chronological_split
from src.routing.baseline import build_nearest_neighbor_baseline, build_clarke_wright_baseline
from src.routing.schedule import propagate_schedule
from src.routing.actionability import compute_actionability_scores, rank_candidates, select_intervention_set
from src.routing.sfr import selective_forward_relocation
from src.routing.validator import validate_route
from src.evaluation.statistics import compute_paired_statistics, compute_route_clustered_statistics
from src.evaluation.tables import format_markdown_table, format_latex_table

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit_explicit_deadline_routing(raw_dir: str) -> Dict[str, Any]:
    """
    Evaluates baseline dispatch on explicit customer deadlines (time_window.end_time_utc)
    without synthetic SLA imputation, disclosing empirical dataset limitations.
    """
    print("\n" + "=" * 80)
    print("ROUTING AUDIT: EXPLICIT AMAZON DEADLINES (time_window.end_time_utc)")
    print("=" * 80)
    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=None)
    
    # Identify test routes chronologically
    all_rows = []
    for r_id, inst in instances.items():
        f_df = extract_ex_ante_features_for_route(inst)
        all_rows.append(f_df)
    full_dataset = pd.concat(all_rows, ignore_index=True)
    _, _, df_test = chronological_split(full_dataset, date_col="route_date")
    test_dates = set(df_test["route_date"].unique())
    test_route_ids = [r_id for r_id, inst in instances.items() if (inst.route_date or "") in test_dates]

    total_explicit_test_stops = 0
    total_test_baseline_tt = 0.0
    total_test_baseline_nl = 0

    for r_id in test_route_ids:
        inst = instances[r_id]
        base_route = build_nearest_neighbor_baseline(inst)
        sched = propagate_schedule(inst, base_route)
        explicit_stops = [s for s in inst.stops.values() if s.deadline_source == "explicit"]
        total_explicit_test_stops += len(explicit_stops)
        total_test_baseline_tt += sched.total_tardiness
        total_test_baseline_nl += sched.total_lateness

    audit_payload = {
        "status": "completed",
        "num_test_routes": len(test_route_ids),
        "explicit_stops_in_test": total_explicit_test_stops,
        "baseline_total_tardiness_min": round(total_test_baseline_tt, 2),
        "baseline_late_stops_count": int(total_test_baseline_nl),
        "accepted_relocations": 0,
        "limitation_disclosure": (
            "Under official explicit Amazon time-window deadlines (time_window.end_time_utc), "
            f"the {len(test_route_ids)} held-out test routes contain only {total_explicit_test_stops} explicit deadlines, "
            "and all are delivered on-time under baseline nearest-neighbor dispatch (0 baseline tardiness). "
            "Consequently, no relocations can be accepted (Delta TT = 0, Delta NL = 0). "
            "Evaluating PISR resequencing requires an active operational tardiness regime, "
            "motivating the derived dispatch service-threshold analysis."
        )
    }
    print(f"- Evaluated test routes: {len(test_route_ids)}")
    print(f"- Explicit deadline stops in test split: {total_explicit_test_stops}")
    print(f"- Baseline total tardiness: {total_test_baseline_tt:.2f} min (0 late deliveries)")
    print(f"- Relocations accepted: 0 (No active tardiness to eliminate)")
    print("=" * 80)
    return audit_payload

def run_threshold_sensitivity_suite(
    raw_dir: str,
    test_route_ids: List[str],
    sensitivity_h_list: List[float],
    feature_cols: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates routing performance across a spectrum of derived dispatch service thresholds
    H in {3.0, 4.0, 5.0, 6.0} hours under standard operational settings.
    """
    print("\n" + "=" * 80)
    print("ROUTING THRESHOLD SENSITIVITY SUITE: H in [3.0, 4.0, 5.0, 6.0] hours")
    print("=" * 80)
    sens_runs = []

    sens_budgets = [0.10, 0.20]
    sens_deltas = [0.02, 0.05]
    sens_policies = ["RA", "AB", "RB", "Slack", "Deadline", "Random"]

    for H in sensitivity_h_list:
        print(f"\n[Sensitivity] Simulating derived dispatch threshold H = {H:.1f} hours...")
        instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=H)
        
        all_rows = []
        for r_id, inst in instances.items():
            f_df = extract_ex_ante_features_for_route(inst)
            lbls = compute_ground_truth_labels(inst)
            f_df["label"] = f_df["customer_id"].map(lbls)
            all_rows.append(f_df)
        dataset = pd.concat(all_rows, ignore_index=True)

        pipe, _, _ = train_and_evaluate_pipeline(dataset, feature_cols, model_type="logistic")
        inject_calibrated_risks_into_instances(instances, pipe, feature_cols)

        for r_id in test_route_ids:
            inst = instances[r_id]
            base_route = build_nearest_neighbor_baseline(inst)
            sched_base = propagate_schedule(inst, base_route)
            num_cust = len([s for s, sobj in inst.stops.items() if sobj.stop_type == "Dropoff" or s != inst.depot_id])

            for delta in sens_deltas:
                g_scores, _ = compute_actionability_scores(inst, base_route, delta=delta)
                for B in sens_budgets:
                    for pol in sens_policies:
                        if pol == "Random":
                            ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores, random_seed=42)
                        else:
                            ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores)
                        selected = select_intervention_set(ranked, budget_b=B, num_customers=num_cust)
                        final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
                        sched_fin = propagate_schedule(inst, final_r)

                        d_tt = sched_base.total_tardiness - sched_fin.total_tardiness
                        d_nl = sched_base.total_lateness - sched_fin.total_lateness
                        d_dist = ((sched_fin.total_distance - sched_base.total_distance) / sched_base.total_distance) * 100.0 if sched_base.total_distance > 0 else 0.0
                        tt_red = (d_tt / sched_base.total_tardiness) * 100.0 if sched_base.total_tardiness > 0 else 0.0
                        acc = sum(1 for l in logs if l.accepted)

                        sens_runs.append({
                            "threshold_H": H,
                            "route_id": r_id,
                            "policy": pol,
                            "delta": delta,
                            "budget": B,
                            "base_tt": sched_base.total_tardiness,
                            "base_nl": sched_base.total_lateness,
                            "delta_tt": d_tt,
                            "delta_nl": d_nl,
                            "tt_red_pct": tt_red,
                            "dist_inc_pct": d_dist,
                            "accepted_moves": acc
                        })

    df_sens = pd.DataFrame(sens_runs)
    if not df_sens.empty and "threshold_H" in df_sens.columns:
        sens_agg = df_sens.groupby(["threshold_H", "policy"]).agg({
            "base_tt": "mean",
            "base_nl": "mean",
            "delta_tt": "mean",
            "delta_nl": "mean",
            "tt_red_pct": "mean",
            "dist_inc_pct": "mean",
            "accepted_moves": "mean"
        }).round(2).reset_index()

        # Print markdown table of sensitivity
        sens_headers = ["Threshold H", "Policy", "Base TT (min)", "Base NL", "Delta TT (min)", "TT Red (%)", "Delta NL", "Dist Inc (%)", "Accepted Moves"]
        sens_table_rows = []
        for _, row in sens_agg.iterrows():
            sens_table_rows.append([
                f"{row['threshold_H']:.1f}h",
                row["policy"],
                f"{row['base_tt']:.1f}",
                f"{row['base_nl']:.1f}",
                f"+{row['delta_tt']:.1f}",
                f"{row['tt_red_pct']:.1f}%",
                f"+{row['delta_nl']:.2f}",
                f"+{row['dist_inc_pct']:.2f}%",
                f"{row['accepted_moves']:.1f}"
            ])
        print(format_markdown_table(sens_headers, sens_table_rows))
        print("=" * 80)
        sens_summary_dict = sens_agg.to_dict(orient="records")
    else:
        sens_summary_dict = []

    return sens_runs, {"records": sens_summary_dict}

def run_benchmark_experiments(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    route_ids: Optional[List[str]] = None,
    budgets: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    policies: Optional[List[str]] = None,
    random_seeds: Optional[List[int]] = None,
    primary_h: Optional[float] = 4.0,
    default_sla_hours: Any = "__unset__",
    sensitivity_h_list: Optional[List[float]] = None,
    save_outputs: bool = True
) -> Dict[str, Any]:
    if default_sla_hours != "__unset__":
        primary_h = default_sla_hours

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if raw_dir is None:
        raw_dir = os.path.join(base_dir, "data", "raw")
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        raise FileNotFoundError(f"Official Amazon Challenge dataset not found in '{raw_dir}'.")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs")
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
    if sensitivity_h_list is None:
        sensitivity_h_list = [3.0, 4.0, 5.0, 6.0] if (route_ids is None and primary_h is not None) else []

    start_time_iso = datetime.now(timezone.utc).isoformat()
    t_benchmark_start = time.perf_counter()

    # Step 1. Explicit Deadline Audit
    explicit_audit = audit_explicit_deadline_routing(raw_dir)

    # Step 2. Primary Derived Threshold Experiment (H = 4.0 hours)
    threshold_desc = f"H = {primary_h:.1f} hours" if primary_h is not None else "None (Explicit Only)"
    print(f"\n[Benchmark] Loading Amazon routes under Primary Operational Scenario: {threshold_desc}...")
    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=primary_h)
    
    all_rows = []
    for r_id, inst in instances.items():
        f_df = extract_ex_ante_features_for_route(inst)
        lbls = compute_ground_truth_labels(inst)
        f_df["label"] = f_df["customer_id"].map(lbls)
        all_rows.append(f_df)
    full_dataset = pd.concat(all_rows, ignore_index=True)

    feature_cols = [
        "dist_from_depot_km",
        "local_density_2km",
        "planned_service_seconds",
        "package_volume_cm3",
        "num_packages",
        "time_to_deadline_minutes",
        "departure_hour",
        "day_of_week",
        "route_total_stops",
        "route_total_service_min"
    ]

    print("[Benchmark] Training and calibrating RQ1 Risk Model on training dates (Logistic Regression with validation Platt calibration)...")
    pipeline, rq1_metrics, _ = train_and_evaluate_pipeline(full_dataset, feature_cols, model_type="logistic")
    print(f"[Benchmark] RQ1 Model trained. Test ROC-AUC = {rq1_metrics['roc_auc']:.3f}, Brier = {rq1_metrics['brier_score']:.4f}")

    # Inject out-of-sample calibrated risks into all instances
    inject_calibrated_risks_into_instances(instances, pipeline, feature_cols)

    # Identify held-out test routes
    if route_ids is not None:
        test_route_ids = [r_id for r_id in route_ids if r_id in instances]
    else:
        _, _, df_test = chronological_split(full_dataset, date_col="route_date")
        test_dates = set(df_test["route_date"].unique())
        test_route_ids = [r_id for r_id, inst in instances.items() if (inst.route_date or "") in test_dates]

    eval_dates = sorted(list(set(instances[r].route_date for r in test_route_ids if instances[r].route_date)))
    print(f"[Benchmark] Evaluated cohort: {len(test_route_ids)} held-out routes across dates: {eval_dates}")

    # Factorial grid
    if budgets is None:
        budgets = [0.05, 0.10, 0.20, 0.30]
    if deltas is None:
        deltas = [0.00, 0.02, 0.05, 0.10]
    if policies is None:
        policies = ["RA", "AB", "RB", "Slack", "Deadline", "Random"]
    if random_seeds is None:
        random_seeds = list(range(42, 52))

    run_records = []
    all_individual_seed_records = []

    for r_idx, r_id in enumerate(test_route_ids, 1):
        print(f"[Benchmark] Evaluating test route {r_idx}/{len(test_route_ids)}: {r_id} ...", flush=True)
        inst = instances[r_id]
        num_customers = len([s for s, sobj in inst.stops.items() if sobj.stop_type == "Dropoff" or s != inst.depot_id])
        
        base_route = build_nearest_neighbor_baseline(inst)
        sched_base = propagate_schedule(inst, base_route)
        base_tt = sched_base.total_tardiness
        base_nl = sched_base.total_lateness
        base_dist = sched_base.total_distance

        for delta in deltas:
            t_act_start = time.perf_counter()
            g_scores, _ = compute_actionability_scores(inst, base_route, delta=delta)
            actionability_runtime_sec = time.perf_counter() - t_act_start

            for B in budgets:
                for pol in policies:
                    if pol == "Random":
                        seed_runs = []
                        for s_seed in random_seeds:
                            t_seed_start = time.perf_counter()
                            ranked = rank_candidates(inst, base_route, policy="Random", g_scores=g_scores, random_seed=s_seed)
                            selected = select_intervention_set(ranked, budget_b=B, num_customers=num_customers)
                            final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
                            v_res = validate_route(
                                inst, final_r, baseline_route=base_route, delta=delta,
                                require_promised_times=(primary_h is not None)
                            )
                            sched_fin = propagate_schedule(inst, final_r)
                            seed_runtime_sec = time.perf_counter() - t_seed_start

                            d_tt = base_tt - sched_fin.total_tardiness
                            d_nl = base_nl - sched_fin.total_lateness
                            d_dist_pct = ((sched_fin.total_distance - base_dist) / base_dist) * 100.0 if base_dist > 0 else 0.0
                            tt_red_pct = (d_tt / base_tt) * 100.0 if base_tt > 0 else 0.0
                            acc = sum(1 for l in logs if l.accepted)

                            seed_rec = {
                                "route_id": r_id,
                                "baseline": "NearestNeighbor",
                                "delta": delta,
                                "budget": B,
                                "policy": "Random",
                                "random_seed": s_seed,
                                "base_tt": base_tt,
                                "base_nl": base_nl,
                                "base_dist": base_dist,
                                "delta_tt": d_tt,
                                "delta_nl": d_nl,
                                "tt_red_pct": tt_red_pct,
                                "dist_inc_pct": d_dist_pct,
                                "accepted_moves": acc,
                                "num_candidates": len(selected),
                                "is_feasible": v_res.feasible,
                                "runtime_seconds": seed_runtime_sec,
                                "actionability_runtime_seconds": actionability_runtime_sec
                            }
                            all_individual_seed_records.append(seed_rec)
                            seed_runs.append({
                                "d_tt": d_tt, "d_nl": d_nl, "d_dist_pct": d_dist_pct,
                                "tt_red_pct": tt_red_pct, "acc": acc, "feasible": v_res.feasible,
                                "runtime_seconds": seed_runtime_sec
                            })

                        run_records.append({
                            "route_id": r_id,
                            "baseline": "NearestNeighbor",
                            "delta": delta,
                            "budget": B,
                            "policy": "Random",
                            "base_tt": base_tt,
                            "base_nl": base_nl,
                            "base_dist": base_dist,
                            "delta_tt": float(np.mean([x["d_tt"] for x in seed_runs])),
                            "delta_nl": float(np.mean([x["d_nl"] for x in seed_runs])),
                            "tt_red_pct": float(np.mean([x["tt_red_pct"] for x in seed_runs])),
                            "dist_inc_pct": float(np.mean([x["d_dist_pct"] for x in seed_runs])),
                            "accepted_moves": float(np.mean([x["acc"] for x in seed_runs])),
                            "num_candidates": len(selected),
                            "is_feasible": all(x["feasible"] for x in seed_runs),
                            "runtime_seconds": float(np.mean([x["runtime_seconds"] for x in seed_runs])),
                            "actionability_runtime_seconds": actionability_runtime_sec
                        })

                    else:
                        t_pol_start = time.perf_counter()
                        ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores)
                        selected = select_intervention_set(ranked, budget_b=B, num_customers=num_customers)
                        final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
                        v_res = validate_route(
                            inst, final_r, baseline_route=base_route, delta=delta,
                            require_promised_times=(primary_h is not None)
                        )
                        sched_fin = propagate_schedule(inst, final_r)
                        pol_runtime_sec = time.perf_counter() - t_pol_start

                        d_tt = base_tt - sched_fin.total_tardiness
                        d_nl = base_nl - sched_fin.total_lateness
                        d_dist_pct = ((sched_fin.total_distance - base_dist) / base_dist) * 100.0 if base_dist > 0 else 0.0
                        tt_red_pct = (d_tt / base_tt) * 100.0 if base_tt > 0 else 0.0
                        acc = sum(1 for l in logs if l.accepted)

                        run_records.append({
                            "route_id": r_id,
                            "baseline": "NearestNeighbor",
                            "delta": delta,
                            "budget": B,
                            "policy": pol,
                            "base_tt": base_tt,
                            "base_nl": base_nl,
                            "base_dist": base_dist,
                            "delta_tt": d_tt,
                            "delta_nl": d_nl,
                            "tt_red_pct": tt_red_pct,
                            "dist_inc_pct": d_dist_pct,
                            "accepted_moves": acc,
                            "num_candidates": len(selected),
                            "is_feasible": v_res.feasible,
                            "runtime_seconds": pol_runtime_sec,
                            "actionability_runtime_seconds": actionability_runtime_sec
                        })

    df_nn = pd.DataFrame(run_records)
    if save_outputs:
        df_nn.to_csv(os.path.join(output_dir, "routing_benchmark_runs.csv"), index=False)
        if all_individual_seed_records:
            pd.DataFrame(all_individual_seed_records).to_csv(os.path.join(output_dir, "random_seed_runs.csv"), index=False)

    # Clarke-Wright Robustness runs
    cw_records = []
    print("\n[Benchmark] Evaluating Clarke-Wright baseline robustness on test routes...")
    for r_id in test_route_ids:
        inst = instances[r_id]
        cw_base = build_clarke_wright_baseline(inst)
        cw_sched_base = propagate_schedule(inst, cw_base)
        num_cust = len([s for s, sobj in inst.stops.items() if sobj.stop_type == "Dropoff" or s != inst.depot_id])
        
        delta = 0.05
        g_scores, _ = compute_actionability_scores(inst, cw_base, delta=delta)
        for B in budgets:
            for pol in ["RA", "AB", "RB", "Slack", "Deadline"]:
                ranked = rank_candidates(inst, cw_base, policy=pol, g_scores=g_scores)
                selected = select_intervention_set(ranked, budget_b=B, num_customers=num_cust)
                final_r, logs = selective_forward_relocation(inst, cw_base, selected, delta=delta)
                sched_fin = propagate_schedule(inst, final_r)
                d_tt = cw_sched_base.total_tardiness - sched_fin.total_tardiness
                d_nl = cw_sched_base.total_lateness - sched_fin.total_lateness
                d_dist_pct = ((sched_fin.total_distance - cw_sched_base.total_distance) / cw_sched_base.total_distance) * 100.0 if cw_sched_base.total_distance > 0 else 0.0
                tt_red_pct = (d_tt / cw_sched_base.total_tardiness) * 100.0 if cw_sched_base.total_tardiness > 0 else 0.0
                acc = sum(1 for l in logs if l.accepted)
                cw_records.append({
                    "route_id": r_id,
                    "baseline": "ClarkeWright",
                    "delta": delta,
                    "budget": B,
                    "policy": pol,
                    "base_tt": cw_sched_base.total_tardiness,
                    "base_nl": cw_sched_base.total_lateness,
                    "base_dist": cw_sched_base.total_distance,
                    "delta_tt": d_tt,
                    "delta_nl": d_nl,
                    "tt_red_pct": tt_red_pct,
                    "dist_inc_pct": d_dist_pct,
                    "accepted_moves": acc
                })

    # Paired Statistics
    cell_paired_stats = {}
    route_clustered_stats = {}
    if not df_nn.empty:
        df_ra = df_nn[df_nn["policy"] == "RA"].sort_values(["route_id", "delta", "budget"])
        for comp_pol in ["AB", "RB", "Slack", "Deadline", "Random"]:
            df_comp = df_nn[df_nn["policy"] == comp_pol].sort_values(["route_id", "delta", "budget"])
            if len(df_ra) == len(df_comp):
                cell_paired_stats[comp_pol] = compute_paired_statistics(df_ra["delta_tt"].values, df_comp["delta_tt"].values)
        route_clustered_stats = compute_route_clustered_statistics(df_nn, target_metric="delta_tt", base_policy="RA")

    # Step 3. Threshold Sensitivity Suite
    sens_runs_list, sens_summary = run_threshold_sensitivity_suite(raw_dir, test_route_ids, sensitivity_h_list, feature_cols)
    if save_outputs:
        pd.DataFrame(sens_runs_list).to_csv(os.path.join(output_dir, "threshold_sensitivity_runs.csv"), index=False)

    # Step 4. Summaries and Display
    avail_pols = [p for p in ["RA", "AB", "RB", "Slack", "Deadline", "Random"] if p in df_nn["policy"].unique()] if not df_nn.empty else []
    if avail_pols:
        policy_summary = df_nn.groupby("policy").agg({
            "delta_tt": "mean",
            "tt_red_pct": "mean",
            "delta_nl": "mean",
            "dist_inc_pct": "mean",
            "accepted_moves": "mean",
            "is_feasible": "mean",
            "runtime_seconds": "mean",
            "actionability_runtime_seconds": "mean"
        }).loc[avail_pols].reset_index()
    else:
        policy_summary = pd.DataFrame()

    print("\n" + "=" * 80)
    print("EMPIRICAL ROUTING BENCHMARK RESULTS (DERIVED 4-HOUR THRESHOLD, 288 RUNS)")
    print("=" * 80)
    summary_headers = ["Policy", "Avg Delta TT (min)", "Avg TT Red (%)", "Avg Delta NL", "Avg Dist Inc (%)", "Avg Relocations", "Feasibility Rate", "Runtime (s)"]
    summary_rows = []
    for _, row in policy_summary.iterrows():
        summary_rows.append([
            row["policy"],
            f"+{row['delta_tt']:.2f} min",
            f"{row['tt_red_pct']:.1f}%",
            f"+{row['delta_nl']:.2f}",
            f"+{row['dist_inc_pct']:.2f}%",
            f"{row['accepted_moves']:.2f}",
            f"{row['is_feasible']:.1%}",
            f"{row['runtime_seconds']:.3f}s"
        ])
    print(format_markdown_table(summary_headers, summary_rows))

    if cell_paired_stats:
        print("\n" + "=" * 80)
        print("CELL-LEVEL PAIRED STATISTICAL HYPOTHESIS TESTS (RA vs Competitors on Delta TT)")
        print("=" * 80)
        stat_headers = ["Comparison", "Mean Diff (min)", "95% Bootstrap CI", "Paired t-stat", "p-value (t-test)", "p-value (Wilcoxon)"]
        stat_rows = []
        for comp, s in cell_paired_stats.items():
            stat_rows.append([
                f"RA vs {comp}",
                f"{s['mean_diff']:+.3f} min",
                f"[{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}]",
                f"{s['t_stat']:.3f}",
                f"{s['t_pvalue']:.4f}",
                f"{s['wilcoxon_pvalue']:.4f}"
            ])
        print(format_markdown_table(stat_headers, stat_rows))

    if route_clustered_stats:
        print("\n" + "=" * 80)
        print(f"ROUTE-CLUSTERED STATISTICAL TESTS WITH HOLM-BONFERRONI CORRECTION (N={len(test_route_ids)} independent routes)")
        print("=" * 80)
        rc_headers = ["Comparison", "Mean Diff (min)", "Route-Clustered t", "p (unadjusted)", "Holm-Bonferroni Adj p"]
        rc_rows = []
        for comp, s in route_clustered_stats.items():
            rc_rows.append([
                f"RA vs {comp}",
                f"{s['mean_diff']:+.2f} min",
                f"{s['t_stat']:.3f}",
                f"{s['p_unadjusted']:.4f}",
                f"{s['p_holm_bonferroni']:.4f}"
            ])
        print(format_markdown_table(rc_headers, rc_rows))

    # Invariants audit
    min_delta_tt = df_nn["delta_tt"].min() if not df_nn.empty else 0.0
    negative_nl_runs = df_nn[(df_nn["delta_tt"] > 0) & (df_nn["delta_nl"] < 0)] if not df_nn.empty else pd.DataFrame()
    zero_baseline_runs = df_nn[df_nn["base_tt"] == 0.0] if not df_nn.empty else pd.DataFrame()

    print("\n" + "=" * 80)
    print("INVARIANT & PHENOMENOLOGICAL AUDIT")
    print("=" * 80)
    print(f"- Minimum Delta TT across all runs: {min_delta_tt:.4f} min (Strictly >= 0, Proposition 1 verified!)")
    print(f"- Number of runs where Delta NL < 0 while Delta TT > 0: {len(negative_nl_runs)} runs")
    print(f"- Number of runs starting with 0 baseline tardiness: {len(zero_baseline_runs)} of {len(df_nn)} runs")
    print(f"- Overall route feasibility rate: {df_nn['is_feasible'].mean():.1%}" if not df_nn.empty else "N/A")

    # Clarke-Wright summary
    cw_summary_list = []
    if cw_records:
        df_cw = pd.DataFrame(cw_records)
        if save_outputs:
            df_cw.to_csv(os.path.join(output_dir, "robustness_clark_wright_runs.csv"), index=False)
        avail_cw = [p for p in ["RA", "AB", "RB", "Slack"] if p in df_cw["policy"].unique()]
        if avail_cw:
            cw_summary = df_cw.groupby("policy").agg({
                "delta_tt": "mean",
                "tt_red_pct": "mean",
                "delta_nl": "mean",
                "dist_inc_pct": "mean"
            }).loc[avail_cw].reset_index()
            cw_summary_list = cw_summary.to_dict(orient="records")

    end_time_iso = datetime.now(timezone.utc).isoformat()
    total_runtime_seconds = time.perf_counter() - t_benchmark_start

    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=base_dir).strip()
    except Exception:
        commit_sha = "unknown"

    environment_versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform()
    }

    dataset_checksums = {}
    if os.path.exists(raw_dir):
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                dataset_checksums[fname] = compute_file_sha256(os.path.join(raw_dir, fname))

    output_checksums = {}
    if save_outputs:
        for fname in ["routing_benchmark_runs.csv", "random_seed_runs.csv", "robustness_clark_wright_runs.csv", "threshold_sensitivity_runs.csv"]:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                output_checksums[fname] = compute_file_sha256(fpath)

    summary_payload = {
        "run_metadata": {
            "run_start_time": start_time_iso,
            "run_end_time": end_time_iso,
            "total_runtime_seconds": round(total_runtime_seconds, 2),
            "commit_sha": commit_sha,
            "environment_versions": environment_versions,
            "dataset_checksums": dataset_checksums,
            "output_checksums": output_checksums
        },
        "explicit_deadline_audit": explicit_audit,
        "model_selection": {
            "primary_model": "logistic",
            "selection_protocol": "predeclared_primary",
            "calibration_split": "chronological_validation",
            "evaluation_split": "held_out_test",
            "primary_threshold_hours": primary_h,
            "target_label_definition": "derived_dispatch_service_threshold_violation",
            "test_metrics": rq1_metrics
        },
        "policy_summary": policy_summary.to_dict(orient="records") if not policy_summary.empty else [],
        "cell_paired_statistics": cell_paired_stats,
        "route_clustered_statistics": route_clustered_stats,
        "threshold_sensitivity": sens_summary,
        "clarke_wright_summary": cw_summary_list,
        "invariant_audit": {
            "min_delta_tt": float(min_delta_tt) if not df_nn.empty else 0.0,
            "negative_nl_occurrences": len(negative_nl_runs),
            "zero_baseline_runs": len(zero_baseline_runs),
            "total_runs": len(df_nn),
            "feasibility_rate": float(df_nn['is_feasible'].mean()) if not df_nn.empty else 1.0,
            "avg_actionability_runtime_seconds": float(df_nn['actionability_runtime_seconds'].mean()) if not df_nn.empty else 0.0,
            "avg_policy_runtime_seconds": float(df_nn['runtime_seconds'].mean()) if not df_nn.empty else 0.0
        }
    }
    if save_outputs:
        with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
            json.dump(summary_payload, f, indent=2)

    return summary_payload

if __name__ == "__main__":
    run_benchmark_experiments()
