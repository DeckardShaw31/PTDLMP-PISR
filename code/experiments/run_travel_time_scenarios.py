"""
M8 pilot: independent travel-time scenario evaluation of frozen PISR decisions.

  1. Planning:   official historical mean travel-time matrix -> p_i, g_i, policy ranking, budget B, SFR with delta.
  2. Freeze:     baseline R0 and final R' of every route x policy x B x delta (x random seed) are written to
                 scenario_frozen_routes.json and never modified afterwards.
  3. Evaluation: frozen R0 / R' are re-timed on mean-preserving lognormal travel-time scenarios T^(s).
                 No p_i / g_i recomputation, no re-selection and no SFR re-run happens in this stage.
  4. Comparison: Delta TT^(s), Delta NL^(s) aggregated by route and scenario; RA vs AB (V_risk) and baselines.

Scenario replications on the same route are NOT independent observations: statistical tests are route-level.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.features.build_features import extract_ex_ante_features_for_route, compute_ground_truth_labels
from src.prediction.train import train_and_evaluate_pipeline, inject_calibrated_risks_into_instances, chronological_split
from src.routing.baseline import build_nearest_neighbor_baseline
from src.routing.schedule import propagate_schedule
from src.routing.actionability import compute_actionability_scores, rank_candidates, select_intervention_set
from src.routing.sfr import selective_forward_relocation
from src.routing.validator import validate_route
from src.evaluation.scenarios import (
    build_route_time_data, draw_standard_normals, scenario_travel_times, matrix_fingerprint,
    encode_sequences, evaluate_sequences, lognormal_sigma
)
from src.evaluation.statistics import compute_route_clustered_statistics
from src.evaluation.tables import format_markdown_table
from experiments.run_rq1_prediction import FEATURE_COLS
from experiments.run_routing_benchmark import compute_file_sha256

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RAW_DIR = os.path.join(CODE_DIR, "data", "raw")
DEFAULT_OUTPUT_DIR = os.path.join(CODE_DIR, "outputs")
DEFAULT_REPORT_PATH = os.path.join(CODE_DIR, "reports", "travel_time_scenario_pilot.md")

DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_DELTAS = [0.00, 0.02, 0.05, 0.10]
DEFAULT_POLICIES = ["RA", "AB", "RB", "Slack", "Deadline", "Random"]
DEFAULT_RANDOM_SEEDS = list(range(42, 52))
DEFAULT_CV_LEVELS = [0.0, 0.10, 0.20, 0.30]
DEFAULT_SCENARIO_SEED = 2026

# Same strict-improvement tolerance as SFR (Eq. 17): differences below it are treated as ties.
EPS_MIN = 1e-6
# Allowed numerical gap between float-second propagation and datetime (microsecond) propagation.
CV0_TT_TOLERANCE_MIN = 1e-3

FROZEN_FILENAME = "scenario_frozen_routes.json"

def _git_commit_sha() -> str:
    """HEAD commit, suffixed with '-dirty' when the working tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=CODE_DIR).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=CODE_DIR).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"

# ----------------------------------------------------------------------
# Stage 1-2: Planning on the mean matrix and freezing R0 / R'
# ----------------------------------------------------------------------
def plan_and_freeze(
    raw_dir: str,
    frozen_path: str,
    primary_h: float = 4.0,
    route_ids: Optional[List[str]] = None,
    budgets: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    policies: Optional[List[str]] = None,
    random_seeds: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Reproduces the routing benchmark planning decisions (same model, same R0, same grid) and freezes them.
    """
    budgets = budgets if budgets is not None else DEFAULT_BUDGETS
    deltas = deltas if deltas is not None else DEFAULT_DELTAS
    policies = policies if policies is not None else DEFAULT_POLICIES
    random_seeds = random_seeds if random_seeds is not None else DEFAULT_RANDOM_SEEDS

    print("\n" + "=" * 80)
    print(f"STAGE 1 - PLANNING ON THE OFFICIAL MEAN TRAVEL-TIME MATRIX (H = {primary_h:.1f} hours)")
    print("=" * 80)
    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=primary_h)

    all_rows = []
    for r_id, inst in instances.items():
        f_df = extract_ex_ante_features_for_route(inst)
        lbls = compute_ground_truth_labels(inst)
        f_df["label"] = f_df["customer_id"].map(lbls)
        all_rows.append(f_df)
    full_dataset = pd.concat(all_rows, ignore_index=True)

    pipeline, rq1_metrics, _ = train_and_evaluate_pipeline(full_dataset, FEATURE_COLS, model_type="logistic")
    inject_calibrated_risks_into_instances(instances, pipeline, FEATURE_COLS)
    print(f"[Planning] Risk model trained. Test ROC-AUC = {rq1_metrics['roc_auc']:.3f}, Brier = {rq1_metrics['brier_score']:.4f}")

    if route_ids is not None:
        test_route_ids = [r_id for r_id in route_ids if r_id in instances]
    else:
        _, _, df_test = chronological_split(full_dataset, date_col="route_date")
        test_dates = set(df_test["route_date"].unique())
        test_route_ids = [r_id for r_id, inst in instances.items() if (inst.route_date or "") in test_dates]

    frozen_routes: Dict[str, Any] = {}
    for r_idx, r_id in enumerate(test_route_ids, 1):
        print(f"[Planning] Route {r_idx}/{len(test_route_ids)}: {r_id} ...", flush=True)
        inst = instances[r_id]
        num_customers = len([s for s, sobj in inst.stops.items() if sobj.stop_type == "Dropoff" or s != inst.depot_id])
        base_route = build_nearest_neighbor_baseline(inst)
        sched_base = propagate_schedule(inst, base_route)

        plans = []
        actionability = {}
        for delta in deltas:
            g_scores, _ = compute_actionability_scores(inst, base_route, delta=delta)
            actionability[f"{delta:.2f}"] = {c: round(g, 6) for c, g in g_scores.items()}
            for B in budgets:
                for pol in policies:
                    seeds = random_seeds if pol == "Random" else [None]
                    for s_seed in seeds:
                        if pol == "Random":
                            ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores, random_seed=s_seed)
                        else:
                            ranked = rank_candidates(inst, base_route, policy=pol, g_scores=g_scores)
                        selected = select_intervention_set(ranked, budget_b=B, num_customers=num_customers)
                        final_r, logs = selective_forward_relocation(inst, base_route, selected, delta=delta)
                        v_res = validate_route(inst, final_r, baseline_route=base_route, delta=delta)
                        sched_fin = propagate_schedule(inst, final_r)

                        plans.append({
                            "policy": pol,
                            "budget": B,
                            "delta": delta,
                            "random_seed": s_seed,
                            "selected": selected,
                            "accepted": [l.customer_id for l in logs if l.accepted],
                            "final_route": final_r,
                            "planning": {
                                "final_tt": sched_fin.total_tardiness,
                                "final_nl": sched_fin.total_lateness,
                                "final_dist": sched_fin.total_distance,
                                "is_feasible": v_res.feasible
                            }
                        })

        frozen_routes[r_id] = {
            "route_date": inst.route_date,
            "baseline_route": base_route,
            "baseline_planning": {
                "tt": sched_base.total_tardiness,
                "nl": sched_base.total_lateness,
                "dist": sched_base.total_distance
            },
            "predicted_risk": {c: round(inst.stops[c].predicted_risk_pi, 6) for c in base_route[1:]},
            "actionability": actionability,
            "plans": plans
        }

    payload = {
        "metadata": {
            "description": "Frozen ex-ante PISR decisions planned on the official mean travel-time matrix (M8 pilot).",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "commit_sha": _git_commit_sha(),
            "threshold_hours": primary_h,
            "baseline_constructor": "NearestNeighbor",
            "budgets": budgets,
            "deltas": deltas,
            "policies": policies,
            "random_seeds": random_seeds,
            "risk_model": "logistic + validation Platt calibration",
            "risk_model_test_metrics": {k: v for k, v in rq1_metrics.items() if not isinstance(v, dict)},
            "dataset_checksums": {
                f: compute_file_sha256(os.path.join(raw_dir, f))
                for f in sorted(os.listdir(raw_dir)) if f.endswith(".json")
            }
        },
        "routes": frozen_routes
    }

    os.makedirs(os.path.dirname(frozen_path), exist_ok=True)
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    num_plans = sum(len(r["plans"]) for r in frozen_routes.values())
    print(f"[Freeze] Wrote {num_plans} frozen plans for {len(frozen_routes)} routes to '{frozen_path}'.")
    return payload

# ----------------------------------------------------------------------
# Stage 3: Scenario evaluation of frozen routes
# ----------------------------------------------------------------------
def _sequence_digest(sequences: List[List[str]]) -> str:
    h = hashlib.sha256()
    for seq in sequences:
        h.update(("|".join(seq) + "\n").encode("utf-8"))
    return h.hexdigest()

def evaluate_frozen_routes(
    raw_dir: str,
    frozen: Dict[str, Any],
    n_scenarios: int,
    cv_levels: List[float],
    base_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Re-times frozen R0 / R' on travel-time scenarios. Only service thresholds, planned service durations and
    the mean matrix are read from the dataset; decisions come exclusively from the frozen payload.

    Returns:
      plan_df: one row per route x CV x scenario x frozen plan (random seeds kept separate)
      fp_df:   matrix fingerprints and multiplier diagnostics per route x CV x scenario
      audit:   freeze integrity, zero-mean edge usage, reproducibility audit
    """
    print("\n" + "=" * 80)
    print(f"STAGE 3 - SCENARIO EVALUATION OF FROZEN ROUTES ({n_scenarios} scenarios x CV {cv_levels})")
    print("=" * 80)
    primary_h = frozen["metadata"]["threshold_hours"]
    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=primary_h)

    plan_frames = []
    fp_records = []
    audit: Dict[str, Any] = {"routes": {}}

    for r_id, r_frozen in frozen["routes"].items():
        inst = instances[r_id]
        data = build_route_time_data(inst)
        plans = r_frozen["plans"]
        sequences = [r_frozen["baseline_route"]] + [p["final_route"] for p in plans]
        digest_before = _sequence_digest(sequences)
        encoded = encode_sequences(data, sequences)

        off_diag = ~np.eye(len(data.stop_ids), dtype=bool)
        zero_mean_edges = off_diag & (data.mean_travel_seconds == 0.0)
        used = np.zeros_like(zero_mean_edges)
        used[encoded[:, :-1], encoded[:, 1:]] = True
        positive_edges = off_diag & (data.mean_travel_seconds > 0.0)

        n_plans = len(plans)
        n_cv = len(cv_levels)
        final_tt = np.empty((n_scenarios, n_cv, n_plans))
        final_nl = np.empty((n_scenarios, n_cv, n_plans))
        base_tt = np.empty((n_scenarios, n_cv))
        base_nl = np.empty((n_scenarios, n_cv))

        for s in range(n_scenarios):
            z = draw_standard_normals(r_id, s, len(data.stop_ids), base_seed)
            for c_idx, cv in enumerate(cv_levels):
                matrix = scenario_travel_times(data.mean_travel_seconds, z, cv)
                # One matrix per route x scenario x CV, shared by R0 and every policy's R'
                tt, nl = evaluate_sequences(data, matrix, encoded)
                base_tt[s, c_idx], base_nl[s, c_idx] = tt[0], nl[0]
                final_tt[s, c_idx], final_nl[s, c_idx] = tt[1:], nl[1:]

                mult = matrix[positive_edges] / data.mean_travel_seconds[positive_edges]
                fp_records.append({
                    "route_id": r_id,
                    "cv": cv,
                    "scenario_id": s,
                    "matrix_sha256_16": matrix_fingerprint(matrix),
                    "multiplier_mean": float(mult.mean()),
                    "multiplier_cv": float(mult.std() / mult.mean()),
                    "multiplier_min": float(mult.min()),
                    "multiplier_max": float(mult.max()),
                    "baseline_tt": float(tt[0]),
                    "baseline_nl": int(nl[0])
                })

        meta = pd.DataFrame({
            "policy": [p["policy"] for p in plans],
            "budget": [p["budget"] for p in plans],
            "delta": [p["delta"] for p in plans],
            "random_seed": [p["random_seed"] if p["random_seed"] is not None else -1 for p in plans],
            "accepted_moves": [len(p["accepted"]) for p in plans],
            "planning_final_tt": [p["planning"]["final_tt"] for p in plans],
            "planning_final_nl": [p["planning"]["final_nl"] for p in plans],
            "dist_inc_pct": [
                (p["planning"]["final_dist"] - r_frozen["baseline_planning"]["dist"]) / r_frozen["baseline_planning"]["dist"] * 100.0
                for p in plans
            ]
        })
        grid = len(meta)
        frame = pd.concat([meta] * (n_scenarios * n_cv), ignore_index=True)
        frame.insert(0, "route_id", r_id)
        frame.insert(1, "cv", np.repeat(np.tile(np.asarray(cv_levels, dtype=float), n_scenarios), grid))
        frame.insert(2, "scenario_id", np.repeat(np.arange(n_scenarios), n_cv * grid))
        frame["base_tt"] = np.repeat(base_tt.reshape(-1), grid)
        frame["base_nl"] = np.repeat(base_nl.reshape(-1), grid)
        frame["final_tt"] = final_tt.reshape(-1)
        frame["final_nl"] = final_nl.reshape(-1)
        frame["planning_base_tt"] = r_frozen["baseline_planning"]["tt"]
        frame["planning_base_nl"] = r_frozen["baseline_planning"]["nl"]
        plan_frames.append(frame)

        # Re-encoding after evaluation must give the same frozen decisions
        digest_after = _sequence_digest([r_frozen["baseline_route"]] + [p["final_route"] for p in plans])
        audit["routes"][r_id] = {
            "num_frozen_plans": n_plans,
            "sequence_digest": digest_before,
            "sequences_unchanged": digest_before == digest_after,
            "zero_mean_offdiag_edges": int(zero_mean_edges.sum()),
            "zero_mean_edges_on_frozen_routes": int((zero_mean_edges & used).sum()),
            "missing_edges": int(np.isnan(data.mean_travel_seconds).sum())
        }
        print(f"[Evaluation] {r_id}: {n_plans} frozen plans x {n_scenarios} scenarios x {n_cv} CV levels evaluated.")

    plan_df = pd.concat(plan_frames, ignore_index=True)
    plan_df["delta_tt"] = plan_df["base_tt"] - plan_df["final_tt"]
    plan_df["delta_nl"] = plan_df["base_nl"] - plan_df["final_nl"]
    fp_df = pd.DataFrame(fp_records)

    # Reproducibility audit: regenerate a subset of matrices and compare fingerprints
    regen_checked, regen_mismatch = 0, 0
    for r_id in frozen["routes"]:
        data = build_route_time_data(instances[r_id])
        for s in sorted({0, n_scenarios // 2, n_scenarios - 1}):
            z = draw_standard_normals(r_id, s, len(data.stop_ids), base_seed)
            for cv in cv_levels:
                fp = matrix_fingerprint(scenario_travel_times(data.mean_travel_seconds, z, cv))
                stored = fp_df[(fp_df["route_id"] == r_id) & (fp_df["cv"] == cv) & (fp_df["scenario_id"] == s)]["matrix_sha256_16"].iloc[0]
                regen_checked += 1
                regen_mismatch += int(fp != stored)
    audit["regenerated_matrices_checked"] = regen_checked
    audit["regenerated_matrix_mismatches"] = regen_mismatch
    return plan_df, fp_df, audit

# ----------------------------------------------------------------------
# Stage 4: Verification checks and aggregation
# ----------------------------------------------------------------------
def collapse_random_seeds(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Averages the seeded Random selections per scenario (same convention as the routing benchmark)."""
    keys = ["route_id", "cv", "scenario_id", "policy", "budget", "delta"]
    value_cols = ["base_tt", "base_nl", "final_tt", "final_nl", "delta_tt", "delta_nl",
                  "accepted_moves", "planning_final_tt", "planning_final_nl", "planning_base_tt",
                  "planning_base_nl", "dist_inc_pct"]
    return plan_df.groupby(keys, sort=False)[value_cols].mean().reset_index()

def run_verification_checks(
    plan_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    audit: Dict[str, Any]
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # (1a) Deterministic planning reproduces the committed routing benchmark
    plan_cells = plan_df[(plan_df["cv"] == 0.0) & (plan_df["scenario_id"] == 0)].copy()
    plan_cells["planning_delta_tt"] = plan_cells["planning_base_tt"] - plan_cells["planning_final_tt"]
    plan_cells["planning_delta_nl"] = plan_cells["planning_base_nl"] - plan_cells["planning_final_nl"]
    bench_path = os.path.join(DEFAULT_OUTPUT_DIR, "routing_benchmark_runs.csv")
    seed_path = os.path.join(DEFAULT_OUTPUT_DIR, "random_seed_runs.csv")
    repro = {"available": os.path.exists(bench_path) and os.path.exists(seed_path)}
    if repro["available"]:
        bench = pd.read_csv(bench_path)
        seeds = pd.read_csv(seed_path)
        keys = ["route_id", "policy", "budget", "delta"]
        det = plan_cells[plan_cells["policy"] != "Random"].merge(bench, on=keys, suffixes=("", "_bench"))
        rnd = plan_cells[plan_cells["policy"] == "Random"].merge(
            seeds, on=keys + ["random_seed"], suffixes=("", "_bench"))
        both = pd.concat([det, rnd], ignore_index=True)
        repro.update({
            "frozen_plans_compared": int(len(both)),
            "frozen_plans_total": int(len(plan_cells)),
            "max_abs_delta_tt_diff_min": float((both["planning_delta_tt"] - both["delta_tt_bench"]).abs().max()) if len(both) else None,
            "max_abs_delta_nl_diff": float((both["planning_delta_nl"] - both["delta_nl_bench"]).abs().max()) if len(both) else None,
            "accepted_moves_mismatches": int((both["accepted_moves"] != both["accepted_moves_bench"]).sum()) if len(both) else None,
            "max_abs_dist_inc_pct_diff": float((both["dist_inc_pct"] - both["dist_inc_pct_bench"]).abs().max()) if len(both) else None
        })
        repro["passed"] = bool(
            len(both) == len(plan_cells) and repro["max_abs_delta_tt_diff_min"] < 1e-9
            and repro["max_abs_delta_nl_diff"] == 0 and repro["accepted_moves_mismatches"] == 0
        )
    checks["deterministic_benchmark_reproduced"] = repro

    # (1b) CV = 0 scenario evaluation equals the deterministic (datetime-based) evaluation
    cv0 = plan_df[plan_df["cv"] == 0.0]
    if len(cv0):
        tt_gap = float(max((cv0["final_tt"] - cv0["planning_final_tt"]).abs().max(),
                           (cv0["base_tt"] - cv0["planning_base_tt"]).abs().max()))
        nl_mismatch = int(((cv0["final_nl"] != cv0["planning_final_nl"]) | (cv0["base_nl"] != cv0["planning_base_nl"])).sum())
        same_matrix_all_s = int(fp_df[fp_df["cv"] == 0.0].groupby("route_id")["matrix_sha256_16"].nunique().max())
        checks["cv0_equivalence"] = {
            "rows_compared": int(len(cv0)),
            "max_abs_tt_diff_min": tt_gap,
            "tolerance_min": CV0_TT_TOLERANCE_MIN,
            "nl_mismatches": nl_mismatch,
            "distinct_cv0_matrices_per_route": same_matrix_all_s,
            "passed": bool(tt_gap <= CV0_TT_TOLERANCE_MIN and nl_mismatch == 0 and same_matrix_all_s == 1)
        }
    else:
        checks["cv0_equivalence"] = {"passed": False, "reason": "CV = 0 not evaluated"}

    # (2) Frozen decisions untouched by evaluation
    checks["frozen_routes_unchanged"] = {
        "routes": {r: a["sequences_unchanged"] for r, a in audit["routes"].items()},
        "passed": all(a["sequences_unchanged"] for a in audit["routes"].values())
    }

    # (3) Same matrix for every policy and reproducible from the seed
    checks["shared_matrix_and_seed_reproducibility"] = {
        "matrices_per_route_scenario_cv": int(fp_df.groupby(["route_id", "cv", "scenario_id"]).size().max()),
        "regenerated_matrices_checked": audit["regenerated_matrices_checked"],
        "regenerated_matrix_mismatches": audit["regenerated_matrix_mismatches"],
        "passed": bool(fp_df.groupby(["route_id", "cv", "scenario_id"]).size().max() == 1
                       and audit["regenerated_matrix_mismatches"] == 0)
    }
    return checks

def _share(x: pd.Series, cond: str) -> float:
    if cond == "pos":
        return float((x > EPS_MIN).mean())
    if cond == "neg":
        return float((x < -EPS_MIN).mean())
    return float((x.abs() <= EPS_MIN).mean())

def summarize_outcomes(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Section 4.1 statistics of Delta TT^(s) and Delta NL^(s) over scenarios (and pooled cells)."""
    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        planning_dtt = g["planning_base_tt"] - g["planning_final_tt"]
        base = g["base_tt"].to_numpy()
        tt_red = np.divide(g["delta_tt"].to_numpy(), base, out=np.zeros(len(g)), where=base > 0) * 100.0
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": int(len(g)),
            "planning_delta_tt": float(planning_dtt.mean()),
            "mean_delta_tt": float(g["delta_tt"].mean()),
            "median_delta_tt": float(g["delta_tt"].median()),
            "p05_delta_tt": float(np.percentile(g["delta_tt"], 5)),
            "p95_delta_tt": float(np.percentile(g["delta_tt"], 95)),
            "mean_tt_red_pct": float(np.mean(tt_red)),
            "mean_delta_nl": float(g["delta_nl"].mean()),
            "median_delta_nl": float(g["delta_nl"].median()),
            "share_delta_tt_pos": _share(g["delta_tt"], "pos"),
            "share_delta_tt_neg": _share(g["delta_tt"], "neg"),
            "share_delta_nl_neg": _share(g["delta_nl"], "neg"),
            "retention_ratio": float(g["delta_tt"].mean() / planning_dtt.mean()) if planning_dtt.mean() > EPS_MIN else float("nan"),
            "accepted_moves": float(g["accepted_moves"].mean()),
            "dist_inc_pct": float(g["dist_inc_pct"].mean())
        })
    return pd.DataFrame(rows)

def compare_ra_ab(scen_df: pd.DataFrame, frozen: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Section 4.2: V_risk^(s) = Delta TT^(s)(RA) - Delta TT^(s)(AB) on identical scenarios,
    plus selection overlap and accepted moves from the frozen plans.
    """
    keys = ["route_id", "cv", "scenario_id", "budget", "delta"]
    ra = scen_df[scen_df["policy"] == "RA"].set_index(keys)
    ab = scen_df[scen_df["policy"] == "AB"].set_index(keys)
    pair = pd.DataFrame({
        "v_risk": ra["delta_tt"] - ab["delta_tt"],
        "v_risk_nl": ra["delta_nl"] - ab["delta_nl"],
        "planning_v_risk": (ra["planning_base_tt"] - ra["planning_final_tt"]) - (ab["planning_base_tt"] - ab["planning_final_tt"])
    }).reset_index()

    plan_meta = []
    for r_id, r_frozen in frozen["routes"].items():
        by_cell: Dict[Tuple[float, float], Dict[str, Any]] = {}
        for p in r_frozen["plans"]:
            if p["policy"] in ("RA", "AB"):
                by_cell.setdefault((p["budget"], p["delta"]), {})[p["policy"]] = p
        for (B, delta), cell in by_cell.items():
            if "RA" not in cell or "AB" not in cell:
                continue
            s_ra, s_ab = cell["RA"]["selected"], cell["AB"]["selected"]
            plan_meta.append({
                "route_id": r_id,
                "budget": B,
                "delta": delta,
                "num_selected": len(s_ra),
                "selection_overlap": len(set(s_ra) & set(s_ab)) / len(s_ra) if s_ra else 1.0,
                "same_selection_order": s_ra == s_ab,
                "identical_final_route": cell["RA"]["final_route"] == cell["AB"]["final_route"],
                "accepted_moves_ra": len(cell["RA"]["accepted"]),
                "accepted_moves_ab": len(cell["AB"]["accepted"])
            })
    meta_df = pd.DataFrame(plan_meta)

    cell_rows = []
    for (r_id, cv, B, delta), g in pair.groupby(["route_id", "cv", "budget", "delta"], sort=False):
        cell_rows.append({
            "route_id": r_id, "cv": cv, "budget": B, "delta": delta,
            "n_scenarios": int(len(g)),
            "planning_v_risk": float(g["planning_v_risk"].iloc[0]),
            "mean_v_risk": float(g["v_risk"].mean()),
            "median_v_risk": float(g["v_risk"].median()),
            "p05_v_risk": float(np.percentile(g["v_risk"], 5)),
            "p95_v_risk": float(np.percentile(g["v_risk"], 95)),
            "share_ra_better": _share(g["v_risk"], "pos"),
            "share_ab_better": _share(g["v_risk"], "neg"),
            "share_tie": _share(g["v_risk"], "tie"),
            "mean_v_risk_nl": float(g["v_risk_nl"].mean())
        })
    cell_df = pd.DataFrame(cell_rows).merge(meta_df, on=["route_id", "budget", "delta"], how="left")
    return pair, cell_df

def route_level_tests(scen_df: pd.DataFrame, cv_levels: List[float]) -> Dict[str, Any]:
    """
    Route-clustered paired tests on scenario-averaged Delta TT (one value per route x B x delta x policy),
    so that scenario replications never inflate the number of independent observations.
    """
    out = {}
    for cv in cv_levels:
        cell_means = (scen_df[scen_df["cv"] == cv]
                      .groupby(["route_id", "budget", "delta", "policy"])["delta_tt"].mean()
                      .reset_index().rename(columns={"delta_tt": "scenario_mean_delta_tt"}))
        out[f"{cv:.2f}"] = compute_route_clustered_statistics(cell_means, target_metric="scenario_mean_delta_tt", base_policy="RA")
    return out

# ----------------------------------------------------------------------
# Pilot report
# ----------------------------------------------------------------------
def _fmt(x: float, nd: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+,.{nd}f}" if signed else f"{x:,.{nd}f}"

def _pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"

def write_pilot_report(summary: Dict[str, Any], overall: pd.DataFrame, ra_ab_cv: pd.DataFrame,
                       ra_ab_route: pd.DataFrame, gen_diag: pd.DataFrame, report_path: str) -> None:
    meta = summary["metadata"]
    checks = summary["checks"]
    lines = [
        "# M8 Pilot Report: Independent Travel-Time Scenario Evaluation of PISR",
        "",
        f"*Generated by `code/experiments/run_travel_time_scenarios.py` on {meta['run_end_time'][:10]} "
        f"(commit `{meta['commit_sha'][:7]}{'-dirty' if meta['commit_sha'].endswith('-dirty') else ''}`). "
        "All numbers below are computed by the script from this run's `scenario_*` output files; do not edit them by hand.*",
        "",
        "## 1. Design",
        "",
        "- **Planning (ex-ante, frozen):** official historical mean travel-time matrix; predicted risk $p_i$ (logistic + Platt), "
        "actionability $g_i$, policy ranking, budget selection and SFR exactly as in the routing benchmark "
        f"(H = {meta['threshold_hours']:.1f} h, NN baseline, B in {meta['budgets']}, delta in {meta['deltas']}, "
        f"Random seeds {meta['random_seeds'][0]}-{meta['random_seeds'][-1]}).",
        f"- **Freeze:** {meta['num_frozen_plans']} frozen R' plus one R0 per route on {meta['num_routes']} held-out routes "
        "(`scenario_frozen_routes.json`). The evaluation stage only reads these sequences; it never recomputes $p_i$/$g_i$, "
        "re-selects customers or re-runs SFR, even when a scenario makes R' worse than R0.",
        "- **Scenarios:** $t_{uv}^{(s)} = \\bar t_{uv}\\,\\exp(\\sigma Z_{uv}^{(s)} - \\sigma^2/2)$, $Z_{uv}^{(s)} \\sim N(0,1)$ i.i.d. per directed edge, "
        "$\\sigma = \\sqrt{\\ln(1 + CV^2)}$, so $E[t_{uv}^{(s)}] = \\bar t_{uv}$. "
        f"CV in {meta['cv_levels']}, {meta['n_scenarios']} scenarios per CV, base seed {meta['scenario_seed']}.",
        "- **Seeding:** $Z^{(s)}$ is drawn from `SeedSequence([base_seed, sha256(route_id), s])`. The same route and scenario ID "
        "therefore give the same matrix for every policy, scenario $s$ is identical whether 20 or 100 scenarios are run, "
        "and the same $Z^{(s)}$ is reused across CV levels (common random numbers).",
        "- **Held fixed:** departure time, planned service durations, service thresholds (H = 4 h), route distances (Haversine), B and delta.",
        "- **Missing edges:** never filled with Haversine; the evaluator raises an error if a frozen route uses a missing edge.",
        "- **Random policy:** outcomes are averaged over the 10 seeded selections within each scenario (benchmark convention).",
        "",
        "These are synthetic sensitivity scenarios, not observed Amazon travel times.",
        "",
        "## 2. Verification checks (feedback 4.3)",
        ""
    ]

    repro = checks["deterministic_benchmark_reproduced"]
    cv0 = checks["cv0_equivalence"]
    shared = checks["shared_matrix_and_seed_reproducibility"]
    check_rows = [
        ["1a. Deterministic results reproduced",
         "Planning Delta TT / Delta NL / moves equal `routing_benchmark_runs.csv` and `random_seed_runs.csv`",
         (f"{repro.get('frozen_plans_compared')}/{repro.get('frozen_plans_total')} plans; max abs dTT diff = "
          f"{repro.get('max_abs_delta_tt_diff_min', float('nan')):.2e} min; max abs dNL diff = {repro.get('max_abs_delta_nl_diff')}; "
          f"moves mismatches = {repro.get('accepted_moves_mismatches')}") if repro.get("available") else "benchmark CSVs not found",
         "PASS" if repro.get("passed") else "FAIL"],
        ["1b. CV = 0 equals deterministic evaluation",
         f"abs TT gap <= {CV0_TT_TOLERANCE_MIN:g} min, NL identical, one CV=0 matrix per route",
         f"max abs TT gap = {cv0.get('max_abs_tt_diff_min', float('nan')):.2e} min over {cv0.get('rows_compared')} rows; "
         f"NL mismatches = {cv0.get('nl_mismatches')}",
         "PASS" if cv0.get("passed") else "FAIL"],
        ["2. Frozen R0 / R' unchanged", "Sequence digests identical before and after evaluation",
         ", ".join(f"{r[8:16]}: {'unchanged' if ok else 'CHANGED'}" for r, ok in checks["frozen_routes_unchanged"]["routes"].items()),
         "PASS" if checks["frozen_routes_unchanged"]["passed"] else "FAIL"],
        ["3. Same matrix for all policies, reproducible seed",
         "One matrix per route x scenario x CV; regenerated fingerprints identical",
         f"{shared['regenerated_matrices_checked']} matrices regenerated, {shared['regenerated_matrix_mismatches']} mismatches",
         "PASS" if shared["passed"] else "FAIL"]
    ]
    lines.append(format_markdown_table(["Check", "Criterion", "Result", "Status"], check_rows))
    lines += [
        "",
        "## 3. Scenario generator diagnostics",
        "",
        "Pooled over all positive-mean directed edges, routes and scenarios. Baseline TT is TT(R0) averaged over routes and scenarios.",
        ""
    ]
    gen_rows = [[f"{r['cv']:.2f}", f"{r['sigma']:.4f}", f"{r['multiplier_mean']:.4f}", f"{r['multiplier_cv']:.4f}",
                 f"{r['multiplier_min']:.3f}", f"{r['multiplier_max']:.3f}", _fmt(r["baseline_tt"], 1), _fmt(r["baseline_nl"], 1)]
                for _, r in gen_diag.iterrows()]
    lines.append(format_markdown_table(
        ["CV", "sigma", "Mean multiplier", "Realized CV", "Min", "Max", "Baseline TT (min)", "Baseline NL"], gen_rows))
    zero_info = summary["zero_mean_edges"]
    lines += [
        "",
        f"Zero-mean off-diagonal edges (co-located stops) stay at zero under every scenario: {zero_info['total']} on the evaluated routes, "
        f"{zero_info['on_frozen_routes']} of them traversed by a frozen R0 or R'. Missing edges on evaluated routes: {zero_info['missing']}.",
        "",
        "## 4. Intervention effect and stability (feedback 4.1)",
        "",
        f"Pooled over {meta['num_routes']} routes x {len(meta['budgets']) * len(meta['deltas'])} (B, delta) cells x {meta['n_scenarios']} scenarios. "
        "Delta TT = TT(R0) - TT(R'), Delta NL = NL(R0) - NL(R') on the same scenario (minutes / stops). "
        "*Planning* is the deterministic mean-matrix result; *retention* = mean scenario Delta TT / planning Delta TT.",
        ""
    ]
    pol_order = [p for p in DEFAULT_POLICIES if p in overall["policy"].unique()]
    for cv in meta["cv_levels"]:
        sub = overall[overall["cv"] == cv].set_index("policy").loc[pol_order].reset_index()
        rows = [[r["policy"], _fmt(r["planning_delta_tt"], 1, True), _fmt(r["mean_delta_tt"], 1, True), _fmt(r["median_delta_tt"], 1, True),
                 f"[{_fmt(r['p05_delta_tt'], 1, True)}, {_fmt(r['p95_delta_tt'], 1, True)}]", _fmt(r["mean_delta_nl"], 2, True),
                 _fmt(r["median_delta_nl"], 1, True), _pct(r["share_delta_tt_pos"]), _pct(r["share_delta_tt_neg"]),
                 _pct(r["share_delta_nl_neg"]), _fmt(r["retention_ratio"], 3)]
                for _, r in sub.iterrows()]
        lines += [f"**CV = {cv:.2f}**", ""]
        lines.append(format_markdown_table(
            ["Policy", "Planning dTT", "Mean dTT", "Median dTT", "5-95% dTT", "Mean dNL", "Median dNL",
             "dTT > 0", "dTT < 0", "dNL < 0", "Retention"], rows))
        lines.append("")

    lines += [
        "## 5. RA vs AB: incremental value of predicted risk (feedback 4.2)",
        "",
        "$V_{risk}^{(s)} = \\Delta TT^{(s)}(RA) - \\Delta TT^{(s)}(AB)$ on identical scenarios; positive means RA reduced more tardiness. "
        "Ties are |V_risk| <= 1e-6 min. Selection overlap and accepted moves come from the frozen plans (identical for every CV).",
        ""
    ]
    rows = [[f"{r['cv']:.2f}", _fmt(r["planning_v_risk"], 2, True), _fmt(r["mean_v_risk"], 2, True), _fmt(r["median_v_risk"], 2, True),
             f"[{_fmt(r['p05_v_risk'], 1, True)}, {_fmt(r['p95_v_risk'], 1, True)}]", _pct(r["share_ra_better"]),
             _pct(r["share_ab_better"]), _pct(r["share_tie"]), _fmt(r["mean_v_risk_nl"], 3, True)]
            for _, r in ra_ab_cv.iterrows()]
    lines.append(format_markdown_table(
        ["CV", "Planning V_risk", "Mean V_risk", "Median V_risk", "5-95% V_risk", "RA better", "AB better", "Tie", "Mean dNL(RA) - dNL(AB)"], rows))

    sel = summary["ra_ab_selection"]
    lines += [
        "",
        f"Frozen-plan comparison over {sel['cells']} route x (B, delta) cells: mean selection overlap {_pct(sel['mean_selection_overlap'])}, "
        f"identical selection order in {sel['same_selection_order']} cells, identical final routes R' in {sel['identical_final_route']} cells, "
        f"mean accepted moves RA {sel['mean_accepted_ra']:.2f} vs AB {sel['mean_accepted_ab']:.2f}. "
        f"Predicted risk on the evaluated stops has median {sel['p_median']:.3f} and interquartile range "
        f"[{sel['p_q25']:.3f}, {sel['p_q75']:.3f}]"
        + (", so for most customers $p_i g_i$ is close to a constant multiple of $g_i$ and RA ranks customers almost like AB."
           if sel["p_q75"] - sel["p_q25"] < 0.05 else "."),
        "",
        "Per-route mean V_risk (averaged over cells and scenarios):",
        ""
    ]
    route_ids = list(dict.fromkeys(ra_ab_route["route_id"]))
    rows = []
    for r_id in route_ids:
        rr = ra_ab_route[ra_ab_route["route_id"] == r_id].set_index("cv")
        rows.append([r_id[8:16]] + [_fmt(rr.loc[cv, "mean_v_risk"], 2, True) for cv in meta["cv_levels"]])
    lines.append(format_markdown_table(["Route"] + [f"CV = {cv:.2f}" for cv in meta["cv_levels"]], rows))

    lines += [
        "",
        "## 6. Route-level paired tests (exploratory)",
        "",
        f"Scenario replications are not independent routes. Each test uses one scenario-averaged Delta TT difference per route "
        f"(N = {meta['num_routes']} routes, averaged over B and delta), a one-sample t-test and Holm-Bonferroni correction.",
        ""
    ]
    rows = []
    for cv_key, comps in summary["route_level_tests"].items():
        for comp, s in comps.items():
            rows.append([cv_key, f"RA vs {comp}", _fmt(s["mean_diff"], 2, True), f"{s['t_stat']:.3f}",
                         f"{s['p_unadjusted']:.4f}", f"{s.get('p_holm_bonferroni', float('nan')):.4f}"])
    lines.append(format_markdown_table(["CV", "Comparison", "Mean diff (min)", "t", "p (unadj.)", "Holm p"], rows))

    lines += [
        "",
        "## 7. Limitations",
        "",
        "- Scenarios are synthetic: independent edge-level lognormal noise with assumed CV levels, not estimated from Amazon data. "
        "Correlated (e.g., time-of-day or area-level) scenarios are a planned robustness check when the cohort is expanded.",
        f"- Only {meta['num_routes']} held-out routes; route-level tests have very low power regardless of the number of scenarios.",
        "- Route distances are Haversine proxies and do not change across scenarios.",
        "- Clarke-Wright baselines and the threshold-sensitivity grid are not re-evaluated in this pilot.",
        ""
    ]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Wrote pilot report to '{report_path}'.")

# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_travel_time_scenarios(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    report_path: Optional[str] = None,
    stage: str = "all",
    n_scenarios: int = 100,
    cv_levels: Optional[List[float]] = None,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
    route_ids: Optional[List[str]] = None,
    budgets: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    policies: Optional[List[str]] = None,
    random_seeds: Optional[List[int]] = None
) -> Dict[str, Any]:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    if report_path is None:
        same_dir = os.path.abspath(output_dir) == os.path.abspath(DEFAULT_OUTPUT_DIR)
        report_path = DEFAULT_REPORT_PATH if same_dir else os.path.join(output_dir, "travel_time_scenario_pilot.md")
    cv_levels = cv_levels if cv_levels is not None else DEFAULT_CV_LEVELS
    if 0.0 not in cv_levels:
        cv_levels = [0.0] + list(cv_levels)
    os.makedirs(output_dir, exist_ok=True)
    frozen_path = os.path.join(output_dir, FROZEN_FILENAME)

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    if stage in ("plan", "all"):
        plan_and_freeze(raw_dir, frozen_path, route_ids=route_ids, budgets=budgets, deltas=deltas,
                        policies=policies, random_seeds=random_seeds)
        if stage == "plan":
            return {"frozen_path": frozen_path}

    if not os.path.exists(frozen_path):
        raise FileNotFoundError(f"Frozen plans not found at '{frozen_path}'. Run with --stage plan first.")
    frozen_sha_before = compute_file_sha256(frozen_path)
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    plan_df, fp_df, audit = evaluate_frozen_routes(raw_dir, frozen, n_scenarios, cv_levels, scenario_seed)
    checks = run_verification_checks(plan_df, fp_df, audit)
    checks["frozen_file_unchanged"] = {"sha256": frozen_sha_before,
                                       "passed": compute_file_sha256(frozen_path) == frozen_sha_before}

    scen_df = collapse_random_seeds(plan_df)
    cell_summary = summarize_outcomes(scen_df, ["route_id", "cv", "policy", "budget", "delta"])
    overall = summarize_outcomes(scen_df, ["cv", "policy"])
    pair, ra_ab_cells = compare_ra_ab(scen_df, frozen)

    ra_ab_cv_rows, ra_ab_route_rows = [], []
    for cv, g in pair.groupby("cv", sort=False):
        ra_ab_cv_rows.append({
            "cv": cv,
            "planning_v_risk": float(g["planning_v_risk"].mean()),
            "mean_v_risk": float(g["v_risk"].mean()),
            "median_v_risk": float(g["v_risk"].median()),
            "p05_v_risk": float(np.percentile(g["v_risk"], 5)),
            "p95_v_risk": float(np.percentile(g["v_risk"], 95)),
            "share_ra_better": _share(g["v_risk"], "pos"),
            "share_ab_better": _share(g["v_risk"], "neg"),
            "share_tie": _share(g["v_risk"], "tie"),
            "mean_v_risk_nl": float(g["v_risk_nl"].mean())
        })
    for (r_id, cv), g in pair.groupby(["route_id", "cv"], sort=False):
        ra_ab_route_rows.append({"route_id": r_id, "cv": cv, "mean_v_risk": float(g["v_risk"].mean()),
                                 "share_ra_better": _share(g["v_risk"], "pos"), "share_ab_better": _share(g["v_risk"], "neg")})
    ra_ab_cv = pd.DataFrame(ra_ab_cv_rows)
    ra_ab_route = pd.DataFrame(ra_ab_route_rows)

    gen_rows = []
    for cv, g in fp_df.groupby("cv", sort=False):
        gen_rows.append({"cv": cv, "sigma": lognormal_sigma(cv), "multiplier_mean": float(g["multiplier_mean"].mean()),
                         "multiplier_cv": float(g["multiplier_cv"].mean()), "multiplier_min": float(g["multiplier_min"].min()),
                         "multiplier_max": float(g["multiplier_max"].max()), "baseline_tt": float(g["baseline_tt"].mean()),
                         "baseline_nl": float(g["baseline_nl"].mean())})
    gen_diag = pd.DataFrame(gen_rows)

    tests = route_level_tests(scen_df, cv_levels)

    risk_values = np.array([v for r in frozen["routes"].values() for v in r["predicted_risk"].values()])
    sel_cells = ra_ab_cells.drop_duplicates(["route_id", "budget", "delta"])
    selection = {
        "cells": int(len(sel_cells)),
        "mean_selection_overlap": float(sel_cells["selection_overlap"].mean()),
        "same_selection_order": int(sel_cells["same_selection_order"].sum()),
        "identical_final_route": int(sel_cells["identical_final_route"].sum()),
        "mean_accepted_ra": float(sel_cells["accepted_moves_ra"].mean()),
        "mean_accepted_ab": float(sel_cells["accepted_moves_ab"].mean()),
        "p_median": float(np.median(risk_values)),
        "p_q25": float(np.percentile(risk_values, 25)),
        "p_q75": float(np.percentile(risk_values, 75))
    }

    # Outputs
    scen_out = scen_df[["route_id", "cv", "scenario_id", "policy", "budget", "delta", "base_tt", "base_nl",
                        "final_tt", "final_nl", "delta_tt", "delta_nl"]].round(4)
    # Scenario-level detail is large (routes x CV x scenarios x cells x policies), so it is gzip-compressed
    scen_out.to_csv(os.path.join(output_dir, "scenario_level_runs.csv.gz"), index=False, compression="gzip")
    cell_summary.round(4).to_csv(os.path.join(output_dir, "scenario_policy_summary.csv"), index=False)
    ra_ab_cells.round(4).to_csv(os.path.join(output_dir, "scenario_ra_vs_ab.csv"), index=False)
    fp_df.round(6).to_csv(os.path.join(output_dir, "scenario_matrix_fingerprints.csv"), index=False)

    end_iso = datetime.now(timezone.utc).isoformat()
    summary = {
        "metadata": {
            "run_start_time": start_iso,
            "run_end_time": end_iso,
            "total_runtime_seconds": round(time.perf_counter() - t0, 2),
            "commit_sha": _git_commit_sha(),
            "environment_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "frozen_plans_file": FROZEN_FILENAME,
            "frozen_plans_sha256": frozen_sha_before,
            "frozen_plans_commit_sha": frozen["metadata"]["commit_sha"],
            "threshold_hours": frozen["metadata"]["threshold_hours"],
            "budgets": frozen["metadata"]["budgets"],
            "deltas": frozen["metadata"]["deltas"],
            "policies": frozen["metadata"]["policies"],
            "random_seeds": frozen["metadata"]["random_seeds"],
            "num_routes": len(frozen["routes"]),
            "num_frozen_plans": int(sum(len(r["plans"]) for r in frozen["routes"].values())),
            "cv_levels": cv_levels,
            "n_scenarios": n_scenarios,
            "scenario_seed": scenario_seed,
            "scenario_model": "mean-preserving lognormal, independent per directed edge, common random numbers across CV"
        },
        "checks": checks,
        "all_checks_passed": bool(all(c.get("passed", False) for c in checks.values())),
        "zero_mean_edges": {
            "total": int(sum(a["zero_mean_offdiag_edges"] for a in audit["routes"].values())),
            "on_frozen_routes": int(sum(a["zero_mean_edges_on_frozen_routes"] for a in audit["routes"].values())),
            "missing": int(sum(a["missing_edges"] for a in audit["routes"].values()))
        },
        "generator_diagnostics": gen_diag.to_dict(orient="records"),
        "policy_outcomes_by_cv": overall.to_dict(orient="records"),
        "ra_vs_ab_by_cv": ra_ab_cv.to_dict(orient="records"),
        "ra_vs_ab_by_route": ra_ab_route.to_dict(orient="records"),
        "ra_ab_selection": selection,
        "route_level_tests": tests
    }
    with open(os.path.join(output_dir, "scenario_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)
    for name, c in checks.items():
        print(f"- {name}: {'PASS' if c.get('passed') else 'FAIL'}")
    print("\n" + "=" * 80)
    print("RA vs AB (V_risk = Delta TT(RA) - Delta TT(AB))")
    print("=" * 80)
    print(format_markdown_table(
        ["CV", "Mean V_risk", "RA better", "AB better", "Tie"],
        [[f"{r['cv']:.2f}", _fmt(r["mean_v_risk"], 2, True), _pct(r["share_ra_better"]), _pct(r["share_ab_better"]), _pct(r["share_tie"])]
         for _, r in ra_ab_cv.iterrows()]))

    write_pilot_report(summary, overall, ra_ab_cv, ra_ab_route, gen_diag, report_path)
    return summary

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M8 pilot: travel-time scenario evaluation of frozen PISR decisions.")
    parser.add_argument("--stage", choices=["plan", "evaluate", "all"], default="all")
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--cv-levels", type=str, default="0,0.1,0.2,0.3")
    parser.add_argument("--scenario-seed", type=int, default=DEFAULT_SCENARIO_SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_travel_time_scenarios(
        output_dir=args.output_dir,
        report_path=args.report_path,
        stage=args.stage,
        n_scenarios=args.n_scenarios,
        cv_levels=[float(x) for x in args.cv_levels.split(",")],
        scenario_seed=args.scenario_seed
    )
