"""
Full-cohort evaluation of PISR on the complete ALMRRC 2021 training set (all routes, not the 13-route pilot).

The pipeline reuses the pilot's code paths route by route, so every number is produced by the same functions:
  features  - (parallel, all routes)  build each route from its shard for H in {explicit, 3, 4, 5, 6} hours and
              extract ex-ante features and historical-sequence lateness labels.
  train     - chronological date split (60/20/20) over all routes; logistic + Platt (primary), random forest and
              prevalence baseline per H; held-out test routes = routes on test dates.
  evaluate  - (parallel, test routes, resumable) inject p_i, run the factorial benchmark grid (evaluate_route_grid),
              Clarke-Wright robustness, threshold sensitivity, and the M8 travel-time scenario evaluation of the
              frozen plans; one checkpoint file per route.
  report    - aggregate checkpoints into CSV / JSON outputs and code/reports/full_cohort_report.md.

  python experiments/run_full_cohort.py --stage all --workers 14
"""
import os
import sys
import json
import gzip
import time
import pickle
import argparse
import platform
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.cohort import read_shard, build_instance_from_shard
from src.features.build_features import extract_ex_ante_features_for_route, compute_ground_truth_labels
from src.prediction.train import train_and_evaluate_pipeline, inject_calibrated_risks_into_instances, chronological_split
from src.prediction.evaluate import evaluate_predictions
from src.routing.baseline import build_nearest_neighbor_baseline
from src.routing.schedule import propagate_schedule
from src.evaluation.scenarios import (
    build_route_time_data, draw_standard_normals, scenario_travel_times, matrix_fingerprint,
    encode_sequences, evaluate_sequences
)
from src.evaluation.statistics import compute_paired_statistics, compute_route_clustered_statistics
from experiments.run_rq1_prediction import FEATURE_COLS
from experiments.run_routing_benchmark import (
    evaluate_route_grid, evaluate_clarke_wright_route, evaluate_sensitivity_route
)
from experiments.run_travel_time_scenarios import (
    DEFAULT_BUDGETS, DEFAULT_DELTAS, DEFAULT_POLICIES, DEFAULT_RANDOM_SEEDS, DEFAULT_CV_LEVELS,
    DEFAULT_SCENARIO_SEED, EPS_MIN, CV0_TT_TOLERANCE_MIN, _sequence_digest, _git_commit_sha
)

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_COHORT_DIR = os.path.join(CODE_DIR, "data", "full_cohort")
DEFAULT_OUTPUT_DIR = os.path.join(CODE_DIR, "outputs", "full_cohort")
DEFAULT_REPORT_PATH = os.path.join(CODE_DIR, "reports", "full_cohort_report.md")

PRIMARY_H = 4.0
SENSITIVITY_H = [3.0, 4.0, 5.0, 6.0]
EXPLICIT_TAG = "explicit"
COMPARISONS = ["AB", "RB", "Slack", "Deadline", "Random"]

def _h_tag(h: Optional[float]) -> str:
    return EXPLICIT_TAG if h is None else f"H{h:.1f}"

def _paths(output_dir: str) -> Dict[str, str]:
    return {
        "features": os.path.join(output_dir, "features"),
        "routes": os.path.join(output_dir, "routes"),
        "models": os.path.join(output_dir, "models.pkl"),
        "split": os.path.join(output_dir, "cohort_split.json"),
        "exclusions": os.path.join(output_dir, "excluded_routes.csv"),
        "explicit_routing": os.path.join(output_dir, "explicit_deadline_routing.csv"),
    }

# ----------------------------------------------------------------------
# Stage 1: features and labels for every route
# ----------------------------------------------------------------------
_WORKER: Dict[str, Any] = {}

def _init_feature_worker(shard_dir: str) -> None:
    _WORKER["shard_dir"] = shard_dir

def _route_features(route_id: str) -> Dict[str, Any]:
    try:
        shard = read_shard(_WORKER["shard_dir"], route_id)
        frames = {}
        explicit_routing = None
        for h in [None] + SENSITIVITY_H:
            inst = build_instance_from_shard(shard, derived_threshold_hours=h)
            f_df = extract_ex_ante_features_for_route(inst)
            lbls = compute_ground_truth_labels(inst)
            f_df["label"] = f_df["customer_id"].map(lbls)
            frames[_h_tag(h)] = f_df
            if h is None:
                # Routing audit under explicit Amazon time windows only (no derived threshold)
                base_route = build_nearest_neighbor_baseline(inst)
                sched = propagate_schedule(inst, base_route)
                explicit_routing = {
                    "route_id": route_id,
                    "route_date": inst.route_date,
                    "explicit_stops": sum(1 for s in inst.stops.values() if s.deadline_source == "explicit"),
                    "baseline_tt": sched.total_tardiness,
                    "baseline_nl": sched.total_lateness
                }
        return {"route_id": route_id, "ok": True, "frames": frames, "explicit_routing": explicit_routing}
    except Exception as e:
        return {"route_id": route_id, "ok": False, "error": f"{type(e).__name__}: {e}"}

def stage_features(cohort_dir: str, output_dir: str, workers: int, limit: Optional[int] = None) -> None:
    paths = _paths(output_dir)
    os.makedirs(paths["features"], exist_ok=True)
    manifest = pd.read_csv(os.path.join(cohort_dir, "manifest.csv"))
    exclusions = []
    for _, row in manifest.iterrows():
        if not row["has_shard"]:
            exclusions.append({"route_id": row["route_id"], "reason": "no travel-time record"})
        elif not row["has_actual_sequence"]:
            exclusions.append({"route_id": row["route_id"], "reason": "no actual sequence"})
        elif not row["has_packages"]:
            exclusions.append({"route_id": row["route_id"], "reason": "no package record"})
    excluded = {e["route_id"] for e in exclusions}
    route_ids = [r for r in manifest["route_id"] if r not in excluded]
    if limit is not None:
        route_ids = route_ids[:limit]

    print(f"[Features] Extracting features for {len(route_ids)} routes with {workers} workers...", flush=True)
    t0 = time.perf_counter()
    collected: Dict[str, List[pd.DataFrame]] = {}
    explicit_rows = []
    shard_dir = os.path.join(cohort_dir, "shards")
    with Pool(workers, initializer=_init_feature_worker, initargs=(shard_dir,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_route_features, route_ids, chunksize=8), 1):
            if res["ok"]:
                for tag, df in res["frames"].items():
                    collected.setdefault(tag, []).append(df)
                explicit_rows.append(res["explicit_routing"])
            else:
                exclusions.append({"route_id": res["route_id"], "reason": res["error"]})
            if k % 250 == 0 or k == len(route_ids):
                print(f"[Features] {k}/{len(route_ids)} routes ({time.perf_counter() - t0:.0f}s)", flush=True)

    for tag, frames in collected.items():
        df = pd.concat(frames, ignore_index=True).sort_values(["route_date", "route_id", "customer_id"]).reset_index(drop=True)
        df.to_parquet(os.path.join(paths["features"], f"features_{tag}.parquet"), index=False)
        print(f"[Features] {tag}: {len(df):,} stops from {df['route_id'].nunique():,} routes.")
    pd.DataFrame(explicit_rows).to_csv(paths["explicit_routing"], index=False)
    pd.DataFrame(exclusions, columns=["route_id", "reason"]).to_csv(paths["exclusions"], index=False)
    print(f"[Features] Done in {time.perf_counter() - t0:.0f}s; {len(exclusions)} routes excluded.", flush=True)

# ----------------------------------------------------------------------
# Stage 2: risk models and the chronological cohort split
# ----------------------------------------------------------------------
def _split_summary(df: pd.DataFrame) -> Dict[str, Any]:
    d_tr, d_va, d_te = chronological_split(df, date_col="route_date")
    out = {}
    for name, part in [("train", d_tr), ("validation", d_va), ("test", d_te)]:
        out[name] = {
            "dates": sorted(part["route_date"].unique().tolist()),
            "num_dates": int(part["route_date"].nunique()),
            "num_routes": int(part["route_id"].nunique()),
            "num_stops": int(len(part)),
            "positive_rate": round(float(part["label"].mean()), 4) if len(part) else None
        }
    return out

def stage_train(output_dir: str) -> None:
    paths = _paths(output_dir)
    feats = {tag: pd.read_parquet(os.path.join(paths["features"], f"features_{tag}.parquet"))
             for tag in [EXPLICIT_TAG] + [_h_tag(h) for h in SENSITIVITY_H]}

    # Explicit-deadline audit (official time windows only)
    exp_df = feats[EXPLICIT_TAG]
    exp_only = exp_df[exp_df["deadline_source"] == "explicit"]
    explicit_audit = {
        "total_stops": int(len(exp_df)),
        "explicit_deadline_stops": int(len(exp_only)),
        "explicit_pct": round(100.0 * len(exp_only) / max(1, len(exp_df)), 2),
        "explicit_positive_labels": int(exp_only["label"].sum()),
        "split": _split_summary(exp_only) if exp_only["route_date"].nunique() >= 3 else None
    }

    models = {}
    rq1 = {}
    for h in SENSITIVITY_H:
        tag = _h_tag(h)
        df = feats[tag]
        print(f"[Train] {tag}: {len(df):,} stops, prevalence {df['label'].mean():.4f}", flush=True)
        d_tr, d_va, d_te = chronological_split(df[df["deadline_source"] != "none"], date_col="route_date")
        train_prev = float(d_tr["label"].mean())
        prev = {
            "test": evaluate_predictions(d_te["label"].values, np.full(len(d_te), train_prev)),
            "validation": evaluate_predictions(d_va["label"].values, np.full(len(d_va), train_prev))
        }
        pipe_lr, m_lr, preds_lr = train_and_evaluate_pipeline(df, FEATURE_COLS, model_type="logistic")
        pipe_rf, m_rf, _ = train_and_evaluate_pipeline(df, FEATURE_COLS, model_type="rf")
        models[tag] = pipe_lr
        p = preds_lr["predicted_risk_pi"]
        rq1[tag] = {
            "H_hours": h,
            "split": _split_summary(df),
            "models": {"PrevalenceBaseline": prev, "LogisticRegression": m_lr, "RandomForest": m_rf},
            "test_risk_distribution": {
                "median": float(p.median()), "q25": float(p.quantile(0.25)), "q75": float(p.quantile(0.75)),
                "p05": float(p.quantile(0.05)), "p95": float(p.quantile(0.95)), "std": float(p.std())
            }
        }
        if abs(h - PRIMARY_H) < 1e-9:
            preds_lr[["route_id", "route_date", "customer_id", "label", "predicted_risk_pi"]].to_parquet(
                os.path.join(output_dir, "rq1_test_predictions_H4.parquet"), index=False)
        print(f"[Train] {tag}: LR test ROC-AUC {m_lr['roc_auc']:.3f}, Brier {m_lr['brier_score']:.4f}, ECE {m_lr['ece']:.4f}; "
              f"RF ROC-AUC {m_rf['roc_auc']:.3f}", flush=True)

    with open(paths["models"], "wb") as f:
        pickle.dump(models, f)

    primary = feats[_h_tag(PRIMARY_H)]
    _, _, d_te = chronological_split(primary, date_col="route_date")
    test_dates = sorted(d_te["route_date"].unique().tolist())
    test_routes = sorted(primary.loc[primary["route_date"].isin(test_dates), "route_id"].unique().tolist())
    split_payload = {
        "primary_threshold_hours": PRIMARY_H,
        "split": _split_summary(primary),
        "test_dates": test_dates,
        "test_route_ids": test_routes
    }
    with open(paths["split"], "w") as f:
        json.dump(split_payload, f, indent=1)
    with open(os.path.join(output_dir, "rq1_prediction_results.json"), "w") as f:
        json.dump({"feature_columns": FEATURE_COLS, "explicit_deadline_audit": explicit_audit,
                   "threshold_sensitivity": rq1}, f, indent=2)
    print(f"[Train] {len(test_routes)} held-out test routes on {len(test_dates)} dates.", flush=True)

# ----------------------------------------------------------------------
# Stage 3: per-route benchmark, robustness and scenario evaluation
# ----------------------------------------------------------------------
def _init_eval_worker(shard_dir: str, models_path: str, routes_dir: str, n_scenarios: int,
                      cv_levels: List[float], scenario_seed: int) -> None:
    with open(models_path, "rb") as f:
        _WORKER["models"] = pickle.load(f)
    _WORKER.update(shard_dir=shard_dir, routes_dir=routes_dir, n_scenarios=n_scenarios,
                   cv_levels=cv_levels, scenario_seed=scenario_seed)

def _route_checkpoint(routes_dir: str, route_id: str) -> str:
    return os.path.join(routes_dir, f"{route_id}.pkl.gz")

def _instance_with_risk(shard: Dict[str, Any], h: float):
    inst = build_instance_from_shard(shard, derived_threshold_hours=h)
    inject_calibrated_risks_into_instances({inst.route_id: inst}, _WORKER["models"][_h_tag(h)], FEATURE_COLS)
    return inst

def _scenario_evaluation(inst, grid: Dict[str, Any], n_scenarios: int, cv_levels: List[float],
                         base_seed: int) -> Dict[str, Any]:
    """M8 evaluation of the frozen R0 / R' of one route (same procedure as run_travel_time_scenarios)."""
    r_id = inst.route_id
    plans = grid["plans"]
    data = build_route_time_data(inst)
    sequences = [grid["base_route"]] + [p["final_route"] for p in plans]
    digest_before = _sequence_digest(sequences)
    encoded = encode_sequences(data, sequences)

    off_diag = ~np.eye(len(data.stop_ids), dtype=bool)
    positive = off_diag & (data.mean_travel_seconds > 0.0)
    zero_mean = off_diag & (data.mean_travel_seconds == 0.0)
    used = np.zeros_like(zero_mean)
    used[encoded[:, :-1], encoded[:, 1:]] = True

    n_cv, n_plans = len(cv_levels), len(plans)
    final_tt = np.empty((n_scenarios, n_cv, n_plans))
    final_nl = np.empty((n_scenarios, n_cv, n_plans))
    base_tt = np.empty((n_scenarios, n_cv))
    base_nl = np.empty((n_scenarios, n_cv))
    fingerprints = {}
    mult_stats = np.zeros((n_cv, 3))  # count, sum, sum of squares of edge multipliers
    mult_min = np.full(n_cv, np.inf)
    mult_max = np.full(n_cv, -np.inf)
    for s in range(n_scenarios):
        z = draw_standard_normals(r_id, s, len(data.stop_ids), base_seed)
        for c, cv in enumerate(cv_levels):
            matrix = scenario_travel_times(data.mean_travel_seconds, z, cv)
            tt, nl = evaluate_sequences(data, matrix, encoded)
            base_tt[s, c], base_nl[s, c] = tt[0], nl[0]
            final_tt[s, c], final_nl[s, c] = tt[1:], nl[1:]
            fingerprints[(s, c)] = matrix_fingerprint(matrix)
            mult = matrix[positive] / data.mean_travel_seconds[positive]
            mult_stats[c] += (mult.size, mult.sum(), np.square(mult).sum())
            mult_min[c] = min(mult_min[c], mult.min())
            mult_max[c] = max(mult_max[c], mult.max())

    regen_checked = regen_mismatch = 0
    for s in sorted({0, n_scenarios // 2, n_scenarios - 1}):
        z = draw_standard_normals(r_id, s, len(data.stop_ids), base_seed)
        for c, cv in enumerate(cv_levels):
            regen_checked += 1
            regen_mismatch += int(matrix_fingerprint(scenario_travel_times(data.mean_travel_seconds, z, cv)) != fingerprints[(s, c)])

    planning_tt = np.array([p["planning"]["final_tt"] for p in plans])
    planning_nl = np.array([p["planning"]["final_nl"] for p in plans])
    cv0 = [c for c, cv in enumerate(cv_levels) if cv == 0.0]
    cv0_gap, cv0_nl_mismatch = None, None
    if cv0:
        c = cv0[0]
        cv0_gap = float(max(np.abs(final_tt[:, c, :] - planning_tt).max(), np.abs(base_tt[:, c] - grid["base_tt"]).max()))
        cv0_nl_mismatch = int((final_nl[:, c, :] != planning_nl).sum() + (base_nl[:, c] != grid["base_nl"]).sum())
    digest_after = _sequence_digest([grid["base_route"]] + [p["final_route"] for p in plans])

    # Collapse the seeded Random selections per scenario (benchmark convention)
    cell_keys: List[Tuple[str, float, float]] = []
    cell_index: Dict[Tuple[str, float, float], List[int]] = {}
    for k, p in enumerate(plans):
        key = (p["policy"], p["budget"], p["delta"])
        if key not in cell_index:
            cell_index[key] = []
            cell_keys.append(key)
        cell_index[key].append(k)
    d_tt = base_tt[:, :, None] - final_tt
    d_nl = base_nl[:, :, None] - final_nl
    cells_tt = np.stack([d_tt[:, :, idx].mean(axis=2) for idx in cell_index.values()], axis=2)
    cells_nl = np.stack([d_nl[:, :, idx].mean(axis=2) for idx in cell_index.values()], axis=2)
    planning_cell_dtt = np.array([np.mean(grid["base_tt"] - planning_tt[idx]) for idx in cell_index.values()])

    ra_ab = []
    by_cell = {(p["budget"], p["delta"], p["policy"]): p for p in plans if p["policy"] in ("RA", "AB")}
    for (B, delta, pol), p_ra in by_cell.items():
        if pol != "RA" or (B, delta, "AB") not in by_cell:
            continue
        p_ab = by_cell[(B, delta, "AB")]
        s_ra, s_ab = p_ra["selected"], p_ab["selected"]
        ra_ab.append({
            "route_id": r_id, "budget": B, "delta": delta,
            "num_selected": len(s_ra),
            "selection_overlap": len(set(s_ra) & set(s_ab)) / len(s_ra) if s_ra else 1.0,
            "same_selection_order": s_ra == s_ab,
            "identical_final_route": p_ra["final_route"] == p_ab["final_route"],
            "accepted_moves_ra": len(p_ra["accepted"]),
            "accepted_moves_ab": len(p_ab["accepted"])
        })

    return {
        "cell_keys": cell_keys,
        "delta_tt": cells_tt,
        "delta_nl": cells_nl,
        "planning_cell_delta_tt": planning_cell_dtt,
        "base_tt": base_tt,
        "base_nl": base_nl,
        "ra_ab_plans": ra_ab,
        "audit": {
            "num_frozen_plans": n_plans,
            "sequence_digest": digest_before,
            "sequences_unchanged": digest_before == digest_after,
            "zero_mean_offdiag_edges": int(zero_mean.sum()),
            "zero_mean_edges_on_frozen_routes": int((zero_mean & used).sum()),
            "missing_edges": int(np.isnan(data.mean_travel_seconds).sum()),
            "distinct_matrices": len(set(fingerprints.values())),
            "regenerated_matrices_checked": regen_checked,
            "regenerated_matrix_mismatches": regen_mismatch,
            "cv0_max_abs_tt_gap": cv0_gap,
            "cv0_nl_mismatches": cv0_nl_mismatch,
            "multiplier_stats": mult_stats.tolist(),
            "multiplier_min": mult_min.tolist(),
            "multiplier_max": mult_max.tolist()
        }
    }

def _evaluate_route(route_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out_path = _route_checkpoint(_WORKER["routes_dir"], route_id)
    try:
        shard = read_shard(_WORKER["shard_dir"], route_id)
        inst = _instance_with_risk(shard, PRIMARY_H)
        t_grid = time.perf_counter()
        grid = evaluate_route_grid(inst, DEFAULT_BUDGETS, DEFAULT_DELTAS, DEFAULT_POLICIES, DEFAULT_RANDOM_SEEDS,
                                   require_promised_times=True, return_plans=True)
        t_cw = time.perf_counter()
        cw = evaluate_clarke_wright_route(inst, DEFAULT_BUDGETS)
        t_scen = time.perf_counter()
        scen = _scenario_evaluation(inst, grid, _WORKER["n_scenarios"], _WORKER["cv_levels"], _WORKER["scenario_seed"])
        t_sens = time.perf_counter()
        sens = []
        for h in SENSITIVITY_H:
            sens.extend(evaluate_sensitivity_route(inst if h == PRIMARY_H else _instance_with_risk(shard, h), h))
        t_end = time.perf_counter()

        risks = np.array([inst.stops[c].predicted_risk_pi for c in grid["base_route"][1:]])
        result = {
            "route_id": route_id,
            "route_date": inst.route_date,
            "station_code": inst.station_code,
            "num_customers": len(grid["base_route"]) - 1,
            "predicted_risk": risks,
            "runs": grid["runs"],
            "seed_runs": grid["seed_runs"],
            "plans": [{k: p[k] for k in ("policy", "budget", "delta", "random_seed", "selected", "accepted", "final_route")}
                      for p in grid["plans"]],
            "base_route": grid["base_route"],
            "clarke_wright": cw,
            "sensitivity": sens,
            "scenario": scen,
            "timing_seconds": {
                "load_and_inject": t_grid - t0, "grid": t_cw - t_grid, "clarke_wright": t_scen - t_cw,
                "scenarios": t_sens - t_scen, "sensitivity": t_end - t_sens, "total": t_end - t0
            }
        }
        tmp = out_path + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_path)
        return {"route_id": route_id, "ok": True, "seconds": t_end - t0}
    except Exception as e:
        return {"route_id": route_id, "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "seconds": time.perf_counter() - t0}

def stage_evaluate(cohort_dir: str, output_dir: str, workers: int, n_scenarios: int, cv_levels: List[float],
                   scenario_seed: int, limit: Optional[int] = None) -> None:
    paths = _paths(output_dir)
    os.makedirs(paths["routes"], exist_ok=True)
    with open(paths["split"]) as f:
        test_routes = json.load(f)["test_route_ids"]
    if limit is not None:
        test_routes = test_routes[:limit]
    todo = [r for r in test_routes if not os.path.exists(_route_checkpoint(paths["routes"], r))]
    print(f"[Evaluate] {len(test_routes)} test routes, {len(test_routes) - len(todo)} already done, "
          f"{len(todo)} to run with {workers} workers.", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    failures = []
    init_args = (os.path.join(cohort_dir, "shards"), paths["models"], paths["routes"], n_scenarios, cv_levels, scenario_seed)
    with Pool(workers, initializer=_init_eval_worker, initargs=init_args, maxtasksperchild=50) as pool:
        for k, res in enumerate(pool.imap_unordered(_evaluate_route, todo, chunksize=1), 1):
            if not res["ok"]:
                failures.append(res)
                print(f"[Evaluate] FAILED {res['route_id']}: {res['error']}", flush=True)
            if k % 25 == 0 or k == len(todo):
                elapsed = time.perf_counter() - t0
                eta = elapsed / k * (len(todo) - k)
                print(f"[Evaluate] {k}/{len(todo)} routes, elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(output_dir, "evaluate_failures.csv"), index=False)
        print(f"[Evaluate] {len(failures)} routes failed; see evaluate_failures.csv", flush=True)

# ----------------------------------------------------------------------
# Stage 4: aggregation and report
# ----------------------------------------------------------------------
def _share(x: np.ndarray, cond: str) -> float:
    if cond == "pos":
        return float((x > EPS_MIN).mean())
    if cond == "neg":
        return float((x < -EPS_MIN).mean())
    return float((np.abs(x) <= EPS_MIN).mean())

def _load_route_results(routes_dir: str, route_ids: List[str]) -> List[Dict[str, Any]]:
    results = []
    for r in route_ids:
        path = _route_checkpoint(routes_dir, r)
        if os.path.exists(path):
            with gzip.open(path, "rb") as f:
                res = pickle.load(f)
            res.pop("plans", None)
            res.pop("base_route", None)
            results.append(res)
    return results

def _benchmark_statistics(df_nn: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cell_stats = {}
    df_ra = df_nn[df_nn["policy"] == "RA"].sort_values(["route_id", "delta", "budget"])
    for comp in COMPARISONS:
        df_comp = df_nn[df_nn["policy"] == comp].sort_values(["route_id", "delta", "budget"])
        if len(df_ra) == len(df_comp):
            cell_stats[comp] = compute_paired_statistics(df_ra["delta_tt"].values, df_comp["delta_tt"].values)
    route_stats = compute_route_clustered_statistics(df_nn, target_metric="delta_tt", base_policy="RA")
    for s in route_stats.values():
        means = np.asarray(s.pop("route_means"))
        s["share_routes_ra_better"] = float((means > EPS_MIN).mean())
        s["share_routes_tied"] = float((np.abs(means) <= EPS_MIN).mean())
        nz = means[np.abs(means) > EPS_MIN]
        if len(nz) >= 2:
            s["wilcoxon_pvalue"] = float(scipy_stats.wilcoxon(nz).pvalue)
    return cell_stats, route_stats

def stage_report(cohort_dir: str, output_dir: str, report_path: str, cv_levels: List[float],
                 n_scenarios: int, scenario_seed: int) -> Dict[str, Any]:
    paths = _paths(output_dir)
    with open(paths["split"]) as f:
        split = json.load(f)
    with open(os.path.join(output_dir, "rq1_prediction_results.json")) as f:
        rq1 = json.load(f)
    manifest = pd.read_csv(os.path.join(cohort_dir, "manifest.csv"))
    exclusions = pd.read_csv(paths["exclusions"])
    results = _load_route_results(paths["routes"], split["test_route_ids"])
    evaluated = [r["route_id"] for r in results]
    print(f"[Report] Aggregating {len(results)} of {len(split['test_route_ids'])} test routes.", flush=True)

    df_nn = pd.DataFrame([rec for r in results for rec in r["runs"]])
    df_seed = pd.DataFrame([rec for r in results for rec in r["seed_runs"]])
    df_cw = pd.DataFrame([rec for r in results for rec in r["clarke_wright"]])
    df_sens = pd.DataFrame([rec for r in results for rec in r["sensitivity"]])
    df_nn.to_csv(os.path.join(output_dir, "routing_benchmark_runs.csv.gz"), index=False)
    df_seed.to_csv(os.path.join(output_dir, "random_seed_runs.csv.gz"), index=False)
    df_cw.to_csv(os.path.join(output_dir, "robustness_clark_wright_runs.csv.gz"), index=False)
    df_sens.to_csv(os.path.join(output_dir, "threshold_sensitivity_runs.csv.gz"), index=False)

    pols = [p for p in DEFAULT_POLICIES if p in df_nn["policy"].unique()]
    policy_summary = df_nn.groupby("policy").agg({
        "delta_tt": "mean", "tt_red_pct": "mean", "delta_nl": "mean", "dist_inc_pct": "mean",
        "accepted_moves": "mean", "is_feasible": "mean", "runtime_seconds": "mean",
        "actionability_runtime_seconds": "mean"
    }).loc[pols].reset_index()
    route_level = df_nn.groupby(["policy", "route_id"])["delta_tt"].mean().unstack("policy")[pols]
    cell_stats, route_stats = _benchmark_statistics(df_nn)

    cw_pols = [p for p in ["RA", "AB", "RB", "Slack", "Deadline"] if p in df_cw["policy"].unique()]
    cw_std = df_cw[(df_cw["budget"] == 0.20) & (df_cw["delta"] == 0.05)]
    cw_summary = {
        "all_budgets": df_cw.groupby("policy")[["delta_tt", "tt_red_pct", "delta_nl", "dist_inc_pct", "accepted_moves"]].mean().loc[cw_pols].reset_index().to_dict(orient="records"),
        "B0.20_delta0.05": cw_std.groupby("policy")[["delta_tt", "tt_red_pct", "delta_nl", "dist_inc_pct", "accepted_moves"]].mean().loc[cw_pols].reset_index().to_dict(orient="records"),
        "route_clustered": compute_route_clustered_statistics(df_cw, target_metric="delta_tt", base_policy="RA",
                                                              comparison_policies=["AB", "RB", "Slack", "Deadline"])
    }
    for s in cw_summary["route_clustered"].values():
        s.pop("route_means", None)
    sens_summary = df_sens.groupby(["threshold_H", "policy"])[["base_tt", "base_nl", "delta_tt", "delta_nl", "tt_red_pct", "dist_inc_pct", "accepted_moves"]].mean().reset_index()
    sens_stats = {}
    for h, g in df_sens.groupby("threshold_H"):
        st = compute_route_clustered_statistics(g, target_metric="delta_tt", base_policy="RA")
        for s in st.values():
            s.pop("route_means", None)
        sens_stats[f"{h:.1f}"] = st

    risks = np.concatenate([r["predicted_risk"] for r in results])
    timing = pd.DataFrame([r["timing_seconds"] for r in results])
    seed_means = df_seed.groupby("random_seed")["delta_tt"].mean()

    # ---- M8 travel-time scenarios ----
    keys = results[0]["scenario"]["cell_keys"]
    for r in results:
        if r["scenario"]["cell_keys"] != keys:
            raise ValueError(f"Inconsistent scenario cell order on route {r['route_id']}")
    key_index = {k: i for i, k in enumerate(keys)}
    D_TT = np.stack([r["scenario"]["delta_tt"] for r in results])   # (R, S, C, K)
    D_NL = np.stack([r["scenario"]["delta_nl"] for r in results])
    PLAN = np.stack([r["scenario"]["planning_cell_delta_tt"] for r in results])  # (R, K)
    BASE_TT = np.stack([r["scenario"]["base_tt"] for r in results])  # (R, S, C)
    audits = [r["scenario"]["audit"] for r in results]
    mstats = np.sum([a["multiplier_stats"] for a in audits], axis=0)

    scen_checks = {
        "cv0_equivalence": {
            "max_abs_tt_gap_min": max(a["cv0_max_abs_tt_gap"] for a in audits),
            "nl_mismatches": int(sum(a["cv0_nl_mismatches"] for a in audits)),
            "tolerance_min": CV0_TT_TOLERANCE_MIN
        },
        "frozen_routes_unchanged": all(a["sequences_unchanged"] for a in audits),
        "distinct_matrices_per_route": {"min": min(a["distinct_matrices"] for a in audits),
                                        "expected": n_scenarios * (len(cv_levels) - (1 if 0.0 in cv_levels else 0)) + (1 if 0.0 in cv_levels else 0)},
        "regenerated_matrices_checked": int(sum(a["regenerated_matrices_checked"] for a in audits)),
        "regenerated_matrix_mismatches": int(sum(a["regenerated_matrix_mismatches"] for a in audits)),
        "missing_edges_on_evaluated_routes": int(sum(a["missing_edges"] for a in audits)),
        "zero_mean_offdiag_edges": int(sum(a["zero_mean_offdiag_edges"] for a in audits)),
        "zero_mean_edges_on_frozen_routes": int(sum(a["zero_mean_edges_on_frozen_routes"] for a in audits))
    }
    scen_checks["cv0_equivalence"]["passed"] = bool(scen_checks["cv0_equivalence"]["max_abs_tt_gap_min"] <= CV0_TT_TOLERANCE_MIN
                                                    and scen_checks["cv0_equivalence"]["nl_mismatches"] == 0)
    scen_checks["shared_and_reproducible_matrices_passed"] = bool(
        scen_checks["regenerated_matrix_mismatches"] == 0
        and scen_checks["distinct_matrices_per_route"]["min"] == scen_checks["distinct_matrices_per_route"]["expected"])

    generator = []
    for c, cv in enumerate(cv_levels):
        n, s1, s2 = mstats[c]
        mean = s1 / n
        generator.append({
            "cv": cv, "mean_multiplier": mean, "realized_cv": float(np.sqrt(max(0.0, s2 / n - mean * mean)) / mean),
            "min": float(min(a["multiplier_min"][c] for a in audits)), "max": float(max(a["multiplier_max"][c] for a in audits)),
            "baseline_tt_mean": float(BASE_TT[:, :, c].mean())
        })

    pooled = []
    for c, cv in enumerate(cv_levels):
        for pol in pols:
            ks = [key_index[k] for k in keys if k[0] == pol]
            x = D_TT[:, :, c, ks].reshape(-1)
            y = D_NL[:, :, c, ks].reshape(-1)
            plan = PLAN[:, ks].mean()
            pooled.append({
                "cv": cv, "policy": pol, "n": int(x.size),
                "planning_delta_tt": float(plan), "mean_delta_tt": float(x.mean()), "median_delta_tt": float(np.median(x)),
                "p05_delta_tt": float(np.percentile(x, 5)), "p95_delta_tt": float(np.percentile(x, 95)),
                "mean_delta_nl": float(y.mean()), "median_delta_nl": float(np.median(y)),
                "share_delta_tt_pos": _share(x, "pos"), "share_delta_tt_neg": _share(x, "neg"), "share_delta_nl_neg": _share(y, "neg"),
                "retention_ratio": float(x.mean() / plan) if plan > EPS_MIN else float("nan"),
                "share_routes_mean_delta_tt_neg": float((D_TT[:, :, c, ks].mean(axis=(1, 2)) < -EPS_MIN).mean())
            })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(os.path.join(output_dir, "scenario_policy_summary.csv"), index=False)

    v_rows = []
    ra_idx = [key_index[k] for k in keys if k[0] == "RA"]
    ab_idx = [key_index[(("AB",) + k[1:])] for k in keys if k[0] == "RA"]
    V = D_TT[..., ra_idx] - D_TT[..., ab_idx]   # (R, S, C, 16)
    V_PLAN = PLAN[:, ra_idx] - PLAN[:, ab_idx]
    for c, cv in enumerate(cv_levels):
        v = V[:, :, c, :].reshape(-1)
        route_v = V[:, :, c, :].mean(axis=(1, 2))
        v_rows.append({
            "cv": cv, "planning_v_risk": float(V_PLAN.mean()), "mean_v_risk": float(v.mean()), "median_v_risk": float(np.median(v)),
            "p05_v_risk": float(np.percentile(v, 5)), "p95_v_risk": float(np.percentile(v, 95)),
            "share_ra_better": _share(v, "pos"), "share_ab_better": _share(v, "neg"), "share_tie": _share(v, "tie"),
            "routes_ra_better": float((route_v > EPS_MIN).mean()), "routes_ab_better": float((route_v < -EPS_MIN).mean())
        })
    v_df = pd.DataFrame(v_rows)
    v_df.to_csv(os.path.join(output_dir, "scenario_ra_vs_ab.csv"), index=False)
    ra_ab_meta = pd.DataFrame([m for r in results for m in r["scenario"]["ra_ab_plans"]])
    ra_ab_meta.to_csv(os.path.join(output_dir, "ra_ab_selection_overlap.csv.gz"), index=False)

    # Route-level tests on scenario-averaged Delta TT (scenarios are never treated as independent routes)
    scen_route_tests = {}
    for c, cv in enumerate(cv_levels):
        rows = []
        for i, r in enumerate(results):
            m = D_TT[i, :, c, :].mean(axis=0)
            for k, (pol, B, delta) in enumerate(keys):
                rows.append({"route_id": r["route_id"], "policy": pol, "budget": B, "delta": delta, "delta_tt": m[k]})
        st = compute_route_clustered_statistics(pd.DataFrame(rows), target_metric="delta_tt", base_policy="RA")
        for s in st.values():
            s.pop("route_means", None)
        scen_route_tests[f"{cv:.2f}"] = st

    # Per-route scenario summary (route x CV x policy, averaged over cells and scenarios)
    per_route = []
    for i, r in enumerate(results):
        for c, cv in enumerate(cv_levels):
            for pol in pols:
                ks = [key_index[k] for k in keys if k[0] == pol]
                x = D_TT[i, :, c, ks]
                per_route.append({"route_id": r["route_id"], "cv": cv, "policy": pol,
                                  "mean_delta_tt": float(x.mean()), "share_delta_tt_neg": _share(x.reshape(-1), "neg"),
                                  "mean_delta_nl": float(D_NL[i, :, c, ks].mean())})
    pd.DataFrame(per_route).to_csv(os.path.join(output_dir, "scenario_route_summary.csv.gz"), index=False)
    np.savez_compressed(os.path.join(output_dir, "scenario_delta_arrays.npz"),
                        route_ids=np.array(evaluated), cell_keys=np.array([f"{p}|{b}|{d}" for p, b, d in keys]),
                        cv_levels=np.array(cv_levels), delta_tt=D_TT.astype(np.float64), delta_nl=D_NL.astype(np.float32))

    summary = {
        "run_metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "commit_sha": _git_commit_sha(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dataset": "Amazon Last Mile Routing Research Challenge 2021, training set (model_build_inputs)",
            "manifest_routes": int(len(manifest)),
            "excluded_routes": int(len(exclusions)),
            "exclusion_reasons": exclusions["reason"].str.split(":").str[0].value_counts().to_dict(),
            "test_routes_planned": len(split["test_route_ids"]),
            "test_routes_evaluated": len(evaluated),
            "scenario_settings": {"n_scenarios": n_scenarios, "cv_levels": cv_levels, "base_seed": scenario_seed}
        },
        "cohort_split": split["split"],
        "rq1": rq1,
        "test_risk_distribution": {"n": int(risks.size), "median": float(np.median(risks)),
                                   "q25": float(np.percentile(risks, 25)), "q75": float(np.percentile(risks, 75)),
                                   "p05": float(np.percentile(risks, 5)), "p95": float(np.percentile(risks, 95))},
        "policy_summary": policy_summary.to_dict(orient="records"),
        "route_level_policy_means": {p: {"mean": float(route_level[p].mean()), "median": float(route_level[p].median())} for p in pols},
        "cell_paired_statistics": cell_stats,
        "route_clustered_statistics": route_stats,
        "clarke_wright": cw_summary,
        "threshold_sensitivity": {"summary": sens_summary.to_dict(orient="records"), "route_clustered": sens_stats},
        "random_seed_means": {"mean": float(seed_means.mean()), "std": float(seed_means.std()),
                              "min": float(seed_means.min()), "max": float(seed_means.max())},
        "invariant_audit": {
            "min_delta_tt": float(df_nn["delta_tt"].min()),
            "total_runs": int(len(df_nn)),
            "feasibility_rate": float(df_nn["is_feasible"].mean()),
            "runs_delta_tt_pos_delta_nl_neg": int(((df_nn["delta_tt"] > 0) & (df_nn["delta_nl"] < 0)).sum()),
            "zero_baseline_runs": int((df_nn["base_tt"] == 0.0).sum())
        },
        "runtime_seconds_per_route": {k: float(v) for k, v in timing.mean().items()},
        "actionability_runtime_by_delta": df_nn.groupby("delta")["actionability_runtime_seconds"].mean().to_dict(),
        "scenarios": {
            "checks": scen_checks,
            "generator": generator,
            "pooled": pooled,
            "v_risk": v_rows,
            "ra_ab_selection": {
                "mean_overlap": float(ra_ab_meta["selection_overlap"].mean()),
                "identical_order_cells": int(ra_ab_meta["same_selection_order"].sum()),
                "identical_final_route_cells": int(ra_ab_meta["identical_final_route"].sum()),
                "cells": int(len(ra_ab_meta)),
                "mean_accepted_ra": float(ra_ab_meta["accepted_moves_ra"].mean()),
                "mean_accepted_ab": float(ra_ab_meta["accepted_moves_ab"].mean())
            },
            "route_level_tests": scen_route_tests
        }
    }
    with open(os.path.join(output_dir, "full_cohort_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_full_cohort_report(summary, report_path)
    print(f"[Report] Wrote '{report_path}'.", flush=True)
    return summary

# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------
def _f(x: Any, nd: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+,.{nd}f}" if signed else f"{x:,.{nd}f}"

def _p(x: float) -> str:
    return f"{100.0 * x:.1f}%"

def _pv(x: float) -> str:
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"

def write_full_cohort_report(s: Dict[str, Any], report_path: str) -> None:
    meta, split, rq1 = s["run_metadata"], s["cohort_split"], s["rq1"]
    L = []
    L.append("# Full-Cohort Evaluation of PISR (ALMRRC 2021 Training Set)")
    L.append("")
    L.append(f"*Generated by `code/experiments/run_full_cohort.py` on {meta['created_utc'][:10]} (commit `{meta['commit_sha'][:7]}"
             f"{'-dirty' if meta['commit_sha'].endswith('-dirty') else ''}`). All numbers are computed from this run's "
             "`code/outputs/full_cohort/` files; do not edit them by hand.*")
    L.append("")
    L.append("## 1. Cohort")
    L.append("")
    L.append(f"- Source: {meta['dataset']}; {meta['manifest_routes']:,} routes in `route_data.json`.")
    L.append(f"- Excluded: {meta['excluded_routes']} routes ({', '.join(f'{k}: {v}' for k, v in meta['exclusion_reasons'].items()) or 'none'}).")
    L.append(f"- Held-out test routes evaluated: {meta['test_routes_evaluated']:,} of {meta['test_routes_planned']:,}.")
    L.append("")
    L.append("| Block | Dates | Routes | Drop-off stops | Late share (H = 4 h) |")
    L.append("| --- | --- | --- | --- | --- |")
    for name in ["train", "validation", "test"]:
        b = split[name]
        L.append(f"| {name.capitalize()} | {b['num_dates']} ({b['dates'][0]} to {b['dates'][-1]}) | {b['num_routes']:,} | {b['num_stops']:,} | {_p(b['positive_rate'])} |")
    L.append("")
    ea = rq1["explicit_deadline_audit"]
    L.append(f"Explicit Amazon time windows cover {ea['explicit_deadline_stops']:,} of {ea['total_stops']:,} drop-off stops "
             f"({ea['explicit_pct']:.2f}%), of which {ea['explicit_positive_labels']:,} are late under the historical sequence.")
    L.append("")
    L.append("## 2. RQ1: risk prediction (held-out test dates)")
    L.append("")
    L.append("| H | Test prevalence | Model | ROC-AUC | PR-AUC | Brier | ECE |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for tag, r in rq1["threshold_sensitivity"].items():
        for m in ["PrevalenceBaseline", "LogisticRegression", "RandomForest"]:
            t = r["models"][m]["test"]
            L.append(f"| {r['H_hours']:.0f} h | {_p(r['split']['test']['positive_rate'])} | {m} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | {t['brier_score']:.4f} | {t['ece']:.4f} |")
    trd = s["test_risk_distribution"]
    L.append("")
    L.append(f"Calibrated logistic p_i on the {trd['n']:,} evaluated test stops (H = 4 h): median {trd['median']:.3f}, "
             f"IQR [{trd['q25']:.3f}, {trd['q75']:.3f}], 5-95% [{trd['p05']:.3f}, {trd['p95']:.3f}].")
    L.append("")
    L.append("## 3. RQ2/RQ3: routing benchmark (H = 4 h, NN baseline, B x delta = 16 cells per route)")
    L.append("")
    L.append("| Policy | Mean dTT (min) | TT red. (%) | Mean dNL | Dist. inc. (%) | Accepted moves | Feasible |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in s["policy_summary"]:
        L.append(f"| {r['policy']} | {_f(r['delta_tt'], 1, True)} | {r['tt_red_pct']:.2f} | {_f(r['delta_nl'], 2, True)} | {_f(r['dist_inc_pct'], 2, True)} | {r['accepted_moves']:.2f} | {_p(r['is_feasible'])} |")
    L.append("")
    L.append(f"Route-clustered paired tests (one mean difference per route, N = {meta['test_routes_evaluated']:,}), Holm-Bonferroni:")
    L.append("")
    L.append("| Comparison | Mean diff (min) | 95% bootstrap CI | t | p (unadj.) | Holm p | Wilcoxon p | Routes RA better |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for comp, st in s["route_clustered_statistics"].items():
        L.append(f"| RA vs {comp} | {_f(st['mean_diff'], 2, True)} | [{_f(st['ci_lower'], 2, True)}, {_f(st['ci_upper'], 2, True)}] | {st['t_stat']:.2f} | "
                 f"{_pv(st['p_unadjusted'])} | {_pv(st['p_holm_bonferroni'])} | {_pv(st.get('wilcoxon_pvalue', float('nan')))} | {_p(st['share_routes_ra_better'])} |")
    inv = s["invariant_audit"]
    L.append("")
    L.append(f"Invariants: min dTT = {inv['min_delta_tt']:.4f} min over {inv['total_runs']:,} runs; feasibility {_p(inv['feasibility_rate'])}; "
             f"{inv['runs_delta_tt_pos_delta_nl_neg']} runs with dTT > 0 and dNL < 0; {inv['zero_baseline_runs']} runs with zero baseline tardiness.")
    rs = s["random_seed_means"]
    L.append(f"Random policy over 10 seeds: mean dTT {rs['mean']:.1f} min (seed means {rs['min']:.1f} to {rs['max']:.1f}, std {rs['std']:.1f}).")
    rt = s["runtime_seconds_per_route"]
    ab = s["actionability_runtime_by_delta"]
    L.append(f"Runtime per route (single-threaded worker): full benchmark grid {rt['grid']:.1f} s; actionability per delta "
             + ", ".join(f"{float(k):.2f}: {v:.2f} s" for k, v in ab.items()) + ".")
    L.append("")
    L.append("## 4. Clarke-Wright baseline (delta = 5%)")
    L.append("")
    L.append("| Policy | dTT at B = 20% | TT red. (%) | dNL | Dist. inc. (%) |")
    L.append("| --- | --- | --- | --- | --- |")
    for r in s["clarke_wright"]["B0.20_delta0.05"]:
        L.append(f"| {r['policy']} | {_f(r['delta_tt'], 1, True)} | {r['tt_red_pct']:.2f} | {_f(r['delta_nl'], 2, True)} | {_f(r['dist_inc_pct'], 2, True)} |")
    L.append("")
    L.append("## 5. Threshold sensitivity (B in {10, 20}%, delta in {2, 5}%)")
    L.append("")
    L.append("| H | Policy | Base TT | dTT | TT red. (%) | dNL | RA - policy (Holm p) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in s["threshold_sensitivity"]["summary"]:
        st = s["threshold_sensitivity"]["route_clustered"][f"{r['threshold_H']:.1f}"].get(r["policy"])
        cmp_txt = f"{_f(st['mean_diff'], 1, True)} ({_pv(st['p_holm_bonferroni'])})" if st else "-"
        L.append(f"| {r['threshold_H']:.0f} h | {r['policy']} | {r['base_tt']:.1f} | {_f(r['delta_tt'], 1, True)} | {r['tt_red_pct']:.2f} | {_f(r['delta_nl'], 2, True)} | {cmp_txt} |")
    L.append("")
    sc = s["scenarios"]
    ch = sc["checks"]
    L.append("## 6. M8 travel-time scenarios on the full test cohort")
    L.append("")
    L.append(f"Frozen plans re-timed on {meta['scenario_settings']['n_scenarios']} mean-preserving lognormal scenarios per CV "
             f"(base seed {meta['scenario_settings']['base_seed']}); Random averaged over its 10 seeds per scenario.")
    L.append("")
    L.append(f"- CV = 0 equals deterministic evaluation: max gap {ch['cv0_equivalence']['max_abs_tt_gap_min']:.2e} min, "
             f"{ch['cv0_equivalence']['nl_mismatches']} NL mismatches -> {'PASS' if ch['cv0_equivalence']['passed'] else 'FAIL'}")
    L.append(f"- Frozen R0 / R' unchanged by evaluation -> {'PASS' if ch['frozen_routes_unchanged'] else 'FAIL'}")
    L.append(f"- One shared matrix per route x scenario x CV, regenerated fingerprints: {ch['regenerated_matrices_checked']:,} checked, "
             f"{ch['regenerated_matrix_mismatches']} mismatches -> {'PASS' if ch['shared_and_reproducible_matrices_passed'] else 'FAIL'}")
    L.append(f"- Missing edges on evaluated routes: {ch['missing_edges_on_evaluated_routes']}; zero-mean off-diagonal edges: "
             f"{ch['zero_mean_offdiag_edges']} ({ch['zero_mean_edges_on_frozen_routes']} traversed).")
    L.append("")
    L.append("| CV | Mean multiplier | Realized CV | Min | Max | Mean baseline TT |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for g in sc["generator"]:
        L.append(f"| {g['cv']:.2f} | {g['mean_multiplier']:.4f} | {g['realized_cv']:.4f} | {g['min']:.3f} | {g['max']:.3f} | {g['baseline_tt_mean']:,.1f} |")
    L.append("")
    L.append("| CV | Policy | Planning dTT | Mean dTT | Median dTT | 5-95% dTT | Mean dNL | dTT > 0 | dTT < 0 | Routes with mean dTT < 0 | Retention |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sc["pooled"]:
        L.append(f"| {r['cv']:.2f} | {r['policy']} | {_f(r['planning_delta_tt'], 1, True)} | {_f(r['mean_delta_tt'], 1, True)} | {_f(r['median_delta_tt'], 1, True)} | "
                 f"[{_f(r['p05_delta_tt'], 1, True)}, {_f(r['p95_delta_tt'], 1, True)}] | {_f(r['mean_delta_nl'], 2, True)} | {_p(r['share_delta_tt_pos'])} | "
                 f"{_p(r['share_delta_tt_neg'])} | {_p(r['share_routes_mean_delta_tt_neg'])} | {r['retention_ratio']:.3f} |")
    L.append("")
    sel = sc["ra_ab_selection"]
    L.append(f"RA vs AB (V_risk = dTT(RA) - dTT(AB); positive favours RA). Frozen plans: mean selection overlap {_p(sel['mean_overlap'])}, "
             f"identical selection order in {sel['identical_order_cells']:,} and identical final routes in {sel['identical_final_route_cells']:,} "
             f"of {sel['cells']:,} route x cell combinations; accepted moves RA {sel['mean_accepted_ra']:.2f} vs AB {sel['mean_accepted_ab']:.2f}.")
    L.append("")
    L.append("| CV | Mean V_risk | Median | 5-95% | RA better | AB better | Tie | Routes RA better | Routes AB better | Route-level Holm p |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sc["v_risk"]:
        st = sc["route_level_tests"][f"{r['cv']:.2f}"]["AB"]
        L.append(f"| {r['cv']:.2f} | {_f(r['mean_v_risk'], 2, True)} | {_f(r['median_v_risk'], 2, True)} | [{_f(r['p05_v_risk'], 1, True)}, {_f(r['p95_v_risk'], 1, True)}] | "
                 f"{_p(r['share_ra_better'])} | {_p(r['share_ab_better'])} | {_p(r['share_tie'])} | {_p(r['routes_ra_better'])} | {_p(r['routes_ab_better'])} | {_pv(st['p_holm_bonferroni'])} |")
    L.append("")
    L.append("Route-level tests on scenario-averaged dTT (RA minus comparison, Holm-Bonferroni):")
    L.append("")
    L.append("| CV | " + " | ".join(f"RA vs {c}" for c in COMPARISONS) + " |")
    L.append("| --- | " + " | ".join("---" for _ in COMPARISONS) + " |")
    for cv, st in sc["route_level_tests"].items():
        L.append(f"| {cv} | " + " | ".join(f"{_f(st[c]['mean_diff'], 1, True)} (p {_pv(st[c]['p_holm_bonferroni'])})" for c in COMPARISONS) + " |")
    L.append("")
    L.append("## 7. Limitations")
    L.append("")
    L.append("- Service thresholds are derived (departure + H); explicit Amazon time windows are too sparse to evaluate routing.")
    L.append("- Lateness labels propagate the historical driver sequence with mean travel times and planned service times.")
    L.append("- Travel-time scenarios are synthetic, independent edge-level lognormal noise; distances are Haversine proxies.")
    L.append("- Test routes come from the latest calendar dates of one season (2018); stations are not held out.")
    L.append("")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["features", "train", "evaluate", "report", "all"], default="all")
    parser.add_argument("--cohort-dir", default=DEFAULT_COHORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--cv-levels", type=float, nargs="+", default=DEFAULT_CV_LEVELS)
    parser.add_argument("--scenario-seed", type=int, default=DEFAULT_SCENARIO_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N routes (smoke tests).")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.stage in ("features", "all"):
        stage_features(args.cohort_dir, args.output_dir, args.workers, args.limit)
    if args.stage in ("train", "all"):
        stage_train(args.output_dir)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(args.cohort_dir, args.output_dir, args.workers, args.n_scenarios, args.cv_levels,
                       args.scenario_seed, args.limit)
    if args.stage in ("report", "all"):
        stage_report(args.cohort_dir, args.output_dir, args.report_path, args.cv_levels, args.n_scenarios, args.scenario_seed)
