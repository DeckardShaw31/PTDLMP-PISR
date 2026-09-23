import os
import sys
import json
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.features.build_features import extract_ex_ante_features_for_route, compute_ground_truth_labels, assert_no_data_leakage, LEGAL_EX_ANTE_FEATURES
from src.features.leakage_guard import check_chronological_integrity, compute_historical_rates_with_cutoff
from src.prediction.train import train_and_evaluate_pipeline, RiskPredictionPipeline, chronological_split
from src.evaluation.tables import format_markdown_table, format_latex_table
from src.prediction.evaluate import evaluate_predictions

FEATURE_COLS = [
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

def run_explicit_deadline_audit(raw_dir: str) -> Dict[str, Any]:
    """
    Performs an explicit audit of customer time-window deadlines (time_window.end_time_utc)
    without synthetic SLA imputation. Discloses dataset sparsity and single-class limitations.
    """
    print("\n" + "=" * 75)
    print("EXPLICIT-DEADLINE AUDIT: OFFICIAL AMAZON TIME WINDOWS (time_window.end_time_utc)")
    print("=" * 75)
    instances = load_official_amazon_dataset(raw_dir, strict_mode=True, derived_threshold_hours=None)
    all_rows = []
    for r_id, inst in instances.items():
        f_df = extract_ex_ante_features_for_route(inst)
        lbls = compute_ground_truth_labels(inst)
        f_df["label"] = f_df["customer_id"].map(lbls)
        all_rows.append(f_df)
    full_df = pd.concat(all_rows, ignore_index=True)

    explicit_df = full_df[full_df["deadline_source"] == "explicit"].copy()
    total_stops = len(full_df)
    total_explicit = len(explicit_df)
    explicit_pct = (total_explicit / total_stops * 100.0) if total_stops > 0 else 0.0

    df_train, df_val, df_test = chronological_split(explicit_df, date_col="route_date")

    audit_res = {
        "total_stops": total_stops,
        "explicit_deadline_stops": total_explicit,
        "explicit_pct": round(explicit_pct, 2),
        "split_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test)
        },
        "split_positive_counts": {
            "train": int(df_train["label"].sum()) if not df_train.empty else 0,
            "val": int(df_val["label"].sum()) if not df_val.empty else 0,
            "test": int(df_test["label"].sum()) if not df_test.empty else 0
        },
        "split_positive_prevalence": {
            "train": round(float(df_train["label"].mean()), 4) if not df_train.empty else 0.0,
            "val": round(float(df_val["label"].mean()), 4) if not df_val.empty else 0.0,
            "test": round(float(df_test["label"].mean()), 4) if not df_test.empty else 0.0
        },
        "single_class_test_limitation": bool(len(df_test) == 0 or df_test["label"].nunique() < 2),
        "limitation_disclosure": (
            "Only 128 of 2,038 drop-off stops possess explicit time-window deadlines in the 13-route cohort. "
            "In the chronological held-out test split, only 13 explicit stops exist and all 13 are on-time (0 positive labels). "
            "Because the test cohort contains strictly one class, discrimination metrics (ROC-AUC, PR-AUC) "
            "and out-of-sample risk model evaluation are mathematically uninformative or undefined. "
            "This empirical finding is disclosed as an explicit dataset limitation."
        )
    }

    print(f"Total drop-off stops: {total_stops}")
    print(f"Explicit-deadline stops (time_window.end_time_utc): {total_explicit} ({explicit_pct:.1f}%)")
    print(f"Chronological split of explicit stops: Train={len(df_train)} (pos={audit_res['split_positive_counts']['train']}), "
          f"Val={len(df_val)} (pos={audit_res['split_positive_counts']['val']}), "
          f"Test={len(df_test)} (pos={audit_res['split_positive_counts']['test']})")
    print(f"Limitation: Single-class test split ({audit_res['split_positive_counts']['test']} late deliveries). "
          "Out-of-sample discrimination metrics cannot be evaluated on explicit deadlines alone.")
    print("=" * 75)
    return audit_res

def run_rq1_experiment(
    data_dir: str,
    output_dir: str,
    primary_h: float = 4.0,
    sensitivity_h_list: Optional[List[float]] = None
) -> Tuple[RiskPredictionPipeline, Dict[str, Any], pd.DataFrame]:
    if sensitivity_h_list is None:
        sensitivity_h_list = [3.0, 4.0, 5.0, 6.0]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Audit explicit deadlines
    explicit_audit = run_explicit_deadline_audit(data_dir)

    # 2. Threshold Sensitivity Experiment across H in {3, 4, 5, 6}
    sensitivity_results = {}
    primary_pipe = None
    primary_results = {}
    primary_test_preds = None

    print("\n" + "=" * 75)
    print(f"RQ1 EXPERIMENT: DERIVED DISPATCH SERVICE THRESHOLDS tau_i(H) = t_dep + H")
    print(f"Primary Predeclared Operational Scenario: H = {primary_h:.1f} hours")
    print(f"Robustness Sensitivity Thresholds: H in {sensitivity_h_list} hours")
    print("=" * 75)

    for H in sensitivity_h_list:
        print(f"\n[RQ1] Evaluating Derived Dispatch Service Threshold: H = {H:.1f} hours...")
        instances = load_official_amazon_dataset(data_dir, strict_mode=True, derived_threshold_hours=H)
        
        all_rows = []
        for r_id, inst in instances.items():
            f_df = extract_ex_ante_features_for_route(inst)
            lbls = compute_ground_truth_labels(inst)
            f_df["label"] = f_df["customer_id"].map(lbls)
            all_rows.append(f_df)
        dataset = pd.concat(all_rows, ignore_index=True)

        assert_no_data_leakage(FEATURE_COLS)

        df_train, df_val, df_test = chronological_split(dataset, date_col="route_date")
        
        # Prevalence Baseline
        train_prev = float(df_train["label"].mean())
        val_prev_probs = np.full(len(df_val), train_prev)
        test_prev_probs = np.full(len(df_test), train_prev)
        val_prev_metrics = evaluate_predictions(df_val["label"].values, val_prev_probs)
        test_prev_metrics = evaluate_predictions(df_test["label"].values, test_prev_probs)
        prev_res = {
            **test_prev_metrics,
            "test": test_prev_metrics,
            "validation": val_prev_metrics
        }

        # Logistic Regression (Predeclared Primary)
        pipe_lr, metrics_lr, test_preds_lr = train_and_evaluate_pipeline(dataset, FEATURE_COLS, model_type="logistic")

        # Random Forest
        pipe_rf, metrics_rf, test_preds_rf = train_and_evaluate_pipeline(dataset, FEATURE_COLS, model_type="rf")

        h_summary = {
            "H_hours": H,
            "total_stops": len(dataset),
            "overall_prevalence": round(float(dataset["label"].mean()), 4),
            "train_prevalence": round(float(df_train["label"].mean()), 4),
            "val_prevalence": round(float(df_val["label"].mean()), 4),
            "test_prevalence": round(float(df_test["label"].mean()), 4),
            "models": {
                "PrevalenceBaseline": prev_res,
                "LogisticRegression": metrics_lr,
                "RandomForest": metrics_rf
            }
        }
        sensitivity_results[f"H_{H:.1f}h"] = h_summary

        if abs(H - primary_h) < 1e-5:
            primary_pipe = pipe_lr
            primary_results = {
                "PrevalenceBaseline": prev_res,
                "LogisticRegression": metrics_lr,
                "RandomForest": metrics_rf
            }
            primary_test_preds = test_preds_lr

    # Print Validation Calibration Fit Table for Primary Scenario
    print("\n" + "=" * 75)
    print(f"RQ1 VALIDATION CALIBRATION FIT (PRIMARY SCENARIO: H = {primary_h:.1f}h)")
    print("=" * 75)
    val_headers = ["Model", "Val ROC-AUC", "Val PR-AUC", "Val Brier Score", "Val Log Loss", "Val ECE"]
    val_rows = []
    for m_name, m_dict in primary_results.items():
        v = m_dict["validation"]
        val_rows.append([
            m_name,
            f"{v['roc_auc']:.3f}",
            f"{v['pr_auc']:.3f}",
            f"{v['brier_score']:.4f}",
            f"{v['log_loss']:.4f}",
            f"{v['ece']:.4f}"
        ])
    print(format_markdown_table(val_headers, val_rows))

    # Print Out-of-Sample Test Evaluation Table for Primary Scenario
    print("\n" + "=" * 75)
    print(f"RQ1 OUT-OF-SAMPLE TEST EVALUATION (PRIMARY SCENARIO: H = {primary_h:.1f}h)")
    print("=" * 75)
    test_headers = ["Model", "Test ROC-AUC", "Test PR-AUC", "Test Brier Score", "Test Log Loss", "Test ECE"]
    test_rows = []
    for m_name, m_dict in primary_results.items():
        t = m_dict["test"]
        test_rows.append([
            m_name,
            f"{t['roc_auc']:.3f}",
            f"{t['pr_auc']:.3f}",
            f"{t['brier_score']:.4f}",
            f"{t['log_loss']:.4f}",
            f"{t['ece']:.4f}"
        ])
    print(format_markdown_table(test_headers, test_rows))

    # Print Sensitivity Summary across H
    print("\n" + "=" * 75)
    print("RQ1 THRESHOLD SENSITIVITY SUMMARY (LOGISTIC REGRESSION OUT-OF-SAMPLE)")
    print("=" * 75)
    sens_headers = ["Threshold H", "Test Prev (%)", "Test ROC-AUC", "Test PR-AUC", "Test Brier", "Test ECE"]
    sens_rows = []
    for h_key, h_data in sensitivity_results.items():
        lr_t = h_data["models"]["LogisticRegression"]["test"]
        sens_rows.append([
            f"{h_data['H_hours']:.1f} hours",
            f"{h_data['test_prevalence'] * 100.0:.1f}%",
            f"{lr_t['roc_auc']:.3f}",
            f"{lr_t['pr_auc']:.3f}",
            f"{lr_t['brier_score']:.4f}",
            f"{lr_t['ece']:.4f}"
        ])
    print(format_markdown_table(sens_headers, sens_rows))

    # Compile overall output payload
    output_payload = {
        "_metadata": {
            "primary_model": "LogisticRegression",
            "model_selection_protocol": "predeclared_primary",
            "calibration_split": "chronological_validation",
            "evaluation_split": "held_out_test",
            "primary_threshold_hours": primary_h,
            "target_label_definition": "derived_dispatch_service_threshold_violation",
            "feature_columns": FEATURE_COLS
        },
        "explicit_deadline_audit": explicit_audit,
        "primary_scenario_h4": {
            "threshold_hours": primary_h,
            "models": primary_results
        },
        "threshold_sensitivity": sensitivity_results,
        "PrevalenceBaseline": primary_results["PrevalenceBaseline"],
        "LogisticRegression": primary_results["LogisticRegression"],
        "RandomForest": primary_results["RandomForest"]
    }

    out_json = os.path.join(output_dir, "rq1_prediction_results.json")
    with open(out_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[RQ1] Saved complete prediction results to '{out_json}'.")

    # Save detailed test predictions CSV for primary scenario
    if primary_test_preds is not None:
        primary_test_preds["route"] = primary_test_preds["route_id"] if "route_id" in primary_test_preds.columns else primary_test_preds["route_date"]
        primary_test_preds["customer"] = primary_test_preds["customer_id"]
        primary_test_preds["probability"] = primary_test_preds["predicted_risk_pi"]
        primary_test_preds["selected_model"] = "LogisticRegression"
        primary_test_preds["model_selection_protocol"] = "predeclared_primary"
        primary_test_preds["target_label_definition"] = "derived_4h_dispatch_service_threshold"
        
        cols_to_save = [
            "route", "customer", "label", "probability", "route_id", "customer_id",
            "predicted_risk_pi", "route_date", "selected_model", "model_selection_protocol",
            "target_label_definition", "deadline_source", "is_deadline_constrained"
        ]
        avail_cols = [c for c in cols_to_save if c in primary_test_preds.columns]
        out_csv = os.path.join(output_dir, "rq1_test_predictions.csv")
        primary_test_preds[avail_cols].to_csv(out_csv, index=False)
        print(f"[RQ1] Saved primary scenario test predictions to '{out_csv}' ({len(primary_test_preds)} rows).")

    return primary_pipe, output_payload, primary_test_preds

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        raise FileNotFoundError(f"Official Amazon Challenge dataset not found in '{raw_dir}'.")
    o_dir = os.path.join(base_dir, "outputs")
    print(f"[RQ1] Selected official dataset directory: {raw_dir}")
    run_rq1_experiment(raw_dir, o_dir)
