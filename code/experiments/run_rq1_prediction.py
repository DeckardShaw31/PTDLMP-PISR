import os
import sys
import json
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.features.build_features import extract_ex_ante_features_for_route, compute_ground_truth_labels, assert_no_data_leakage, LEGAL_EX_ANTE_FEATURES
from src.features.leakage_guard import check_chronological_integrity, compute_historical_rates_with_cutoff
from src.prediction.train import train_and_evaluate_pipeline, RiskPredictionPipeline, chronological_split
from src.evaluation.tables import format_markdown_table, format_latex_table

def run_rq1_experiment(
    data_dir: str,
    output_dir: str
) -> Tuple[RiskPredictionPipeline, Dict[str, Any], pd.DataFrame]:
    print(f"[RQ1] Loading Amazon dataset from '{data_dir}'...")
    instances = load_official_amazon_dataset(data_dir, strict_mode=True, default_sla_hours=4.0)
    print(f"[RQ1] Successfully loaded {len(instances)} routes.")

    # 1. Build dataset rows across all routes
    all_rows = []
    for r_id, inst in instances.items():
        feat_df = extract_ex_ante_features_for_route(inst)
        labels = compute_ground_truth_labels(inst)
        feat_df["label"] = feat_df["customer_id"].map(labels)
        all_rows.append(feat_df)

    dataset = pd.concat(all_rows, ignore_index=True)
    print(f"[RQ1] Built dataset with {len(dataset)} customer delivery stops across {dataset['route_date'].nunique()} dates.")
    print(f"[RQ1] Base late delivery prevalence: {dataset['label'].mean():.2%}")

    # 2. Strict leakage guard audit
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
    assert_no_data_leakage(feature_cols)
    print("[RQ1] Leakage guard verified: all features are strictly ex-ante.")

    # 3. Chronological train/val/test split
    df_train, df_val, df_test = chronological_split(dataset, date_col="route_date")
    print(f"[RQ1] Chronological split: Train={len(df_train)} stops ({df_train['route_date'].nunique()} dates), "
          f"Val={len(df_val)} stops ({df_val['route_date'].nunique()} dates), "
          f"Test={len(df_test)} stops ({df_test['route_date'].nunique()} dates)")

    # 4. Compare Models: Prevalence Baseline, Logistic Regression, Random Forest
    results = {}
    from src.prediction.evaluate import evaluate_predictions

    # Prevalence Baseline (predicts constant training prevalence)
    train_prev = float(df_train["label"].mean())
    val_prev_probs = np.full(len(df_val), train_prev)
    test_prev_probs = np.full(len(df_test), train_prev)
    val_prev_metrics = evaluate_predictions(df_val["label"].values, val_prev_probs)
    test_prev_metrics = evaluate_predictions(df_test["label"].values, test_prev_probs)
    results["PrevalenceBaseline"] = {
        **test_prev_metrics,
        "test": test_prev_metrics,
        "validation": val_prev_metrics
    }

    # Logistic Regression
    pipe_lr, metrics_lr, test_preds_lr = train_and_evaluate_pipeline(dataset, feature_cols, model_type="logistic")
    results["LogisticRegression"] = metrics_lr

    # Random Forest
    pipe_rf, metrics_rf, test_preds_rf = train_and_evaluate_pipeline(dataset, feature_cols, model_type="rf")
    results["RandomForest"] = metrics_rf

    print("\n" + "=" * 70)
    print("RQ1 VALIDATION MODEL SELECTION COMPARISON (CHRONOLOGICAL SPLIT)")
    print("=" * 70)
    val_headers = ["Model", "Val ROC-AUC", "Val PR-AUC", "Val Brier Score", "Val Log Loss", "Val ECE"]
    val_rows = []
    for m_name, m_dict in results.items():
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

    # Model Selection Rule: select model with lowest validation Brier score
    brier_lr = metrics_lr["validation"]["brier_score"]
    brier_rf = metrics_rf["validation"]["brier_score"]
    selected_name = "LogisticRegression" if brier_lr <= brier_rf else "RandomForest"
    selected_pipe = pipe_lr if selected_name == "LogisticRegression" else pipe_rf
    selected_preds = test_preds_lr if selected_name == "LogisticRegression" else test_preds_rf
    print(f"\n[RQ1] Model Selection Decision: '{selected_name}' selected based on validation Brier ({min(brier_lr, brier_rf):.4f}).")

    print("\n" + "=" * 70)
    print("RQ1 PREDICTION EVALUATION RESULTS (OUT-OF-SAMPLE TEST COHORT)")
    print("=" * 70)
    table_headers = ["Model", "Test ROC-AUC", "Test PR-AUC", "Test Brier Score", "Test Log Loss", "Test ECE"]
    table_rows = []
    for m_name, m_dict in results.items():
        t = m_dict["test"]
        table_rows.append([
            m_name,
            f"{t['roc_auc']:.3f}",
            f"{t['pr_auc']:.3f}",
            f"{t['brier_score']:.4f}",
            f"{t['log_loss']:.4f}",
            f"{t['ece']:.4f}"
        ])
    print(format_markdown_table(table_headers, table_rows))

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "rq1_prediction_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed test predictions CSV (route, customer, label, probability)
    selected_preds["route"] = selected_preds["route_id"] if "route_id" in selected_preds.columns else selected_preds["route_date"]
    selected_preds["customer"] = selected_preds["customer_id"]
    selected_preds["probability"] = selected_preds["predicted_risk_pi"]
    selected_preds["selected_model"] = selected_name
    selected_preds["target_label_definition"] = "route_propagated_promised_time_violation_proxy"
    out_csv = os.path.join(output_dir, "rq1_test_predictions.csv")
    cols_to_save = ["route", "customer", "label", "probability", "route_id", "customer_id", "predicted_risk_pi", "route_date", "selected_model", "target_label_definition"]
    avail_cols = [c for c in cols_to_save if c in selected_preds.columns]
    selected_preds[avail_cols].to_csv(out_csv, index=False)
    print(f"[RQ1] Saved predictions to '{out_csv}' ({len(selected_preds)} rows).")

    return selected_pipe, results, dataset

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    if not os.path.exists(os.path.join(raw_dir, "route_data.json")):
        raise FileNotFoundError(f"Official Amazon Challenge dataset not found in '{raw_dir}'.")
    o_dir = os.path.join(base_dir, "outputs")
    print(f"[RQ1] Selected official dataset directory: {raw_dir}")
    run_rq1_experiment(raw_dir, o_dir)
