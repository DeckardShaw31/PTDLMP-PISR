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
    instances = load_official_amazon_dataset(data_dir)
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
    
    # Logistic Regression
    pipe_lr, metrics_lr, _ = train_and_evaluate_pipeline(dataset, feature_cols, model_type="logistic")
    results["LogisticRegression"] = metrics_lr

    # Random Forest
    pipe_rf, metrics_rf, test_preds = train_and_evaluate_pipeline(dataset, feature_cols, model_type="rf")
    results["RandomForest"] = metrics_rf

    print("\n" + "=" * 60)
    print("RQ1 PREDICTION EVALUATION RESULTS (OUT-OF-SAMPLE TEST COHORT)")
    print("=" * 60)
    table_headers = ["Model", "ROC-AUC", "PR-AUC", "Brier Score", "Log Loss", "ECE"]
    table_rows = []
    for m_name, m_dict in results.items():
        table_rows.append([
            m_name,
            f"{m_dict['roc_auc']:.3f}",
            f"{m_dict['pr_auc']:.3f}",
            f"{m_dict['brier_score']:.4f}",
            f"{m_dict['log_loss']:.4f}",
            f"{m_dict['ece']:.4f}"
        ])
    print(format_markdown_table(table_headers, table_rows))

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "rq1_prediction_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    return pipe_rf, results, dataset

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    challenge_dir = os.path.join(base_dir, "data", "challenge")
    d_dir = raw_dir if os.path.exists(os.path.join(raw_dir, "route_data.json")) else challenge_dir
    o_dir = os.path.join(base_dir, "outputs")
    print(f"[RQ1] Selected dataset directory: {d_dir}")
    run_rq1_experiment(d_dir, o_dir)
