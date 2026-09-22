import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import assert_no_data_leakage, LEGAL_EX_ANTE_FEATURES
from src.features.leakage_guard import check_chronological_integrity, compute_historical_rates_with_cutoff
from src.prediction.evaluate import evaluate_predictions, compute_expected_calibration_error
from src.prediction.calibrate import PostHocCalibrator
from src.prediction.train import chronological_split, RiskPredictionPipeline

def test_leakage_guard_rejects_forbidden_features():
    clean_features = ["dist_from_depot_km", "num_packages", "planned_service_seconds"]
    # Should pass without exception
    assert_no_data_leakage(clean_features)

    # Should raise ValueError
    leaky_features_1 = ["dist_from_depot_km", "actual_sequence_number"]
    with pytest.raises(ValueError, match="Data Leakage Violation"):
        assert_no_data_leakage(leaky_features_1)

    leaky_features_2 = ["realized_delay", "num_packages"]
    with pytest.raises(ValueError, match="Data Leakage Violation"):
        assert_no_data_leakage(leaky_features_2)

def test_chronological_split():
    dates = ["2018-07-01", "2018-07-02", "2018-07-03", "2018-07-04", "2018-07-05"]
    rows = []
    for d in dates:
        for i in range(10):
            rows.append({"route_date": d, "val": i, "label": i % 2})
    df = pd.DataFrame(rows)

    df_train, df_val, df_test = chronological_split(df, date_col="route_date")

    # Dates must be strictly separated
    train_dates = set(df_train["route_date"].unique())
    val_dates = set(df_val["route_date"].unique())
    test_dates = set(df_test["route_date"].unique())

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    # Earliest dates in train, latest in test
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)

def test_probability_calibrator():
    np.random.seed(42)
    # Generate poorly calibrated probabilities (overconfident)
    raw_probs = np.random.uniform(0.1, 0.9, size=100)
    y = (raw_probs > 0.5).astype(int)

    calibrator = PostHocCalibrator(method="sigmoid")
    calibrator.fit(raw_probs, y)

    calibrated = calibrator.predict_proba(raw_probs)
    assert len(calibrated) == 100
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)

def test_end_to_end_prediction_pipeline():
    np.random.seed(42)
    dates = ["2018-06-01", "2018-06-02", "2018-06-03", "2018-06-04", "2018-06-05", "2018-06-06"]
    rows = []
    for d in dates:
        for i in range(20):
            dist = np.random.uniform(1.0, 15.0)
            time_to_deadline = np.random.uniform(30.0, 240.0)
            # High dist and low time_to_deadline => higher chance of being late
            logit = 0.2 * dist - 0.02 * time_to_deadline - 0.5
            prob = 1.0 / (1.0 + np.exp(-logit))
            y = int(np.random.uniform(0, 1) < prob)

            rows.append({
                "route_date": d,
                "dist_from_depot_km": dist,
                "time_to_deadline_minutes": time_to_deadline,
                "planned_service_seconds": float(np.random.choice([60, 120, 180])),
                "package_volume_cm3": float(np.random.uniform(500, 5000)),
                "num_packages": int(np.random.choice([1, 2, 3])),
                "label": y
            })

    df = pd.DataFrame(rows)
    feature_cols = [
        "dist_from_depot_km", "time_to_deadline_minutes",
        "planned_service_seconds", "package_volume_cm3", "num_packages"
    ]

    df_train, df_val, df_test = chronological_split(df, date_col="route_date")
    X_train, y_train = df_train[feature_cols], df_train["label"].values
    X_val, y_val = df_val[feature_cols], df_val["label"].values
    X_test, y_test = df_test[feature_cols], df_test["label"].values

    pipeline = RiskPredictionPipeline(model_type="rf", feature_cols=feature_cols)
    pipeline.fit(X_train, y_train, X_val, y_val)

    test_probs = pipeline.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, test_probs)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

def test_route_clustered_statistics():
    from src.evaluation.statistics import compute_route_clustered_statistics
    # Construct mock factorial DataFrame with 3 routes
    rows = []
    for r in ["R1", "R2", "R3"]:
        for b in [0.1, 0.2]:
            for d in [0.0, 0.05]:
                rows.append({"route_id": r, "budget": b, "delta": d, "policy": "RA", "delta_tt": 50.0})
                rows.append({"route_id": r, "budget": b, "delta": d, "policy": "AB", "delta_tt": 48.0})
                rows.append({"route_id": r, "budget": b, "delta": d, "policy": "RB", "delta_tt": 10.0})
    df_runs = pd.DataFrame(rows)

    stats_res = compute_route_clustered_statistics(df_runs, target_metric="delta_tt", base_policy="RA", comparison_policies=["AB", "RB"])
    assert "AB" in stats_res
    assert "RB" in stats_res
    assert stats_res["AB"]["n_routes"] == 3
    assert stats_res["RB"]["mean_diff"] == 40.0
    assert "p_holm_bonferroni" in stats_res["AB"]

