import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .calibrate import PostHocCalibrator
from .evaluate import evaluate_predictions
from ..data.schemas import RouteInstance
from ..features.build_features import extract_ex_ante_features_for_route, assert_no_data_leakage

def chronological_split(
    df: pd.DataFrame,
    date_col: str = "route_date",
    train_frac: float = 0.6,
    val_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions DataFrame strictly chronologically by unique dates (Section 8.4 of design.md).
    Earliest 60% of dates -> Train
    Next 20% of dates -> Validation (for calibration)
    Latest 20% of dates -> Test (held out)
    """
    unique_dates = sorted(df[date_col].unique())
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError(f"Need at least 3 distinct dates for chronological splitting, got {n_dates}")

    n_train = max(1, int(math.floor(train_frac * n_dates)))
    n_val = max(1, int(math.floor(val_frac * n_dates)))
    # Ensure test has at least 1 date
    if n_train + n_val >= n_dates:
        n_train = max(1, n_dates - 2)
        n_val = 1

    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]

    df_train = df[df[date_col].isin(train_dates)].copy()
    df_val = df[df[date_col].isin(val_dates)].copy()
    df_test = df[df[date_col].isin(test_dates)].copy()

    return df_train, df_val, df_test

class RiskPredictionPipeline:
    def __init__(
        self,
        model_type: str = "rf",
        feature_cols: Optional[List[str]] = None,
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
        random_state: int = 42
    ):
        self.model_type = model_type
        self.feature_cols = feature_cols
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.random_state = random_state

        self.scaler = None
        self.base_model = None
        self.calibrator = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        assert_no_data_leakage(list(X_train.columns))

        if self.model_type == "logistic":
            self.scaler = StandardScaler()
            X_tr_mat = self.scaler.fit_transform(X_train)
            X_val_mat = self.scaler.transform(X_val)
            self.base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
            self.base_model.fit(X_tr_mat, y_train)
            val_raw_probs = self.base_model.predict_proba(X_val_mat)[:, 1]
        elif self.model_type == "rf":
            self.base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                random_state=self.random_state
            )
            self.base_model.fit(X_train, y_train)
            val_raw_probs = self.base_model.predict_proba(X_val)[:, 1]
        elif self.model_type == "hist_gb":
            self.base_model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                random_state=self.random_state
            )
            self.base_model.fit(X_train, y_train)
            val_raw_probs = self.base_model.predict_proba(X_val)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'")

        if self.calibrate:
            self.calibrator = PostHocCalibrator(method=self.calibration_method)
            self.calibrator.fit(val_raw_probs, y_val)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert_no_data_leakage(list(X.columns))
        if self.model_type == "logistic":
            X_mat = self.scaler.transform(X)
            raw_probs = self.base_model.predict_proba(X_mat)[:, 1]
        else:
            raw_probs = self.base_model.predict_proba(X)[:, 1]

        if self.calibrate and self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    model_type: str = "rf",
    calibration_method: str = "sigmoid"
) -> Tuple[RiskPredictionPipeline, Dict[str, float], pd.DataFrame]:
    """
    Executes full chronological training, calibration, and test evaluation.
    Returns: (fitted_pipeline, test_metrics, test_predictions_df)
    """
    df_train, df_val, df_test = chronological_split(df, date_col="route_date")

    X_train, y_train = df_train[feature_cols], df_train[target_col].values
    X_val, y_val = df_val[feature_cols], df_val[target_col].values
    X_test, y_test = df_test[feature_cols], df_test[target_col].values

    pipeline = RiskPredictionPipeline(
        model_type=model_type,
        feature_cols=feature_cols,
        calibrate=True,
        calibration_method=calibration_method
    )
    pipeline.fit(X_train, y_train, X_val, y_val)

    test_probs = pipeline.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, test_probs)

    df_test_preds = df_test.copy()
    df_test_preds["predicted_risk_pi"] = test_probs

    return pipeline, metrics, df_test_preds

def inject_calibrated_risks_into_instances(
    instances: Dict[str, RouteInstance],
    pipeline: RiskPredictionPipeline,
    feature_cols: List[str],
    historical_station_rates: Optional[Dict[str, float]] = None
) -> None:
    """
    Populates Stop.predicted_risk_pi on all RouteInstance objects using the trained model.
    Enforces strict ex-ante evaluation.
    """
    for route_id, inst in instances.items():
        feat_df = extract_ex_ante_features_for_route(inst, historical_station_rates=historical_station_rates)
        if not feat_df.empty:
            X = feat_df[feature_cols]
            probs = pipeline.predict_proba(X)
            for idx, c_id in enumerate(feat_df["customer_id"]):
                inst.stops[c_id].predicted_risk_pi = float(probs[idx])
