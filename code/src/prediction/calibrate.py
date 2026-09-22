import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class PostHocCalibrator(BaseEstimator, ClassifierMixin):
    """
    Fits post-hoc probability calibration on validation split probabilities
    using either Platt scaling (logistic) or Isotonic regression (Section 8.3 & 8.4 of design.md).
    """
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator_ = None

    def fit(self, val_uncalibrated_probs: np.ndarray, y_val: np.ndarray):
        val_probs = np.clip(np.asarray(val_uncalibrated_probs, dtype=float), 1e-5, 1.0 - 1e-5)
        y_val = np.asarray(y_val, dtype=int)

        if len(np.unique(y_val)) < 2:
            self.calibrator_ = None
            return self

        if self.method == "sigmoid":
            # Platt scaling: log-odds as input to 1D LogisticRegression
            log_odds = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(log_odds, y_val)
            self.calibrator_ = clf
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_probs, y_val)
            self.calibrator_ = iso
        else:
            raise ValueError(f"Unknown calibration method: '{self.method}'")

        return self

    def predict_proba(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(uncalibrated_probs, dtype=float), 1e-5, 1.0 - 1e-5)
        if self.calibrator_ is None:
            return probs

        if self.method == "sigmoid":
            log_odds = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            calibrated = self.calibrator_.predict_proba(log_odds)[:, 1]
        else:
            calibrated = self.calibrator_.predict(probs)

        return np.clip(calibrated, 1e-4, 1.0 - 1e-4)
