import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # In bin
        if i == num_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Evaluates risk predictions against ground-truth labels.
    Handles potential class imbalance or edge cases gracefully.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)

    # If only one class present in evaluation slice
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": float(np.mean(y_true)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, y_prob)),
            "ece": compute_expected_calibration_error(y_true, y_prob)
        }

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": ll,
        "ece": ece
    }
