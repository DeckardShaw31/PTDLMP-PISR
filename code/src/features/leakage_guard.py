from typing import List, Dict, Any
import pandas as pd
from .build_features import FORBIDDEN_LEAKAGE_SUBSTRINGS, assert_no_data_leakage

def check_chronological_integrity(df: pd.DataFrame, date_col: str = "route_date") -> bool:
    """
    Verifies that the dataset can be strictly partitioned chronologically without date overlap.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")
    unique_dates = sorted(df[date_col].unique())
    return len(unique_dates) >= 3

def compute_historical_rates_with_cutoff(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    date_col: str = "route_date"
) -> Dict[str, Dict[str, float]]:
    """
    Computes expanding-window historical mean rates strictly for dates prior to t (cutoff < t).
    Guarantees zero target leakage across dates.
    Returns: mapping (date, group_val) -> historical_rate.
    """
    sorted_df = df.sort_values(date_col)
    unique_dates = sorted(sorted_df[date_col].unique())
    rate_map: Dict[str, Dict[str, float]] = {}

    for d in unique_dates:
        # Strictly earlier dates only: date < d
        prior = sorted_df[sorted_df[date_col] < d]
        if prior.empty:
            global_mean = 0.15
            rates = {}
        else:
            global_mean = float(prior[target_col].mean())
            grouped = prior.groupby(group_col)[target_col].agg(["count", "mean"])
            # Empirical Bayes / m-estimate smoothing: (count*mean + 5*global_mean) / (count + 5)
            rates = {}
            for g_val, row in grouped.iterrows():
                cnt = row["count"]
                m = row["mean"]
                smoothed = (cnt * m + 5.0 * global_mean) / (cnt + 5.0)
                rates[g_val] = float(smoothed)
            rates["__GLOBAL__"] = global_mean

        rate_map[d] = rates

    return rate_map
