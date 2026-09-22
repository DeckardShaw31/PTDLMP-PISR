from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy import stats

def compute_paired_statistics(
    sample_a: List[float],
    sample_b: List[float],
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Computes paired t-test, Wilcoxon signed-rank test, and bootstrap 95% CI
    for paired differences: diff = sample_a - sample_b.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    diff = a - b
    n = len(diff)

    if n < 2:
        return {
            "n": n,
            "mean_diff": float(np.mean(diff)) if n > 0 else 0.0,
            "ci_lower": float(np.mean(diff)) if n > 0 else 0.0,
            "ci_upper": float(np.mean(diff)) if n > 0 else 0.0,
            "t_stat": 0.0,
            "t_pvalue": 1.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_pvalue": 1.0
        }

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # Paired t-test
    if np.all(diff == diff[0]):
        t_stat, t_pval = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(a, b, alternative=alternative)
        t_stat = float(t_res.statistic)
        t_pval = float(t_res.pvalue)

    # Wilcoxon signed-rank test
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) == 0:
        w_stat, w_pval = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(nonzero_diff, alternative=alternative)
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(1000):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(diff[idx]))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "t_pvalue": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval
    }

def compute_route_clustered_statistics(
    runs_df: Any,
    target_metric: str = "delta_tt",
    base_policy: str = "RA",
    comparison_policies: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Computes route-level clustered paired statistics to guard against pseudo-replication
    across repeated budget-tolerance experimental cells.

    Aggregates cell observations to the route level:
        D_r = mean_{b, delta} (Metric(base_policy) - Metric(comp_policy))
    Then conducts a 1-sample t-test on route means and applies Holm-Bonferroni correction.
    """
    import pandas as pd
    if comparison_policies is None:
        comparison_policies = ["AB", "RB", "Slack", "Deadline", "Random"]

    df = runs_df.copy()
    piv = df.pivot(index=["route_id", "budget", "delta"], columns="policy", values=target_metric).reset_index()

    raw_results = {}
    p_values_to_correct = []
    comp_order = []

    for comp in comparison_policies:
        if comp not in piv.columns or base_policy not in piv.columns:
            continue
        piv[f"diff_{comp}"] = piv[base_policy] - piv[comp]
        route_means = piv.groupby("route_id")[f"diff_{comp}"].mean()
        n_routes = len(route_means)

        if n_routes < 2:
            raw_results[comp] = {
                "n_routes": n_routes,
                "mean_diff": float(route_means.mean()) if n_routes > 0 else 0.0,
                "route_means": [float(x) for x in route_means.values],
                "t_stat": 0.0,
                "p_unadjusted": 1.0,
                "p_holm_bonferroni": 1.0
            }
            continue

        mean_val = float(route_means.mean())
        std_val = float(route_means.std(ddof=1))
        t_stat, p_val = stats.ttest_1samp(route_means, 0.0)

        # Bootstrap 95% CI across routes
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(1000):
            idx = rng.randint(0, n_routes, size=n_routes)
            boot.append(np.mean(route_means.values[idx]))
        ci_lower = float(np.percentile(boot, 2.5))
        ci_upper = float(np.percentile(boot, 97.5))

        raw_results[comp] = {
            "n_routes": n_routes,
            "mean_diff": mean_val,
            "std_diff": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "route_means": [float(x) for x in route_means.values],
            "t_stat": float(t_stat),
            "p_unadjusted": float(p_val)
        }
        comp_order.append(comp)
        p_values_to_correct.append(float(p_val))

    # Apply Holm-Bonferroni correction
    m = len(p_values_to_correct)
    if m > 0:
        sorted_indices = np.argsort(p_values_to_correct)
        adjusted_p = np.zeros(m)
        cum_max = 0.0
        for rank, idx in enumerate(sorted_indices):
            multiplier = m - rank
            adj_p = min(1.0, p_values_to_correct[idx] * multiplier)
            cum_max = max(cum_max, adj_p)
            adjusted_p[idx] = min(1.0, cum_max)

        for i, comp in enumerate(comp_order):
            raw_results[comp]["p_holm_bonferroni"] = float(adjusted_p[i])

    return raw_results

