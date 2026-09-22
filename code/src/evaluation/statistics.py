from typing import List, Dict, Tuple, Any
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
