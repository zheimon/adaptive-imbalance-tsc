"""Statistical significance tests for comparing method results."""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def effect_size_r(statistic: float, n: int) -> float:
    """Compute effect size r = Z / sqrt(N) from Wilcoxon signed-rank test.

    Parameters
    ----------
    statistic : float
        Wilcoxon test statistic W.
    n : int
        Number of paired observations.

    Returns
    -------
    float — effect size r.
    """
    # Compute Z approximation from W statistic
    # E[W] = n(n+1)/4, Var[W] = n(n+1)(2n+1)/24
    if n <= 0:
        return 0.0
    mean_w = n * (n + 1) / 4
    var_w = n * (n + 1) * (2 * n + 1) / 24
    if var_w <= 0:
        return 0.0
    z = (statistic - mean_w) / np.sqrt(var_w)
    return float(abs(z) / np.sqrt(n))


def wilcoxon_test(
    scores_a: List[float], scores_b: List[float]
) -> Dict[str, float]:
    """Wilcoxon signed-rank test between two paired score lists.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Paired scores (e.g., per-seed results for two methods).

    Returns
    -------
    dict with keys:
        'statistic': float — Wilcoxon W statistic
        'pvalue': float
        'significant_at_05': bool
        'effect_size': float — r = |Z| / sqrt(N)
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    # Handle edge cases: all differences zero
    diffs = a - b
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "significant_at_05": False,
            "effect_size": 0.0,
        }

    result = stats.wilcoxon(a, b)
    n = len(a)
    r = effect_size_r(float(result.statistic), n)

    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "significant_at_05": bool(result.pvalue < 0.05),
        "effect_size": r,
    }


def pairwise_wilcoxon(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank test across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Columns: ['method', 'seed', metric]. Each (method, seed) pair
        gives one score.
    metric : str
        Column name for the score to compare.

    Returns
    -------
    pd.DataFrame — K×K p-value matrix, indexed and columned by method name.
    """
    methods = results_df["method"].unique().tolist()
    k = len(methods)
    pmat = np.ones((k, k), dtype=np.float64)

    # Build method -> scores dict (sorted by seed for alignment)
    method_scores: Dict[str, List[float]] = {}
    for m in methods:
        sub = results_df[results_df["method"] == m].sort_values("seed")
        method_scores[m] = sub[metric].tolist()

    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                pmat[i, j] = 1.0
                continue
            result = wilcoxon_test(method_scores[mi], method_scores[mj])
            pmat[i, j] = result["pvalue"]

    return pd.DataFrame(pmat, index=methods, columns=methods)
