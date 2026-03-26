"""Statistical tests and significance analysis."""

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                          n_bootstrap: int = 10000, seed: int = 42) -> dict[str, float]:
    """Paired bootstrap test for comparing two systems.

    Returns p-value for the hypothesis that system B is better than system A.
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert len(b) == n

    delta = b - a
    observed_delta = delta.mean()
    count_better = 0

    for _ in range(n_bootstrap):
        sample = rng.choice(delta, size=n, replace=True)
        if sample.mean() > 0:
            count_better += 1

    p_value = 1.0 - (count_better / n_bootstrap)
    return {
        "observed_delta": float(observed_delta),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
    }


def compute_confidence_interval(scores: list[float],
                                confidence: float = 0.95) -> dict[str, float]:
    """Compute confidence interval for a set of scores."""
    arr = np.array(scores)
    mean = arr.mean()
    se = stats.sem(arr)
    h = se * stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
    return {
        "mean": float(mean),
        "std": float(arr.std()),
        "ci_lower": float(mean - h),
        "ci_upper": float(mean + h),
        "confidence": confidence,
        "n": len(scores),
    }


def summarize_experiment_results(results: dict[str, list[float]]) -> dict[str, dict]:
    """Summarize results across multiple metrics with CIs."""
    summary = {}
    for metric_name, scores in results.items():
        summary[metric_name] = compute_confidence_interval(scores)
    return summary
