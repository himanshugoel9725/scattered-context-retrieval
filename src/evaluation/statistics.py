"""Statistical testing utilities for experiment analysis.

Provides bootstrap significance tests, effect sizes, and confidence intervals
for comparing metric distributions across experimental conditions.
"""

import math
from typing import NamedTuple

import numpy as np


class BootstrapResult(NamedTuple):
    """Result of a paired bootstrap significance test."""
    p_value: float
    observed_diff: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


class MetricSummary(NamedTuple):
    """Summary statistics for a metric distribution."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap significance test (two-sided).

    Tests H0: mean(scores_a) == mean(scores_b).
    Uses the bootstrap estimation of the distribution of the difference in means.

    Parameters
    ----------
    scores_a : list[float]
        Metric scores for condition A (NaN values are excluded pairwise).
    scores_b : list[float]
        Metric scores for condition B (must be same length as scores_a).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Named tuple with p_value, observed_diff, ci_lower, ci_upper.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    assert len(a) == len(b), f"Score lists must be same length: {len(a)} vs {len(b)}"

    # Exclude NaN pairwise
    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return BootstrapResult(p_value=1.0, observed_diff=0.0,
                               ci_lower=0.0, ci_upper=0.0, n_bootstrap=0)

    observed_diff = float(np.mean(a) - np.mean(b))
    diffs = a - b

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_bootstrap)
    n = len(diffs)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_diffs[i] = np.mean(sample)

    # Two-sided p-value: fraction of bootstrap samples on the opposite side of zero
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return BootstrapResult(
        p_value=p_value,
        observed_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
    )


def compute_effect_size(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Returns NaN if either group has zero variance or fewer than 2 values.
    """
    a = np.array([x for x in scores_a if not math.isnan(x)], dtype=float)
    b = np.array([x for x in scores_b if not math.isnan(x)], dtype=float)

    if len(a) < 2 or len(b) < 2:
        return math.nan

    pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                          (len(b) - 1) * np.var(b, ddof=1)) /
                         (len(a) + len(b) - 2))
    if pooled_std == 0:
        return math.nan

    return float((np.mean(a) - np.mean(b)) / pooled_std)


def summarize_with_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> MetricSummary:
    """Compute mean, std, and bootstrap confidence interval for a metric.

    Parameters
    ----------
    scores : list[float]
        Metric values (NaN values are excluded).
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MetricSummary
        Named tuple with mean, std, ci_lower, ci_upper, n.
    """
    arr = np.array([x for x in scores if not math.isnan(x)], dtype=float)
    n = len(arr)

    if n == 0:
        return MetricSummary(mean=math.nan, std=math.nan,
                             ci_lower=math.nan, ci_upper=math.nan, n=0)
    if n == 1:
        return MetricSummary(mean=float(arr[0]), std=0.0,
                             ci_lower=float(arr[0]), ci_upper=float(arr[0]), n=1)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # Bootstrap CI for the mean
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return MetricSummary(mean=mean, std=std, ci_lower=ci_lower,
                         ci_upper=ci_upper, n=n)


def format_metric(summary: MetricSummary, fmt: str = ".3f") -> str:
    """Format a MetricSummary as 'mean ± std [CI_lower, CI_upper]'."""
    if math.isnan(summary.mean):
        return "N/A"
    return (f"{summary.mean:{fmt}} ± {summary.std:{fmt}} "
            f"[{summary.ci_lower:{fmt}}, {summary.ci_upper:{fmt}}]")


def significance_annotation(p_value: float) -> str:
    """Return significance stars for a p-value (for figure annotations)."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."
