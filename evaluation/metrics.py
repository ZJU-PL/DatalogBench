import math
import random

from collections import defaultdict
from typing import Dict, List, Tuple


# Two-sided 95% critical values of Student's t distribution, indexed by degrees
# of freedom. Falls back to the normal approximation (1.96) for large df.
_T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T_CRITICAL_95.get(df, 1.96)


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, sample standard deviation) of the values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def mean_std_ci95(values: List[float]) -> Dict[str, float]:
    """Mean, sample std, and half-width of the 95% CI of the mean (t-based).

    The half-width is the error-bar size: report mean +/- ci95_half.
    """
    n = len(values)
    mean, std = mean_std(values)
    if n <= 1:
        return {"mean": mean, "std": std, "ci95_half": 0.0, "n": n}
    half = t_critical_95(n - 1) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_half": half, "n": n}


def paired_permutation_test(diffs: List[float], iterations: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences.

    Under H0 (no difference between the two systems) each per-case difference
    is symmetric around 0, so its sign can be flipped at random. Returns the
    p-value for the observed mean difference.
    """
    n = len(diffs)
    if n == 0 or all(d == 0 for d in diffs):
        return 1.0
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed:
            extreme += 1
    # Add-one smoothing keeps the p-value away from an impossible exact 0.
    return (extreme + 1) / (iterations + 1)


def bootstrap_ci_mean(values: List[float], iterations: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI (low, high) for the mean of the values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        total = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(total / n)
    means.sort()
    low = means[int(0.025 * iterations)]
    high = means[min(int(0.975 * iterations), iterations - 1)]
    return low, high


def bootstrap_mean_ci(values: List[float], iterations: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Mean plus a percentile bootstrap 95% CI, with a half-width for +/- reporting.

    Use this over the *per-case* values of a metric: it answers "how would this
    number move if we had drawn a different sample of benchmark tasks", which is
    the uncertainty a benchmark number should carry. Unlike the across-run CI it
    is computable from a single run.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci95_half": 0.0, "n": 0}
    mean = sum(values) / n
    low, high = bootstrap_ci_mean(values, iterations=iterations, seed=seed)
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        # asymmetric in general; report the wider side so "mean +/- half" never understates
        "ci95_half": max(high - mean, mean - low),
        "n": n,
    }


def per_case_values(details: List[Dict], field: str) -> List[float]:
    """Collapse detail rows to one value per case (mean over repeated runs)."""
    per_case = defaultdict(list)
    for row in details:
        value = row.get(field)
        if value is None:
            continue
        per_case[row["case_id"]].append(float(bool(value)) if isinstance(value, bool) else float(value))
    return [sum(v) / len(v) for v in per_case.values()]


def case_level_bootstrap(details: List[Dict], field: str, iterations: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap CI over benchmark cases for one metric field."""
    return bootstrap_mean_ci(per_case_values(details, field), iterations=iterations, seed=seed)


def calculate_precision_recall_f1(expected_count: int, predicted_count: int, true_positive_count: int) -> Dict[str, float]:
    fp = max(predicted_count - true_positive_count, 0)
    fn = max(expected_count - true_positive_count, 0)

    precision = true_positive_count / (true_positive_count + fp) if (true_positive_count + fp) > 0 else 0.0
    recall = true_positive_count / (true_positive_count + fn) if (true_positive_count + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_from_confusion(tp: int, fp: int, fn: int) -> Dict[str, float]:
    expected_count = tp + fn
    predicted_count = tp + fp
    return calculate_precision_recall_f1(expected_count, predicted_count, tp)
