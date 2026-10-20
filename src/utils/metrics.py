"""Metrics used for Safe-CEG evaluation."""

from __future__ import annotations

import math
from statistics import fmean, median, stdev


def q_error(estimate: float, true_cardinality: float) -> float:
    """Return the q-error between an estimate and the true cardinality."""
    if estimate <= 0:
        raise ValueError("Estimate must be positive")
    if true_cardinality <= 0:
        raise ValueError("True cardinality must be positive")
    return max(estimate / true_cardinality, true_cardinality / estimate)


def summarize_q_errors(q_errors: list[float]) -> dict[str, float]:
    """Return a compact q-error summary dictionary."""
    if not q_errors:
        raise ValueError("At least one q-error is required")

    sorted_errors = sorted(q_errors)
    return {
        "mean": float(fmean(sorted_errors)),
        "median": float(median(sorted_errors)),
        "p95": _percentile(sorted_errors, 0.95),
        "p99": _percentile(sorted_errors, 0.99),
    }


def safety_violation_rate(
    ml_estimates: list[float], bounds: list[float], tau: float
) -> float:
    """Fraction of estimates exceeding the *threshold* ``tau * bound``.

    NOTE: this is a *policy/threshold* metric (how often the fallback gate would
    trigger), NOT a guarantee violation against the true cardinality. The real
    safety-relevant violation (``bound < |Q|``) is measured separately by
    ``bound_validity_violation_rate``. Reports should not read this as a
    true-cardinality guarantee violation.
    """
    if tau <= 0:
        raise ValueError("Tau must be positive")
    if len(ml_estimates) != len(bounds):
        raise ValueError("Estimate and bound collections must have the same length")
    if not ml_estimates:
        raise ValueError("At least one estimate is required")

    violations = sum(
        1
        for estimate, bound in zip(ml_estimates, bounds, strict=True)
        if estimate > tau * bound
    )
    return violations / len(ml_estimates)


def bound_validity_violation_rate(
    bounds: list[float], true_cardinalities: list[float]
) -> float:
    """Fraction of queries where the upper bound is *invalid* (``bound < |Q|``).

    This is the real safety-relevant violation: a sound upper certificate must
    satisfy ``bound >= true_cardinality``.
    """
    if len(bounds) != len(true_cardinalities):
        raise ValueError("Bound and true-cardinality collections must match length")
    if not bounds:
        raise ValueError("At least one bound is required")
    violations = sum(
        1
        for bound, true in zip(bounds, true_cardinalities, strict=True)
        if bound < true
    )
    return violations / len(bounds)


def aggregate_multi_seed(
    per_seed_results: list[dict[str, object]],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    """Aggregate numeric results across seeds into mean ± std with 95% CI.

    Returns ``{key: {"mean": ..., "std": ..., "ci95_low": ..., "ci95_high": ...}}``
    for each requested key.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = []
        for result in per_seed_results:
            val = result.get(key)
            if isinstance(val, (int, float)):
                values.append(float(val))
        if not values:
            continue
        n = len(values)
        m = fmean(values)
        s = stdev(values) if n > 1 else 0.0
        # 95% CI via t-distribution critical values for common n
        t_table = {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.228,
        }
        z = t_table.get(n, 1.96 if n >= 30 else 2.0)
        margin = z * s / math.sqrt(n) if n > 1 else 0.0
        ci_low = m - margin
        ci_high = m + margin
        # Bounded metrics (rates/accuracies in [0, 1]) must not report CI
        # outside [0, 1]; clip rather than emit nonsensical negative bounds.
        if values and min(values) >= 0.0 and max(values) <= 1.0:
            ci_low = max(0.0, ci_low)
            ci_high = min(1.0, ci_high)
        aggregated[key] = {
            "mean": m,
            "std": s,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "n_seeds": float(n),
        }
    return aggregated


def format_mean_std(mean: float, std: float) -> str:
    """Format a metric as 'mean ± std'."""
    if std < 0.01:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile (matches ``numpy.quantile`` default)."""
    if not sorted_values:
        raise ValueError("At least one value is required")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must lie in [0, 1]")

    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = fraction * (n - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[low])
    weight = pos - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)
