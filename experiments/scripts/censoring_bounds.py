"""What the censored JOB queries could do to the reported cost ratio.

The real-JOB run evaluates 53 of 113 templates; 58 exceed the per-query budget
and 2 do not parse. The 58 are not a random sample --- they are the largest
joins --- so the reported median cost ratio is a statistic about the queries
that *finished*, and quoting it as a statistic about JOB would be wrong.

This script says how wrong it could be, in two ways.

**Worst-case (Manski) bounds.** With 58 of 113 values unknown, the median of the
full set is not point-identified: if every censored query sat below the smallest
observed ratio the median would fall to that value, and if every one sat above
the largest it would rise to that. These bounds assume nothing and are
correspondingly wide; we report them so the reader sees the identification
limit rather than a false precision.

**Direction of the bias.** Assumption-free bounds are uninformative, but the
sign of the bias is not. If the cost ratio grows with query size on the queries
we can see, then the censored (larger) queries plausibly sit above the observed
ones, and the reported median understates the full-benchmark value. We test that
association directly, with a query-level bootstrap interval on the correlation.

Usage::

    PYTHONHASHSEED=0 python -m experiments.scripts.censoring_bounds
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from statistics import median
from typing import Any

from experiments.scripts.bootstrap_ci import JOB_LOWER_DRAWS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (ties averaged)."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else float("nan")


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _measured_block(path: Path, censored: int) -> dict[str, Any] | None:
    """Manski bounds on the measured one-draw ratios, if the re-timing exists."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    ratios = sorted(float(r["ratio_1draw"]) for r in payload["per_template"])
    return {
        "source": str(path),
        "evaluated": len(ratios),
        "observed_median_cost_ratio": median(ratios),
        "share_of_evaluated_above_one": sum(1 for r in ratios if r > 1.0) / len(ratios),
        "manski_lower": median([ratios[0]] * censored + ratios),
        "manski_upper": median(ratios + [ratios[-1]] * censored),
        "note": (
            "One-draw certificate re-timed on the same templates "
            "(verify_draw_cost); the censored count is the as-run one, since "
            "the re-timing covered exactly the templates that finished."
        ),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("results/job_full/job_full.json")
    )
    parser.add_argument(
        "--one-draw",
        type=Path,
        default=Path("results/job_full/job_full_1draw_verification.json"),
        help="measured one-draw re-timing (``verify_draw_cost``); optional",
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/job_full/censoring_bounds.json")
    )
    args = parser.parse_args()

    run = json.loads(args.run.read_text())
    observed = run["per_query"]
    total = run["templates_total"]
    censored = len(run["skipped"].get("timeout", []))
    unparsed = len(run["skipped"].get("unparsed", []))

    ratios = [r["cost_ratio"] for r in observed if r["cost_ratio"] == r["cost_ratio"]]
    sizes = [r["full_count_ms"] for r in observed]
    n_eval = len(ratios)

    # Manski bounds: place every censored value at the extreme, then re-median.
    evaluable = n_eval + censored
    lower_set = [min(ratios)] * censored + sorted(ratios)
    upper_set = sorted(ratios) + [max(ratios)] * censored
    manski_low = median(lower_set)
    manski_high = median(upper_set)

    rho = spearman(sizes, ratios)
    rng = random.Random(args.seed)
    draws: list[float] = []
    for _ in range(args.replicates):
        idx = [rng.randrange(n_eval) for _ in range(n_eval)]
        value = spearman([sizes[i] for i in idx], [ratios[i] for i in idx])
        if value == value:
            draws.append(value)

    share_above_one = sum(1 for r in ratios if r > 1.0) / n_eval
    payload: dict[str, Any] = {
        "experiment": "censoring_sensitivity_real_job",
        "templates_total": total,
        "evaluated": n_eval,
        "censored_at_budget": censored,
        "unparsed": unparsed,
        "observed_median_cost_ratio": median(ratios),
        "observed_min": min(ratios),
        "observed_max": max(ratios),
        "share_of_evaluated_above_one": share_above_one,
        "manski_bounds_on_median_over_evaluable": {
            "n_evaluable": evaluable,
            "lower": manski_low,
            "upper": manski_high,
            "note": (
                "Assumption-free: every censored query placed at the observed "
                "minimum (lower) or maximum (upper). Wide by construction; "
                "reported to show what the data alone can identify."
            ),
        },
        "per_draw": {
            "draws_aggregated_in_run": JOB_LOWER_DRAWS,
            "observed_median_cost_ratio": median(ratios) / JOB_LOWER_DRAWS,
            "share_of_evaluated_above_one": sum(
                1 for r in ratios if r / JOB_LOWER_DRAWS > 1.0
            )
            / n_eval,
            "manski_lower": manski_low / JOB_LOWER_DRAWS,
            "manski_upper": manski_high / JOB_LOWER_DRAWS,
            "note": (
                "The real-JOB run aggregated three sampling draws per query "
                "(--lower-seeds 1 2 3); the certificate's cost is linear in the "
                "draws, so these are the as-run figures divided by the draw "
                "count, the quantity comparable to the one-draw STATS-CEB run."
            ),
        },
        "one_draw_measured": _measured_block(args.one_draw, censored),
        "size_association": {
            "statistic": "Spearman(full exact-count time, cost ratio)",
            "rho": rho,
            "ci_low": percentile(draws, 0.025),
            "ci_high": percentile(draws, 0.975),
            "n": n_eval,
            "interpretation": (
                "A positive rho means the certificate gets relatively more "
                "expensive on larger joins, so the censored (larger) queries "
                "would sit above the observed ones and the reported median "
                "understates the full-benchmark value."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "evaluated %d/%d (censored %d, unparsed %d)", n_eval, total, censored, unparsed
    )
    logger.info(
        "observed median cost ratio %.3f  (range %.3f--%.3f, %.1f%% above 1.0)",
        median(ratios),
        min(ratios),
        max(ratios),
        100.0 * share_above_one,
    )
    logger.info(
        "Manski bounds on the median over the %d evaluable: [%.3f, %.3f]",
        evaluable,
        manski_low,
        manski_high,
    )
    logger.info(
        "Spearman(size, cost ratio) = %+.3f  [%+.3f, %+.3f]  (n=%d)",
        rho,
        percentile(draws, 0.025),
        percentile(draws, 0.975),
        n_eval,
    )
    logger.info("-> %s", args.output)


if __name__ == "__main__":
    main()
