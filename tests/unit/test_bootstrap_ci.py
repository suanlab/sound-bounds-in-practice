"""Tests for the query-clustered bootstrap."""

from __future__ import annotations

from statistics import mean

from experiments.scripts.bootstrap_ci import cluster_bootstrap, percentile


def test_percentile_interpolates_between_order_statistics() -> None:
    """Percentiles are linearly interpolated, not nearest-rank."""
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_interval_brackets_the_point_estimate() -> None:
    """A percentile interval must contain the statistic it is built around."""
    rows = [{"query_id": i // 3, "x": float(i)} for i in range(30)]
    result = cluster_bootstrap(
        rows, lambda r: mean(row["x"] for row in r), replicates=500
    )
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert result["n_clusters"] == 10
    assert result["n_rows"] == 30


def test_clustering_widens_the_interval() -> None:
    """Resampling whole queries must not be narrower than resampling rows.

    This is the property the paper depends on: rows within a query are
    duplicates of one another here, so treating them as independent would
    understate the spread. Clustering by the real query id must therefore give
    an interval at least as wide as clustering by a unique per-row id.
    """
    clustered = [{"query_id": i // 10, "x": float(i // 10)} for i in range(100)]
    per_row = [{"query_id": i, "x": float(i // 10)} for i in range(100)]

    wide = cluster_bootstrap(
        clustered, lambda r: mean(row["x"] for row in r), replicates=800
    )
    narrow = cluster_bootstrap(
        per_row, lambda r: mean(row["x"] for row in r), replicates=800
    )

    assert wide["ci_high"] - wide["ci_low"] > narrow["ci_high"] - narrow["ci_low"]


def test_is_deterministic_given_a_seed() -> None:
    """Committed intervals must be reproducible."""
    rows = [{"query_id": i // 2, "x": float(i)} for i in range(20)]
    statistic = lambda r: mean(row["x"] for row in r)  # noqa: E731
    first = cluster_bootstrap(rows, statistic, replicates=200, seed=7)
    second = cluster_bootstrap(rows, statistic, replicates=200, seed=7)
    assert first == second
