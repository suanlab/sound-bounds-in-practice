"""Regenerate the human-readable results snapshot from canonical result JSON.

This rewrites ``docs/paper_results_snapshot.md`` directly from the *same*
``results/*.json`` files that produce the canonical paper tables (see
``docs/REPRODUCE.md`` for the table -> command -> JSON map). It therefore
cannot drift from the paper the way the old hand-curated snapshot did
(AUDIT finding 4).

The snapshot is a convenience mirror, **not** a source of truth: the canonical
artifacts are ``docs/vldb_2027_vision/paper.tex`` and ``docs/REPRODUCE.md``.
Missing inputs are reported, not fatal.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical inputs (mirrors docs/REPRODUCE.md).
INPUTS: dict[str, Path] = {
    "bound": Path("results/bound_verification/bound_validity.json"),
    "shift": Path("results/shift/shift_results.json"),
    "ablation": Path("results/ablation/ablation.json"),
    "real": Path("results/real_data/job_medium_5seed.json"),
    "runtime": Path("results/scalability/runtime.json"),
    "cert": Path("results/certificate_upgrade/cert_upgrade.json"),
    "lower": Path("results/lower_certificate/lower_cert.json"),
    "learned": Path("results/learned_baseline/learned_baseline.json"),
    "tau": Path("results/tau_sweep/tau_sweep.json"),
    "diversity": Path("results/config_diversity/config_diversity.json"),
}

_METHOD_LABELS = {
    "safe_ceg_gnn": "Safe-CEG (GNN)",
    "safe_ceg_mlp": "Safe-CEG (MLP)",
    "safe_ceg_bucketed": "Safe-CEG (bucketed)",
    "ml_only_gnn": "ML-only (GNN)",
    "fixed_optimistic": "Fixed optimistic",
    "pes_only": "Pessimistic only",
    "independence": "Independence (AVI)",
    "oracle_best": "Oracle-best",
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("missing input: %s", path)
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _num(value: object) -> str:
    """Format a number compactly (commas for large ints, else a few sig figs)."""
    if not isinstance(value, (int, float)):
        return str(value)
    number = float(value)
    if number != number:  # NaN
        return "n/a"
    if abs(number) >= 1e6:
        return f"{number:.2e}"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}"


def _int(value: object) -> str:
    """Format a count as a comma-grouped integer (e.g. 5.0 -> ``5``)."""
    if isinstance(value, (int, float)) and value == value:
        return f"{round(float(value)):,}"
    return str(value)


def _stat(node: Any, *keys: str) -> object:
    """Walk a nested dict by ``keys``; return the leaf (or its ``mean``)."""
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    if isinstance(node, dict) and "mean" in node:
        return node["mean"]
    return node


def _pct(value: object) -> str:
    if isinstance(value, (int, float)) and value == value:
        return f"{float(value) * 100:.0f}%"
    return "-"


def _section_bound(payload: dict[str, Any]) -> list[str]:
    lines = ["## Bound validity (Table 1)", ""]
    lines.append(
        f"- Queries tested: {_int(payload.get('total_queries'))}; "
        f"violations: {_int(payload.get('num_violations'))}; "
        f"valid: {float(payload.get('valid_rate', 0.0)) * 100:.1f}%; "
        f"max true/bound ratio: {_num(payload.get('max_ratio'))}"
    )
    per_family = payload.get("per_family", {})
    if isinstance(per_family, dict) and per_family:
        lines += [
            "",
            "| Family | Queries | Violations | Valid |",
            "|---|---:|---:|---:|",
        ]
        for family, info in per_family.items():
            tested = info.get("tested", info.get("queries", "?"))
            viol = info.get("violations", "?")
            valid = info.get("valid_rate", info.get("valid"))
            valid_pct = (
                f"{float(valid) * 100:.0f}%" if isinstance(valid, (int, float)) else "?"
            )
            lines.append(f"| {family} | {_int(tested)} | {_int(viol)} | {valid_pct} |")
    return lines


def _section_ablation(payload: dict[str, Any]) -> list[str]:
    lines = [f"## Ablation (Tables 3-4, {_int(payload.get('n_seeds'))} seeds)", ""]
    for split_key, title in (
        ("in_distribution", "In-distribution"),
        ("cross_workload", "Cross-workload"),
    ):
        split = payload.get(split_key, {})
        if not isinstance(split, dict) or not split:
            continue
        lines += [
            f"### {title} (paper Tables 3-4 report the mean column)",
            "",
            "| Method | Mean q-error | Median q-error | Fallback |",
            "|---|---:|---:|---:|",
        ]
        for method, label in _METHOD_LABELS.items():
            if method not in split:
                continue
            mean = _stat(split[method], "mean")
            median = _stat(split[method], "median")
            fallback = _stat(split[method], "fallback_rate")
            lines.append(
                f"| {label} | {_num(mean)} | {_num(median)} | {_pct(fallback)} |"
            )
        lines.append("")
    return lines


def _section_shift(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"## Distribution shift (Table 2, {_int(payload.get('n_seeds'))} seeds)",
        "",
        "| Scenario | Test set | Median q-error | Fallback |",
        "|---|---|---:|---:|",
    ]
    for scenario, data in payload.get("scenarios", {}).items():
        for test_name, metrics in data.get("test_on", {}).items():
            median = _stat(metrics, "median")
            fallback = _stat(metrics, "fallback_rate")
            lines.append(
                f"| {scenario} | {test_name} | {_num(median)} | {_pct(fallback)} |"
            )
    return lines


def _section_real(payload: dict[str, Any]) -> list[str]:
    agg = payload.get("aggregated", {})
    accuracy = _stat(agg, "config_accuracy")
    return [
        f"## Real data: {payload.get('dataset', '?')} "
        f"(Table 5, {_int(payload.get('n_seeds'))} seeds, "
        f"{_int(payload.get('num_queries'))} queries)",
        "",
        f"- Safe-CEG mean q-error: {_num(_stat(agg, 'mean'))}; "
        f"pessimistic: {_num(_stat(agg, 'pes_mean'))}; "
        f"oracle: {_num(_stat(agg, 'oracle_mean'))}; "
        f"config accuracy: {_pct(accuracy)}",
        "- Synthetic IMDb-like fixture; true cardinalities via exact join execution.",
    ]


def _section_runtime(payload: dict[str, Any]) -> list[str]:
    hw = payload.get("hardware", {})
    lines = [
        f"## Runtime (Table 6, {_int(payload.get('queries_per_size'))} queries/size)",
        "",
        f"- Hardware: {hw.get('platform', '?')}, CPython {hw.get('python', '?')}",
        "",
        "| Relations | 9-cfg (ms) | ML inf. (ms) | Bound (ms) |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| {_int(row.get('n_relations'))} "
            f"| {_num(_stat(row, 'all_9_configs_ms', 'median'))} "
            f"| {_num(_stat(row, 'ml_inference_ms', 'median'))} "
            f"| {_num(_stat(row, 'pessimistic_bound_ms', 'median'))} |"
        )
    return lines


def _section_cert(payload: dict[str, Any]) -> list[str]:
    agg = payload.get("aggregated", {})
    unsound = agg.get("geo_bound_violation_rate")
    unsound_pct = (
        f"{float(unsound) * 100:.1f}%" if isinstance(unsound, (int, float)) else "n/a"
    )
    return [
        "## Certificate upgrade: MOLP vs geometric (Table 7, CAUTIONARY)",
        "",
        f"- Geo fallback q-error {_num(agg.get('geo_fallback_qerr_median'))} vs "
        f"MOLP {_num(agg.get('molp_fallback_qerr_median'))} "
        f"({_num(agg.get('qerr_median_improvement_factor'))}x tighter), BUT",
        f"- **Geometric bound is unsound: violates true cardinality {unsound_pct}** "
        "of the time (MOLP 0%) on correlated cycles. Not a certificate.",
    ]


def _section_lower(payload: dict[str, Any]) -> list[str]:
    lines = [
        "## Lower-bound certificate (Section 5.9)",
        "",
        f"- Pessimistic (upper) q-error median: "
        f"{_num(payload.get('pessimistic_qerr_median'))}",
        "",
        "| Sample rate | l-hat median q-err | Sound LB validity |",
        "|---:|---:|---:|",
    ]
    for rate, info in payload.get("per_rate", {}).items():
        validity = info.get("sound_lb_validity")
        v = (
            f"{float(validity) * 100:.0f}%"
            if isinstance(validity, (int, float))
            else "?"
        )
        lines.append(f"| {rate} | {_num(info.get('lhat_qerr_median'))} | {v} |")
    return lines


def _section_learned(payload: dict[str, Any]) -> list[str]:
    agg = payload.get("aggregated", {})
    return [
        f"## Learned-regression baseline (Section 6, {len(payload.get('seeds', []))} "
        "seeds)",
        "",
        f"- Learned regressor median q-error {_num(agg.get('learned_median'))} vs "
        f"CEG oracle {_num(agg.get('oracle_median'))} vs "
        f"pessimistic {_num(agg.get('pessimistic_mean'))}.",
        "- Accuracy is not the differentiator; the sound guarantee is.",
    ]


def _section_tau(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("per_tau", [])
    qerrs = {round(float(r["mean_q_error"]), 3) for r in rows}
    invariant = "invariant" if len(qerrs) == 1 else "varying"
    lines = [
        f"## Safety-accuracy tradeoff: tau sweep (Section 5.6, "
        f"{_int(payload.get('n_seeds'))} seeds)",
        "",
        f"- On the vacuous-certificate (cyclic) regime, q-error is {invariant} "
        "across tau (fallback 0% throughout):",
        "",
        "| tau | Mean q-error | Fallback |",
        "|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tau']} | {_num(r['mean_q_error'])} | {_pct(r['mean_fallback'])} |"
        )
    return lines


def _section_diversity(payload: dict[str, Any]) -> list[str]:
    names = payload.get("config_names", [])
    lines = ["## Config diversity (Figure 1)", ""]
    for family, fractions in payload.get("oracle_best_fraction", {}).items():
        ranked = sorted(zip(names, fractions, strict=False), key=lambda kv: -kv[1])[:3]
        top = ", ".join(f"{n}={v:.2f}" for n, v in ranked if v > 0)
        lines.append(f"- **{family}**: {top}")
    return lines


_SECTIONS = [
    ("bound", _section_bound),
    ("ablation", _section_ablation),
    ("shift", _section_shift),
    ("real", _section_real),
    ("runtime", _section_runtime),
    ("cert", _section_cert),
    ("lower", _section_lower),
    ("learned", _section_learned),
    ("tau", _section_tau),
    ("diversity", _section_diversity),
]


def render_snapshot(payloads: dict[str, dict[str, Any] | None]) -> str:
    """Render the full markdown snapshot from loaded canonical payloads."""
    out = [
        "# Safe-CEG Results Snapshot (AUTO-GENERATED)",
        "",
        "> **Generated** by `experiments/scripts/export_paper_tables.py` from the "
        "canonical `results/*.json` files.",
        "> This is a convenience mirror and **not** a source of truth. The "
        "canonical artifacts are",
        "> `docs/vldb_2027_vision/paper.tex` and `docs/REPRODUCE.md`. Do not "
        "hand-edit numbers here.",
        "",
    ]
    for key, renderer in _SECTIONS:
        payload = payloads.get(key)
        if payload is None:
            out += [f"<!-- {key}: input missing, section skipped -->", ""]
            continue
        out.extend(renderer(payload))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/paper_results_snapshot.md")
    )
    args = parser.parse_args()

    payloads = {key: _load(path) for key, path in INPUTS.items()}
    markdown = render_snapshot(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
