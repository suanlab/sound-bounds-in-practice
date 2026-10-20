from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize when Safe-CEG helps relative to ML-only and CEG_M."
    )
    parser.add_argument(
        "--shift-result",
        type=Path,
        default=Path("results/shift/shift_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/safe_ceg_help_analysis.md"),
    )
    args = parser.parse_args()

    payload = json.loads(args.shift_result.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Expected shift result payload to be a JSON object")

    markdown = render_markdown(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)


def render_markdown(payload: dict[str, object]) -> str:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Shift result payload is missing scenarios")

    lines = [
        "# Safe-CEG Help Analysis",
        "",
        (
            "This file summarizes when Safe-CEG changes the final answer, "
            "when fallback is actually helpful, and when the method is merely "
            "safe but inaccurate."
        ),
        "",
        (
            "| Scenario | Test workload | Mean q-error | Fallback rate | "
            "Final differs from CEG_M | Final better than ML-only | "
            "Final better than CEG_M | Helpful fallback | "
            "ML better than CEG_M | Interpretation |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for scenario_name, scenario_payload in scenarios.items():
        if not isinstance(scenario_payload, dict):
            continue
        test_on = scenario_payload.get("test_on")
        if not isinstance(test_on, dict):
            continue
        for test_name, metrics in test_on.items():
            if not isinstance(metrics, dict):
                continue
            interpretation = classify(metrics)
            lines.append(
                (
                    "| {scenario} | {test} | {mean:.4f} | {fallback:.4f} | "
                    "{diff_pes:.4f} | {better_ml:.4f} | {better_pes:.4f} | "
                    "{helpful_fb:.4f} | {ml_better_pes:.4f} | {interp} |"
                ).format(
                    scenario=scenario_name,
                    test=test_name,
                    mean=float(metrics.get("mean", 0.0)),
                    fallback=float(metrics.get("fallback_rate", 0.0)),
                    diff_pes=float(
                        metrics.get("final_differs_from_pessimistic_rate", 0.0)
                    ),
                    better_ml=float(metrics.get("final_better_than_ml_rate", 0.0)),
                    better_pes=float(
                        metrics.get("final_better_than_pessimistic_rate", 0.0)
                    ),
                    helpful_fb=float(metrics.get("helpful_fallback_rate", 0.0)),
                    ml_better_pes=float(
                        metrics.get("ml_better_than_pessimistic_rate", 0.0)
                    ),
                    interp=interpretation,
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            (
                "- `fallback-dominant recovery`: Safe-CEG improves on ML-only "
                "mainly by rejecting unsafe proposals and returning CEG_M."
            ),
            (
                "- `safe but inaccurate`: the final estimate stays on the "
                "optimistic branch, but remains inaccurate without violating "
                "the safety threshold."
            ),
            "- `mixed`: neither pattern dominates strongly.",
        ]
    )
    return "\n".join(lines) + "\n"


def classify(metrics: dict[str, object]) -> str:
    fallback = cast("float", metrics.get("fallback_rate", 0.0))
    helpful_fallback = cast("float", metrics.get("helpful_fallback_rate", 0.0))
    better_pes = cast("float", metrics.get("final_better_than_pessimistic_rate", 0.0))

    if fallback >= 0.75 and helpful_fallback >= 0.75:
        return "fallback-dominant recovery"
    if fallback == 0.0 and better_pes <= 0.1:
        return "safe but inaccurate"
    return "mixed"


if __name__ == "__main__":
    main()
