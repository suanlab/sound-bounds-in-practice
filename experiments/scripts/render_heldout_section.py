"""Render the paper's held-out paragraph from the confirmatory results.

The wording was fixed before the JOB-range results existed (2026-10-20); only
the numbers and the per-test verdicts (``confirmed`` / ``falsified`` /
``inconclusive`` / ``not testable``, computed by ``heldout_jobrange analyze``)
are filled in. The audit checks that the paper contains exactly this text.

Usage::

    python -m experiments.scripts.render_heldout_section
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS = Path("results/heldout_jobrange/confirmatory_results.json")


def pct(value: float) -> str:
    """A share as the paper writes it."""
    return f"{100 * value:.1f}\\%"


def exclusions(excluded: dict[str, int]) -> str:
    """The registered exclusions that removed queries, in registered order."""
    names = {
        "final_not_listed": "full query not in the sub-plan list",
        "missing_count": "a sub-plan without an exact count",
        "missing_estimate": "a missing in-context estimate",
        "missing_plan": "a missing native plan",
        "plan_outside_list": "a native plan outside the sub-plan list",
        "fewer_than_3": "fewer than three relations",
    }
    parts = [f"{n} for {names[k]}" for k, n in excluded.items() if n]
    return "; ".join(parts) if parts else "none"


def render(data: dict[str, Any]) -> str:
    """The held-out paragraph."""
    post = data["a_posteriori"]
    s = post["shares"]
    t = data["tests"]
    changed = data["plan_changed_by_k"]
    if t["C7"] == "not testable":
        c7 = (
            "planning on the true sizes lowered $C_{out}$ by only "
            f"${data['truth_gain']:.2f}\\times$, below the registered $1.05\\times$, "
            "so C7 was not testable"
        )
    else:
        shares = data["share_of_truth_gain_log"]
        c7 = (
            "both guards recovered "
            f"{pct(shares['2'])} of the true-size $C_{{out}}$ gain at $k=2$ and "
            f"{pct(shares['16'])} at $k=16$ (C7, at least half and less than half: "
            f"{t['C7']})"
        )
    testable = [v for v in t.values() if v != "not testable"]
    return (
        "To test the post-hoc findings on data we had not seen, we pre-registered "
        "seven predictions before measuring a seeded random sample of 200 of the "
        "artifact's 1{,}000 JOB-range queries (range filters over the JOB-light "
        "tables; the registration, two addenda filed before the measurements and "
        "the code hashes are in the supplement). The registered exclusions removed "
        f"{exclusions(post['excluded'])}, leaving {post['n_queries']} queries for the "
        f"certificates and {data['n_sweep_queries']} for the tightness sweep. After "
        "execution, with the final result in $C_{out}$, PostgreSQL's plan is "
        f"certified within $2\\times$ on {pct(s['cout|none'])} of queries with no "
        f"sample (C1, predicted at least 70\\%: {t['C1']}) and on "
        f"{pct(s['cout|sample'])} with the sampled floor (C2, at least 80\\%: "
        f"{t['C2']}); without the final result on {pct(s['cout_minus_final|none'])} "
        f"without a sample (C3, at most 5\\%: {t['C3']}) and on "
        f"{pct(s['cout_minus_final|sample'])} with it (C4, between 10\\% and 70\\%: "
        f"{t['C4']}); the certificate's soundness check held on every query (C5: "
        f"{t['C5']}). Under both synthetic guards on PostgreSQL's own estimates, the "
        f"share of plans that change falls from {pct(changed['2'])} at $k=2$ to "
        f"{pct(changed['16'])} at $k=16$ (C6, at least halving and never rising by "
        f"more than 5 points: {t['C6']}); {c7}. Of the {len(testable)} testable "
        f"predictions, {sum(v == 'confirmed' for v in testable)} were confirmed."
    )


def main() -> None:
    """CLI entry point."""
    logger.info("%s", render(json.loads(RESULTS.read_text())))


if __name__ == "__main__":
    main()
