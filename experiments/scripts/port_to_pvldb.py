"""Port the ICDE paper source to the official PVLDB Volume 20 template.

Reads ``docs/icde_2027/paper.tex`` and writes ``docs/vldb_2027/paper.tex``:
IEEEtran becomes ``acmart`` (sigconf, nonacm) with the mandatory PVLDB block,
the category tag moves to the end of the title as PVLDB requires, the author
block and abstract follow acmart's order, and the bibliography uses the ACM
style. The body text is copied unchanged.

Usage::

    python -m experiments.scripts.port_to_pvldb
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SOURCE = Path("docs/icde_2027/paper.tex")
TARGET = Path("docs/vldb_2027/paper.tex")
URL = "https://github.com/suanlab/sound-bounds-in-practice"

PREAMBLE = rf"""% === PVLDB Volume 20 (VLDB 2027): Experiment, Analysis & Benchmark ===
% Rules (vldb.org/2027/submission-guidelines.html, checked 2026-10-20):
%   * <= 12 pages excluding references; appendices and acknowledgements COUNT
%   * official PVLDB template (github.com/vldbproceedings/VLDB-Template, 39c95f5);
%     mandatory blocks rendered by \vldbtopmatter -- do not remove
%   * single-blind: names and affiliations on page 1
%   * title suffix "[Experiment, Analysis & Benchmark]" (removed on acceptance)
%   * EA&B: link to the full reproducibility package in the initial submission
%   * monthly deadline 1st 5pm PT, abstract by the 25th before; target 2026-12-01
% Generated from docs/icde_2027/paper.tex by experiments/scripts/port_to_pvldb.py;
% edit this file from now on.
% ================================================================================
\documentclass[sigconf, nonacm]{{acmart}}

%%% do not modify the following VLDB block %%
%%% VLDB block start %%%
\usepackage{{pvldb}}
%%% VLDB block end %%%

\renewcommand\vldbdoi{{XX.XX/XXX.XX}}
\renewcommand\vldbpages{{XXX-XXX}}
\renewcommand\vldbavailabilityurl{{{URL}}}

\usepackage{{booktabs}}

\begin{{document}}

\title{{Sound Cardinality Bounds in Practice: Guards, Estimates, and Plan
Certificates [Experiment, Analysis \& Benchmark]}}

\author{{Suan Lee}}
\orcid{{0000-0002-3047-1167}}
\affiliation{{%
  \institution{{Semyung University}}
  \city{{Jecheon}}
  \country{{Republic of Korea}}
}}
\email{{suanlee@semyung.ac.kr}}

"""


def main() -> None:
    """Write the PVLDB version of the paper (refuses to overwrite it)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an edited PVLDB paper"
    )
    args = parser.parse_args()
    if TARGET.exists() and not args.force:
        msg = f"{TARGET} exists and is edited by hand; pass --force to overwrite it"
        raise SystemExit(msg)
    src = SOURCE.read_text()
    abstract = src[src.index("\\begin{abstract}") : src.index("\\end{abstract}")]
    abstract += "\\end{abstract}\n"
    body_start = src.index("\\end{IEEEkeywords}") + len("\\end{IEEEkeywords}")
    body = src[body_start : src.index("\\bibliographystyle{IEEEtran}")]
    out = (
        PREAMBLE
        + abstract
        + "\n\\maketitle\n\n%%% do not modify the following VLDB block %%\n"
        "%%% VLDB block start %%%\n\\vldbtopmatter\n%%% VLDB block end %%%\n"
        + body
        + "\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{../references}\n\n"
        "\\end{document}\n"
    )
    TARGET.write_text(out)
    logger.info("wrote %s (%d characters)", TARGET, len(out))


if __name__ == "__main__":
    main()
