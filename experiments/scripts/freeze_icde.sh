#!/usr/bin/env bash
# ICDE 2027 freeze: build the paper, run every gate, build the supplement, and
# write a tarball + per-file SHA256 manifest. Refuses to freeze while the paper
# holds a TODO(author) or PENDING marker (pass --dry-run to run every gate and
# build everything anyway, without the final tarball name claiming a freeze).
# Usage (repo root): bash experiments/scripts/freeze_icde.sh [--dry-run]
set -euo pipefail
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
export LC_ALL=C PYTHONHASHSEED=0
PAPER=docs/icde_2027
PREFIX=$([[ $DRY -eq 1 ]] && echo icde2027_dryrun || echo icde2027_submission)
STAMP="${PREFIX}_$(date +%Y-%m-%d_%H%M)"

fail() { echo "FREEZE ABORTED: $*" >&2; exit 1; }

if grep -nE 'TODO\(author\)|PENDING' "$PAPER/paper.tex"; then
    if [[ $DRY -eq 1 ]]; then
        echo "warning: open TODO/PENDING markers (dry run)"
    else
        fail "open TODO/PENDING markers in $PAPER/paper.tex"
    fi
fi

(
    cd "$PAPER"
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    bibtex paper >/dev/null
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
)
LOG="$PAPER/paper.log"
[[ $(grep -a -c '^! ' "$LOG") -eq 0 ]] || fail "LaTeX errors"
[[ $(grep -a -c 'Overfull' "$LOG") -eq 0 ]] || fail "overfull boxes"
[[ $(grep -a -ci 'undefined' "$LOG") -eq 0 ]] || fail "undefined references"
[[ $(pdffonts "$PAPER/paper.pdf" | grep -ci 'type 3') -eq 0 ]] || fail "Type 3 fonts"
PAGES=$(pdfinfo "$PAPER/paper.pdf" | awk '/^Pages:/{print $2}')
REF_PAGE=0
for ((i = 1; i <= PAGES; i++)); do
    # Capture first: grep -q would close the pipe early and trip pipefail.
    page_text=$(pdftotext -f "$i" -l "$i" "$PAPER/paper.pdf" -)
    if grep -qE '^R ?EFERENCES$' <<<"$page_text"; then REF_PAGE=$i; break; fi
done
# The body may use 12 pages; references may start at the top of page 13 at most.
[[ $REF_PAGE -ge 1 && $REF_PAGE -le 13 ]] || fail "References start on page $REF_PAGE"
first_page=$(pdftotext -f 1 -l 1 "$PAPER/paper.pdf" -)
grep -q '\[Experiment, Analysis, and Benchmark\]' <<<"$first_page" \
    || fail "title lacks [Experiment, Analysis, and Benchmark]"

python3 -m experiments.scripts.audit_icde_numbers | tail -1
pytest_out=$(python3 -m pytest tests/ -q 2>&1) || fail "tests"
tail -1 <<<"$pytest_out"
ruff check experiments/ tests/ >/dev/null || fail "ruff"

bash experiments/scripts/build_icde_supplement.sh freeze

tar --exclude='*.pt' --exclude='*.ckpt' --exclude='__pycache__' \
    --exclude='experiments/pg_injection/build' \
    --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
    --exclude="$PAPER/paper.aux" --exclude="$PAPER/paper.log" \
    --exclude="$PAPER/paper.out" --exclude="$PAPER/paper.blg" \
    -czf "freeze/$STAMP.tar.gz" docs results experiments src tests figures \
    CLAUDE.md REVIEW.md TOP_PLAN.md AUDIT.md AUDIT_2026-06-20.md \
    pyproject.toml requirements.txt requirements-lock.txt
tar -tzf "freeze/$STAMP.tar.gz" | grep -v '/$' | xargs -d '\n' sha256sum >"freeze/$STAMP.sha256"
sha256sum -c --quiet "freeze/$STAMP.sha256" || fail "manifest does not verify"
sha256sum "freeze/$STAMP.tar.gz" "$PAPER/paper.pdf" freeze/icde2027_supplement.zip \
    | tee "freeze/$STAMP.upload.sha256"
echo "references start on page $REF_PAGE of $PAGES; frozen as freeze/$STAMP.tar.gz"
