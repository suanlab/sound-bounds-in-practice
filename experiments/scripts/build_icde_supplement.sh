#!/usr/bin/env bash
# Build the ICDE 2027 supplemental zip (code + results + provenance docs).
# Usage (repo root): bash experiments/scripts/build_icde_supplement.sh [OUT_DIR]
set -euo pipefail
OUT_DIR="${1:-freeze}"
NAME="icde2027_supplement"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
mkdir -p "$DEST/docs/icde_2027"
rsync -a --exclude '__pycache__' --exclude '*.pt' --exclude '*.ckpt' \
    --exclude 'pg_injection/build' --exclude 'audit_paper_numbers.py' \
    --exclude 'build_supplement.sh' --exclude 'freeze_submission.sh' \
    src experiments tests results figures "$DEST/"
cp pyproject.toml requirements.txt requirements-lock.txt LICENSE "$DEST/"
mkdir -p "$DEST/LICENSES" && cp LICENSES/LpBound-MIT.txt "$DEST/LICENSES/"
cp docs/references.bib docs/c4_preregistration.md docs/c1_preregistration.md \
    docs/AUDIT_REPAIR_LOG.md "$DEST/docs/"
cp docs/icde_2027/{paper.tex,paper.pdf,paper.bbl,CLAIMS_EVIDENCE.md,REPRODUCE.md} "$DEST/docs/icde_2027/"
cp docs/icde_2027/SUPPLEMENT_README.md "$DEST/README.md"
# Predictions frozen before the post-hoc mechanism analyses, their results, and
# the held-out pre-registration with its addendum (each with its SHA-256 file).
cp docs/icde_2027/{PREDICTIONS.md,PREDICTIONS.sha256,PREDICTIONS_RESULTS.md,HELDOUT_RENDERER.sha256} "$DEST/docs/icde_2027/"
cp docs/icde_2027/PREREGISTRATION_JOBRANGE{.md,.sha256,_ADDENDUM.md,_ADDENDUM.sha256,_ADDENDUM2.md,_ADDENDUM2.sha256} "$DEST/docs/icde_2027/"
if find "$DEST" \( -name '*.pt' -o -name '*.ckpt' -o -name 'postgresql-13.1*' \) | grep -q .; then
    echo "excluded artefact leaked into the supplement" >&2; exit 1
fi
# Only provenance logs, the frozen registration and a cited venue may name the earlier draft.
if grep -rIl -i 'edbt' "$DEST" | grep -v -E 'references.bib|c4_preregistration.md|c1_preregistration.md|AUDIT_REPAIR_LOG.md|SUPERSEDED.md|build_icde_supplement.sh' | grep -q .; then
    echo "unexpected EDBT reference in the supplement:" >&2
    grep -rIl -i 'edbt' "$DEST" | grep -v -E 'references.bib|c4_preregistration.md|c1_preregistration.md|AUDIT_REPAIR_LOG.md|SUPERSEDED.md|build_icde_supplement.sh' >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
(cd "$STAGE" && zip -qr -X "$NAME.zip" "$NAME")
mv "$STAGE/$NAME.zip" "$OUT_DIR/"
sha256sum "$OUT_DIR/$NAME.zip"
unzip -l "$OUT_DIR/$NAME.zip" | tail -1
