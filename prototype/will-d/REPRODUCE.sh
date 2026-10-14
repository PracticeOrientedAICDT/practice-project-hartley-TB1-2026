#!/usr/bin/env bash
# Reproducible migration: Week 2 Exp -> prototype/will-d (no tests).
#
# Usage:
#   SRC="/Users/ky23812/Documents/Practise Projects/Week 2/Exp" bash prototype/will-d/REPRODUCE.sh
#   # or: SRC=<path-to-week2-exp> bash prototype/will-d/REPRODUCE.sh
#
# What it does:
#   1. rsyncs the working tree at $SRC into prototype/will-d/, excluding
#      tests/, data/, results/, logs/, site/, .venv/, .git/ and all caches,
#      checkpoints (*.pt/*.pth) and OS artefacts.
#   2. removes the placeholder prototype/will-d/.gitkeep once real files land.
#   3. regenerates prototype/will-d/SHA256SUMS so a rerun is verifiable.
#
# Idempotent: rerunning with the same $SRC yields zero file diffs
# (verify with `git status --porcelain -- prototype/will-d`).
set -euo pipefail

SRC="${SRC:-/Users/ky23812/Documents/Practise Projects/Week 2/Exp}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SRC/PROTOCOL.md" ] || [ ! -d "$SRC/src/latency_ml" ]; then
  echo "error: SRC does not look like Week 2 Exp: $SRC" >&2
  exit 1
fi

echo "SRC : $SRC"
echo "DEST: $DEST"
echo "SRC commit: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "SRC status: $(git -C "$SRC" status -sb 2>/dev/null | head -n 5 | tr '\n' ';')"

rsync -a --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ty/' \
  --exclude='.uv/' \
  --exclude='tests/' \
  --exclude='data/' \
  --exclude='results/' \
  --exclude='logs/' \
  --exclude='site/' \
  --exclude='results/videos/' \
  --exclude='src/latency_ml/finance/' \
  --exclude='configs/finance.yaml' \
  --exclude='configs/finance_v2b.yaml' \
  --exclude='configs/finance_v2b_amzn.yaml' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='.DS_Store' \
  --exclude='.gitkeep' \
  --exclude='REPRODUCE.sh' \
  --exclude='SHA256SUMS' \
  --exclude='MIGRATION.md' \
  "$SRC/" "$DEST/"

rm -f "$DEST/.gitkeep"

# Reproducible checksum manifest (excludes itself + the two meta files).
(cd "$DEST" && find . -type f \
  ! -name 'SHA256SUMS' ! -name 'MIGRATION.md' ! -name 'REPRODUCE.sh' \
  ! -path './.git/*' ! -path './.ruff_cache/*' ! -path './.pytest_cache/*' \
  ! -path './.mypy_cache/*' ! -path '*/__pycache__/*' \
  ! -name '.DS_Store' ! -name '*.pyc' \
  | sort | xargs shasum -a 256 > SHA256SUMS)

echo "done: $(wc -l < "$DEST/SHA256SUMS") files, see $DEST/SHA256SUMS"
echo "rerun check: git status --porcelain -- prototype/will-d"
