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
#      checkpoints (*.pt/*.pth) and OS artefacts; finance code and Phase 4
#      shared-trunk scaffolding are excluded too (out of scope, see
#      MIGRATION.md).
#   2. removes the placeholder prototype/will-d/.gitkeep once real files land.
#   3. strips `#` comments from the imported `.py` files (docstrings kept;
#      ruff/mypy directives like `# noqa`, `# type:`, `# pragma:` kept).
#   4. regenerates prototype/will-d/SHA256SUMS so a rerun is verifiable.
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
  --exclude='src/latency_ml/models/shared_trunk.py' \
  --exclude='src/latency_ml/experiments/shared_runner.py' \
  --exclude='src/latency_ml/analysis/shared_analysis.py' \
  --exclude='src/latency_ml/analysis/transfer_init.py' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='.DS_Store' \
  --exclude='.gitkeep' \
  --exclude='REPRODUCE.sh' \
  --exclude='SHA256SUMS' \
  --exclude='MIGRATION.md' \
  "$SRC/" "$DEST/"

rm -f "$DEST/.gitkeep"

# Belt-and-braces: drop out-of-scope trees even if a previous import predates
# the excludes above.
rm -rf \
  "$DEST/src/latency_ml/finance" \
  "$DEST/configs/finance.yaml" \
  "$DEST/configs/finance_v2b.yaml" \
  "$DEST/configs/finance_v2b_amzn.yaml" \
  "$DEST/src/latency_ml/models/shared_trunk.py" \
  "$DEST/src/latency_ml/experiments/shared_runner.py" \
  "$DEST/src/latency_ml/analysis/shared_analysis.py" \
  "$DEST/src/latency_ml/analysis/transfer_init.py"

# Strip `#` comments from imported Python files. Token-based (never touches
# `#` inside strings); keeps docstrings, shebangs, encoding cookies and
# tool directives (`noqa`, `type:`, `fmt:`, `isort:`, `pragma:`).
python3 - "$DEST" <<'PYEOF'
import io
import sys
import tokenize
from pathlib import Path

DEST = Path(sys.argv[1])
KEPT_FIRST_WORDS = {"noqa", "type", "fmt", "isort", "pragma"}


def keep_comment(text: str, row: int) -> bool:
    if row == 1 and text.startswith("#!"):
        return True
    body = text[1:].strip()
    if "coding" in body and ("-*-" in text or "coding:" in body or "coding=" in body):
        return True
    first = body.split()[0].rstrip(":") if body.split() else ""
    return first in KEPT_FIRST_WORDS


changed = 0
for path in sorted(DEST.rglob("*.py")):
    src = path.read_bytes().decode("utf-8")
    lines = src.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        print(f"skip (untokenizable): {path.relative_to(DEST)}: {exc}")
        continue
    edits = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        (row, col) = tok.start
        if keep_comment(tok.string, row):
            continue
        edits.append((row, col))
    if not edits:
        continue
    for row, col in edits:
        line = lines[row - 1]
        ending = "\n" if line.endswith("\n") else ""
        lines[row - 1] = line[:col].rstrip() + ending
    path.write_bytes("".join(lines).encode("utf-8"))
    changed += 1
print(f"comment-stripped {changed} files")
PYEOF

# Canonicalize with the repo toolchain when available (same ruff CI uses).
# Keeps reruns byte-identical to the committed state.
REPO_ROOT="$(cd "$DEST/../.." && pwd)"
if command -v uv >/dev/null 2>&1 && [ -f "$REPO_ROOT/pyproject.toml" ]; then
  (cd "$REPO_ROOT" && uv run --no-sync --frozen ruff check --fix --quiet prototype/will-d \
    && uv run --no-sync --frozen ruff format --quiet prototype/will-d) \
    || echo "warning: ruff normalization failed; output may differ from canonical form" >&2
else
  echo "warning: uv unavailable; skipping ruff canonicalization" >&2
fi

# Reproducible checksum manifest (excludes itself + the two meta files).
(cd "$DEST" && find . -type f \
  ! -name 'SHA256SUMS' ! -name 'MIGRATION.md' ! -name 'REPRODUCE.sh' \
  ! -path './.git/*' ! -path './.ruff_cache/*' ! -path './.pytest_cache/*' \
  ! -path './.mypy_cache/*' ! -path '*/__pycache__/*' \
  ! -name '.DS_Store' ! -name '*.pyc' \
  | sort | xargs shasum -a 256 > SHA256SUMS)

echo "done: $(wc -l < "$DEST/SHA256SUMS") files, see $DEST/SHA256SUMS"
echo "rerun check: git status --porcelain -- prototype/will-d"
