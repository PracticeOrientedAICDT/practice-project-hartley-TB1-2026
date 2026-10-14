# Migration: Week 2 Exp → `prototype/will-d`

Standalone snapshot of the Week 2 experiment repo, imported verbatim
(working tree, not a squash) so the activation comparison stays attributable.
No `tests/` — per request this folder ships code + docs only.

## Source

- Path: `/Users/ky23812/Documents/Practise Projects/Week 2/Exp`
- Commit: `42b1ef320e2ad9c94b30a05ae13074c2e0f81c19` (`main`)
- Working-tree state at import: `M .gitignore, M Makefile, ?? scripts/`
  (uncommitted `results/videos/` + `site/` ignores, `videos`/`site` Makefile
  targets, and untracked `scripts/build_site.py`, `scripts/render_videos.py`
  are all included as copied).
- Upstream identity: `latency-ml v2`, Hartley bounded ReLU (`clip 2.0`,
  `src/latency_ml/models/activations.py`). See `PROTOCOL.md` §2/§8–§9 and
  `RESULTS.md` §1/§6 for the parity gate (tensor-for-tensor vs v1 checkpoint)
  and the inconclusive activation verdict.

## Destination

- Branch: `feature/will-d-week2`, off `dev` at `0c97c3e`
  (which includes latest `origin/main` `0c97c3e`; `main` was fast-forwarded
  `d9d2081..0c97c3e` then `dev` fast-forwarded `e7f985e..0c97c3e` before branching).
- Target: `prototype/will-d/` in
  `github.com/PracticeOrientedAICDT/practice-project-hartley-TB1-2026`
  (the GitHub repo, not the GitLab `hartley-prototype`).
- PR: draft, blank body, base `dev` (per repo rule `feature/* → dev`).

## What was copied (67 files, ~0.95 MB)

`src/latency_ml/` (all packages: `analysis/`, `envs/`, `experiments/`,
`finance/`, `hardware/`, `models/`, `rl/`, `cli.py`, `config.py`,
`seeding.py`, `timing.py`), `configs/` (6 yamls), `scripts/`
(`build_site.py`, `render_videos.py`), `docs/ENVIRONMENT.md`, `PROTOCOL.md`,
`RESULTS.md`, `README.md`, `Makefile`, `pyproject.toml`, `uv.lock`,
`.python-version`, `.gitignore`.

Full manifest with sha256: `SHA256SUMS` (67 lines).

## What was deliberately excluded

- `tests/` (entire — requested "no tests"; includes 4.6 MB `tests/fixtures/v1/`
  frozen anchors, so the parity gate cannot run from here).
- `data/` (LOBSTER sessions, gitignored at source, ~10 sessions).
- `results/` (25 MB; `*.pt` checkpoints gitignored at source and regenerable
  via `latency-ml sweep`; `ledger.json` progress state; `videos/`/`site/`
  derived artefacts — rebuild with `make videos` / `make site`).
- `logs/`, `site/`, `.venv/`, `.git/`, `__pycache__/`, `*.pyc`,
  `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.DS_Store`.
- Destination placeholder `prototype/will-d/.gitkeep` (removed on import).

The destination `.gitignore` does not ignore `*.pt`/`data/` — nothing of
that kind was copied, so no ignore change was needed. If checkpoints are
added later, add ignores then.

## Reproduce

```sh
SRC="/Users/ky23812/Documents/Practise Projects/Week 2/Exp" bash prototype/will-d/REPRODUCE.sh
git status --porcelain -- prototype/will-d   # expect only MIGRATION.md/SHA256SUMS diffs on first run, empty on rerun
```

Rerun is idempotent (`rsync --delete` + sorted `SHA256SUMS`).
`REPRODUCE.sh` skips itself, `MIGRATION.md` and `SHA256SUMS` during rsync
so metadata never contaminates the checksum manifest.

## Standalone vs repo wiring

- `pyproject.toml` / `uv.lock` / `.python-version` are kept as a **reference**
  for the snapshot's own pins (`torch==2.14.0`, `gymnasium==1.3.0`, …).
  They are **not** wired into the repo-root `uv sync --frozen` / CI —
  root CI uses the root lock. To run the snapshot standalone, copy this
  folder out and `uv sync --all-extras` inside it.
- Root CI scope note: `ruff check .` / `ty check` walk `prototype/will-d/`,
  but `pytest tests/unit|integration` do not (no tests added, existing suites
  unaffected). Verification at import (repo root, `uv run`):
  `ruff check prototype/will-d` — pass; `ruff format --check` — 57 files
  already formatted; `pytest tests/unit`, `pytest tests/integration` — pass
  (1 each, pre-existing); `ty check prototype/will-d` — 79 diagnostics
  (snapshot was written against `mypy --strict`, not `ty`; left verbatim —
  fixes belong in a follow-up).

## Results pointers (not copied)

- Primary finding: 25.0% out-of-sample edge decay (0.1→100 ms) vs 14.6%
  oracle decay; capture −3.41; lift +0.0082 unresolved (2 SE 0.041);
  L200−L20 Δlift −0.0443, 95% CI [−0.0651, −0.0235]. (`RESULTS.md`, `PROTOCOL.md` §12–§16.)
- Regenerate: `make check-all`, then the sweep/evaluate/shared-trunk
  commands in `RESULTS.md` §8 (130 training cells, ~2 h 20 m on 15 cores).
