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

## What was copied (50 files, ~0.8 MB)

`src/latency_ml/` (packages: `analysis/`, `envs/`, `experiments/`,
`hardware/`, `models/`, `rl/`, plus `cli.py`, `config.py`, `seeding.py`,
`timing.py`), `configs/` (`base.yaml`, `cartpole.yaml`, `lunarlander.yaml`),
`scripts/` (`build_site.py`, `render_videos.py`), `docs/ENVIRONMENT.md`,
`PROTOCOL.md`, `RESULTS.md`, `README.md`, `Makefile`, `pyproject.toml`,
`uv.lock`, `.python-version`, `.gitignore`.

Full manifest with sha256: `SHA256SUMS` (50 lines).

## What was removed after import

- `src/latency_ml/finance/` (LOB/market/replay/target/selectivity/stage_a —
  14 files) and `configs/finance.yaml`, `configs/finance_v2b.yaml`,
  `configs/finance_v2b_amzn.yaml`. Out of scope for this prototype.
  `REPRODUCE.sh` excludes them so a rerun keeps this state.
- Left as-is (verbatim, non-functional without the package): finance
  references in `cli.py` (finance subcommands/imports), `config.py`
  (`FinanceConfig`), and the `finance:` block in `configs/base.yaml`.
  `PROTOCOL.md`/`RESULTS.md` still document the finance (V2B/L1) findings
  as the historical record.

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
- Root CI scope note: `ruff check .` / `ruff format --check` cover
  `prototype/will-d/` and pass; `pytest tests/unit|integration` do not touch
  it (no tests added, existing suites unaffected — 1 pass each, pre-existing).
  `ty` cannot resolve the snapshot's third-party imports (`torch`, …) from
  the repo-root environment, so the repo-root `pyproject.toml` sets
  `[tool.ty.src] exclude = ["prototype"]` (exploratory spikes carry their own
  stacks; ruff still lints them). Verification at fix time, repo root via
  `uv run`: `ruff check .` — pass; `ruff format --check .` — 62 files
  formatted; `ty check` — pass.

## Results pointers (not copied)

- Primary finding: 25.0% out-of-sample edge decay (0.1→100 ms) vs 14.6%
  oracle decay; capture −3.41; lift +0.0082 unresolved (2 SE 0.041);
  L200−L20 Δlift −0.0443, 95% CI [−0.0651, −0.0235]. (`RESULTS.md`, `PROTOCOL.md` §12–§16.)
- Regenerate: `make check-all`, then the sweep/evaluate/shared-trunk
  commands in `RESULTS.md` §8 (130 training cells, ~2 h 20 m on 15 cores).
