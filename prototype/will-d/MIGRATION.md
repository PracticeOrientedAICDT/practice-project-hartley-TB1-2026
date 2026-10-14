# will-d prototype: origin and curation log

Standalone control-latency prototype (CartPole/LunarLander PPO sweeps under
Hartley's bounded ReLU). It began as an import from a Week 2 experiment repo
and has since been curated in place — **nothing syncs back anywhere**:
there is no import script, no manifest, and no dependency on the source tree.
Edit files here directly.

## Origin (history, not a live link)

- Source was `/Users/ky23812/Documents/Practise Projects/Week 2/Exp`
  at `42b1ef3` (`latency-ml v2`, bounded ReLU `clip 2.0`).
- Imported without `tests/`, `data/`, `results/`, caches, or checkpoints.
- `PROTOCOL.md` / `RESULTS.md` are the historical record (finance V2B/L1 and
  Phase 4 findings included); the code below no longer implements those branches.

## What was removed during curation

- Finance: `src/latency_ml/finance/` (14 files), `configs/finance*.yaml`,
  `FinanceConfig` (+ its `ProjectConfig` field), and all finance CLI
  subcommands (`historical`, `diagnose-target`, `stage-a`, `selectivity`,
  `paired-lookback`, `mechanism`, `capacity`) with their parser blocks and
  dispatch branches.
- Phase 4 shared-trunk scaffolding: `models/shared_trunk.py`,
  `experiments/shared_runner.py`, `analysis/shared_analysis.py`,
  `analysis/transfer_init.py`, the `shared-trunk` CLI subcommand, and
  `scripts/render_videos.py` (Phase-4-only renderer).
- Kept deliberately: `analysis/parity.py` (the kept control sweep
  `experiments/runner.py` imports `task_protocol` from it);
  `analysis/power.py` / `analysis/timescale.py` (generic stats and
  control-task dynamics, not phase-specific).
- Dangling but harmless (snapshot is not executed by repo CI):
  `models/__init__.py` re-exports `build_shared_trunk*`,
  `models/architectures.py` keeps `build_shared_trunk`,
  `rl/task_policy.py` keeps a lazy shared-trunk facade, and
  `configs/base.yaml` keeps an inert `finance:` block.

## Comment stripping

All `#` comments were stripped from the `.py` files (token-based, so `#`
inside strings is untouched). Kept: docstrings, shebangs, encoding cookies,
and tool directives (`noqa`, `type:`, `fmt:`, `isort:`, `pragma:`). The tree
was then canonicalized with the repo toolchain (`ruff check --fix`,
`ruff format`). `Makefile` help comments are functional (the `help` target
greps them) and were left alone.

## Files that are curated here (not imported)

- `README.md` — the simple How-to-train / How-to-view front page.
- `view.sh` — builds `site/` and serves it on localhost.
- `scripts/build_site.py` — stdlib-only static site generator over
  `results/*/evaluation.json` (placeholder page when nothing trained yet).
- `.gitignore` — training/site artefacts (`results/`, `site/`, `data/`,
  `*.pt`, `*.pth`, `*.mp4`) stay local.
- `Makefile` — `site` / `view` targets route through `./view.sh`; the
  `videos` target is gone with its script.

## Repo wiring

- `pyproject.toml` / `uv.lock` / `.python-version` pin this folder's own
  stack (`torch==2.14.0`, `gymnasium==1.3.0`, …). They are **not** wired into
  the repo-root `uv sync --frozen` / CI — root CI uses the root lock. Train
  from inside this folder (`uv sync --all-extras`, see README).
- Root CI scope: `ruff check .` / `ruff format --check` cover this folder
  and pass; `pytest tests/unit|integration` do not touch it (no tests here).
  `ty` cannot resolve the snapshot's third-party imports (`torch`, …) from
  the repo-root environment, so the repo-root `pyproject.toml` sets
  `[tool.ty.src] exclude = ["prototype"]`; ruff still lints it.
