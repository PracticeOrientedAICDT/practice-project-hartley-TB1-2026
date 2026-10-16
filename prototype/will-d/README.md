# will-d latency prototype

Control-latency PPO sweeps (CartPole, LunarLander, shared trunk) under ReLU vs.
capped ReLU vs. Hartree (private hmlp-ti emulator, cartpole dedicated only).
Latency is injected by holding each action for the number of env
frames that best covers the requested control latency; the five-point sweep
grid runs 20-100 ms at 10 seeds per (latency, activation).

## Train

```bash
uv sync --extra prototype-will-d
uv run latency-ml train --phase cartpole --activation capped-relu --sweep-id cp-capped --workers 12 --threads 1
uv run latency-ml train --phase cartpole --activation hartree --sweep-id cp-hartree --workers 4 --threads 1
```

Phases: `cartpole`, `lunarlander` (dedicated policies), `shared` (one trunk with
per-task adapter/head, both tasks per iteration). Sweeps resume by default:
completed `<latency>ns/seed<n>/policy.pt` checkpoints are skipped.
Activations: `relu`, `capped-relu`, `hartree` (cartpole dedicated only).

## Hartree arm (private emulator)

The Hartree column trains a `HartreePolicy` around the private
hmlp-ti 1.1.0 emulator. Only the actor is constrained through analog
physics: obs 4 -> Linear encoder -> sigmoid*255 IN codes (256)
-> frozen `DeviceTorch` (f64, draws=False, inference-only) -> y volts (256)
-> Linear actor head (2). The critic is a normal clean MLP
(obs -> 128 -> 128 -> 1, ReLU) so value learning is never hobbled by the
hardware; it trains normally and is never deployed. PPO trains encoder +
actor head + critic; Hartree codes are frozen and never updated.

The emulator is private and never committed to this public repo:

- copy `hmlp_ti/` (+ `wheels/*.whl`) into `prototype/will-d/vendor/` (gitignored)
- install locally: `uv pip install --python .venv/bin/python vendor/wheels/hartree_net-*.whl vendor/wheels/hartree_emu-*.whl`
- without `vendor/`, `import latency_ml.models.hartree_backend` still works
  (`HARTREE_AVAILABLE=False`); only `--activation hartree` fails with a clear message
- `results/`, `site/`, `vendor/` are all gitignored, so trained Hartree
  checkpoints and videos stay local

Smoke (fast, proves the wiring; ~3 min/cell): `--latencies 20000000 --seeds 0 --steps 4096 --eval-episodes 20`.
Full sweep uses the default 4M steps/cell from `configs/base.yaml`.

## Push phase (non-saturated measure)

Return saturates once a policy survives the whole episode, so after training,
measure **time to stability**: shove the cart with a brief external force
(±15 N for 6 env frames / 120 ms, added to the policy's own force while it
keeps acting) and time how long the policy needs to bring the pole back
into the stable band (|theta| < 0.12 rad, |theta-dot| < 1.0 rad/s, held for 20
env steps). The band sits just outside the nominal hover envelope of a trained
policy, so recovery time carries signal while the episode return does not.

```bash
uv run latency-ml push --sweep-id cp-capped --activation capped-relu
uv run latency-ml push --sweep-id sh-capped --kind shared --activation capped-relu
```

Writes `results/<sweep>/push/<kind>/<latency>ns/<seed>.json` trials plus
`results/<sweep>/push/<kind>/summary.json` aggregates (mean time to stability
in env frames and ms, success rate per latency).

## Render

Record one rollout video per sweep / task / latency with the median-seed
policy. `--mode control` records plain episodes (shown from control tables),
`--mode push` replays a push-phase trial (first knock-out + recovery) with a
visible force pulse after a stable hover (shown from push tables). All
CartPole videos share the same rules: a per-frame STABLE/UNSTABLE pill, and a
fall plays out past the 12-degree termination until the pole is horizontal.
Existing files are skipped.

```bash
uv run latency-ml render                  # all sweeps, both modes
uv run latency-ml render --sweep-id cp-relu --mode push
```

Writes `results/<sweep>/videos/<task>_<latency>ms_seed<seed>.mp4`.

## View

```bash
uv run latency-ml view --build-only   # writes site/index.html (+ site/videos/)
uv run latency-ml view                # build + serve on http://localhost:8000
```

The site is a single offline HTML page with two tabs. **Charts** shows one
comparison card per task (relu in blue vs capped-relu in orange vs hartree in
green on the same chart) for return-vs-latency (with seed spread) and
time-to-stability-vs-latency; clicking a table row opens the rollout videos
for that latency. **Details** documents the learning setup, model sizes, and
evaluation protocols.

## Layout

```
configs/    base.yaml (shared PPO/latency grid) + per-task overrides
src/latency_ml/
  cli.py              train / push / render / view commands
  config.py           YAML config loading with extends chains
  envs/               latency hold wrapper + task registry
  models/             dedicated MLP actor-critic + shared trunk + hartree (private vendor/)
  rl/                 PPO trainer + duration-aware rollout buffer
  experiments/        sweeps, evaluation, shared-trunk training, push phase, render
scripts/build_site.py static site generator
vendor/               private hmlp-ti copy (gitignored, never committed)
```
