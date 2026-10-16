# will-d latency prototype

Control-latency PPO sweeps (CartPole, LunarLander, shared trunk) under ReLU vs.
capped ReLU. Latency is injected by holding each action for the number of env
frames that best covers the requested control latency; the five-point sweep
grid runs 20-100 ms at 10 seeds per (latency, activation).

## Train

```bash
uv sync --extra prototype-will-d
uv run latency-ml train --phase cartpole --activation capped-relu --sweep-id cp-capped --workers 12 --threads 1
```

Phases: `cartpole`, `lunarlander` (dedicated policies), `shared` (one trunk with
per-task adapter/head, both tasks per iteration). Sweeps resume by default:
completed `<latency>ns/seed<n>/policy.pt` checkpoints are skipped.

## Push phase (non-saturated measure)

Return saturates once a policy survives the whole episode, so after training,
measure **time to stability**: kick the pole with a random impulse
(theta, theta-dot, x, x-dot) and time how long the policy needs to bring it back
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
`--mode push` replays push-phase trial 0 with a visible kick (shown from push
tables). Existing files are skipped.

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
comparison card per task (relu in blue vs capped-relu in orange on the same
chart) for return-vs-latency (with seed spread) and time-to-stability-vs-latency;
clicking a table row opens both rollout videos for that latency.
**Details** documents the learning setup, model sizes, and evaluation protocols.

## Layout

```
configs/    base.yaml (shared PPO/latency grid) + per-task overrides
src/latency_ml/
  cli.py              train / push / render / view commands
  config.py           YAML config loading with extends chains
  envs/               latency hold wrapper + task registry
  models/             dedicated MLP actor-critic + shared trunk
  rl/                 PPO trainer + duration-aware rollout buffer
  experiments/        sweeps, evaluation, shared-trunk training, push phase, render
scripts/build_site.py static site generator
```
