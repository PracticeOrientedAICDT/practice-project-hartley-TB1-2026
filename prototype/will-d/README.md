# will-d latency prototype

Control-latency PPO sweeps (CartPole, LunarLander) under Hartley's bounded
ReLU: one policy trains per latency condition at equal simulated task time,
and effectiveness is compared across the grid. Background and findings live
in `PROTOCOL.md` / `RESULTS.md`; import provenance in `MIGRATION.md`.

# How to train

```bash
cd prototype/will-d
uv sync --all-extras
uv run latency-ml sweep --config configs/cartpole.yaml --sweep-id demo \
  --latencies 20000000 --seeds 0 --steps 200000
uv run latency-ml evaluate --config configs/cartpole.yaml --sweep-id demo
```

Drop `--steps` / `--latencies` / `--seeds` for the full grid (50 cells per
task, hours on many cores). LunarLander needs the `box2d` extra, included in
`--all-extras`.

# How to view

```bash
cd prototype/will-d
./view.sh
```

Then open http://localhost:8000. The site builds from
`results/<sweep-id>/evaluation.json`; with nothing trained yet it shows a
placeholder. `./view.sh --build-only` rebuilds `site/` without serving.
