# will-d latency prototype

Control-latency PPO sweeps (CartPole, LunarLander, and a shared trunk) under
ReLU vs. Hartley's bounded ReLU: one policy trains per latency condition at
equal simulated task time, and effectiveness is compared across the grid.

# How to train

```bash
cd prototype/will-d
uv sync --extra prototype-will-d
uv run latency-ml train --phase cartpole --activation capped-relu \
  --sweep-id demo --steps 200000 --seeds 0 --latencies 20000000
```

`--phase` selects `cartpole`, `lunarlander`, or `shared`; `--activation`
selects `relu` or `capped-relu`. Omit `--steps` / `--latencies` / `--seeds`
for the full grid. Training evaluates automatically and writes
`results/<sweep-id>/evaluation.json`.

# How to view

```bash
cd prototype/will-d
./view.sh
```

Then open http://localhost:8000. The site builds from
`results/<sweep-id>/evaluation.json`; `./view.sh --build-only` rebuilds `site/`
without serving.
