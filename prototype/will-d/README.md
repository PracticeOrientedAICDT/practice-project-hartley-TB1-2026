# will-d latency prototype

Control-latency PPO sweeps (CartPole, LunarLander, shared trunk) under ReLU vs. capped ReLU.

Train:
```bash
cd prototype/will-d
uv sync --extra prototype-will-d
uv run python -m latency_ml.cli train --phase cartpole --activation capped-relu \
  --sweep-id demo --steps 200000 --seeds 0 --latencies 20000000
```
or from repo root:
```bash
uv run latency-ml train --phase cartpole --activation capped-relu --steps 200000
```

View:
```bash
uv run prototype/will-d/view.py
```
