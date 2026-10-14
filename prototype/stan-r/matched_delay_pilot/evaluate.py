"""Paired-seed latency sweep; includes uncontrolled baseline and post-kick metrics."""
import argparse, csv, json
from pathlib import Path
from dataclasses import replace
import numpy as np
from envs.beam_env import BeamEnv, BeamConfig


def rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    reward = 0.0
    xs = []
    amps = []
    tracked = []
    crossings = 0
    prev = None
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        reward += r
        if info["kicked"]:
            x = info["x"]
            xs.append(abs(x))
            amps.append(info["amplitude"])
            tracked.append(info["on_target"])
            if prev is not None and prev * x < 0:
                crossings += 1
            prev = x
        if term or trunc:
            break
    # Require remaining below amplitude threshold for the rest of the episode.
    bad = np.flatnonzero(np.asarray(amps) >= env.cfg.target_radius)
    settled = int(bad[-1]) + 1 if len(bad) else 0
    settle_us = (
        (settled + 1) / env.cfg.sample_hz * 1e6
        if settled < len(amps) and not term
        else float("nan")
    )
    return dict(
        reward=reward,
        mean_error=float(np.mean(xs)) if xs else float("nan"),
        max_error=float(max(xs)) if xs else float("nan"),
        tracking=float(np.mean(tracked)) if tracked else float("nan"),
        crossings=crossings,
        survival=env.t,
        lost=float(term),
        settle_us=settle_us,
        final_amplitude=amps[-1] if amps else float("nan"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="PPO paths and/or zero")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--obs-delays", type=int, nargs="+", default=[0, 1, 2, 5, 8])
    p.add_argument(
        "--act-delays", type=float, nargs="+", default=[0, 1, 2, 5, 7.56, 10, 16]
    )
    p.add_argument(
        "--inference-us",
        type=float,
        default=0.0,
        help="Additional modeled compute latency; do not double count",
    )
    p.add_argument("--out", default="evaluation.csv")
    a = p.parse_args()
    if a.episodes < 1 or a.inference_us < 0:
        p.error("Invalid episodes/latency")
    rows = []
    for path in a.models:
        if path == "zero":
            base = BeamConfig()
            policy = lambda obs: np.zeros(1, np.float32)
        else:
            import torch
            from stable_baselines3 import PPO

            torch.set_num_threads(1)
            model = PPO.load(path, device="cpu")
            base = BeamConfig(
                **json.loads(Path(path).with_suffix(".json").read_text())["config"]
            )
            policy = lambda obs: model.predict(obs, deterministic=True)[0]
        for od in a.obs_delays:
            for ad in a.act_delays:
                total = ad + a.inference_us * base.sample_hz / 1e6
                cfg = replace(
                    base,
                    obs_delay_range=(od, od),
                    action_delay_range=(total, total),
                    max_obs_delay=max(base.max_obs_delay, od),
                    max_action_delay=max(base.max_action_delay, total),
                )
                env = BeamEnv(cfg)
                episodes = [rollout(env, policy, 10000 + i) for i in range(a.episodes)]
                row = dict(
                    model=path,
                    obs_delay_steps=od,
                    action_delay_steps=total,
                    obs_delay_us=od / base.sample_hz * 1e6,
                    action_delay_us=total / base.sample_hz * 1e6,
                )
                for key in episodes[0]:
                    vals = np.array([e[key] for e in episodes])
                    valid = vals[np.isfinite(vals)]
                    row[key] = float(valid.mean()) if len(valid) else float("nan")
                    row[key + "_std"] = (
                        float(valid.std(ddof=1)) if len(valid) > 1 else float("nan")
                    )
                row["settled_fraction"] = float(
                    np.mean([np.isfinite(e["settle_us"]) for e in episodes])
                )
                rows.append(row)
                env.close()
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} conditions to {a.out}")


if __name__ == "__main__":
    main()
