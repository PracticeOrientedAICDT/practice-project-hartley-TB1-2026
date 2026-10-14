import argparse, json, time, platform
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from actor import NumpyActor
from envs.beam_env import BeamEnv, BeamConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--calls", type=int, default=10000)
    p.add_argument("--out", default="latency.json")
    a = p.parse_args()
    if a.calls < 100:
        p.error("--calls must be >=100")
    torch.set_num_threads(1)
    model = PPO.load(a.model, device="cpu")
    meta = json.loads(Path(a.model).with_suffix(".json").read_text())
    env = BeamEnv(BeamConfig(**meta["config"]))
    obs, _ = env.reset(seed=123)
    actor = NumpyActor.from_ppo(model)
    # Compare actual rollout observations, including disturbances and saturation.
    observations = []

    for _ in range(512):
        expected, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(actor(obs), expected, atol=2e-6, rtol=2e-5)
        observations.append(obs.copy())
        obs, _, term, trunc, _ = env.step(expected)
        if term or trunc:
            obs, _ = env.reset()
    results = {}

    for name, fn in [
        ("sb3_predict", lambda x: model.predict(x, deterministic=True)[0]),
        ("numpy_actor", actor),
    ]:
        for i in range(1000):
            fn(observations[i % len(observations)])
        times = []
        for i in range(a.calls):
            x = observations[i % len(observations)]
            start = time.perf_counter_ns()
            fn(x)
            times.append((time.perf_counter_ns() - start) / 1000)
        results[name] = {
            k: float(v)
            for k, v in zip(
                ["p50_us", "p95_us", "p99_us", "max_us"],
                [*np.percentile(times, [50, 95, 99]), max(times)],
            )
        }

    actor.save(Path(a.model).with_suffix(".npz"))
    report = dict(
        model=a.model,
        calls=a.calls,
        platform=platform.platform(),
        processor=platform.processor(),
        numpy=np.__version__,
        torch=torch.__version__,
        results=results,
        warning="Host timing only; excludes sensor/transport/DAC. Not a hard-real-time guarantee.",
    )
    
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
