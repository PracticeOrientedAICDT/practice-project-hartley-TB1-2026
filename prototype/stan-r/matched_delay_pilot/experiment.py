"""Shared controller loading, environment factories, rollouts and timing."""
from dataclasses import replace
from pathlib import Path
import json, platform, time
import numpy as np
from actor import NumpyActor
from envs.accelerator_env import AcceleratorConfig, AcceleratorEnv
from envs.laser_target_env import LaserTargetConfig, LaserTargetEnv

BUDGETS = [0, 10, 50, 100, 250, 500, 1000, 2000, 2800, 5000]
ARCHS = {
    "paper12": [12],
    "paper16": [16],
    "small": [8, 8],
    "medium": [64, 64],
    "max": [256, 256, 256],
}
ENVIRONMENTS = ("accelerator", "laser")


def config_class(environment):
    return {"accelerator": AcceleratorConfig, "laser": LaserTargetConfig}[environment]


def environment_class(environment):
    return {"accelerator": AcceleratorEnv, "laser": LaserTargetEnv}[environment]


def make_config(environment, **kwargs):
    return config_class(environment)(**kwargs)


def make_env(environment, config):
    return environment_class(environment)(config)


def model_stem(path):
    p = Path(path)
    return p.with_suffix("") if p.suffix in (".zip", ".npz") else p


def load_controller(path, backend="numpy", environment=None):
    if path == "zero":
        environment = environment or "accelerator"
        cfg = make_config(environment)
        shape = environment_class(environment)(cfg).action_space.shape
        return (
            lambda obs: np.zeros(shape, np.float32),
            cfg,
            {
                "name": "zero",
                "environment": environment,
                "trained_inference_ns": np.nan,
            },
        )
    stem = model_stem(path)
    meta = json.loads(stem.with_suffix(".json").read_text())
    saved_environment = meta.get("environment", "accelerator")
    if environment is not None and environment != saved_environment:
        raise ValueError(f"Checkpoint is for {saved_environment}, not {environment}")
    cfg = config_class(saved_environment)(**meta["config"])
    if backend == "numpy":
        return NumpyActor.load(stem.with_suffix(".npz")), cfg, meta
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    if backend == "sb3-cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but CUDA is unavailable")
    model = PPO.load(str(stem), device="cuda" if backend == "sb3-cuda" else "cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0], cfg, meta


def benchmark(policy, observations, calls=2000, warmup=200):
    if calls < 1 or not observations:
        raise ValueError("Need observations and positive calls")
    for i in range(warmup):
        policy(observations[i % len(observations)])
    times = np.empty(calls)
    for i in range(calls):
        obs = observations[i % len(observations)]
        start = time.perf_counter_ns()
        policy(obs)
        times[i] = time.perf_counter_ns() - start
    return dict(
        zip(
            ["p50_ns", "p95_ns", "p99_ns", "max_ns"],
            map(float, [*np.percentile(times, [50, 95, 99]), times.max()]),
        )
    )


def rollout(policy, cfg, seed=10000, record=False, environment=None):
    environment = environment or (
        "laser" if isinstance(cfg, LaserTargetConfig) else "accelerator"
    )
    env = make_env(environment, cfg)
    obs, info = env.reset(seed=seed)
    observations, trace, samples = [], [], []
    total = 0.0
    while True:
        observations.append(obs.copy())
        obs, reward, term, trunc, info = env.step(policy(obs))
        total += reward
        row = dict(
            info,
            step=getattr(env, "steps", getattr(env, "t", 0)),
            reward=reward,
            terminated=term,
            truncated=trunc,
        )
        if record:
            trace.append(row)
        # Accelerator excludes pre-kick state. Laser evaluates the entire tracking run.
        if environment == "laser" or info.get("kicked", False):
            samples.append(row)
        if term or trunc:
            break
    horizon = getattr(env, "steps", getattr(env, "t", 0))
    env.close()
    dist = np.asarray([x["distance"] for x in samples], float)
    amps = np.asarray([x["amplitude"] for x in samples], float)
    bad = np.flatnonzero(dist >= cfg.target_radius)
    settle = int(bad[-1]) + 1 if len(bad) else 0
    settled = bool(len(dist) and settle < len(dist) and not term)
    metrics = dict(
        reward=total,
        mean_error=float(np.mean(dist)) if len(dist) else np.nan,
        rms_error=float(np.sqrt(np.mean(dist**2))) if len(dist) else np.nan,
        p95_amplitude=float(np.percentile(amps, 95)) if len(amps) else np.nan,
        final_amplitude=float(amps[-1]) if len(amps) else np.nan,
        tracking=float(np.mean(dist < cfg.target_radius)) if len(dist) else np.nan,
        settle_us=(settle + 1) / cfg.sample_hz * 1e6 if settled else np.nan,
        settled=float(settled),
        lost=float(term),
        survival=horizon,
    )
    if environment == "accelerator":
        valid = amps > max(cfg.noise_std, 1e-8)
        rate = np.nan
        if valid.sum() > 8:
            ts = np.arange(len(amps))[valid] / cfg.sample_hz * 1e6
            rate = float(-np.polyfit(ts, np.log(amps[valid]), 1)[0])
        metrics["damping_per_us"] = rate
    else:
        metrics["mean_speed"] = (
            float(np.mean([x["speed"] for x in samples])) if samples else np.nan
        )
    return metrics, trace, observations


def eval_config(cfg, inference_ns, transport_ns=None, inference_jitter_ns=None):
    transport = cfg.transport_ns if transport_ns is None else transport_ns
    kwargs = dict(
        inference_ns=float(inference_ns),
        inference_range_ns=None,
        transport_ns=float(transport),
    )
    if inference_jitter_ns is not None and hasattr(cfg, "inference_jitter_ns"):
        kwargs["inference_jitter_ns"] = float(inference_jitter_ns)
    if isinstance(cfg, AcceleratorConfig):
        kwargs["action_delay_range"] = (0.0, 0.0)
        kwargs["max_action_delay"] = max(
            cfg.max_action_delay,
            (float(inference_ns) + transport) * cfg.sample_hz * 1e-9,
        )
    else:
        kwargs["max_action_delay_steps"] = max(
            cfg.max_action_delay_steps,
            (
                float(inference_ns)
                + transport
                + 3 * getattr(cfg, "inference_jitter_ns", 0)
            )
            * cfg.sample_hz
            * 1e-9,
        )
    return replace(cfg, **kwargs)


def host_info():
    return dict(
        platform=platform.platform(),
        processor=platform.processor(),
        numpy=np.__version__,
        warning="Host Python timing is not a hard-real-time or hardware guarantee.",
    )
