from __future__ import annotations

from typing import Any

import gymnasium as gym

from latency_ml.envs.latency_env import LatencyHoldEnv
from latency_ml.timing import NATIVE_DT_ENV_NS

CARTPOLE_DT_NS: int = 20000000
LUNARLANDER_DT_NS: int = 20000000

_TASK_SPECS: dict[str, dict[str, Any]] = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "obs_dim": 4,
        "n_actions": 2,
        "dt_ns": CARTPOLE_DT_NS,
        "kwargs": {},
    },
    "lunarlander": {
        "env_id": "LunarLander-v3",
        "obs_dim": 8,
        "n_actions": 4,
        "dt_ns": LUNARLANDER_DT_NS,
        "kwargs": {"continuous": False, "gravity": -10.0, "enable_wind": False},
    },
}


class UnsupportedTaskError(ValueError):
    pass


def known_tasks() -> tuple[str, ...]:
    return tuple(_TASK_SPECS)


def task_spec(task: str) -> dict[str, Any]:
    if task not in _TASK_SPECS:
        raise UnsupportedTaskError(f"unknown task {task!r}; known tasks: {', '.join(known_tasks())}")
    return _TASK_SPECS[task]


def make_base_env(task: str, render_mode: str | None = None, **overrides: Any) -> gym.Env[Any, Any]:
    spec = task_spec(task)
    kwargs = dict(spec["kwargs"])
    kwargs.update(overrides)
    return gym.make(spec["env_id"], render_mode=render_mode, **kwargs)


def probe_dt_env_ns(task: str) -> int:
    env = make_base_env(task)
    try:
        unrewrapped = env.unwrapped
        tau = getattr(unrewrapped, "tau", None)
        if tau is not None:
            return round(float(tau) * 1000000000)
        fps = getattr(unrewrapped, "FPS", None)
        if fps is not None:
            return round(1000000000 / float(fps))
        metadata_fps = env.metadata.get("render_fps") if env.metadata else None
        if metadata_fps:
            return round(1000000000 / float(metadata_fps))
    finally:
        env.close()
    return task_spec(task)["dt_ns"]


def probe_env_dims(task: str) -> tuple[int, int]:
    env = make_base_env(task)
    try:
        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n) if isinstance(env.action_space, gym.spaces.Discrete) else int(getattr(env.action_space, "shape", (0,))[0])
        return (obs_dim, n_actions)
    finally:
        env.close()


def make_latency_env(task: str, latency_ns: int, *, dt_env_ns: int | None = None, **overrides: Any) -> LatencyHoldEnv:
    dt = dt_env_ns if dt_env_ns is not None else probe_dt_env_ns(task)
    return LatencyHoldEnv(make_base_env(task, **overrides), latency_ns=latency_ns, dt_env_ns=dt)


__all__ = [
    "NATIVE_DT_ENV_NS",
    "UnsupportedTaskError",
    "make_base_env",
    "make_latency_env",
    "probe_dt_env_ns",
    "probe_env_dims",
    "task_spec",
]
