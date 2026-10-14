from __future__ import annotations

from typing import Any

import gymnasium as gym

from latency_ml.envs.latency_env import LatencyHoldEnv, LatencyHoldEnvT
from latency_ml.timing import NATIVE_DT_ENV_NS

CARTPOLE_DT_NS: int = 20000000
LUNARLANDER_DT_NS: int = 20000000
CARTPOLE_MAX_EPISODE_STEPS: int = 500
LUNARLANDER_SOLVED_SCORE: float = 200.0


class UnsupportedTaskError(ValueError):
    pass


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


def known_tasks() -> tuple[str, ...]:
    return tuple(_TASK_SPECS)


def task_spec(task: str) -> dict[str, Any]:
    if task not in _TASK_SPECS:
        raise UnsupportedTaskError(
            f"unknown task {task!r}; known tasks: {', '.join(known_tasks())}"
        )
    return _TASK_SPECS[task]


def make_base_env(task: str, **overrides: Any) -> gym.Env[Any, Any]:
    spec = task_spec(task)
    kwargs = dict(spec["kwargs"])
    kwargs.update(overrides)
    return gym.make(spec["env_id"], render_mode=None, **kwargs)


def probe_dt_env_ns(task: str) -> int:
    spec = task_spec(task)
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
    return int(spec["dt_ns"])


def probe_env_dims(task: str) -> tuple[int, int]:
    env = make_base_env(task)
    try:
        obs_dim = int(env.observation_space.shape[0])
        n_actions = (
            int(env.action_space.n)
            if isinstance(env.action_space, gym.spaces.Discrete)
            else int(getattr(env.action_space, "shape", (0,))[0])
        )
        return (obs_dim, n_actions)
    finally:
        env.close()


def probe_max_episode_steps(task: str) -> int | None:
    env = make_base_env(task)
    try:
        spec = env.spec
        return int(spec.max_episode_steps) if spec and spec.max_episode_steps else None
    finally:
        env.close()


def make_latency_env(
    task: str, latency_ns: int, *, dt_env_ns: int | None = None, **overrides: Any
) -> LatencyHoldEnvT:
    dt = dt_env_ns if dt_env_ns is not None else probe_dt_env_ns(task)
    env = make_base_env(task, **overrides)
    return LatencyHoldEnv(env, latency_ns=latency_ns, dt_env_ns=dt)


def expected_dt_env_ns(task: str) -> int:
    return int(task_spec(task)["dt_ns"])


def assert_dt_env_ns_matches(task: str) -> int:
    probed = probe_dt_env_ns(task)
    expected = expected_dt_env_ns(task)
    assert probed == expected, (
        f"{task}: probed timestep {probed} ns but expected {expected} ns. Gymnasium's integration timestep changed, which invalidates the latency-to-action-step mapping (PLAN.md section 5.1)."
    )
    return probed


def make_vector_env(task: str, num_envs: int, latency_ns: int) -> gym.vector.SyncVectorEnv:
    pass

    def make_one() -> LatencyHoldEnvT:
        return make_latency_env(task, latency_ns)

    return gym.vector.SyncVectorEnv([make_one for _ in range(num_envs)])


__all__ = [
    "CARTPOLE_DT_NS",
    "CARTPOLE_MAX_EPISODE_STEPS",
    "LUNARLANDER_DT_NS",
    "LUNARLANDER_SOLVED_SCORE",
    "NATIVE_DT_ENV_NS",
    "LatencyHoldEnv",
    "UnsupportedTaskError",
    "assert_dt_env_ns_matches",
    "expected_dt_env_ns",
    "known_tasks",
    "make_base_env",
    "make_latency_env",
    "make_vector_env",
    "probe_dt_env_ns",
    "probe_env_dims",
    "probe_max_episode_steps",
    "task_spec",
]
