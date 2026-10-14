"""Measured task-response timescale for the control tasks, independent of PPO.

Nothing here imports the trainer, a policy, or a checkpoint. Only the raw
simulator is stepped, with the action held constant. The quantity is a property
of the environment's dynamics, not of the controller, which is what the
ratio hypothesis needs: ``tau_knee / T_task`` is only meaningful if ``T_task``
does not already depend on the policy that produced ``tau_knee``.

Definition (fixed before any latency result was inspected)
--------------------------------------------------------
From a fixed reset state, hold one discrete action and measure the time until
the continuous state has moved one unit away from where it started, where each
state variable is normalised by its own initial magnitude:

    T_task = dt * min { t : || (x_t - x_0) / max(|x_0|, 1) || >= 1 }

Reported per action, aggregated by the median over reset seeds. Median rather
than mean because the metric is a hitting time over a threshold and is therefore
skewed.

Why this definition and not the linearised one
----------------------------------------------
A spectral-radius e-folding time, ``T = dt / ln(rho)``, was implemented first and
is the textbook answer. It fails here for LunarLander. At reset the lander is in
near-free-fall, so its one-step Jacobian is a double integrator whose dominant
eigenvalue is 1 *exactly*; the measured ``rho`` came out 1.001-1.0014, and
``dt/ln(rho)`` amplifies that difference from roundoff into a 2-17 s timescale
with no physical content. CartPole, being already linear, gave a clean
256-268 ms, which is exactly the kind of asymmetry that makes a definition
unusable for a cross-task comparison. A first attempt to difference the
*observation* instead of the raw state was worse still (``rho`` in the
thousands), because LunarLander normalises its observation by the viewport scale
and the perturbation vanished into float32 rounding.

The hitting-time measure is well conditioned for both tasks, needs no
derivative, and reduces to the e-folding time in the linear limit.

Termination is suppressed during measurement
---------------------------------------------
Both tasks terminate under any constant action, and that termination is a
*reward-shaping convention* rather than a property of the physics: CartPole
declares failure when the pole leaves a 12-degree window, and the lander
declares it on ground contact. Measuring with termination active therefore
returns "how long a frozen policy survives", which folds the control problem
into the quantity meant to be independent of it. Suppressing the flag lets the
dynamics run, and makes the measurement comparable across tasks whose
termination rules have nothing in common. It also removes the earlier candidate
-- time for the state to deviate by one e-fold under a held action -- for the
same reason.

Both suppressions are measurement-only. The environments used for training and
evaluation are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]


DEFAULT_THRESHOLD: float = 1.0


DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


MAX_STEPS: int = 2000

_ENV_ID: dict[str, str] = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3"}

_ENV_KWARGS: dict[str, dict[str, Any]] = {
    "cartpole": {},
    "lunarlander": {"continuous": False, "gravity": -10.0, "enable_wind": False},
}

_NOOP_STEP: Any = None


def _cartpole_state(env: gym.Env[Any, Any]) -> FloatVec:
    unwrapped: Any = env.unwrapped
    return np.asarray(unwrapped.state, dtype=np.float64)


def _lunarlander_state(env: gym.Env[Any, Any]) -> FloatVec:
    """Continuous state only: pose, linear velocity, angle, angular velocity.

    The two leg-contact flags in the observation are discrete and are excluded.
    """
    unwrapped: Any = env.unwrapped
    lander = unwrapped.lander
    return np.array(
        [
            lander.position.x,
            lander.position.y,
            lander.linearVelocity.x,
            lander.linearVelocity.y,
            lander.angle,
            lander.angularVelocity,
        ],
        dtype=np.float64,
    )


_STATE_IO: dict[str, Callable[[gym.Env[Any, Any]], FloatVec]] = {
    "cartpole": _cartpole_state,
    "lunarlander": _lunarlander_state,
}


def _make(task: str, seed: int) -> gym.Env[Any, Any]:
    """Reset an environment with its termination flag suppressed."""
    env = gym.make(_ENV_ID[task], **_ENV_KWARGS[task])
    env.reset(seed=seed)
    env.reset(seed=seed)
    unwrapped: Any = env.unwrapped
    original_step = unwrapped.step

    def step_without_termination(action: Any) -> Any:
        obs, reward, _, truncated, info = original_step(action)
        return obs, reward, False, truncated, info

    unwrapped.step = step_without_termination
    return env


@dataclass(frozen=True)
class TaskTimescale:
    """Measured task-response timescale under one held action.

    Attributes:
        task: Task name.
        action: Held discrete action.
        efold_ns: Median over seeds of the time to unit normalised displacement.
        per_seed_ns: The same per seed, for spread.
        n_reached: Seeds that reached the threshold within ``MAX_STEPS``.
    """

    task: str
    action: int
    efold_ns: int
    per_seed_ns: tuple[int, ...]
    n_reached: int

    @property
    def efold_ms(self) -> float:
        return self.efold_ns / 1e6


def measure_efold_time(
    task: str,
    action: int,
    *,
    dt_ns: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    threshold: float = DEFAULT_THRESHOLD,
    max_steps: int = MAX_STEPS,
) -> TaskTimescale:
    """Time for the held action to displace the state by one normalised unit.

    Args:
        task: ``"cartpole"`` or ``"lunarlander"``.
        action: Discrete action to hold.
        dt_ns: Environment integration timestep in nanoseconds.
        seeds: Reset seeds; the reported value is their median.
        threshold: Normalised displacement defining the timescale.
        max_steps: Per-seed step cap.

    Returns:
        The measured timescale and its per-seed spread.

    Raises:
        ValueError: If the task is unknown.
    """
    if task not in _ENV_ID:
        raise ValueError(f"unknown task {task!r}; known: {', '.join(sorted(_ENV_ID))}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    getter = _STATE_IO[task]
    per_seed: list[int] = []
    for seed in seeds:
        env = _make(task, seed)
        try:
            start = getter(env)
            scale = np.maximum(np.abs(start), 1.0)
            steps_to_threshold: int | None = None
            for t in range(1, max_steps + 1):
                env.step(action)
                current = getter(env)
                if not np.all(np.isfinite(current)):
                    break
                if float(np.linalg.norm((current - start) / scale)) >= threshold:
                    steps_to_threshold = t
                    break
            if steps_to_threshold is not None:
                per_seed.append(steps_to_threshold * dt_ns)
        finally:
            env.close()

    if not per_seed:
        raise RuntimeError(
            f"no seed for task {task!r} action {action} reached the threshold "
            f"within {max_steps} steps; the timescale is undefined"
        )
    return TaskTimescale(
        task=task,
        action=action,
        efold_ns=int(np.median(per_seed)),
        per_seed_ns=tuple(per_seed),
        n_reached=len(per_seed),
    )


def measure_task_timescale(
    task: str,
    *,
    dt_ns: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[int, TaskTimescale]:
    """Timescale for every discrete action of a task.

    Per-action values are returned rather than averaged. The timescale is a
    property of the action-conditioned dynamics, and collapsing the spread would
    hide exactly the variation a cross-task comparison needs to see.
    """
    env = _make(task, seeds[0])
    try:
        action_space: Any = env.action_space
        n_actions = int(action_space.n)
    finally:
        env.close()
    return {
        action: measure_efold_time(task, action, dt_ns=dt_ns, seeds=seeds, threshold=threshold)
        for action in range(n_actions)
    }


def summarise(timescales: dict[int, TaskTimescale]) -> dict[str, float]:
    """Min/median/max timescale in milliseconds across held actions."""
    values = sorted(t.efold_ms for t in timescales.values())
    n = len(values)
    return {"min_ms": values[0], "median_ms": values[n // 2], "max_ms": values[-1]}
