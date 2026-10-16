from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from latency_ml.timing import NATIVE_DT_ENV_NS, derive_icdi

HOLD_STEPS_KEY = "hold_steps"


class LatencyHoldEnv(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Hold each action for the number of env frames that best covers ``latency_ns``."""

    def __init__(self, env: gym.Env[ObsType, ActType], latency_ns: int, dt_env_ns: int = NATIVE_DT_ENV_NS) -> None:
        super().__init__(env)
        self.latency_ns = latency_ns
        self.dt_env_ns = dt_env_ns
        self.icdi = derive_icdi(latency_ns, dt_env_ns)
        self._pending_terminal: tuple[Any, dict[str, Any]] | None = None

    @property
    def action_steps(self) -> int:
        return self.icdi.action_steps

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._pending_terminal = None
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._pending_terminal is not None:
            raise RuntimeError(
                "LatencyHoldEnv.step called with an unreleased terminal observation from the previous hold. The underlying episode ended mid-hold; call release_terminal() before continuing."
            )
        total_reward = 0.0
        hold_steps = 0
        obs: Any = None
        terminated = False
        truncated = False
        for _ in range(self.action_steps):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            hold_steps += 1
            if terminated or truncated:
                break
        if terminated or truncated:
            self._pending_terminal = (obs, {})
        return obs, total_reward, terminated, truncated, {HOLD_STEPS_KEY: hold_steps}

    def release_terminal(self) -> None:
        """Drop a terminal observation the caller saw mid-hold."""
        self._pending_terminal = None
