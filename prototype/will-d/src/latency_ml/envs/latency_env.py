from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from latency_ml.timing import ICDI, NATIVE_DT_ENV_NS, Resolution, derive_icdi

HOLD_STEPS_KEY = "hold_steps"
SIM_TIME_NS_KEY = "sim_time_ns"
ACTION_SWITCHED_KEY = "action_switched"


class LatencyHoldEnv(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(
        self, env: gym.Env[ObsType, ActType], latency_ns: int, dt_env_ns: int = NATIVE_DT_ENV_NS
    ) -> None:
        super().__init__(env)
        self.latency_ns = latency_ns
        self.dt_env_ns = dt_env_ns
        self.icdi: ICDI = derive_icdi(latency_ns, dt_env_ns)
        self._last_action: Any = None
        self._have_last_action = False
        self._pending_terminal: tuple[Any, dict[str, Any]] | None = None
        self._last_record: TransitionRecord | None = None

    @property
    def action_steps(self) -> int:
        return self.icdi.action_steps

    def is_valid_measurement(self) -> bool:
        return self.icdi.is_independent_measurement

    def last_record(self) -> TransitionRecord | None:
        return self._last_record

    def pending_terminal(self) -> tuple[Any, dict[str, Any]] | None:
        return self._pending_terminal

    def release_terminal(self) -> tuple[Any, dict[str, Any]] | None:
        result = self._pending_terminal
        self._pending_terminal = None
        return result

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._last_action = None
        self._have_last_action = False
        self._pending_terminal = None
        self._last_record = None
        obs, info = self.env.reset(**kwargs)
        return (obs, info)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._pending_terminal is not None:
            raise RuntimeError(
                "LatencyHoldEnv.step called with an unreleased terminal observation from the previous hold. The underlying episode ended mid-hold; call release_terminal() before continuing."
            )
        switched = self._switched(action)
        self._last_action = self._as_comparable(action)
        self._have_last_action = True
        total_reward = 0.0
        hold_steps = 0
        obs: Any = None
        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(self.action_steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            hold_steps += 1
            if terminated or truncated:
                break
        sim_time_ns = hold_steps * self.dt_env_ns
        step_info: dict[str, Any] = dict(info)
        step_info[HOLD_STEPS_KEY] = hold_steps
        step_info[SIM_TIME_NS_KEY] = sim_time_ns
        step_info[ACTION_SWITCHED_KEY] = switched
        step_info["requested_latency_ns"] = self.latency_ns
        step_info["effective_latency_ns"] = self.icdi.effective_ns
        step_info["resolution"] = self.icdi.resolution.value
        self._last_record = TransitionRecord(
            hold_steps=hold_steps,
            requested_steps=self.action_steps,
            sim_time_ns=sim_time_ns,
            requested_ns=self.latency_ns,
            effective_ns=self.icdi.effective_ns,
            resolution=self.icdi.resolution,
            switched=switched,
            terminated=terminated,
            truncated=truncated,
        )
        if terminated or truncated:
            self._pending_terminal = (obs, dict(info))
        return (obs, total_reward, terminated, truncated, step_info)

    def _switched(self, action: Any) -> bool:
        if not self._have_last_action:
            return True
        return not np.array_equal(
            np.asarray(self._as_comparable(action)), np.asarray(self._last_action)
        )

    def _as_comparable(self, action: Any) -> Any:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return np.int64(action)
        return np.asarray(action)


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    hold_steps: int
    requested_steps: int
    sim_time_ns: int
    requested_ns: int
    effective_ns: int
    resolution: Resolution
    switched: bool
    terminated: bool
    truncated: bool

    @property
    def is_short_hold(self) -> bool:
        return self.hold_steps < self.requested_steps


LatencyHoldEnvT = LatencyHoldEnv[Any, Any]
