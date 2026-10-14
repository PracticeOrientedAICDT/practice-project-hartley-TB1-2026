"""The action-hold latency wrapper.

This is where inference latency becomes physics. A policy observes, decides
once, and that action is held for ``N`` environment integration steps, where
``N`` is derived from the requested latency and the simulator timestep
(`PLAN.md` section 4).

Three properties this wrapper must get right, each of which a naive
implementation gets wrong:

1. **Reward accumulates across the whole hold.** `PLAN.md` section 25 Risk 3:
   the learner sees one transition whose reward is the sum over all held
   steps. Returning only the last step's reward would silently rescale the
   return per latency condition.

2. **Termination and truncation stay distinct.** CartPole terminates when the
   pole falls and truncates at the 500-step limit. Only the first ends the
   episode, and only the second should bootstrap.

3. **The final hold is usually shorter than requested.** The 500-step limit is
   not divisible by ``N``, so the last interval of an episode almost never
   runs its full length. Semi-Markov discounting must use the *actual* hold
   length, not the requested one, or every episode's last transition is
   mis-discounted.

``info`` carries the actual hold length so (3) reaches the learner as data
rather than being inferred.
"""

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
    """Hold each action for ``N`` integration steps.

    Wraps a Gymnasium environment whose integration timestep is known. Each
    wrapper transition covers a whole held-action interval and is treated by
    the learner as a semi-Markov transition of that duration.

    Args:
        env: The environment to wrap.
        latency_ns: Requested inference latency in nanoseconds.
        dt_env_ns: The environment's integration timestep in nanoseconds.
            Defaults to the native Gymnasium 20 ms.

    Attributes:
        icdi: The derived :class:`~latency_ml.timing.ICDI` for this condition.

    Note:
        An :attr:`Resolution.UNRESOLVED` condition is still constructible, so
        it can be carried through configuration and recorded. It is not an
        independent measurement; use :meth:`is_valid_measurement` to gate
        reporting.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        latency_ns: int,
        dt_env_ns: int = NATIVE_DT_ENV_NS,
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
        """``N``, integration steps per held action."""
        return self.icdi.action_steps

    def is_valid_measurement(self) -> bool:
        """Whether this condition is an independent, honestly-measured point."""
        return self.icdi.is_independent_measurement

    def last_record(self) -> TransitionRecord | None:
        """The most recent transition record, or ``None`` before the first step."""
        return self._last_record

    def pending_terminal(self) -> tuple[Any, dict[str, Any]] | None:
        """Terminal observation/info from an episode-ending hold, if unreleased.

        The wrapper keeps the terminal observation so a rollout can record the
        underlying environment's final diagnostics without the wrapper losing
        them.
        """
        return self._pending_terminal

    def release_terminal(self) -> tuple[Any, dict[str, Any]] | None:
        """Take the pending terminal observation/info, clearing it."""
        result = self._pending_terminal
        self._pending_terminal = None
        return result

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset and clear hold state."""
        self._last_action = None
        self._have_last_action = False
        self._pending_terminal = None
        self._last_record = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one held-action transition.

        Args:
            action: The action to hold.

        Returns:
            ``(obs, total_reward, terminated, truncated, info)`` where the
            reward is summed over the hold and ``info`` carries
            :data:`HOLD_STEPS_KEY` and :data:`SIM_TIME_NS_KEY`.

        Raises:
            RuntimeError: If a terminal observation from a previous hold has
                not been released. That indicates a caller not handling
                episode boundaries, which this wrapper will not paper over.
        """
        if self._pending_terminal is not None:
            raise RuntimeError(
                "LatencyHoldEnv.step called with an unreleased terminal "
                "observation from the previous hold. The underlying episode "
                "ended mid-hold; call release_terminal() before continuing."
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

        return obs, total_reward, terminated, truncated, step_info

    def _switched(self, action: Any) -> bool:
        """Whether this action differs from the previously held one."""
        if not self._have_last_action:
            return True
        return not np.array_equal(
            np.asarray(self._as_comparable(action)), np.asarray(self._last_action)
        )

    def _as_comparable(self, action: Any) -> Any:
        """Normalise an action to a comparable array form."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return np.int64(action)
        return np.asarray(action)


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """What one wrapper transition actually did.

    Returned by :meth:`LatencyHoldEnv.last_record` so evaluation and training
    can log duration and switching without reaching into wrapper internals.

    Attributes:
        hold_steps: Integration steps actually executed. This is what
            semi-Markov discounting must use.
        requested_steps: The configured ``N``. Recorded explicitly rather than
            recomputed, because the wrapper knows it and a derivation from
            ``requested_ns``/``effective_ns`` is wrong -- ``effective_ns``
            already contains the timestep, so such a ratio collapses to 1.
    """

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
        """Whether an episode boundary cut the hold short.

        True when fewer integration steps ran than requested, which is the
        normal case for an episode's final transition and the reason
        semi-Markov discounting must use :attr:`hold_steps` rather than ``N``.
        """
        return self.hold_steps < self.requested_steps


LatencyHoldEnvT = LatencyHoldEnv[Any, Any]
