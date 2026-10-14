"""Control evaluation metrics (`PLAN.md` sections 6.7 and 9.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, pstdev

import numpy as np

from latency_ml.envs.factory import (
    CARTPOLE_MAX_EPISODE_STEPS,
    LUNARLANDER_SOLVED_SCORE,
)

#: CartPole observation layout: [cart_position, cart_velocity, pole_angle, pole_velocity].
CARTPOLE_THETA_INDEX = 2
CARTPOLE_X_INDEX = 0

#: LunarLander observation layout: [x, y, vx, vy, angle, angular_velocity, leg1, leg2].
LANDER_ANGLE_INDEX = 4
LANDER_X_INDEX = 0


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Per-episode metrics for one task.

    Fields common to both tasks are always populated. Landing-quality fields
    apply to LunarLander; CartPole leaves them at their neutral defaults.
    """

    return_total: float
    episode_steps: int
    sim_time_ns: int
    survival_ns: int
    terminated: bool
    truncated: bool
    max_abs_angle: float
    rms_angle: float
    max_abs_position: float
    rms_position: float
    action_switches: int
    decision_count: int
    final_x: float = 0.0
    final_angle: float = 0.0

    @property
    def seconds_survived(self) -> float:
        """Survival in simulated seconds."""
        return self.survival_ns / 1_000_000_000

    @property
    def decisions_per_simulated_second(self) -> float:
        """Neural policy invocations per simulated second.

        Falls with latency by construction, since each decision consumes a
        fixed simulated interval. Reported because `PLAN.md` section 6.7 lists
        it, and it makes the action-interval effect explicit rather than
        implicit.
        """
        if self.survival_ns <= 0:
            return 0.0
        return self.decision_count / self.seconds_survived

    @property
    def integration_steps_per_decision(self) -> float:
        """Mean integration steps per policy invocation."""
        return self.episode_steps / self.decision_count if self.decision_count else 0.0


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """Aggregated metrics across episodes for one (task, latency, seed).

    `PLAN.md` section 20 is explicit that the unit of analysis is the
    independently trained policy, not the episode. So episode-level spread and
    seed-level spread are kept separate rather than pooled.
    """

    task: str
    latency_ns: int
    effective_ns: int
    action_steps: int
    resolution: str
    seed: int
    mean_return: float
    median_return: float
    std_return: float
    ci95_return: float
    mean_survival_s: float
    failure_rate: float
    truncation_rate: float
    success_rate: float
    episodes: int
    mean_max_abs_angle: float
    mean_rms_angle: float
    mean_action_switches_per_s: float
    mean_decisions_per_s: float

    @property
    def is_independent_measurement(self) -> bool:
        """Whether this condition may be plotted as a distinct point."""
        return self.resolution == "exact"


def _ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval of the mean.

    Uses the population standard deviation, appropriate here because episodes
    are the full set evaluated for a given policy rather than a sample from a
    larger population.
    """
    n = len(values)
    if n < 2:
        return 0.0
    # 1.96 approximates the normal 97.5th percentile; at 200 episodes per
    # seed the normal approximation is well justified.
    spread = float(pstdev(values))
    result = 1.96 * spread / (n**0.5)
    return float(result)


def solved_score(task: str) -> float:
    """Return threshold above which an episode counts as solved.

    `PLAN.md` section 9.5 defines LunarLander's success rate as
    ``return >= 200``, matching Gymnasium's documented solved threshold. CartPole
    has no such documented criterion, so its "success" is reaching the 500-step
    ceiling -- the only condition under which the task is not failed. The two
    are different claims and are not merged.

    Raises:
        ValueError: If the task is unknown.
    """
    if task == "lunarlander":
        return LUNARLANDER_SOLVED_SCORE
    if task == "cartpole":
        return CARTPOLE_MAX_EPISODE_STEPS
    raise ValueError(f"unknown task {task!r}")


def summarise_episodes(
    task: str,
    latency_ns: int,
    effective_ns: int,
    action_steps: int,
    resolution: str,
    seed: int,
    episodes: list[EpisodeMetrics],
) -> ConditionSummary:
    """Aggregate episodes for one trained policy.

    Raises:
        ValueError: If no episodes are supplied.
    """
    if not episodes:
        raise ValueError("cannot summarise an empty episode list")

    returns = [e.return_total for e in episodes]
    survival = [e.seconds_survived for e in episodes]
    failures = [1.0 if e.terminated else 0.0 for e in episodes]
    truncations = [1.0 if e.truncated else 0.0 for e in episodes]
    solved = solved_score(task)
    successes = [1.0 if e.return_total >= solved else 0.0 for e in episodes]

    return ConditionSummary(
        task=task,
        latency_ns=latency_ns,
        effective_ns=effective_ns,
        action_steps=action_steps,
        resolution=resolution,
        seed=seed,
        mean_return=mean(returns),
        median_return=median(returns),
        std_return=pstdev(returns) if len(returns) > 1 else 0.0,
        ci95_return=_ci95(returns),
        mean_survival_s=mean(survival),
        failure_rate=mean(failures),
        truncation_rate=mean(truncations),
        success_rate=mean(successes),
        episodes=len(episodes),
        mean_max_abs_angle=mean(e.max_abs_angle for e in episodes),
        mean_rms_angle=mean(e.rms_angle for e in episodes),
        mean_action_switches_per_s=mean(
            e.action_switches / e.seconds_survived for e in episodes if e.seconds_survived > 0
        ),
        mean_decisions_per_s=mean(e.decisions_per_simulated_second for e in episodes),
    )


@dataclass(slots=True)
class EpisodeRecorder:
    """Accumulates per-step observations and flags into an episode record.

    Mutable by design: it is fed one integration step at a time by the
    evaluation loop, then frozen into an :class:`EpisodeMetrics` by
    :meth:`finish`. Metric definitions live here rather than being scattered
    through the rollout code.
    """

    dt_env_ns: int
    angle_index: int = CARTPOLE_THETA_INDEX
    position_index: int = CARTPOLE_X_INDEX
    angles: list[float] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    total_reward: float = 0.0
    steps: int = 0
    decisions: int = 0
    switches: int = 0
    terminated: bool = False
    truncated: bool = False

    def observe(self, obs: np.ndarray) -> None:
        """Record the angle and position from one integration step."""
        self.angles.append(float(abs(obs[self.angle_index])))
        self.positions.append(float(abs(obs[self.position_index])))

    def accumulate_reward(self, reward: float) -> None:
        """Add the reward from one integration step."""
        self.total_reward += float(reward)

    def count_step(self, hold_steps: int = 1) -> None:
        """Count the integration steps a wrapper transition actually covered.

        Args:
            hold_steps: Integration steps executed by this transition, from the
                wrapper's ``HOLD_STEPS_KEY``. Must be at least 1.

        A wrapper transition is one *decision* spanning ``N`` integration
        steps, so passing 1 here would under-count simulated time by a factor
        of ``N``. The error is silent: it makes survival time fall with
        latency while the return stays at the 500-step ceiling, which reads
        as a control effect but is pure bookkeeping. The last transition of an
        episode is usually short, so the wrapper's reported value is used
        rather than the configured hold length.
        """
        assert hold_steps >= 1, f"hold_steps must be >= 1, got {hold_steps}"
        self.steps += hold_steps

    def count_decision(self, switched: bool) -> None:
        """Count one policy invocation and whether its action changed."""
        self.decisions += 1
        if switched:
            self.switches += 1

    def finish(self, terminated: bool, truncated: bool, final_obs: np.ndarray) -> EpisodeMetrics:
        """Close the episode and produce its frozen metrics."""
        self.observe(final_obs)
        self.terminated = terminated
        self.truncated = truncated
        angles = self.angles or [0.0]
        positions = self.positions or [0.0]
        survival_ns = self.steps * self.dt_env_ns
        return EpisodeMetrics(
            return_total=self.total_reward,
            episode_steps=self.steps,
            sim_time_ns=survival_ns,
            survival_ns=survival_ns,
            terminated=terminated,
            truncated=truncated,
            max_abs_angle=max(angles),
            rms_angle=float(np.sqrt(np.mean(np.square(angles)))),
            max_abs_position=max(positions),
            rms_position=float(np.sqrt(np.mean(np.square(positions)))),
            action_switches=self.switches,
            decision_count=self.decisions,
            final_x=float(final_obs[self.position_index]),
            final_angle=float(final_obs[self.angle_index]),
        )
