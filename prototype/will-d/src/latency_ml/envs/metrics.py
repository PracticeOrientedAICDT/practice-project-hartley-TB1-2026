from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, pstdev

import numpy as np

from latency_ml.envs.factory import CARTPOLE_MAX_EPISODE_STEPS, LUNARLANDER_SOLVED_SCORE

CARTPOLE_THETA_INDEX = 2
CARTPOLE_X_INDEX = 0
LANDER_ANGLE_INDEX = 4
LANDER_X_INDEX = 0


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
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
        return self.survival_ns / 1000000000

    @property
    def decisions_per_simulated_second(self) -> float:
        if self.survival_ns <= 0:
            return 0.0
        return self.decision_count / self.seconds_survived

    @property
    def integration_steps_per_decision(self) -> float:
        return self.episode_steps / self.decision_count if self.decision_count else 0.0


@dataclass(frozen=True, slots=True)
class ConditionSummary:
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
        return self.resolution == "exact"


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    spread = float(pstdev(values))
    result = 1.96 * spread / n**0.5
    return float(result)


def solved_score(task: str) -> float:
    if task == "lunarlander":
        return LUNARLANDER_SOLVED_SCORE
    if task == "cartpole":
        return CARTPOLE_MAX_EPISODE_STEPS
    raise ValueError(f"unknown task {task!r}")


def summarise_episodes(task: str, latency_ns: int, effective_ns: int, action_steps: int, resolution: str, seed: int, episodes: list[EpisodeMetrics]) -> ConditionSummary:
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
        mean_action_switches_per_s=mean(e.action_switches / e.seconds_survived for e in episodes if e.seconds_survived > 0),
        mean_decisions_per_s=mean(e.decisions_per_simulated_second for e in episodes),
    )


@dataclass(slots=True)
class EpisodeRecorder:
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
        self.angles.append(float(abs(obs[self.angle_index])))
        self.positions.append(float(abs(obs[self.position_index])))

    def accumulate_reward(self, reward: float) -> None:
        self.total_reward += float(reward)

    def count_step(self, hold_steps: int = 1) -> None:
        assert hold_steps >= 1, f"hold_steps must be >= 1, got {hold_steps}"
        self.steps += hold_steps

    def count_decision(self, switched: bool) -> None:
        self.decisions += 1
        if switched:
            self.switches += 1

    def finish(self, terminated: bool, truncated: bool, final_obs: np.ndarray) -> EpisodeMetrics:
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
