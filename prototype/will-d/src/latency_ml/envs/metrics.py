from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    return_total: float
    survival_ns: int
    terminated: bool
    truncated: bool
    decision_count: int


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
    failure_rate: float
    truncation_rate: float
    episodes: int
    mean_survival_s: float


def summarise_episodes(task, latency_ns, effective_ns, action_steps, resolution, seed, episodes):
    if not episodes:
        raise ValueError("empty")
    returns = [e.return_total for e in episodes]
    survival = [e.survival_ns / 1e9 for e in episodes]
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
        failure_rate=mean(1.0 if e.terminated else 0.0 for e in episodes),
        truncation_rate=mean(1.0 if e.truncated else 0.0 for e in episodes),
        episodes=len(episodes),
        mean_survival_s=mean(survival),
    )


CARTPOLE_THETA_INDEX = 2
CARTPOLE_X_INDEX = 0
LANDER_ANGLE_INDEX = 4
LANDER_X_INDEX = 0


@dataclass(slots=True)
class EpisodeRecorder:
    dt_env_ns: int
    total_reward: float = 0.0
    steps: int = 0
    decisions: int = 0
    switches: int = 0
    terminated: bool = False
    truncated: bool = False

    def observe(self, obs):
        pass

    def accumulate_reward(self, reward: float):
        self.total_reward += reward

    def count_step(self, hold_steps: int = 1):
        self.steps += hold_steps

    def count_decision(self, switched: bool):
        self.decisions += 1
        if switched:
            self.switches += 1

    def finish(self, terminated, truncated, final_obs):
        self.terminated = terminated
        self.truncated = truncated
        return EpisodeMetrics(return_total=self.total_reward, survival_ns=self.steps * self.dt_env_ns, terminated=terminated, truncated=truncated, decision_count=self.decisions)
