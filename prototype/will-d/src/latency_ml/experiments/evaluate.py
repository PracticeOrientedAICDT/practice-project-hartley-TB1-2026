from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from latency_ml.config import ProjectConfig
from latency_ml.envs.factory import make_latency_env
from latency_ml.envs.latency_env import HOLD_STEPS_KEY
from latency_ml.envs.metrics import ConditionSummary, EpisodeMetrics, EpisodeRecorder, summarise_episodes
from latency_ml.models.architectures import MLPPolicy, build_control_policy
from latency_ml.rl.task_policy import TaskPolicy
from latency_ml.seeding import seed_env
from latency_ml.timing import derive_icdi


def greedy_action(policy: TaskPolicy, obs: np.ndarray) -> int:
    with torch.no_grad():
        action = policy.action_only(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(action.item())


def run_episode(policy: TaskPolicy, task: str, latency_ns: int, seed: int) -> EpisodeMetrics:
    env = make_latency_env(task, latency_ns)
    try:
        recorder = EpisodeRecorder(dt_env_ns=env.dt_env_ns)
        obs, _ = seed_env(env, seed)
        while True:
            action = greedy_action(policy, np.asarray(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.accumulate_reward(float(reward))
            recorder.count_step(int(info.get(HOLD_STEPS_KEY, 1)))
            if terminated or truncated:
                return recorder.finish(terminated, truncated, np.asarray(obs))
    finally:
        env.close()


def evaluate_policy(policy: TaskPolicy, config: ProjectConfig, latency_ns: int, train_seed: int, episodes: int | None = None) -> ConditionSummary:
    n_episodes = episodes if episodes is not None else config.evaluation.control_episodes_per_seed
    icdi = derive_icdi(latency_ns)
    records = [run_episode(policy, config.task, latency_ns, config.evaluation.episode_seed(index)) for index in range(n_episodes)]
    return summarise_episodes(
        task=config.task, latency_ns=latency_ns, effective_ns=icdi.effective_ns, action_steps=icdi.action_steps, resolution=icdi.resolution.value, seed=train_seed, episodes=records
    )


def summary_to_dict(summary: ConditionSummary) -> dict[str, Any]:
    return {
        "task": summary.task,
        "latency_ns": summary.latency_ns,
        "effective_ns": summary.effective_ns,
        "action_steps": summary.action_steps,
        "resolution": summary.resolution,
        "seed": summary.seed,
        "mean_return": summary.mean_return,
        "median_return": summary.median_return,
        "std_return": summary.std_return,
        "mean_survival_s": summary.mean_survival_s,
        "failure_rate": summary.failure_rate,
        "truncation_rate": summary.truncation_rate,
        "episodes": summary.episodes,
    }


@dataclass(frozen=True, slots=True)
class LatencyAggregate:
    latency_ns: int
    effective_ns: int
    action_steps: int
    resolution: str
    n_seeds: int
    mean_return: float
    std_across_seeds: float
    mean_survival_s: float
    per_seed_returns: tuple[float, ...]
    median_return: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    truncation_rate: float = 0.0
    mean_action_switches_per_s: float = 0.0

    @property
    def is_independent_measurement(self) -> bool:
        return self.resolution == "exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ns": self.latency_ns,
            "effective_ns": self.effective_ns,
            "action_steps": self.action_steps,
            "resolution": self.resolution,
            "n_seeds": self.n_seeds,
            "mean_return": self.mean_return,
            "std_across_seeds": self.std_across_seeds,
            "mean_survival_s": self.mean_survival_s,
            "median_return": self.median_return,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "truncation_rate": self.truncation_rate,
            "mean_action_switches_per_s": self.mean_action_switches_per_s,
            "per_seed_returns": list(self.per_seed_returns),
            "is_independent_measurement": self.is_independent_measurement,
        }


def aggregate_by_latency(summaries: Sequence[ConditionSummary]) -> list[LatencyAggregate]:
    if not summaries:
        raise ValueError("cannot aggregate an empty summary list")
    by_latency: dict[int, list[ConditionSummary]] = {}
    for summary in summaries:
        by_latency.setdefault(summary.latency_ns, []).append(summary)
    aggregates = []
    for latency in sorted(by_latency):
        group = sorted(by_latency[latency], key=lambda s: s.seed)
        returns = tuple(s.mean_return for s in group)
        aggregates.append(
            LatencyAggregate(
                latency_ns=latency,
                effective_ns=group[0].effective_ns,
                action_steps=group[0].action_steps,
                resolution=group[0].resolution,
                n_seeds=len(group),
                mean_return=mean(returns),
                std_across_seeds=pstdev(returns) if len(returns) > 1 else 0.0,
                mean_survival_s=mean(s.mean_survival_s for s in group),
                per_seed_returns=returns,
                median_return=mean(s.median_return for s in group),
                failure_rate=mean(s.failure_rate for s in group),
                truncation_rate=mean(s.truncation_rate for s in group),
            )
        )
    return aggregates


@dataclass(slots=True)
class CrossLatencyCell:
    train_latency_ns: int
    eval_latency_ns: int
    train_seed: int
    mean_return: float

    @property
    def is_matched(self) -> bool:
        return self.train_latency_ns == self.eval_latency_ns

    def to_dict(self) -> dict[str, Any]:
        return {"train_latency_ns": self.train_latency_ns, "eval_latency_ns": self.eval_latency_ns, "train_seed": self.train_seed, "mean_return": self.mean_return, "is_matched": self.is_matched}


def cross_latency_matrix(policies: dict[int, dict[int, MLPPolicy]], config: ProjectConfig, eval_latencies_ns: Sequence[int], episodes: int | None = None) -> list[CrossLatencyCell]:
    if not policies:
        raise ValueError("cross_latency_matrix needs at least one policy")
    seed_sets = {lat: set(seeds.keys()) for lat, seeds in policies.items()}
    distinct = {frozenset(s) for s in seed_sets.values()}
    if len(distinct) > 1:
        raise ValueError(f"unbalanced training seeds across latencies: {seed_sets}")
    cells: list[CrossLatencyCell] = []
    for train_latency in sorted(policies):
        for train_seed in sorted(policies[train_latency]):
            policy = policies[train_latency][train_seed]
            for eval_latency in eval_latencies_ns:
                summary = evaluate_policy(policy, config, eval_latency, train_seed, episodes)
                cells.append(CrossLatencyCell(train_latency_ns=train_latency, eval_latency_ns=eval_latency, train_seed=train_seed, mean_return=summary.mean_return))
    return cells


def matched_diagonal(cells: Sequence[CrossLatencyCell]) -> list[CrossLatencyCell]:
    return [c for c in cells if c.is_matched]


def summarise_matrix(cells: Sequence[CrossLatencyCell], episodes_per_cell: int) -> dict[str, Any]:
    matched = matched_diagonal(cells)
    by_eval: dict[int, list[float]] = {}
    for cell in cells:
        by_eval.setdefault(cell.eval_latency_ns, []).append(cell.mean_return)
    return {
        "n_cells": len(cells),
        "n_matched": len(matched),
        "episodes_per_cell": episodes_per_cell,
        "matched_mean_return": mean(c.mean_return for c in matched) if matched else None,
        "by_eval_latency": {latency: mean(values) for latency, values in sorted(by_eval.items())},
        "grand_mean_return": mean(c.mean_return for c in cells) if cells else None,
    }


def load_policy(config: ProjectConfig, task: str, device: str = "cpu", activation: str | None = None) -> MLPPolicy:
    from latency_ml.experiments.runner import make_activation

    act = make_activation(activation)
    policy = build_control_policy(task, activation=act)
    policy.to(torch.device(device))
    return policy


__all__ = [
    "CrossLatencyCell",
    "LatencyAggregate",
    "aggregate_by_latency",
    "cross_latency_matrix",
    "evaluate_policy",
    "greedy_action",
    "load_policy",
    "matched_diagonal",
    "run_episode",
    "summarise_matrix",
    "summary_to_dict",
]
