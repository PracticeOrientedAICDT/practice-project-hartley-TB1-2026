"""Matched control evaluation (`PLAN.md` sections 6.7, 17, 19).

Evaluates trained policies and reduces them to a latency response. Two
properties of this module are what make the comparison mean anything.

**Evaluation is deterministic.** A policy is evaluated greedily, not sampled.
Sampling would add policy noise to the metric and, worse, would make the
comparison across latency conditions depend on the sampling stream. Greedy
evaluation turns the metric into a function of the weights alone.

**Evaluation is paired.** Episode ``k`` uses seed ``eval_seed_base + k`` for
*every* policy and *every* latency condition. A policy trained at 20 ms and
one trained at 100 ms therefore meet the same episode initial states. Section 19
requires this, and it is what allows a within-condition paired difference rather
than a comparison of two independent means -- the difference matters most
exactly where the effect is small and the variance is large.

Cross-latency evaluation is *secondary*. The primary design (section 17) is
matched training: a policy trained at its own latency, evaluated at that
latency. A policy trained at 20 ms and evaluated at 100 ms is a
train/test-mismatch condition, useful for showing the effect of latency at
inference, but it is not the headline number. :func:`evaluate_matrix` marks
which is which rather than pooling them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from latency_ml.config import ProjectConfig
from latency_ml.envs.factory import make_latency_env
from latency_ml.envs.latency_env import ACTION_SWITCHED_KEY, HOLD_STEPS_KEY
from latency_ml.envs.metrics import (
    CARTPOLE_THETA_INDEX,
    CARTPOLE_X_INDEX,
    LANDER_ANGLE_INDEX,
    LANDER_X_INDEX,
    ConditionSummary,
    EpisodeMetrics,
    EpisodeRecorder,
    summarise_episodes,
)
from latency_ml.models.architectures import MLPPolicy, build_control_policy
from latency_ml.rl.task_policy import TaskPolicy
from latency_ml.seeding import seed_env
from latency_ml.timing import derive_icdi

#: Observation indices per task, for the recorder.
_TASK_INDICES: dict[str, tuple[int, int]] = {
    "cartpole": (CARTPOLE_THETA_INDEX, CARTPOLE_X_INDEX),
    "lunarlander": (LANDER_ANGLE_INDEX, LANDER_X_INDEX),
}


def greedy_action(policy: TaskPolicy, obs: np.ndarray) -> int:
    """The policy's argmax action for one observation.

    Deterministic by construction, so the metric is a function of the weights
    alone and a paired comparison cannot pick up sampling noise.
    ``torch.no_grad`` because this runs inside the environment loop, where an
    accumulating graph would cost memory for the whole episode.

    ``action_only`` already applies the argmax, so this reads its result
    directly rather than taking a second argmax of an action index.
    """
    with torch.no_grad():
        action = policy.action_only(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(action.item())


def run_episode(
    policy: TaskPolicy,
    task: str,
    latency_ns: int,
    seed: int,
) -> EpisodeMetrics:
    """Run one deterministic evaluation episode and record its metrics.

    Args:
        policy: Trained policy.
        task: Task name.
        latency_ns: Latency condition to evaluate under. This is the
            *evaluation* latency and need not match the training latency.
        seed: Episode seed, shared across conditions for pairing.

    Returns:
        The episode's metrics.
    """
    env = make_latency_env(task, latency_ns)
    try:
        angle_index, position_index = _TASK_INDICES[task]
        recorder = EpisodeRecorder(
            dt_env_ns=env.dt_env_ns,
            angle_index=angle_index,
            position_index=position_index,
        )
        obs, _ = seed_env(env, seed)
        recorder.observe(np.asarray(obs))

        while True:
            action = greedy_action(policy, np.asarray(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.accumulate_reward(float(reward))
            # Count the integration steps this hold actually covered, not 1.
            # One wrapper step is one decision spanning N integration steps.
            recorder.count_step(int(info.get(HOLD_STEPS_KEY, 1)))
            recorder.count_decision(bool(info.get(ACTION_SWITCHED_KEY, False)))
            recorder.observe(np.asarray(obs))
            if terminated or truncated:
                return recorder.finish(terminated, truncated, np.asarray(obs))
    finally:
        env.close()


def evaluate_policy(
    policy: TaskPolicy,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    episodes: int | None = None,
) -> ConditionSummary:
    """Evaluate one trained policy under one latency condition.

    Args:
        policy: The trained policy.
        config: Experiment config, supplying the evaluation seed base and
            episode count.
        latency_ns: Evaluation latency.
        train_seed: The replicate index, recorded on the summary so a row can
            be traced to the policy that produced it.
        episodes: Override the configured episode count. Pilots use a small
            number; sweeps use 200.

    Returns:
        The aggregated summary for this (policy, latency) pair.
    """
    n_episodes = episodes if episodes is not None else config.evaluation.control_episodes_per_seed
    icdi = derive_icdi(latency_ns)

    records = [
        run_episode(
            policy,
            config.task,
            latency_ns,
            config.evaluation.episode_seed(index),
        )
        for index in range(n_episodes)
    ]
    return summarise_episodes(
        task=config.task,
        latency_ns=latency_ns,
        effective_ns=icdi.effective_ns,
        action_steps=icdi.action_steps,
        resolution=icdi.resolution.value,
        seed=train_seed,
        episodes=records,
    )


def summary_to_dict(summary: ConditionSummary) -> dict[str, Any]:
    """Summary-friendly representation."""
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
        "ci95_return": summary.ci95_return,
        "mean_survival_s": summary.mean_survival_s,
        "failure_rate": summary.failure_rate,
        "truncation_rate": summary.truncation_rate,
        "episodes": summary.episodes,
        "mean_max_abs_angle": summary.mean_max_abs_angle,
        "mean_rms_angle": summary.mean_rms_angle,
        "mean_action_switches_per_s": summary.mean_action_switches_per_s,
        "mean_decisions_per_s": summary.mean_decisions_per_s,
        "is_independent_measurement": summary.is_independent_measurement,
    }


@dataclass(frozen=True, slots=True)
class LatencyAggregate:
    """Latency-response curve at one evaluation latency.

    Reduced over training seeds. `PLAN.md` section 20 makes the independently
    trained policy the unit of analysis, so the spread here is across policies,
    not pooled over episodes. The episode-level spread lives in each
    :class:`ConditionSummary`.

    Attributes:
        latency_ns: The evaluation latency.
        effective_ns: What was actually realised.
        action_steps: Integration steps per decision.
        resolution: Resolution state, for honest plotting.
        n_seeds: Number of trained policies averaged.
        mean_return: Mean over seeds of each policy's mean return.
        std_across_seeds: Population spread of those per-seed means. This is
            the uncertainty that matters for the latency response.
        mean_survival_s: Mean survival over seeds.
        per_seed_returns: Each seed's mean return, kept for paired analysis.
    """

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
        """Whether this point may be plotted as a distinct measurement."""
        return self.resolution == "exact"

    def to_dict(self) -> dict[str, Any]:
        """Aggregate-friendly representation."""
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
    """Reduce per-seed summaries to one point per latency, ordered by latency.

    Raises:
        ValueError: If no summaries are supplied.
    """
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
                success_rate=mean(s.success_rate for s in group),
                failure_rate=mean(s.failure_rate for s in group),
                truncation_rate=mean(s.truncation_rate for s in group),
                mean_action_switches_per_s=mean(s.mean_action_switches_per_s for s in group),
            )
        )
    return aggregates


@dataclass(slots=True)
class CrossLatencyCell:
    """One (train latency, eval latency) pair in the cross-latency matrix.

    Attributes:
        train_latency_ns: Latency the policy was trained under.
        eval_latency_ns: Latency it is being evaluated under.
        train_seed: Replicate index.
        mean_return: The policy's mean return at this evaluation latency.
    """

    train_latency_ns: int
    eval_latency_ns: int
    train_seed: int
    mean_return: float

    @property
    def is_matched(self) -> bool:
        """Whether this is the primary matched-training condition (section 17).

        The diagonal. Off-diagonal cells are train/test mismatch and are
        reported separately, because pooling them with the diagonal would
        average two different experimental designs.
        """
        return self.train_latency_ns == self.eval_latency_ns

    def to_dict(self) -> dict[str, Any]:
        """Cell-friendly representation."""
        return {
            "train_latency_ns": self.train_latency_ns,
            "eval_latency_ns": self.eval_latency_ns,
            "train_seed": self.train_seed,
            "mean_return": self.mean_return,
            "is_matched": self.is_matched,
        }


def cross_latency_matrix(
    policies: dict[int, dict[int, MLPPolicy]],
    config: ProjectConfig,
    eval_latencies_ns: Sequence[int],
    episodes: int | None = None,
) -> list[CrossLatencyCell]:
    """Evaluate every policy at every evaluation latency.

    Args:
        policies: ``{train_latency_ns: {train_seed: policy}}``. Every training
            seed must be present for each latency, or the aggregate would be
            unbalanced.
        config: Experiment config.
        eval_latencies_ns: Latencies to evaluate at.
        episodes: Episode count per cell.

    Returns:
        One cell per (train latency, eval latency, seed).

    Raises:
        ValueError: If a latency has an unbalanced set of training seeds, or
            there are no policies.
    """
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
                cells.append(
                    CrossLatencyCell(
                        train_latency_ns=train_latency,
                        eval_latency_ns=eval_latency,
                        train_seed=train_seed,
                        mean_return=summary.mean_return,
                    )
                )
    return cells


def matched_diagonal(cells: Sequence[CrossLatencyCell]) -> list[CrossLatencyCell]:
    """The primary cells, where evaluation latency equals training latency."""
    return [c for c in cells if c.is_matched]


def summarise_matrix(cells: Sequence[CrossLatencyCell], episodes_per_cell: int) -> dict[str, Any]:
    """Reduce a cross-latency matrix to rows and column means.

    Returns:
        A mapping with the matched diagonal, the mean over training latencies
        for each evaluation latency, and the grand mean. All three are kept so a
        reader can see the matched result without also inheriting the mismatch
        cells.
    """
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
        "grand_mean_return": (mean(c.mean_return for c in cells) if cells else None),
    }


def load_policy(config: ProjectConfig, task: str, device: str = "cpu") -> MLPPolicy:
    """Build a policy with the architecture the config specifies.

    Wraps
    :func:`~latency_ml.models.architectures.build_control_policy` so
    evaluation can load weights into a policy with the right shapes without
    duplicating the construction logic.
    """
    policy = build_control_policy(task)
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
