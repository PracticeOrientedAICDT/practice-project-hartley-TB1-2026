from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any

import numpy as np
import torch

from latency_ml.config import ProjectConfig
from latency_ml.envs.factory import make_latency_env
from latency_ml.envs.latency_env import HOLD_STEPS_KEY
from latency_ml.rl.task_policy import TaskPolicy
from latency_ml.seeding import seed_env


def evaluate_policy(
    policy: TaskPolicy,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    episodes: int | None = None,
) -> dict[str, Any]:
    """Greedy-policy return over paired control episodes at one latency."""
    n = episodes if episodes is not None else int(config.evaluation.control_episodes_per_seed)
    probe = make_latency_env(config.task, latency_ns)
    action_steps = probe.action_steps
    dt_env_ns = probe.dt_env_ns
    probe.close()
    returns: list[float] = []
    survivals: list[float] = []
    failures = 0
    truncated = 0
    for i in range(n):
        env = make_latency_env(config.task, latency_ns)
        obs, _ = seed_env(env, config.evaluation.episode_seed(train_seed * 100 + i))
        total = 0.0
        sim_ns = 0
        term = False
        trunc = False
        while not (term or trunc):
            action = int(policy.action_only(torch.as_tensor(np.asarray(obs), dtype=torch.float32)).item())
            obs, reward, term, trunc, info = env.step(action)
            total += float(reward)
            sim_ns += int(info.get(HOLD_STEPS_KEY, action_steps)) * dt_env_ns
        returns.append(total)
        survivals.append(sim_ns / 1e9)
        failures += int(term)
        truncated += int(trunc)
        env.close()
    return {
        "task": config.task,
        "latency_ns": latency_ns,
        "action_steps": action_steps,
        "seed": train_seed,
        "mean_return": float(mean(returns)) if returns else 0.0,
        "median_return": float(median(returns)) if returns else 0.0,
        "std_return": float(pstdev(returns)) if len(returns) > 1 else 0.0,
        "mean_survival_s": float(mean(survivals)) if survivals else 0.0,
        "failure_rate": float(failures / len(returns)) if returns else 0.0,
        "truncation_rate": float(truncated / len(returns)) if returns else 0.0,
        "episodes": len(returns),
    }


def aggregate_by_latency(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[int, list[dict[str, Any]]] = {}
    for s in summaries:
        by.setdefault(int(s["latency_ns"]), []).append(s)
    out = []
    for lat in sorted(by):
        group = sorted(by[lat], key=lambda s: s["seed"])
        returns = tuple(s["mean_return"] for s in group)
        out.append(
            {
                "latency_ns": lat,
                "action_steps": group[0]["action_steps"],
                "n_seeds": len(group),
                "mean_return": float(mean(returns)),
                "std_across_seeds": float(pstdev(returns)) if len(returns) > 1 else 0.0,
                "mean_survival_s": float(mean(s["mean_survival_s"] for s in group)),
                "failure_rate": float(mean(s["failure_rate"] for s in group)),
                "truncation_rate": float(mean(s["truncation_rate"] for s in group)),
                "per_seed_returns": list(returns),
            }
        )
    return out


def load_policy(config: ProjectConfig, task: str, activation: Any = None) -> TaskPolicy:
    if isinstance(activation, str):
        from latency_ml.experiments.runner import make_activation

        activation = make_activation(activation)
    if activation == "hartree":
        from latency_ml.models.hartree import build_hartree_policy

        return build_hartree_policy(task)
    from latency_ml.models.architectures import build_control_policy

    return build_control_policy(task, activation=activation)
