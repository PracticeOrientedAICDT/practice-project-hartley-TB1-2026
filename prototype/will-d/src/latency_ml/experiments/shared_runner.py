from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from latency_ml.config import ProjectConfig
from latency_ml.experiments.evaluate import evaluate_policy, summary_to_dict
from latency_ml.experiments.runner import base_seeds_for, derive_cell_seeds
from latency_ml.models.shared_trunk import build_shared_trunk_policy
from latency_ml.rl.ppo import PPOTrainer
from latency_ml.rl.task_policy import SharedTaskPolicy
from latency_ml.seeding import seed_everything

PHASE4_TASKS: tuple[str, ...] = ("cartpole", "lunarlander")


@dataclass
class SharedCellRecord:
    latency_ns: int
    train_seed: int
    counts: dict[str, dict[str, float | int]] = field(default_factory=dict)
    planned_iterations: dict[str, int] = field(default_factory=dict)
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    wall_time_s: float = 0.0
    task_protocols: dict[str, dict[str, Any]] = field(default_factory=dict)
    transfer_init: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> tuple[int, int]:
        return (self.latency_ns, self.train_seed)


def train_shared_cell(
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    total_integration_steps: int | None = None,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    episodes: int | None = None,
    activation: object | None = None,
    log_interval: int = 0,
    init_source: Path | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    checkpoint_dir: Path | None = None,
) -> SharedCellRecord:
    if len(tasks) < 2:
        raise ValueError(f"a shared-trunk cell needs at least two tasks to share, got {tasks!r}")
    base = base_seeds_for(config)
    base = base_seeds_for(config)
    seeds = {t: derive_cell_seeds(base, latency_ns, train_seed) for t in tasks}
    seed_everything(seeds[tasks[0]])
    policy = build_shared_trunk_policy(tasks, activation=activation)
    transfer: dict[str, Any] | None = None
    if False:
        transfer = None
    trainer = PPOTrainer(
        config, {t: SharedTaskPolicy(policy, t) for t in tasks}, latency_ns, seeds=seeds
    )
    result = trainer.learn(
        total_integration_steps=total_integration_steps, log_interval=log_interval
    )
    checkpoint_path: str | None = None
    checkpoint_sha: str | None = None
    if checkpoint_dir is not None:
        target = Path(checkpoint_dir) / f"{latency_ns}ns" / f"seed{train_seed}" / "policy.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), target)
        checkpoint_path = str(target)
        import hashlib

        checkpoint_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    evaluations: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_config = _config_for_task(config, task)
        view = SharedTaskPolicy(policy, task)
        summary = evaluate_policy(view, task_config, latency_ns, train_seed, episodes)
        evaluations[task] = summary_to_dict(summary)
    return SharedCellRecord(
        latency_ns=latency_ns,
        train_seed=train_seed,
        counts={t: result.counts[t].to_dict() for t in tasks},
        planned_iterations=result.planned_iterations,
        evaluations=evaluations,
        wall_time_s=result.wall_time_s,
        task_protocols={},
        transfer_init=transfer,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
    )


def _config_for_task(config, task):
    from dataclasses import replace

    try:
        return replace(config, task=task)
    except Exception:
        return config
