from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from latency_ml.config import ProjectConfig, load_config
from latency_ml.experiments.evaluate import evaluate_policy
from latency_ml.experiments.runner import _init, base_seeds_for, derive_cell_seeds, make_activation
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
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_shared_cell(
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    total_integration_steps: int | None = None,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    episodes: int | None = None,
    activation: object | None = None,
    log_interval: int = 0,
    checkpoint_dir: Path | None = None,
) -> SharedCellRecord:
    """Train one shared-trunk cell: every task advances the same trunk per iteration."""
    if len(tasks) < 2:
        raise ValueError(f"a shared-trunk cell needs at least two tasks to share, got {tasks!r}")
    base = base_seeds_for(config)
    seeds = {t: derive_cell_seeds(base, latency_ns, train_seed) for t in tasks}
    seed_everything(seeds[tasks[0]])
    policy = build_shared_trunk_policy(tasks, activation=activation)
    trainer = PPOTrainer(config, {t: SharedTaskPolicy(policy, t) for t in tasks}, latency_ns, seeds=seeds)
    result = trainer.learn(total_integration_steps=total_integration_steps, log_interval=log_interval)
    checkpoint_path: str | None = None
    checkpoint_sha: str | None = None
    if checkpoint_dir is not None:
        target = Path(checkpoint_dir) / f"{latency_ns}ns" / f"seed{train_seed}" / "policy.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), target)
        checkpoint_path = str(target)
        checkpoint_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    evaluations: dict[str, dict[str, Any]] = {}
    for task in tasks:
        summary = evaluate_policy(SharedTaskPolicy(policy, task), _config_for_task(config, task), latency_ns, train_seed, episodes)
        evaluations[task] = summary
    return SharedCellRecord(
        latency_ns=latency_ns,
        train_seed=train_seed,
        counts={t: result.counts[t].to_dict() for t in tasks},
        planned_iterations=result.planned_iterations,
        evaluations=evaluations,
        wall_time_s=result.wall_time_s,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
    )


def _config_for_task(config: ProjectConfig, task: str) -> ProjectConfig:
    from dataclasses import replace

    return replace(config, task=task)


def _run_shared_cell_worker(payload: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", str(payload.get("worker_threads", 2)))
    config = load_config(payload["config_path"])
    rid = f"{payload['sweep_id']}/shared/{payload['latency_ns']}ns/seed{payload['train_seed']}"
    try:
        record = train_shared_cell(
            config,
            payload["latency_ns"],
            payload["train_seed"],
            total_integration_steps=payload["total_integration_steps"],
            tasks=tuple(payload["tasks"]),
            episodes=payload.get("episodes"),
            activation=make_activation(payload.get("activation")),
            log_interval=payload.get("log_interval", 0),
            checkpoint_dir=Path(payload["checkpoint_dir"]),
        )
    except Exception as error:  # one bad cell must not kill the sweep
        return {"run_id": rid, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    return {"run_id": rid, "status": "completed", "record": record.to_dict()}


def run_shared_sweep(
    *,
    config_path,
    sweep_id: str,
    latencies_ns,
    train_seeds,
    total_integration_steps: int | None = None,
    max_workers: int = 4,
    worker_threads: int = 2,
    root: Path = Path("results"),
    resume: bool = True,
    verbose: bool = True,
    activation: str | None = None,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    episodes: int | None = None,
    log_interval: int = 0,
) -> list[dict]:
    """Train every (latency, seed) cell of a shared-trunk sweep in parallel."""
    load_config(config_path)
    checkpoint_dir = Path(root) / sweep_id / "shared"
    outcomes: list[dict] = []
    payloads: list[dict] = []
    for latency_ns in latencies_ns:
        for seed in train_seeds:
            rid = f"{sweep_id}/shared/{latency_ns}ns/seed{seed}"
            if resume and (checkpoint_dir / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt").exists():
                outcomes.append({"run_id": rid, "status": "skipped"})
                continue
            payloads.append(
                {
                    "config_path": str(config_path),
                    "sweep_id": sweep_id,
                    "latency_ns": latency_ns,
                    "train_seed": seed,
                    "total_integration_steps": total_integration_steps,
                    "worker_threads": worker_threads,
                    "activation": activation,
                    "tasks": tasks,
                    "episodes": episodes,
                    "log_interval": log_interval,
                    "checkpoint_dir": str(checkpoint_dir),
                }
            )
    if verbose:
        print(f"shared sweep {sweep_id}: {len(payloads) + len(outcomes)} cells, {len(payloads)} to run, {max_workers} workers x {worker_threads} threads", flush=True)
    if payloads:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init, initargs=(worker_threads,)) as pool:
            for outcome in pool.map(_run_shared_cell_worker, payloads):
                outcomes.append(outcome)
                if verbose:
                    print(f"  {outcome['status']:>9}  {outcome['run_id']}  {outcome.get('error') or 'ok'}", flush=True)
    return outcomes
