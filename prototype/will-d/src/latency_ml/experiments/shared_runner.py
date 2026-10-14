"""Sweep runner for the Phase 4 shared-trunk generality experiment.

One cell is one (latency, seed) pair. The shared trunk is trained **once** per
cell, then each task is evaluated under that cell's latency with exactly the
protocol the dedicated Phase 1/2 baselines used -- same episode count, same
seed derivation, same metrics, same ``SharedTaskPolicy`` interface into the same
evaluation code.

That reuse is the point. If the shared run used its own evaluation path, any
difference in the reported return could come from the evaluation rather than from
the network, and the comparison would stop being controlled.

Each cell writes both tasks' summaries into one record keyed by ``(latency,
seed)``, so the paired analysis against the dedicated baselines is a direct
lookup rather than a join across two differently-shaped result trees.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from latency_ml.analysis.parity import task_protocol
from latency_ml.analysis.transfer_init import (
    file_sha256,
    load_source_policy,
    transfer_from_dedicated,
)
from latency_ml.config import ProjectConfig, load_config
from latency_ml.experiments.evaluate import evaluate_policy, summary_to_dict
from latency_ml.experiments.runner import (
    DEFAULT_WORKER_THREADS,
    base_seeds_for,
    derive_cell_seeds,
)
from latency_ml.models.shared_trunk import (
    SharedTrunkPolicy,
    build_shared_trunk_policy,
)
from latency_ml.rl.ppo import PPOTrainer
from latency_ml.rl.task_policy import SharedTaskPolicy
from latency_ml.seeding import seed_everything

#: The two tasks whose representations are shared. Held as a constant so the
#: experiment cannot silently change which tasks participate.
PHASE4_TASKS: tuple[str, ...] = ("cartpole", "lunarlander")


@dataclass
class SharedCellRecord:
    """One trained shared trunk and both tasks' evaluations of it.

    Attributes:
        latency_ns: Latency condition, for training and evaluation alike.
        train_seed: Replicate index, matching the dedicated baseline's.
        counts: Per-task run counts, including each task's PPO update count.
        planned_iterations: Iterations planned per task, for the invariant check.
        evaluations: Per-task evaluation summaries.
        wall_time_s: Training wall clock.
        task_protocols: Per-task protocol fingerprint, recorded so the shared
            arm can be compared against the dedicated controls from the
            artifacts rather than by assertion.
    """

    latency_ns: int
    train_seed: int
    counts: dict[str, dict[str, float | int]] = field(default_factory=dict)
    planned_iterations: dict[str, int] = field(default_factory=dict)
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    wall_time_s: float = 0.0
    task_protocols: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Phase 4D transfer initialisation audit, when the cell was run with one.
    #: Absent on Phase 4C cells, which are from-scratch by construction. Keeping
    #: the initial state on the same record as the trained result is what makes
    #: "did the representation survive training" answerable per cell.
    transfer_init: dict[str, Any] | None = None
    #: Joint training curve, when curve recording was enabled.
    curve: list[dict[str, Any]] = field(default_factory=list)
    #: Path and SHA-256 of the saved final policy, when one was written.
    #:
    #: Added after Phase 4D, whose cells saved nothing. Without a checkpoint a
    #: shared cell cannot be re-evaluated independently, and two seeds returning
    #: identical metrics could not be distinguished into "identical policy" and
    #: "identical evaluation" -- a distinction that decides whether a duplicate
    #: result is a convergence finding or a seeding bug.
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> tuple[int, int]:
        """The (latency, seed) coordinates matched against dedicated baselines."""
        return (self.latency_ns, self.train_seed)


def train_shared_cell(
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    total_integration_steps: int | None = None,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    episodes: int | None = None,
    log_interval: int = 0,
    init_source: Path | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    checkpoint_dir: Path | None = None,
) -> SharedCellRecord:
    """Train one shared trunk and evaluate both tasks under ``latency_ns``.

    Args:
        config: Project configuration supplying frozen PPO hyperparameters and
            the evaluation protocol.
        latency_ns: Latency condition for training and evaluation.
        train_seed: Replicate index. Seeds are derived per task from this and the
            latency, exactly as the dedicated baseline derives them, so the two
            are not merely equal in count but equal in derivation.
        total_integration_steps: Per-task budget. Defaults to the config value.
        tasks: Tasks sharing the trunk.
        episodes: Evaluation episodes per task. Defaults to the config value,
            which is what the dedicated baselines used.
        log_interval: Progress printing interval.
        init_source: Phase 4D. Path of a solved dedicated checkpoint whose
            representation is transferred into the trunk after seeding and before
            training. ``None`` gives Phase 4C's from-scratch initialisation.
        curve_every: Record a curve point every this many integration steps.
            0 disables.
        curve_episodes: Episodes per task per curve point.
        checkpoint_dir: Directory to write ``policy.pt`` into. ``None`` skips the
            checkpoint, which is the historical behaviour and is why Phase 4D's
            cells could not be re-evaluated afterwards.

    Returns:
        The cell record, including the per-task counts needed to verify the
        update-accounting invariant from the result alone.

    Raises:
        ValueError: If fewer than two tasks are given. The trainer itself accepts
            one task -- that is the dedicated case -- but this entry point exists
            to share a trunk, and a one-task "shared" run is a dedicated run that
            would be filed under a shared label and then compared against a
            dedicated baseline as if it were a sharing result.
    """
    if len(tasks) < 2:
        raise ValueError(f"a shared-trunk cell needs at least two tasks to share, got {tasks!r}")

    base = base_seeds_for(config)
    # Exactly the derivation the dedicated runner uses, including train_seed.
    #
    # The first version of this omitted train_seed and produced three "seeds"
    # that trained byte-identical models, so the sweep reported the same number
    # three times and no spread at all. Replication only exists if each replicate
    # draws a different training stream, so this must go through the same helper.
    base = base_seeds_for(config)
    seeds = {t: derive_cell_seeds(base, latency_ns, train_seed) for t in tasks}

    seed_everything(seeds[tasks[0]])
    policy = build_shared_trunk_policy(tasks)

    transfer: dict[str, Any] | None = None
    if init_source is not None:
        transfer = _apply_transfer(
            config, policy, init_source, latency_ns, train_seed, episodes, tasks
        )

    # One trainer, several tasks. The dedicated cell uses the same class with one
    # entry in the map, so the two arms cannot drift apart in their training code.
    trainer = PPOTrainer(
        config,
        {t: SharedTaskPolicy(policy, t) for t in tasks},
        latency_ns,
        seeds=seeds,
    )
    result = trainer.learn(
        total_integration_steps=total_integration_steps,
        log_interval=log_interval,
        curve_every=curve_every,
        curve_episodes=curve_episodes,
    )

    checkpoint_path: str | None = None
    checkpoint_sha: str | None = None
    if checkpoint_dir is not None:
        # Written before the final evaluation, so a completed record always
        # implies a readable checkpoint. The hash is recorded because a
        # duplicate result cannot be diagnosed without knowing the weights
        # actually differed.
        target = Path(checkpoint_dir) / f"{latency_ns}ns" / f"seed{train_seed}" / "policy.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), target)
        checkpoint_path = str(target)
        checkpoint_sha = file_sha256(target)

    evaluations: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_config = _config_for_task(config, task)
        view = SharedTaskPolicy(policy, task)
        summary = evaluate_policy(
            # evaluate_policy reads config.task, so pass the task-specific copy.
            view,
            task_config,
            latency_ns,
            train_seed,
            episodes,
        )
        evaluations[task] = summary_to_dict(summary)

    return SharedCellRecord(
        latency_ns=latency_ns,
        train_seed=train_seed,
        counts={t: result.counts[t].to_dict() for t in tasks},
        planned_iterations=result.planned_iterations,
        evaluations=evaluations,
        wall_time_s=result.wall_time_s,
        task_protocols={
            t: task_protocol(
                config, t, effective_integration_steps=total_integration_steps
            ).to_dict()
            for t in tasks
        },
        transfer_init=transfer,
        curve=[p.to_dict() for p in result.curve],
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
    )


def _apply_transfer(
    config: ProjectConfig,
    policy: SharedTrunkPolicy,
    init_source: Path,
    latency_ns: int,
    train_seed: int,
    episodes: int | None,
    tasks: tuple[str, ...],
) -> dict[str, Any]:
    """Transfer the solved representation and evaluate the model before training.

    The pre-training evaluation is what separates the ways Phase 4D can turn out.
    Without it a poor final LunarLander score is ambiguous between a transfer that
    never carried useful features and a transfer that carried them and was then
    destroyed by joint training. Those are different findings with different
    causes, so the initial state is recorded rather than inferred.

    Evaluated at the cell's own ``latency_ns`` and ``train_seed`` rather than at a
    fixed condition, so the initial and final numbers for a cell share an
    evaluation protocol and can be compared directly.
    """
    report = transfer_from_dedicated(
        policy,
        load_source_policy(Path(init_source)),
        source_path=Path(init_source),
        seed=train_seed,
    )

    initial: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_config = _config_for_task(config, task)
        view = SharedTaskPolicy(policy, task)
        summary = evaluate_policy(
            view,
            task_config,
            latency_ns,
            train_seed,
            episodes,
        )
        initial[task] = summary_to_dict(summary)

    report.initial_evaluations = initial
    return report.to_dict()


def _config_for_task(config: ProjectConfig, task: str) -> ProjectConfig:
    """A copy of the config pointing at ``task``.

    ``evaluate_policy`` reads ``config.task`` to build the environment, and the
    shared run trains several tasks from one config. Rather than add a task
    parameter to the evaluation path -- which would put the shared and dedicated
    evaluation paths at risk of diverging -- the config is copied per task so the
    evaluation code is untouched.
    """
    from dataclasses import replace

    if config.task == task:
        return config
    return replace(config, task=task)


def run_shared_sweep(
    config: ProjectConfig,
    latencies: tuple[int, ...],
    seeds: tuple[int, ...],
    out_dir: Path,
    total_integration_steps: int | None = None,
    episodes: int | None = None,
    log_interval: int = 0,
    max_workers: int = 1,
    worker_threads: int = DEFAULT_WORKER_THREADS,
    config_path: str | Path | None = None,
    init_source: Path | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    save_checkpoints: bool = True,
) -> list[SharedCellRecord]:
    """Train and evaluate every (latency, seed) cell, writing results as it goes.

    Results are written after each cell so an interrupted sweep keeps its
    completed cells instead of losing everything, and a later invocation
    **resumes** from them rather than retraining cells that already have a
    record.

    Args:
        config: Project configuration.
        latencies: Latency conditions.
        seeds: Training seeds.
        out_dir: Results directory.
        total_integration_steps: Per-task budget.
        episodes: Evaluation episodes per task.
        log_interval: Trainer progress interval.
        max_workers: Worker processes. ``1`` runs serially in this process.
        worker_threads: Torch threads per worker.
        config_path: Path to the config, required when ``max_workers > 1`` so
            each worker can parse its own copy.
        init_source: Phase 4D transfer source. Recorded in the run identity below,
            because a sweep whose cells were transferred and a sweep whose cells
            were not are different experiments that must not be mixed, and a
            resume has to know which one it is continuing.
        curve_every: Record a curve point every this many integration steps.
            0 disables.
        curve_episodes: Episodes per task per curve point.
        save_checkpoints: Write each cell's final policy into
            ``out_dir/checkpoints`` and record its SHA-256.

    Returns:
        Every in-scope record, completed now or in an earlier invocation, in
        canonical (latency, seed) order.

    Raises:
        ValueError: If ``max_workers > 1`` without ``config_path``.
    """
    if max_workers > 1 and config_path is None:
        raise ValueError("config_path is required when max_workers > 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "shared_trunk.json"
    budget = total_integration_steps or config.training.total_integration_steps
    episode_count = episodes or config.evaluation.control_episodes_per_seed
    scope = [(latency_ns, seed) for latency_ns in latencies for seed in seeds]

    # Resume guard. Without this, pointing a transfer run at an existing
    # from-scratch directory would silently keep the from-scratch cells and train
    # only the remainder, producing a mixed grid that looks complete. Refusing is
    # the only safe outcome: the two experiments answer different questions.
    existing_path = out_dir / "shared_trunk.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text()).get("init_source") or ""
        except json.JSONDecodeError:
            # Truncated by an interrupted write. There are no trustworthy
            # completed cells either, so there is nothing to be incompatible
            # with; the cells re-run below.
            existing = str(init_source) if init_source is not None else ""
        wanted = str(init_source) if init_source is not None else ""
        if existing != wanted:
            raise ValueError(
                f"{existing_path} holds cells from a different initialisation "
                f"({existing or 'from-scratch'} vs {wanted or 'from-scratch'}). Point "
                f"the transfer run at its own output directory, or remove the stale "
                f"one. Mixing transferred and from-scratch cells would produce a "
                f"complete-looking grid that answers neither question."
            )

    completed = completed_shared_cells(out_dir)
    todo = pending_shared_cells(latencies, seeds, completed)

    if completed and len(todo) < len(scope):
        print(
            f"Resuming shared sweep: {len(scope) - len(todo)}/{len(scope)} complete, "
            f"{len(todo)} pending",
            flush=True,
        )
    else:
        print(
            f"sweep {len(scope)} cells, {len(todo)} to run, "
            f"{max_workers} worker(s) x {worker_threads} threads",
            flush=True,
        )

    started = time.perf_counter()

    def _write(records: Sequence[SharedCellRecord]) -> None:
        by_key = {record.key: record for record in records}
        ordered = [by_key[key] for key in scope if key in by_key]
        payload = {
            "tasks": list(PHASE4_TASKS),
            "latencies_ns": list(latencies),
            "seeds": list(seeds),
            "per_task_budget": budget,
            "episodes_per_task": episode_count,
            "init_source": str(init_source) if init_source is not None else "",
            "cells": [r.to_dict() for r in ordered],
            "elapsed_s": time.perf_counter() - started,
        }
        results_path.write_text(json.dumps(payload, indent=2))

    def _report(record: SharedCellRecord) -> None:
        summary = " | ".join(
            f"{t}={record.evaluations[t].get('mean_return', float('nan')):.1f}"
            for t in PHASE4_TASKS
        )
        print(
            f"  completed  latency {record.latency_ns / 1e6:5.0f} ms seed "
            f"{record.train_seed} | {summary} | {record.wall_time_s:.0f}s",
            flush=True,
        )

    done: dict[tuple[int, int], SharedCellRecord] = {
        key: record for key, record in completed.items() if key in set(scope)
    }

    if max_workers <= 1:
        for latency_ns, seed in todo:
            record = train_shared_cell(
                config,
                latency_ns,
                seed,
                total_integration_steps=total_integration_steps,
                episodes=episodes,
                log_interval=log_interval,
                init_source=init_source,
                curve_every=curve_every,
                curve_episodes=curve_episodes,
                checkpoint_dir=(out_dir / "checkpoints") if save_checkpoints else None,
            )
            done[record.key] = record
            _write(list(done.values()))
            _report(record)
    else:
        checkpoint_dir = (out_dir / "checkpoints") if save_checkpoints else None
        payloads = [
            {
                "config_path": str(config_path),
                "latency_ns": latency_ns,
                "train_seed": seed,
                "total_integration_steps": total_integration_steps,
                "episodes": episodes,
                "log_interval": log_interval,
                "worker_threads": worker_threads,
                "init_source": str(init_source) if init_source is not None else "",
                "curve_every": curve_every,
                "curve_episodes": curve_episodes,
                "checkpoint_dir": checkpoint_dir,
            }
            for latency_ns, seed in todo
        ]
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_shared_worker_init, initargs=(worker_threads,)
        ) as pool:
            for record in pool.map(_train_shared_cell_in_worker, payloads):
                done[record.key] = record
                _write(list(done.values()))
                _report(record)

    print(
        f"shared sweep: {len(done)}/{len(scope)} complete, {len(scope) - len(done)} failed",
        flush=True,
    )
    by_key = done
    return [by_key[key] for key in scope if key in by_key]


def load_shared_records(path: Path) -> list[SharedCellRecord]:
    """Read a shared-trunk sweep file back into records."""
    payload = json.loads(Path(path).read_text())
    return [SharedCellRecord(**cell) for cell in payload["cells"]]


def completed_shared_cells(out_dir: Path) -> dict[tuple[int, int], SharedCellRecord]:
    """Cells already present in a partial results file, keyed by coordinates.

    Returns an empty mapping when no file exists, so a fresh sweep behaves
    exactly as before.
    """
    path = out_dir / "shared_trunk.json"
    if not path.exists():
        return {}
    try:
        records = load_shared_records(path)
    except (json.JSONDecodeError, KeyError, TypeError):
        # A truncated file from an interrupted write must not abort the sweep;
        # the affected cells simply re-run.
        return {}
    return {record.key: record for record in records}


def pending_shared_cells(
    latencies: tuple[int, ...],
    seeds: tuple[int, ...],
    completed: dict[tuple[int, int], SharedCellRecord],
) -> list[tuple[int, int]]:
    """``(latency, seed)`` pairs in scope that still need training.

    Records outside the requested scope are ignored rather than silently
    retained, so narrowing the seed set cannot leave a stale cell looking
    current.
    """
    return [
        (latency_ns, seed)
        for latency_ns in latencies
        for seed in seeds
        if (latency_ns, seed) not in completed
    ]


def _shared_worker_init(num_threads: int) -> None:
    """Pin a worker's thread budget before torch initialises its thread pool.

    Mirrors the dedicated runner's initialiser. Environment variables must be
    set before the first torch operation in the process, so this cannot live in
    the task body.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)


def _train_shared_cell_in_worker(payload: dict[str, Any]) -> SharedCellRecord:
    """Pool entry point: reload the config and train one shared cell.

    The config travels as a path, matching the dedicated runner, so each worker
    parses its own copy and never receives a config object mutated in the parent.
    """
    _shared_worker_init(int(payload["worker_threads"]))
    config = load_config(str(payload["config_path"]))
    source = payload.get("init_source") or None
    return train_shared_cell(
        config,
        int(payload["latency_ns"]),
        int(payload["train_seed"]),
        total_integration_steps=payload["total_integration_steps"],
        episodes=payload["episodes"],
        log_interval=int(payload["log_interval"]),
        init_source=Path(source) if source else None,
        curve_every=int(payload.get("curve_every", 0)),
        curve_episodes=int(payload.get("curve_episodes", 20)),
        checkpoint_dir=(Path(payload["checkpoint_dir"]) if payload.get("checkpoint_dir") else None),
    )
