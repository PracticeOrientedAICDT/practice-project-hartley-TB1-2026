"""Parallel sweep execution.

Runs the cells of a :class:`~latency_ml.experiments.manifest.SweepPlan`, one
worker process per cell, recording a manifest and result for each.

The parallelism is across *runs*, not across environments within a run. A single
PPO rollout is 512 transitions, so spreading one run over subprocesses would
spend more on IPC than on the task; running whole cells in parallel keeps each
process's environment a plain synchronous loop.

Process-level rather than thread-level, because PyTorch's intra-op parallelism
already uses the available cores. Two threads each launching OpenMP would
oversubscribe and the sweep would get *slower*. Each worker instead gets a
fixed thread budget chosen so workers x threads stays near the core count.

Determinism survives the parallelism change: a cell derives its own seeds from
``(base seed, latency, train_seed)`` inside a fresh interpreter, so its result
does not depend on which worker picked it up or in what order. A test runs one
cell serially and in a pool and compares the recorded counts.

Failure handling: a failed cell is recorded with its error and the sweep
continues. One bad cell should not discard the other 49, and a visibly failed
cell is safer than a silently missing one, which would shift every downstream
mean without any trace.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_ml.analysis.parity import task_protocol
from latency_ml.config import ProjectConfig, load_config
from latency_ml.envs.factory import probe_dt_env_ns, probe_env_dims
from latency_ml.experiments.manifest import (
    DEFAULT_RESULTS_ROOT,
    RunManifest,
    SweepCell,
    SweepLedger,
    SweepPlan,
    make_sweep_plan,
    mark_completed,
    mark_failed,
    read_manifest,
    run_id,
    utc_now,
    write_json_atomic,
    write_manifest,
)
from latency_ml.models.architectures import build_control_policy
from latency_ml.rl.ppo import CurvePoint, PPOTrainer, RunCounts
from latency_ml.seeding import Seeds, derive_seed, seed_everything
from latency_ml.timing import derive_icdi

#: Torch intra-op threads per worker, so workers x this stays within the core
#: budget instead of oversubscribing.
DEFAULT_WORKER_THREADS: int = 2

#: Default workers. Modest because each holds a torch context and 4 envs.
DEFAULT_MAX_WORKERS: int = 4


class CellError(RuntimeError):
    """Raised when a cell cannot be set up consistently."""


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """What happened to one sweep cell.

    Attributes:
        run_id: The cell's identifier.
        status: ``"completed"`` or ``"failed"``.
        counts: Run accounting, present only on success.
        error: Failure reason, present only on failure.
    """

    run_id: str
    status: str
    counts: RunCounts | None = None
    error: str | None = None


def derive_cell_seeds(base_seeds: Seeds, latency_ns: int, train_seed: int) -> Seeds:
    """Seeds for one cell.

    Three roles, derived differently on purpose.

    ``training`` is the cell's ``train_seed`` **verbatim**. Latency is
    deliberately *not* mixed in, so every latency condition at a given replicate
    draws the same weight initialisation and the same minibatch order. This is
    common random numbers: it pairs the optimisation noise across conditions, so
    a difference between two latencies is not partly a difference in which
    minibatches happened to be drawn. Each condition still trains on entirely
    different data, because the action-hold factor changes the state trajectory,
    so the runs are genuinely distinct -- only the torch stream is held fixed.

    An earlier version of this docstring claimed latency was mixed into the
    training seed, so that latency conditions drew different initialisations.
    The code never did that, and pairing them is the better experiment. Recorded
    because a comment asserting behaviour the code does not have is the same
    failure mode as the trainer bug in `RESULTS.md` section 4.1: plausible,
    silent, and wrong.

    ``env`` *is* mixed with both latency and ``train_seed``, so each condition
    sees its own reset sequence.

    ``eval`` is deliberately not mixed with anything. `PLAN.md` section 19
    requires evaluation episodes paired across conditions, and mixing latency in
    would break that pairing.
    """
    return Seeds(
        training=train_seed,
        env=derive_seed(base_seeds.env, latency_ns, train_seed),
        eval=base_seeds.eval,
    )


def base_seeds_for(config: ProjectConfig) -> Seeds:
    """The three seed roles a sweep derives its per-cell seeds from."""
    return Seeds(
        training=config.seed,
        env=config.seed + 10_000,
        eval=config.evaluation.eval_seed_base,
    )


def run_cell(
    *,
    config: ProjectConfig,
    cell: SweepCell,
    sweep_id: str,
    total_integration_steps: int | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    root: Path = DEFAULT_RESULTS_ROOT,
) -> CellOutcome:
    """Train one cell, writing its manifest and result.

    Callable from a worker process: it builds its own trainer and environments
    and takes all inputs by value, inheriting no mutable parent state.

    The manifest is written before training and marked ``running``, so an
    interrupted cell is distinguishable from one that never started.
    """
    dt_env_ns = probe_dt_env_ns(config.task)
    seeds = derive_cell_seeds(base_seeds_for(config), cell.latency_ns, cell.train_seed)

    rid = run_id(sweep_id, config.task, cell.latency_ns, cell.train_seed)
    manifest = RunManifest(
        run_id=rid,
        sweep_id=sweep_id,
        task=config.task,
        latency_ns=cell.latency_ns,
        train_seed=cell.train_seed,
        config=config,
        seeds=seeds,
        icdi=derive_icdi(cell.latency_ns, dt_env_ns),
        dt_env_ns=dt_env_ns,
        started_at=utc_now(),
        # Recorded so a later shared-vs-dedicated comparison can verify parity
        # from the artifacts rather than from trust in the code.
        extra={
            "task_protocol": task_protocol(
                config, config.task, effective_integration_steps=total_integration_steps
            ).to_dict()
        },
    )
    write_manifest(manifest, root)

    try:
        obs_dim, n_actions = probe_env_dims(config.task)
        # Seed *before* the policy is constructed, not merely before training.
        # Weight initialisation draws from the global torch RNG, so a policy built
        # from whatever state the process happened to be in gets initial weights
        # that its seed does not determine -- and re-running the cell would not
        # reproduce them. The trainer seeds for the rollout stream; this line is
        # what makes the initial network itself reproducible.
        seed_everything(seeds)
        # One entry in the task map: a dedicated cell. The shared-trunk arm is the
        # same trainer with several, so nothing about this path is specialised.
        policy = build_control_policy(config.task)
        trainer = PPOTrainer(
            config,
            {config.task: policy},
            cell.latency_ns,
            seeds={config.task: seeds},
        )
        if (policy.obs_dim, policy.n_actions) != (obs_dim, n_actions):
            raise CellError(
                f"{rid}: policy is built for obs_dim={policy.obs_dim}, "
                f"n_actions={policy.n_actions} but the environment reports "
                f"obs_dim={obs_dim}, n_actions={n_actions}"
            )
        result = trainer.learn(
            total_integration_steps=total_integration_steps,
            log_interval=0,
            curve_every=curve_every,
            curve_episodes=curve_episodes,
        )
        counts = result.counts[config.task]
        if result.curve:
            _write_curve(rid, cell.latency_ns, cell.train_seed, result.curve, root)
        # Save weights alongside the manifest so the cell can be evaluated
        # without retraining. Written before the manifest is marked complete,
        # so a completed manifest always implies a readable checkpoint.
        checkpoint = root / rid / "policy.pt"
        policy.save(checkpoint)
    except BaseException as error:
        mark_failed(manifest, error, root)
        return CellOutcome(run_id=rid, status="failed", error=f"{type(error).__name__}: {error}")

    manifest.checkpoint_path = checkpoint.name
    mark_completed(manifest, counts, root)
    return CellOutcome(run_id=rid, status="completed", counts=counts)


def _write_curve(
    run_id_str: str,
    latency_ns: int,
    train_seed: int,
    curve: list[CurvePoint],
    root: Path,
) -> Path:
    """Persist a run's training curve next to its manifest.

    Written separately from the manifest because it is a diagnostic artefact
    of a specific question -- is the latency effect merely fewer optimiser
    updates -- and is not present for ordinary sweep cells.
    """
    path = root / run_id_str / "curve.json"
    write_json_atomic(
        path,
        {
            "run_id": run_id_str,
            "latency_ns": latency_ns,
            "train_seed": train_seed,
            "points": [p.to_dict() for p in curve],
        },
    )
    return path


def _worker_init(num_threads: int) -> None:
    """Pin a worker's thread budget before torch initialises its thread pool.

    An initialiser rather than a call inside the task, because the environment
    variables have to be set before the first torch operation in the process.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)


def _run_cell_in_worker(payload: dict[str, Any]) -> CellOutcome:
    """Pool entry point: reload the config and run one cell.

    The config travels as a path rather than an object so each worker parses it
    itself. That keeps the pickled payload small and, more importantly, means a
    worker never receives a config object mutated in the parent.
    """
    _worker_init(int(payload["worker_threads"]))
    config = load_config(str(payload["config_path"]))
    return run_cell(
        config=config,
        cell=SweepCell(
            latency_ns=int(payload["latency_ns"]), train_seed=int(payload["train_seed"])
        ),
        sweep_id=str(payload["sweep_id"]),
        total_integration_steps=payload["total_integration_steps"],
        curve_every=payload["curve_every"],
        curve_episodes=payload["curve_episodes"],
        root=Path(payload["root"]),
    )


def run_sweep(
    *,
    config_path: str | Path,
    sweep_id: str,
    latencies_ns: Sequence[int] | None = None,
    train_seeds: Sequence[int] | None = None,
    total_integration_steps: int | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    max_workers: int = DEFAULT_MAX_WORKERS,
    worker_threads: int = DEFAULT_WORKER_THREADS,
    root: Path = DEFAULT_RESULTS_ROOT,
    resume: bool = True,
    verbose: bool = True,
) -> list[CellOutcome]:
    """Run a latency x seed sweep, in parallel across cells.

    Args:
        config_path: YAML config, parsed in each worker.
        sweep_id: Identifier, also the results subdirectory.
        latencies_ns: Conditions. Defaults to the config's sweep grid.
        train_seeds: Replicates per condition. Defaults to
            ``range(control_training_seeds)``.
        total_integration_steps: Budget override, for pilots.
        curve_every: Record a training-curve point every this many integration
            steps. 0 disables. Used by the fewer-optimiser-updates diagnostic.
        curve_episodes: Episodes per curve point.
        max_workers: Worker processes.
        worker_threads: Torch threads per worker.
        root: Results root.
        resume: Skip cells already recorded in the ledger. Lets an
            interrupted sweep continue instead of restarting.
        verbose: Print per-cell progress.

    Returns:
        One :class:`CellOutcome` per cell that ran in this invocation. Cells
        skipped by ``resume`` are absent, so the length is the work done here,
        not the sweep size.
    """
    config = load_config(config_path)
    grid = tuple(latencies_ns) if latencies_ns is not None else config.latency.sweep_grid_ns
    seeds = (
        tuple(train_seeds)
        if train_seeds is not None
        else tuple(range(config.training.control_training_seeds))
    )

    plan = make_sweep_plan(
        sweep_id=sweep_id,
        task=config.task,
        latencies_ns=grid,
        train_seeds=seeds,
        config=config,
        total_integration_steps=total_integration_steps,
        root=root,
    )

    ledger = SweepLedger(root, sweep_id)
    todo = pending_cells(plan, ledger) if resume else list(plan.cells)
    if verbose:
        print(
            f"sweep {sweep_id}: {len(plan.cells)} cells, "
            f"{len(todo)} to run, {max_workers} workers x {worker_threads} threads",
            flush=True,
        )

    outcomes: list[CellOutcome] = []
    if not todo:
        return outcomes

    payloads = [
        {
            "config_path": str(config_path),
            "sweep_id": sweep_id,
            "latency_ns": cell.latency_ns,
            "train_seed": cell.train_seed,
            "total_integration_steps": total_integration_steps,
            "curve_every": curve_every,
            "curve_episodes": curve_episodes,
            "root": str(root),
            "worker_threads": worker_threads,
        }
        for cell in todo
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_worker_init, initargs=(worker_threads,)
    ) as pool:
        for outcome in pool.map(_run_cell_in_worker, payloads):
            outcomes.append(outcome)
            ledger.record(load_manifest(outcome.run_id, root), root)
            if verbose:
                counts = outcome.counts
                detail = (
                    f"{counts.integration_steps} steps in {counts.wall_time_s:.1f}s"
                    if counts
                    else outcome.error or "failed"
                )
                print(f"  {outcome.status:>9}  {outcome.run_id}  {detail}", flush=True)

    if verbose:
        failed = [o for o in outcomes if o.status == "failed"]
        print(f"sweep {sweep_id}: {len(outcomes)} ran, {len(failed)} failed", flush=True)
    return outcomes


def pending_cells(plan: SweepPlan, ledger: SweepLedger) -> list[SweepCell]:
    """Cells in the plan with no terminal ledger entry."""
    done = set(ledger.completed())
    return [
        cell
        for cell in plan.cells
        if run_id(plan.sweep_id, plan.task, cell.latency_ns, cell.train_seed) not in done
    ]


def load_manifest(run_id_str: str, root: Path = DEFAULT_RESULTS_ROOT) -> RunManifest:
    """Read a manifest back from disk, for the ledger and for auditing.

    Re-reads rather than reusing the in-memory manifest, so the ledger stays
    consistent with what was actually persisted.

    Raises:
        FileNotFoundError: If the run has no manifest.
    """
    return read_manifest(run_id_str, root)


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_WORKER_THREADS",
    "CellError",
    "CellOutcome",
    "base_seeds_for",
    "derive_cell_seeds",
    "load_manifest",
    "pending_cells",
    "run_cell",
    "run_sweep",
]
