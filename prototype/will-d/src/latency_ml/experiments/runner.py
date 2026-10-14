from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    write_manifest,
)
from latency_ml.models.architectures import build_control_policy
from latency_ml.rl.ppo import PPOTrainer, RunCounts
from latency_ml.seeding import Seeds, derive_seed, seed_everything
from latency_ml.timing import derive_icdi

DEFAULT_WORKER_THREADS: int = 2
DEFAULT_MAX_WORKERS: int = 4


class CellError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CellOutcome:
    run_id: str
    status: str
    counts: RunCounts | None = None
    error: str | None = None


def derive_cell_seeds(base_seeds: Seeds, latency_ns: int, train_seed: int) -> Seeds:
    return Seeds(training=train_seed, env=derive_seed(base_seeds.env, latency_ns, train_seed), eval=base_seeds.eval)


def base_seeds_for(config: ProjectConfig) -> Seeds:
    return Seeds(training=config.seed, env=config.seed + 10000, eval=config.evaluation.eval_seed_base)


def run_cell(
    *,
    config: ProjectConfig,
    cell: SweepCell,
    sweep_id: str,
    total_integration_steps: int | None = None,
    curve_every: int = 0,
    curve_episodes: int = 20,
    root: Path = DEFAULT_RESULTS_ROOT,
    activation: str | None = None,
) -> CellOutcome:
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
    )
    write_manifest(manifest, root)
    try:
        obs_dim, n_actions = probe_env_dims(config.task)
        seed_everything(seeds)
        import torch.nn as _nn

        from latency_ml.models.activations import bounded_relu as _bounded

        if activation == "relu":
            _act = _nn.ReLU()
        elif activation in (None, "", "capped-relu", "bounded-relu"):
            _act = _bounded()
        else:
            _act = _bounded()
        policy = build_control_policy(config.task, activation=_act)
        trainer = PPOTrainer(config, {config.task: policy}, cell.latency_ns, seeds={config.task: seeds})
        if (policy.obs_dim, policy.n_actions) != (obs_dim, n_actions):
            raise CellError(f"{rid}: policy is built for obs_dim={policy.obs_dim}, n_actions={policy.n_actions} but the environment reports obs_dim={obs_dim}, n_actions={n_actions}")
        result = trainer.learn(total_integration_steps=total_integration_steps)
        counts = result.counts[config.task]
        checkpoint = root / rid / "policy.pt"
        policy.save(checkpoint)
    except BaseException as error:
        mark_failed(manifest, error, root)
        return CellOutcome(run_id=rid, status="failed", error=f"{type(error).__name__}: {error}")
    manifest.checkpoint_path = checkpoint.name
    mark_completed(manifest, counts, root)
    return CellOutcome(run_id=rid, status="completed", counts=counts)


def _worker_init(num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)


def _run_cell_in_worker(payload: dict[str, Any]) -> CellOutcome:
    _worker_init(int(payload["worker_threads"]))
    config = load_config(str(payload["config_path"]))
    return run_cell(
        config=config,
        cell=SweepCell(latency_ns=int(payload["latency_ns"]), train_seed=int(payload["train_seed"])),
        sweep_id=str(payload["sweep_id"]),
        total_integration_steps=payload["total_integration_steps"],
        curve_every=payload["curve_every"],
        curve_episodes=payload["curve_episodes"],
        root=Path(payload["root"]),
        activation=payload.get("activation"),
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
    activation: str | None = None,
) -> list[CellOutcome]:
    config = load_config(config_path)
    grid = tuple(latencies_ns) if latencies_ns is not None else config.latency.sweep_grid_ns
    seeds = tuple(train_seeds) if train_seeds is not None else tuple(range(config.training.control_training_seeds))
    plan = make_sweep_plan(sweep_id=sweep_id, task=config.task, latencies_ns=grid, train_seeds=seeds, config=config, total_integration_steps=total_integration_steps, root=root)
    ledger = SweepLedger(root, sweep_id)
    todo = pending_cells(plan, ledger) if resume else list(plan.cells)
    if verbose:
        print(f"sweep {sweep_id}: {len(plan.cells)} cells, {len(todo)} to run, {max_workers} workers x {worker_threads} threads", flush=True)
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
            "activation": activation,
        }
        for cell in todo
    ]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(worker_threads,)) as pool:
        for outcome in pool.map(_run_cell_in_worker, payloads):
            outcomes.append(outcome)
            ledger.record(load_manifest(outcome.run_id, root), root)
            if verbose:
                counts = outcome.counts
                detail = f"{counts.integration_steps} steps in {counts.wall_time_s:.1f}s" if counts else outcome.error or "failed"
                print(f"  {outcome.status:>9}  {outcome.run_id}  {detail}", flush=True)
    if verbose:
        failed = [o for o in outcomes if o.status == "failed"]
        print(f"sweep {sweep_id}: {len(outcomes)} ran, {len(failed)} failed", flush=True)
    return outcomes


def pending_cells(plan: SweepPlan, ledger: SweepLedger) -> list[SweepCell]:
    done = set(ledger.completed())
    return [cell for cell in plan.cells if run_id(plan.sweep_id, plan.task, cell.latency_ns, cell.train_seed) not in done]


def load_manifest(run_id_str: str, root: Path = DEFAULT_RESULTS_ROOT) -> RunManifest:
    return read_manifest(run_id_str, root)


__all__ = ["DEFAULT_MAX_WORKERS", "DEFAULT_WORKER_THREADS", "CellError", "CellOutcome", "base_seeds_for", "derive_cell_seeds", "load_manifest", "pending_cells", "run_cell", "run_sweep"]


def make_activation(name):
    if name == "relu":
        import torch.nn as nn

        return nn.ReLU()
    from latency_ml.models.activations import bounded_relu

    return bounded_relu()
