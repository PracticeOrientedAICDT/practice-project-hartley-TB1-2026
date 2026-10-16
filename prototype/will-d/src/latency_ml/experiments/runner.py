from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from latency_ml.config import ProjectConfig, load_config
from latency_ml.envs.factory import probe_dt_env_ns
from latency_ml.models.architectures import build_control_policy
from latency_ml.rl.ppo import PPOTrainer, RunCounts
from latency_ml.seeding import Seeds, derive_seed, seed_everything


@dataclass(frozen=True, slots=True)
class SweepCell:
    latency_ns: int
    train_seed: int


@dataclass
class CellOutcome:
    run_id: str
    status: str
    counts: RunCounts | None = None
    error: str | None = None


def base_seeds_for(config: ProjectConfig) -> Seeds:
    return Seeds(training=config.seed, env=config.seed + 10000, eval=config.evaluation.eval_seed_base)


def derive_cell_seeds(base: Seeds, latency_ns: int, train_seed: int) -> Seeds:
    return Seeds(training=train_seed, env=derive_seed(base.env, latency_ns, train_seed), eval=base.eval)


def make_activation(name: str | None):
    if name == "relu":
        import torch.nn as nn

        return nn.ReLU()
    from latency_ml.models.architectures import bounded_relu

    return bounded_relu()


def cell_run_id(sweep_id: str, task: str, latency_ns: int, train_seed: int) -> str:
    return f"{sweep_id}/{task}/{latency_ns}ns/seed{train_seed}"


def run_cell(
    *,
    config: ProjectConfig,
    cell: SweepCell,
    sweep_id: str,
    total_integration_steps: int | None = None,
    root: Path = Path("results"),
    activation: object | None = None,
) -> CellOutcome:
    """Train one (latency, seed) cell of a dedicated single-task sweep."""
    probe_dt_env_ns(config.task)
    rid = cell_run_id(sweep_id, config.task, cell.latency_ns, cell.train_seed)
    seed = derive_cell_seeds(base_seeds_for(config), cell.latency_ns, cell.train_seed)
    dirp = Path(root) / rid
    dirp.mkdir(parents=True, exist_ok=True)
    try:
        seed_everything(seed)
        act = make_activation(activation) if isinstance(activation, str) or activation is None else activation
        policy = build_control_policy(config.task, activation=act)
        trainer = PPOTrainer(config, {config.task: policy}, cell.latency_ns, seeds={config.task: seed})
        trainer.learn(total_integration_steps=total_integration_steps)
        policy.save(dirp / "policy.pt")
        counts = trainer.states[config.task].counts if hasattr(trainer, "states") else None
        return CellOutcome(run_id=rid, status="completed", counts=counts)
    except Exception as error:  # one bad cell must not kill the sweep
        return CellOutcome(run_id=rid, status="failed", error=f"{type(error).__name__}: {error}")


def _run_cell_worker(payload: dict) -> CellOutcome:
    os.environ.setdefault("OMP_NUM_THREADS", str(payload.get("worker_threads", 2)))
    config = load_config(payload["config_path"])
    return run_cell(
        config=config,
        cell=SweepCell(payload["latency_ns"], payload["train_seed"]),
        sweep_id=payload["sweep_id"],
        total_integration_steps=payload["total_integration_steps"],
        root=Path(payload["root"]),
        activation=payload.get("activation"),
    )


def run_sweep(
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
) -> list[CellOutcome]:
    """Train every (latency, seed) cell of a dedicated sweep, resuming what exists."""
    config = load_config(config_path)
    task = config.task
    cells = [SweepCell(latency_ns=latency, train_seed=seed) for latency in latencies_ns for seed in train_seeds]
    outcomes: list[CellOutcome] = []
    payloads: list[dict] = []
    for cell in cells:
        rid = cell_run_id(sweep_id, task, cell.latency_ns, cell.train_seed)
        if resume and (Path(root) / rid / "policy.pt").exists():
            outcomes.append(CellOutcome(run_id=rid, status="skipped"))
            continue
        payloads.append(
            {
                "config_path": str(config_path),
                "sweep_id": sweep_id,
                "latency_ns": cell.latency_ns,
                "train_seed": cell.train_seed,
                "total_integration_steps": total_integration_steps,
                "root": str(root),
                "worker_threads": worker_threads,
                "activation": activation,
            }
        )
    if verbose:
        print(f"sweep {sweep_id}: {len(cells)} cells, {len(payloads)} to run, {max_workers} workers x {worker_threads} threads", flush=True)
    if payloads:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init, initargs=(worker_threads,)) as pool:
            for outcome in pool.map(_run_cell_worker, payloads):
                outcomes.append(outcome)
                if verbose:
                    print(f"  {outcome.status:>9}  {outcome.run_id}  {outcome.error or 'ok'}", flush=True)
    if verbose:
        failed = [o for o in outcomes if o.status == "failed"]
        print(f"sweep {sweep_id}: {len(outcomes)} cells, {len(failed)} failed", flush=True)
    return outcomes


def _init(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
