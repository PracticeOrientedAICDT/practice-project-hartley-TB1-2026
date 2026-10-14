from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from latency_ml.config import load_config
from latency_ml.envs.factory import probe_dt_env_ns
from latency_ml.models.architectures import build_control_policy
from latency_ml.rl.ppo import PPOTrainer, RunCounts
from latency_ml.seeding import Seeds, derive_seed, seed_everything


@dataclass(frozen=True, slots=True)
class SweepCell:
    latency_ns: int
    train_seed: int


@dataclass
class RunManifest:
    run_id: str


@dataclass(frozen=True, slots=True)
class CellOutcome:
    run_id: str
    status: str
    counts: RunCounts | None = None
    error: str | None = None


def base_seeds_for(config):
    return Seeds(training=config.seed, env=config.seed + 10000, eval=config.evaluation.eval_seed_base)


def derive_cell_seeds(base, latency_ns, train_seed):
    return Seeds(training=train_seed, env=derive_seed(base.env, latency_ns, train_seed), eval=base.eval)


def make_activation(name):
    if name == "relu":
        import torch.nn as nn

        return nn.ReLU()
    from latency_ml.models.architectures import bounded_relu

    return bounded_relu()


def run_cell(
    *,
    config,
    cell,
    sweep_id,
    total_integration_steps=None,
    curve_every=0,
    curve_episodes=20,
    root=Path("results"),
    activation=None,
):
    probe_dt_env_ns(config.task)
    rid = f"{sweep_id}/{config.task}/{cell.latency_ns}ns/seed{cell.train_seed}"
    seed = derive_cell_seeds(base_seeds_for(config), cell.latency_ns, cell.train_seed)
    dirp = Path(root) / rid
    dirp.mkdir(parents=True, exist_ok=True)
    try:
        seed_everything(seed)
        act = make_activation(activation) if isinstance(activation, str) else activation
        if act is None and isinstance(activation, str):
            act = make_activation(activation) if activation else make_activation("capped-relu")
        act = make_activation("capped-relu") if not hasattr(act, "forward") else act
        policy = build_control_policy(config.task, activation=act)
        trainer = PPOTrainer(config, {config.task: policy}, cell.latency_ns, seeds={config.task: seed})
        trainer.learn(total_integration_steps=total_integration_steps)
        policy.save(dirp / "policy.pt")
        return CellOutcome(
            run_id=rid,
            status="completed",
            counts=(next(iter(trainer.states.values())).counts if hasattr(trainer, "states") else None),
        )
    except Exception as error:
        return CellOutcome(run_id=rid, status="failed", error=f"{type(error).__name__}: {error}")


def _run_cell_worker(payload):
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
    sweep_id,
    latencies_ns,
    train_seeds,
    total_integration_steps=None,
    max_workers=4,
    worker_threads=2,
    root=Path("results"),
    resume=True,
    verbose=True,
    activation=None,
    **kwargs,
):
    load_config(config_path)
    cells = [SweepCell(latency_ns=latency, train_seed=seed) for latency in latencies_ns for seed in train_seeds]
    payloads = [
        {
            "config_path": str(config_path),
            "sweep_id": sweep_id,
            "latency_ns": c.latency_ns,
            "train_seed": c.train_seed,
            "total_integration_steps": total_integration_steps,
            "root": str(root),
            "worker_threads": worker_threads,
            "activation": activation,
        }
        for c in cells
    ]
    if verbose:
        print(
            f"sweep {sweep_id}: {len(cells)} cells, {len(payloads)} to run, {max_workers} workers x {worker_threads} threads",
            flush=True,
        )
    outcomes = []
    if not payloads:
        return outcomes
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init, initargs=(worker_threads,)) as pool:
        for o in pool.map(_run_cell_worker, payloads):
            outcomes.append(o)
            if verbose:
                print(f"  {o.status:>9}  {o.run_id}  {o.error or 'ok'}", flush=True)
    if verbose:
        failed = [o for o in outcomes if o.status == "failed"]
        print(f"sweep {sweep_id}: {len(outcomes)} ran, {len(failed)} failed", flush=True)
    return outcomes


def _init(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


# append correct make_activation already defined above via alias
