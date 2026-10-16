"""Random-push phase: time-to-stability after a random kick.

Return-based control metrics saturate once a policy survives the full
episode, so they cannot separate good from great policies at high latency.
Pushing the pole with a random kick and measuring how long the policy needs
to bring it back to the stable manifold is a non-saturated measure: it keeps
discriminating between latencies and activations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

from latency_ml.config import ProjectConfig, load_config
from latency_ml.envs.factory import make_latency_env
from latency_ml.envs.latency_env import HOLD_STEPS_KEY
from latency_ml.seeding import derive_seed, seed_env

DEFAULT_TRIALS = 20
DEFAULT_HORIZON = 500
# Stability sits just outside the nominal hover envelope of a trained policy
# (|theta| <= ~0.105 rad, |theta-dot| <= ~0.94 rad/s) and well inside the
# termination envelope (|theta| > 0.2095 rad ends the episode).
STABLE_THETA_RAD = 0.12
STABLE_THETA_DOT = 1.0
STABLE_HOLD_STEPS = 20
# The impulse kick is scaled to exceed the stability band so recovery is
# measurable, and to fall short of the termination angle often enough that
# strong kicks fail and censored trials carry signal.
KICK_THETA_RAD = 0.15
KICK_THETA_DOT = 1.2
KICK_X = 0.15
KICK_X_DOT = 0.3


@dataclass(frozen=True)
class PushTrial:
    seed: int
    tts_frames: int | None
    decisions_to_stable: int | None
    fell: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "tts_frames": self.tts_frames,
            "decisions_to_stable": self.decisions_to_stable,
            "fell": self.fell,
            "censored": self.tts_frames is None,
        }


@dataclass(frozen=True)
class PushResult:
    task: str
    latency_ns: int
    train_seed: int
    action_steps: int
    dt_env_ns: int
    trials: list[PushTrial]

    @property
    def successes(self) -> list[PushTrial]:
        return [t for t in self.trials if t.tts_frames is not None]

    def summary(self) -> dict[str, Any]:
        tts = [t.tts_frames for t in self.successes]
        n = len(self.trials)
        return {
            "task": self.task,
            "latency_ns": self.latency_ns,
            "train_seed": self.train_seed,
            "action_steps": self.action_steps,
            "dt_env_ns": self.dt_env_ns,
            "n_trials": n,
            "success_rate": float(len(tts) / n) if n else 0.0,
            "mean_tts_frames": float(mean(tts)) if tts else None,
            "median_tts_frames": float(median(tts)) if tts else None,
            "mean_tts_decisions": float(mean(t.decisions_to_stable for t in self.successes)) if tts else None,
            "mean_tts_ms": float(mean(tts) * self.dt_env_ns / 1e6) if tts else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "trials": [t.to_dict() for t in self.trials]}


def _stable(obs: np.ndarray) -> bool:
    theta, theta_dot = float(obs[2]), float(obs[3])
    return abs(theta) <= STABLE_THETA_RAD and abs(theta_dot) <= STABLE_THETA_DOT


def run_push_trials(
    policy: Any,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    trials: int = DEFAULT_TRIALS,
    horizon: int = DEFAULT_HORIZON,
    dt_env_ns: int | None = None,
) -> PushResult:
    """Kick a CartPole episode with a random state perturbation and time recovery."""
    env = make_latency_env(config.task, latency_ns, dt_env_ns=dt_env_ns)
    action_steps = env.action_steps
    dt = env.dt_env_ns
    records: list[PushTrial] = []
    for trial in range(trials):
        seed = derive_seed(config.seed + 500000, latency_ns, train_seed, trial)
        rng = np.random.default_rng(seed)
        obs, _ = seed_env(env, seed)
        kick = [
            rng.uniform(-KICK_X, KICK_X),
            rng.uniform(-KICK_X_DOT, KICK_X_DOT),
            rng.uniform(-KICK_THETA_RAD, KICK_THETA_RAD),
            rng.uniform(-KICK_THETA_DOT, KICK_THETA_DOT),
        ]
        env.unwrapped.state = np.asarray(kick, dtype=np.float32)
        obs = np.asarray(kick, dtype=np.float32)
        held_frames = 0
        frames = 0
        tts: int | None = None
        decisions_to_stable: int | None = None
        fell = False
        decisions = 0
        while frames < horizon:
            with torch.no_grad():
                action = int(policy.action_only(torch.as_tensor(obs, dtype=torch.float32)).item())
            decisions += 1
            obs, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            hold = int(info.get(HOLD_STEPS_KEY, 1))
            frames += hold
            if _stable(obs):
                held_frames += hold
                if tts is None and held_frames >= STABLE_HOLD_STEPS:
                    tts = frames - held_frames
                    decisions_to_stable = decisions
            else:
                held_frames = 0
            if terminated or truncated:
                fell = fell or bool(terminated)
                break
        records.append(PushTrial(seed=seed, tts_frames=tts, decisions_to_stable=decisions_to_stable, fell=fell))
    env.close()
    return PushResult(
        task=config.task,
        latency_ns=latency_ns,
        train_seed=train_seed,
        action_steps=action_steps,
        dt_env_ns=dt,
        trials=records,
    )


def aggregate_push(results: list[PushResult]) -> list[dict[str, Any]]:
    """Mean per-seed statistics grouped by latency."""
    by: dict[int, list[PushResult]] = {}
    for r in results:
        by.setdefault(r.latency_ns, []).append(r)
    out: list[dict[str, Any]] = []
    for latency_ns in sorted(by):
        group = sorted(by[latency_ns], key=lambda r: r.train_seed)
        summaries = [r.summary() for r in group]
        tts_means = [s["mean_tts_frames"] for s in summaries if s["mean_tts_frames"] is not None]
        ms_means = [s["mean_tts_ms"] for s in summaries if s["mean_tts_ms"] is not None]
        dec_means = [s["mean_tts_decisions"] for s in summaries if s["mean_tts_decisions"] is not None]
        out.append(
            {
                "latency_ns": latency_ns,
                "action_steps": group[0].action_steps,
                "n_seeds": len(group),
                "n_trials": sum(s["n_trials"] for s in summaries),
                "success_rate": float(mean(s["success_rate"] for s in summaries)),
                "mean_tts_frames": float(mean(tts_means)) if tts_means else None,
                "mean_tts_decisions": float(mean(dec_means)) if dec_means else None,
                "mean_tts_ms": float(mean(ms_means)) if ms_means else None,
                "per_seed_tts_frames": [s["mean_tts_frames"] for s in summaries],
            }
        )
    return out


def _load_dedicated_policy(config: ProjectConfig, root: Path, sweep_id: str, latency_ns: int, seed: int, activation: Any) -> Any:
    from latency_ml.experiments.evaluate import load_policy

    path = root / sweep_id / config.task / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt"
    policy = load_policy(config, config.task, activation=activation)
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return policy


def _load_shared_policy(config: ProjectConfig, root: Path, sweep_id: str, latency_ns: int, seed: int, activation: Any) -> Any:
    from latency_ml.models.shared_trunk import build_shared_trunk_policy
    from latency_ml.rl.task_policy import SharedTaskPolicy

    path = root / sweep_id / "shared" / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt"
    model = build_shared_trunk_policy(("cartpole", "lunarlander"), activation=activation)
    model.load_shared(path)
    model.eval()
    return SharedTaskPolicy(model, config.task)


def run_push_sweep(
    *,
    sweep_id: str,
    task: str = "cartpole",
    kind: str = "auto",
    config_path: str | Path | None = None,
    root: Path = Path("results"),
    latencies_ns: list[int] | None = None,
    train_seeds: list[int] | None = None,
    activation: str | None = None,
    trials: int = DEFAULT_TRIALS,
    horizon: int = DEFAULT_HORIZON,
    verbose: bool = True,
) -> Path:
    """Run the push phase over an existing sweep's checkpoints and write summaries."""
    from latency_ml.experiments.runner import make_activation

    config = load_config(config_path) if config_path else _default_config(task)
    if kind == "auto":
        kind = "shared" if (Path(root) / sweep_id / "shared").exists() else "dedicated"
    if kind not in ("dedicated", "shared"):
        raise ValueError(f"unknown push kind {kind!r}")
    load = _load_dedicated_policy if kind == "dedicated" else _load_shared_policy
    act = make_activation(activation or "capped-relu")
    latencies = latencies_ns if latencies_ns is not None else list(config.latency.sweep_grid_ns)
    seeds = train_seeds if train_seeds is not None else list(range(config.training.control_training_seeds))
    out_dir = Path(root) / sweep_id / "push" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PushResult] = []
    for latency_ns in latencies:
        for seed in seeds:
            try:
                policy = load(config, Path(root), sweep_id, latency_ns, seed, act)
            except FileNotFoundError:
                if verbose:
                    print(f"  missing {sweep_id}/{kind}/{latency_ns}ns/seed{seed}", flush=True)
                continue
            result = run_push_trials(policy, config, latency_ns, seed, trials=trials, horizon=horizon)
            results.append(result)
            (out_dir / f"{latency_ns}ns").mkdir(parents=True, exist_ok=True)
            (out_dir / f"{latency_ns}ns" / f"seed{seed}.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            if verbose:
                s = result.summary()
                tts = "   n/a" if s["mean_tts_frames"] is None else f"{s['mean_tts_frames']:6.1f}"
                print(f"  {latency_ns / 1000000:5.0f}ms seed{seed:<3d} tts={tts} steps  success={s['success_rate']:.2f}", flush=True)
    aggregates = aggregate_push(results)
    payload = {
        "sweep_id": sweep_id,
        "task": task,
        "kind": kind,
        "activation": activation or "capped-relu",
        "per_latency": aggregates,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if verbose:
        print(f"push sweep {sweep_id}/{kind}: {len(results)} policies, wrote {summary_path}", flush=True)
    return summary_path


def _default_config(task: str) -> ProjectConfig:
    return load_config(Path(__file__).resolve().parents[3] / "configs" / f"{task}.yaml")
