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
from latency_ml.seeding import derive_seed, seed_env

DEFAULT_TRIALS = 20
DEFAULT_HORIZON = 500
# Stability sits just outside the nominal hover envelope of a trained policy
# (|theta| <= ~0.105 rad, |theta-dot| <= ~0.94 rad/s) and well inside the
# termination envelope (|theta| > 0.2095 rad ends the episode).
STABLE_THETA_RAD = 0.12
STABLE_THETA_DOT = 1.0
STABLE_HOLD_STEPS = 20
# The kick is an external horizontal force on the cart, added to the policy's
# own force for a short pulse. 15 N for 6 env frames (120 ms) reliably knocks a
# trained 20 ms policy out of the band (max |theta| 0.12-0.19 rad, inside the
# 0.2095 rad termination angle) while staying recoverable. Pose is continuous
# across the kick: positions integrate from pre-kick velocities, only the
# force (hence acceleration, then velocity) jumps.
KICK_FORCE_N = 15.0
KICK_STEPS = 6
# A fall video keeps rolling past termination until the pole is visibly down:
# CartPole ends the episode at 12 degrees, while the pole still looks upright.
POLE_DOWN_THETA_RAD = 1.5  # ~86 degrees, pole visually horizontal
FALL_EXTENSION_FRAMES = 150  # safety cap, ~5 s


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


def kick_force_for_trial(config: ProjectConfig, latency_ns: int, train_seed: int, trial: int) -> tuple[int, float]:
    """The exact external force push-phase trial ``trial`` uses: seed plus Newtons."""
    seed = derive_seed(config.seed + 500000, latency_ns, train_seed, trial)
    rng = np.random.default_rng(seed)
    return seed, float(KICK_FORCE_N * rng.choice([-1.0, 1.0]))


def step_with_external_force(env: Any, action: Any, external_force: float) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Step the base env with ``external_force`` Newtons added to the policy's force.

    CartPole derives its force from ``force_mag`` and the discrete action, so an
    external push is applied by shifting that magnitude for one step: the pose
    integrates from pre-step velocities exactly as undisturbed physics would.
    """
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "force_mag"):
        return env.step(action)
    base_mag = float(unwrapped.force_mag)
    unwrapped.force_mag = base_mag + external_force if int(action) == 1 else base_mag - external_force
    try:
        return env.step(action)
    finally:
        unwrapped.force_mag = base_mag


@dataclass(frozen=True)
class PushRollout:
    frames: list[np.ndarray]
    flags: list[bool]
    survived: int
    recovered: bool
    knocked_out: bool
    tts_frames: int | None
    decisions_to_stable: int | None
    fell: bool
    final_theta: float


def run_push_rollout(
    policy: Any,
    env: Any,
    obs: np.ndarray,
    action_steps: int,
    max_frames: int,
    force_at: Any,
    disturb_from: int = 0,
    render: bool = False,
    extend_fall: bool = False,
) -> PushRollout:
    """Underlying-frame rollout with an external-force schedule.

    ``force_at(frame)`` gives Newtons of external push for that env frame (0
    outside the pulse); ``disturb_from`` marks the first disturbed frame so
    knock-out and recovery are measured post-kick. The policy keeps choosing
    actions every ``action_steps`` frames, so it fights the shove in real time.
    With ``extend_fall``, a fall keeps rolling past termination (which fires at
    12 degrees, pole still upright) until the pole is visibly down.
    """
    import logging
    import warnings

    frames: list[np.ndarray] = []
    flags: list[bool] = []
    held, decisions = 0, 0
    tts: int | None = None
    decisions_to_stable: int | None = None
    recovered, knocked_out, fell = False, False, False
    frame = 0
    with torch.no_grad():
        action = int(policy.action_only(torch.as_tensor(np.asarray(obs, dtype=np.float32))).item())
    decisions += 1
    while frame < max_frames:
        if frame == disturb_from:
            held = 0
            recovered = False
            tts = None
            decisions_to_stable = None
        obs, _, terminated, truncated, _ = step_with_external_force(env, action, float(force_at(frame)))
        obs = np.asarray(obs, dtype=np.float32)
        frame += 1
        stable = _stable(obs)
        flags.append(stable)
        if frame - 1 >= disturb_from and not stable:
            knocked_out = True
        if stable:
            held += 1
            if held >= STABLE_HOLD_STEPS and not recovered:
                recovered = True
                tts = frame - disturb_from - held
                decisions_to_stable = decisions
        else:
            held = 0
        if render:
            frames.append(np.asarray(env.render(), dtype=np.uint8))
        if terminated or truncated:
            fell = bool(terminated)
            break
        if frame < max_frames and frame % action_steps == 0:
            with torch.no_grad():
                action = int(policy.action_only(torch.as_tensor(obs)).item())
            decisions += 1
    if extend_fall and fell and abs(float(obs[2])) > STABLE_THETA_RAD:
        gym_logger = logging.getLogger("gymnasium")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev, gym_logger.disabled = gym_logger.disabled, True
            try:
                for _ in range(FALL_EXTENSION_FRAMES):
                    obs, _, _, _, _ = step_with_external_force(env, action, 0.0)
                    obs = np.asarray(obs, dtype=np.float32)
                    frame += 1
                    flags.append(False)
                    if render:
                        frames.append(np.asarray(env.render(), dtype=np.uint8))
                    if abs(float(obs[2])) >= POLE_DOWN_THETA_RAD or abs(float(obs[0])) > 3.5:
                        break
            finally:
                gym_logger.disabled = prev
    return PushRollout(
        frames=frames,
        flags=flags,
        survived=frame,
        recovered=recovered,
        knocked_out=knocked_out,
        tts_frames=tts,
        decisions_to_stable=decisions_to_stable,
        fell=fell,
        final_theta=float(obs[2]),
    )


def run_push_trials(
    policy: Any,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    trials: int = DEFAULT_TRIALS,
    horizon: int = DEFAULT_HORIZON,
    dt_env_ns: int | None = None,
) -> PushResult:
    """Kick a CartPole episode with a random external-force pulse and time recovery."""
    from latency_ml.envs.factory import make_base_env, probe_dt_env_ns
    from latency_ml.timing import derive_icdi

    dt = dt_env_ns if dt_env_ns is not None else probe_dt_env_ns(config.task)
    action_steps = derive_icdi(latency_ns, dt).action_steps
    records: list[PushTrial] = []
    for trial in range(trials):
        seed, force = kick_force_for_trial(config, latency_ns, train_seed, trial)
        env = make_base_env(config.task)
        try:
            obs, _ = seed_env(env, seed)
            roll = run_push_rollout(
                policy,
                env,
                np.asarray(obs, dtype=np.float32),
                action_steps,
                horizon,
                lambda frame, force=force: force if frame < KICK_STEPS else 0.0,
            )
        finally:
            env.close()
        records.append(PushTrial(seed=seed, tts_frames=roll.tts_frames, decisions_to_stable=roll.decisions_to_stable, fell=roll.fell))
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
