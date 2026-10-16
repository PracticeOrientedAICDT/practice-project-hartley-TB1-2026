"""Rollout videos: record a trained policy as an mp4 for the results viewer.

The loop below replays decisions through a rendered base env with the same
action-hold semantics as :class:`LatencyHoldEnv` (hold each action for
``action_steps`` frames, stop the hold early on termination), so what you see
is what the policy saw. One video per sweep / task / latency, using the
median-seed policy, keeps the videos tab comparable across latencies.
"""

from __future__ import annotations

import json
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latency_ml.config import ProjectConfig, load_config
from latency_ml.envs.factory import make_base_env, probe_dt_env_ns
from latency_ml.seeding import seed_env
from latency_ml.timing import derive_icdi

DEFAULT_FPS = 30
DEFAULT_MAX_FRAMES = 360
DEFAULT_PUSH_MAX_FRAMES = 420
DEFAULT_PREAMBLE_FRAMES = 60
DEFAULT_EPISODE_SEED = 7
VIDEO_WIDTH = 480


def _pick_push_trial(simulate, max_candidates: int = 10) -> int:
    """First trial knocked out of the band that recovers; else first recovery; else longest survival."""
    best, best_survived = 0, -1
    first_recovered = None
    for trial in range(max_candidates):
        survived, recovered, knocked_out = simulate(trial)
        if recovered and first_recovered is None:
            first_recovered = trial
        if recovered and knocked_out:
            return trial
        if survived > best_survived:
            best, best_survived = trial, survived
    return first_recovered if first_recovered is not None else best


def _simulate_push_trial(
    policy: Any,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    trial: int,
    max_frames: int,
    preamble_frames: int = DEFAULT_PREAMBLE_FRAMES,
) -> tuple[int, bool, bool]:
    """Outcome of the exact video rollout without rendering: (survived, recovered post-kick, knocked out post-kick)."""
    from latency_ml.experiments.push import KICK_STEPS, kick_force_for_trial, run_push_rollout
    from latency_ml.seeding import derive_seed

    action_steps = derive_icdi(latency_ns, probe_dt_env_ns(config.task)).action_steps
    env = make_base_env(config.task)
    try:
        pre_seed = derive_seed(config.seed + 600000, latency_ns, train_seed)
        obs, _ = seed_env(env, pre_seed)
        _, force = kick_force_for_trial(config, latency_ns, train_seed, trial)
        pre_target = min(preamble_frames, max_frames)
        roll = run_push_rollout(
            policy,
            env,
            np.asarray(obs, dtype=np.float32),
            action_steps,
            max_frames,
            lambda frame, force=force: force if pre_target <= frame < pre_target + KICK_STEPS else 0.0,
            disturb_from=pre_target,
        )
        return roll.survived, roll.recovered, roll.knocked_out
    finally:
        env.close()


def select_median_seed(evaluation: dict[str, Any], latency_ns: int) -> int:
    """Train seed whose mean return is the median at ``latency_ns``."""
    for row in evaluation.get("latency_response", []):
        if int(row["latency_ns"]) == latency_ns:
            returns = [float(v) for v in row.get("per_seed_returns", [])]
            if not returns:
                raise ValueError(f"no per-seed returns for latency {latency_ns}")
            order = sorted(range(len(returns)), key=lambda i: returns[i])
            return order[len(order) // 2]
    raise KeyError(f"latency {latency_ns} not in evaluation")


def rollout_frames(
    policy: Any,
    task: str,
    latency_ns: int,
    *,
    episode_seed: int = DEFAULT_EPISODE_SEED,
    max_frames: int = DEFAULT_MAX_FRAMES,
    dt_env_ns: int | None = None,
    env: Any = None,
) -> tuple[list[np.ndarray], list[bool]]:
    """Run one episode and return (rendered frames, per-frame stable flags).

    CartPole shares the push rollout core, so control videos carry the same
    STABLE/UNSTABLE pills and the same fall rule: a fall plays out past the
    12-degree termination until the pole is visibly down. Other tasks return no
    flags. The first frame is the reset, as before.
    """
    from latency_ml.experiments.push import _stable, run_push_rollout

    dt = dt_env_ns if dt_env_ns is not None else probe_dt_env_ns(task)
    action_steps = derive_icdi(latency_ns, dt).action_steps
    own = env is None
    if env is None:
        env = make_base_env(task, render_mode="rgb_array")
    try:
        obs, _ = seed_env(env, episode_seed)
        first = np.asarray(env.render(), dtype=np.uint8)
        if task == "cartpole":
            roll = run_push_rollout(
                policy,
                env,
                np.asarray(obs, dtype=np.float32),
                action_steps,
                max(0, max_frames - 1),
                lambda frame: 0.0,
                render=True,
                extend_fall=True,
            )
            return [first, *roll.frames], [_stable(np.asarray(obs, dtype=np.float32)), *roll.flags]
        frames = [first]
        done = False
        while len(frames) < max_frames and not done:
            with torch.no_grad():
                action = int(policy.action_only(torch.as_tensor(np.asarray(obs, dtype=np.float32))).item())
            for _ in range(action_steps):
                obs, _, terminated, truncated, _ = env.step(action)
                frames.append(np.asarray(env.render(), dtype=np.uint8))
                if terminated or truncated:
                    done = True
                    break
                if len(frames) >= max_frames:
                    break
        return frames, []
    finally:
        if own:
            env.close()


def rollout_push_frames(
    policy: Any,
    config: ProjectConfig,
    latency_ns: int,
    train_seed: int,
    *,
    trial: int = 0,
    preamble_frames: int = DEFAULT_PREAMBLE_FRAMES,
    max_frames: int = DEFAULT_PUSH_MAX_FRAMES,
    dt_env_ns: int | None = None,
    env: Any = None,
) -> tuple[list[np.ndarray], list[bool], int]:
    """Stable hover, then the push-phase force pulse, then recovery: (frames, stable flags, kick index)."""
    from latency_ml.experiments.push import KICK_STEPS, kick_force_for_trial, run_push_rollout
    from latency_ml.seeding import derive_seed

    dt = dt_env_ns if dt_env_ns is not None else probe_dt_env_ns(config.task)
    action_steps = derive_icdi(latency_ns, dt).action_steps
    own = env is None
    if env is None:
        env = make_base_env(config.task, render_mode="rgb_array")
    try:
        pre_seed = derive_seed(config.seed + 600000, latency_ns, train_seed)
        obs, _ = seed_env(env, pre_seed)
        _, force = kick_force_for_trial(config, latency_ns, train_seed, trial)
        pre_target = min(preamble_frames, max_frames)
        roll = run_push_rollout(
            policy,
            env,
            np.asarray(obs, dtype=np.float32),
            action_steps,
            max_frames,
            lambda frame, force=force: force if pre_target <= frame < pre_target + KICK_STEPS else 0.0,
            disturb_from=pre_target,
            render=True,
            extend_fall=True,
        )
        return roll.frames, roll.flags, min(pre_target, roll.survived)
    finally:
        if own:
            env.close()


def write_video(
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: int = DEFAULT_FPS,
    label: str = "",
    flash: str | None = None,
    flash_at: int = 0,
    flash_frames: int = 15,
    stable: list[bool] | None = None,
) -> Path:
    """Encode ``frames`` as mp4 with an optional burned-in label (and kick flash)."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("no frames to write")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    for frame in frames:
        img = Image.fromarray(frame)
        if img.width > VIDEO_WIDTH:
            img = img.resize((VIDEO_WIDTH, round(img.height * VIDEO_WIDTH / img.width)))
        thumbs.append(img.convert("RGB"))
    if label:
        draw = ImageDraw.Draw(thumbs[0])
        draw.rectangle([6, 6, 6 + draw.textlength(label) + 12, 24], fill=(0, 0, 0))
        draw.text((12, 10), label, fill=(255, 255, 255))
        banner = thumbs[0].copy()
        for img in thumbs[1:]:
            img.paste(banner.crop((0, 0, banner.width, 28)), (0, 0))
    if flash:
        for img in thumbs[flash_at : flash_at + flash_frames]:
            draw = ImageDraw.Draw(img)
            cx = img.width / 2
            draw.rectangle([cx - draw.textlength(flash) / 2 - 10, 6, cx + draw.textlength(flash) / 2 + 10, 26], fill=(200, 30, 30))
            draw.text((cx - draw.textlength(flash) / 2, 10), flash, fill=(255, 255, 255))
    if stable is not None:
        for img, ok in zip(thumbs, stable, strict=True):
            draw = ImageDraw.Draw(img)
            text = "STABLE" if ok else "UNSTABLE"
            color = (30, 150, 60) if ok else (200, 30, 30)
            x0 = img.width - draw.textlength(text) - 22
            draw.rounded_rectangle([x0, 6, img.width - 10, 26], radius=6, fill=color)
            draw.text((x0 + 6, 10), text, fill=(255, 255, 255))
    w, h = thumbs[0].size
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.run(cmd, input=b"".join(img.tobytes() for img in thumbs), capture_output=True)
    proc.check_returncode()
    return out


def _activation_for(sweep_id: str, root: Path) -> str:
    if sweep_id.endswith("-capped"):
        return "capped-relu"
    if sweep_id.endswith("-relu"):
        return "relu"
    for summary in (Path(root) / sweep_id / "push").glob("*/summary.json"):
        act = json.loads(summary.read_text()).get("activation")
        if act:
            return str(act)
    raise ValueError(f"cannot infer activation for sweep {sweep_id!r}")


def _sweep_tasks(sweep_dir: Path) -> tuple[str, ...]:
    if (sweep_dir / "shared").exists():
        return ("cartpole", "lunarlander")
    for child in sorted(sweep_dir.iterdir()):
        if child.is_dir() and child.name not in ("push", "videos"):
            return (child.name,)
    raise FileNotFoundError(f"no task checkpoints in {sweep_dir}")


def _eval_for(root: Path, sweep_id: str, kind: str, task: str) -> dict[str, Any]:
    sub = root / f"{sweep_id}-shared-{task}" if kind == "shared" else root / sweep_id
    return json.loads((sub / "evaluation.json").read_text())


def render_missing(
    root: str | Path = Path("results"),
    sweep_ids: list[str] | None = None,
    episode_seed: int = DEFAULT_EPISODE_SEED,
    fps: int = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    modes: tuple[str, ...] = ("control", "push"),
    verbose: bool = True,
) -> list[Path]:
    """Render control and/or push-trial videos per sweep / task / latency, skipping existing files."""
    from latency_ml.experiments.push import _load_dedicated_policy, _load_shared_policy
    from latency_ml.experiments.runner import make_activation

    root = Path(root)
    if sweep_ids is None:
        sweep_ids = sorted(p.parent.name for p in root.glob("*/evaluation.json") if "-shared-" not in p.parent.name)
    written: list[Path] = []
    for sweep_id in sweep_ids:
        sweep_dir = root / sweep_id
        kind = "shared" if (sweep_dir / "shared").exists() else "dedicated"
        activation = _activation_for(sweep_id, root)
        act = make_activation(activation)
        load = _load_dedicated_policy if kind == "dedicated" else _load_shared_policy
        for task in _sweep_tasks(sweep_dir):
            config = load_config(Path(__file__).resolve().parents[3] / "configs" / f"{task}.yaml")
            evaluation = _eval_for(root, sweep_id, kind, task)
            latencies = [int(r["latency_ns"]) for r in evaluation.get("latency_response", [])]
            for latency_ns in latencies:
                seed = select_median_seed(evaluation, latency_ns)
                try:
                    policy = load(config, root, sweep_id, latency_ns, seed, act)
                except FileNotFoundError:
                    if verbose:
                        print(f"  missing {sweep_id}/{task}/{latency_ns}ns/seed{seed}", flush=True)
                    continue
                ms = latency_ns // 1_000_000
                if "control" in modes:
                    out = sweep_dir / "videos" / f"{task}_{ms}ms_seed{seed}.mp4"
                    if not out.exists():
                        label = f"{activation} - {task} - {ms}ms - seed{seed}"
                        frames, flags = rollout_frames(policy, task, latency_ns, episode_seed=episode_seed, max_frames=max_frames)
                        write_video(frames, out, fps=fps, label=label, stable=flags or None)
                        written.append(out)
                        if verbose:
                            print(f"  wrote {out} ({len(frames)} frames)", flush=True)
                if "push" in modes and task == "cartpole":
                    simulate = partial(_simulate_push_trial, policy, config, latency_ns, seed, max_frames=DEFAULT_PUSH_MAX_FRAMES)
                    trial = _pick_push_trial(simulate)
                    out = sweep_dir / "videos" / f"{task}_{ms}ms_push_t{trial}_seed{seed}.mp4"
                    if not out.exists():
                        label = f"{activation} - {task} - {ms}ms - push trial {trial} - seed{seed}"
                        frames, flags, kick_index = rollout_push_frames(policy, config, latency_ns, seed, trial=trial)
                        write_video(frames, out, fps=fps, label=label, flash="PUSH", flash_at=kick_index, stable=flags)
                        written.append(out)
                        if verbose:
                            print(f"  wrote {out} ({len(frames)} frames)", flush=True)
    return written
