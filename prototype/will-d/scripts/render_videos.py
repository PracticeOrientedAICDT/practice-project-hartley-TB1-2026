"""Render latency-condition rollouts of the trained policies as videos.

Companion to the frozen evaluation path, not a part of it. ``PLAN.md`` section 18
forbids rendering inside training and evaluation because render FPS is playback
speed and has nothing to do with physics timing; this script runs *after* the
fact, from saved checkpoints, and changes no measurement.

## What one video shows

The experiment's variable is inference latency, which the action-hold wrapper
turns into physics: the policy decides once and its action is held for ``N``
integration steps, where ``N`` is derived from the requested latency against the
simulator timestep. The five conditions of the latency grid are therefore also
five *timestep* conditions -- 20 ms acts every step, 100 ms acts every fifth --
and this script renders each of them playing each task.

Three properties keep the comparison honest, mirroring sections 6.7, 17 and 19:

1. **Each panel is matched.** The panel at latency ``L`` plays the checkpoint
   trained at ``L`` and is evaluated at ``L``. That is the diagonal of the
   matrix, the primary design, so the ladder shows what the experiment actually
   measured rather than a train/test mismatch.

2. **The start is paired.** Every panel resets with the same evaluation seed, so
   all five meet the same initial state and the only thing that differs between
   them is how often they are allowed to act.

3. **Playback is real time.** One frame per 20 ms integration step, 50 fps. The
   video's clock is the simulated clock, so a degradation visible in the frames
   is the same degradation the metrics record.

## Layout

For each task the script writes:

- ``<task>_latency_ladder.mp4`` -- the five latency conditions side by side,
  time-aligned on one timeline. This is the figure to watch.
- ``<task>_<latency>ms.mp4`` -- each condition alone.
- ``<task>_contact_sheet.png`` -- the five conditions' final states.

Requires a video encoder. The project itself does not depend on one; if ffmpeg is
absent the script installs nothing and says so. ``uv pip install imageio-ffmpeg``
ships a static binary without touching system state or the lockfile.

Usage::

    uv run python scripts/render_videos.py --tasks cartpole,lunarlander
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from latency_ml.experiments.shared_runner import PHASE4_TASKS
from latency_ml.models.shared_trunk import SharedTrunkPolicy, build_shared_trunk_policy
from PIL import Image, ImageDraw, ImageFont

from latency_ml.config import EvaluationConfig
from latency_ml.envs.factory import probe_dt_env_ns, probe_max_episode_steps, task_spec
from latency_ml.experiments.evaluate import greedy_action
from latency_ml.rl.task_policy import SharedTaskPolicy
from latency_ml.seeding import seed_env
from latency_ml.timing import derive_icdi

FPS = 50


DEFAULT_LATENCIES_MS: tuple[int, ...] = (20, 40, 60, 80, 100)


TASK_ACTIONS: dict[str, tuple[str, ...]] = {
    "cartpole": ("push left", "push right"),
    "lunarlander": ("do nothing", "left engine", "main engine", "right engine"),
}

TASK_TITLES: dict[str, str] = {
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3",
}


LATENCY_COLOURS: dict[int, tuple[int, int, int]] = {
    20: (46, 164, 79),
    40: (201, 162, 39),
    60: (224, 123, 57),
    80: (214, 69, 65),
    100: (176, 42, 48),
}

_BG = (24, 26, 32)
_INK = (235, 238, 245)
_DIM = (150, 156, 170)
_BAD = (214, 69, 65)

_PANEL_W, _PANEL_H = 480, 320
_GAP = 12
_MARGIN = 20
_TITLE_H = 64
_PANEL_HEADER_H = 46
_PANEL_FOOTER_H = 46

_FONT_REGULAR = "Arial.ttf"
_FONT_BOLD = "Arial Bold.ttf"


@dataclass
class PanelState:
    """Live rollout state for one latency condition."""

    task: str
    latency_ms: int
    hold_steps: int
    step: int = 0
    t_s: float = 0.0
    ret: float = 0.0
    action: int = 0
    decisions: int = 0
    done: bool = False
    outcome: str = ""
    last_frame: np.ndarray | None = None

    @property
    def final_frame(self) -> np.ndarray:
        """The episode's last rendered frame.

        Always set: the first integration step renders before anything can end
        the episode, so a rollout that has yielded at least once has a frame.
        """
        if self.last_frame is None:
            raise RuntimeError("rollout produced no frames")
        return self.last_frame

    @property
    def action_label(self) -> str:
        return TASK_ACTIONS[self.task][self.action]

    @property
    def decided_this_step(self) -> bool:
        """Whether this integration step began a new hold.

        A decision lands on steps 0, N, 2N, ...; ``step`` counts steps already
        taken, so a fresh decision is visible one integration step later. Only
        meaningful when the episode has not ended.
        """
        return self.hold_steps > 1 and self.step % self.hold_steps == 1


@dataclass(frozen=True)
class RolloutSummary:
    """What one rendered episode turned out to be, for the manifest and stdout."""

    task: str
    latency_ns: int
    eval_seed: int
    checkpoint: str
    checkpoint_sha256: str
    steps: int
    decisions: int
    return_total: float
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "latency_ns": self.latency_ns,
            "eval_seed": self.eval_seed,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "integration_steps": self.steps,
            "decisions": self.decisions,
            "return": self.return_total,
            "outcome": self.outcome,
        }


@dataclass
class TaskRender:
    """Everything one task's render produced, for the manifest."""

    task: str
    eval_seed: int
    latencies_ns: list[int]
    fps: int
    frames: int
    ladder_path: str
    contact_sheet_path: str
    individual_paths: dict[int, str] = field(default_factory=dict)
    summaries: list[RolloutSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "eval_seed": self.eval_seed,
            "latencies_ns": self.latencies_ns,
            "fps": self.fps,
            "frames": self.frames,
            "ladder_path": self.ladder_path,
            "contact_sheet_path": self.contact_sheet_path,
            "individual_paths": {str(key): value for key, value in self.individual_paths.items()},
            "conditions": [summary.to_dict() for summary in self.summaries],
        }


def load_policy(checkpoint: Path) -> SharedTrunkPolicy:
    """Load a saved shared-trunk checkpoint.

    The checkpoint holds the whole trunk -- both tasks' adapters and heads --
    so one file plays every task in the experiment.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    policy = build_shared_trunk_policy(PHASE4_TASKS)
    policy.load_state_dict(torch.load(checkpoint, weights_only=True))
    policy.eval()
    return policy


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(root: Path, latency_ns: int, seed: int) -> Path:
    """The sweep's checkpoint layout: ``<root>/<latency>ns/seed<n>/policy.pt``."""
    return root / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt"


def render_env(task: str) -> gym.Env[Any, Any]:
    """Build a rendering environment.

    Deliberately not :func:`make_base_env`, whose headless default is
    load-bearing for the measurement path. Constructed here from the same
    ``task_spec`` so the physics and kwargs are identical and only
    ``render_mode`` differs.
    """
    spec = task_spec(task)
    return gym.make(spec["env_id"], render_mode="rgb_array", **spec["kwargs"])


def rollout(
    task: str,
    latency_ns: int,
    policy: SharedTrunkPolicy,
    dt_env_ns: int,
    seed: int,
    max_steps: int,
) -> Iterator[tuple[np.ndarray, PanelState]]:
    """Roll out one greedy episode, yielding a frame per integration step.

    The hold pattern reproduces the action-hold wrapper exactly -- decide, then
    hold the action for ``N`` steps -- but steps the environment directly so a
    frame can be captured after every integration step rather than after every
    decision. At ``N = 5`` that is the difference between a 10 fps video and a
    50 fps one covering the same ten simulated seconds.

    The generator keeps yielding the frozen final frame after the episode ends,
    so several conditions can run in lockstep on one shared timeline instead of
    each video ending wherever its own episode stopped.

    Args:
        task: Task name.
        latency_ns: Inference latency condition.
        policy: Loaded shared trunk, played through this task's view.
        dt_env_ns: Environment integration timestep.
        seed: Reset seed, shared across conditions for pairing.
        max_steps: Integration-step cap, the environment's own limit.

    Yields:
        ``(frame, state)`` per integration step. ``state.last_frame`` is the
        same array.
    """
    view = SharedTaskPolicy(policy, task)
    icdi = derive_icdi(latency_ns, dt_env_ns)
    hold = icdi.action_steps
    dt_s = dt_env_ns / 1e9

    env = render_env(task)
    try:
        state = PanelState(task=task, latency_ms=latency_ns // 1_000_000, hold_steps=hold)
        obs, _ = seed_env(env, seed)

        for step in range(max_steps):
            if not state.done:
                if step % hold == 0:
                    state.action = greedy_action(view, np.asarray(obs))
                    state.decisions += 1
                obs, reward, terminated, truncated, _ = env.step(state.action)
                state.ret += float(reward)
                state.step = step + 1
                state.t_s = (step + 1) * dt_s
                state.last_frame = np.asarray(env.render())
                if terminated or truncated:
                    state.done = True
                    state.outcome = "terminated" if terminated else "truncated"
            yield state.final_frame, state
    finally:
        env.close()


def summarise(state: PanelState, checkpoint: Path, eval_seed: int) -> RolloutSummary:
    """A finished panel's record, with the checkpoint it played."""
    return RolloutSummary(
        task=state.task,
        latency_ns=state.latency_ms * 1_000_000,
        eval_seed=eval_seed,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_of(checkpoint),
        steps=state.step,
        decisions=state.decisions,
        return_total=state.ret,
        outcome=state.outcome,
    )


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_BOLD if bold else _FONT_REGULAR, size)


def _panel_card(frame: np.ndarray, state: PanelState) -> Image.Image:
    """One frame with this condition's header and footer strips.

    Used for both the ladder panels and the standalone videos, so the two
    layouts stay consistent by construction.
    """
    colour = LATENCY_COLOURS.get(state.latency_ms, _DIM)
    image = Image.fromarray(frame).resize((_PANEL_W, _PANEL_H), Image.Resampling.BILINEAR)
    card = Image.new("RGB", (_PANEL_W, _PANEL_H + _PANEL_HEADER_H + _PANEL_FOOTER_H), _BG)
    card.paste(image, (0, _PANEL_HEADER_H))
    draw = ImageDraw.Draw(card)

    draw.rectangle([0, 0, _PANEL_W, _PANEL_HEADER_H], fill=colour)
    hold_text = f"{state.latency_ms} ms  ·  act every {state.hold_steps} step"
    if state.hold_steps > 1:
        hold_text += "s"
    draw.text((14, 12), hold_text, font=_font(22, bold=True), fill=(255, 255, 255))

    footer_y = _PANEL_HEADER_H + _PANEL_H + 8
    draw.text(
        (14, footer_y),
        f"t {state.t_s:4.2f} s    return {state.ret:6.1f}",
        font=_font(19),
        fill=_INK,
    )
    right = f"{'DECIDE  ' if state.decided_this_step else ''}{state.action_label}"
    draw.text(
        (_PANEL_W - 14, footer_y),
        right,
        font=_font(19, bold=state.decided_this_step),
        fill=_INK if state.decided_this_step else _DIM,
        anchor="ra",
    )

    if state.done:
        draw.rectangle(
            [8, _PANEL_HEADER_H + 8, _PANEL_W - 8, _PANEL_HEADER_H + _PANEL_H - 8],
            outline=_BAD,
            width=4,
        )
        label = f"{state.outcome.upper()}  {state.t_s:.2f} s"
        badge_w = int(draw.textlength(label, font=_font(20, bold=True))) + 28
        badge = Image.new("RGB", (badge_w, 34), _BAD)
        card.paste(badge, ((_PANEL_W - badge_w) // 2, _PANEL_HEADER_H + 12))
        draw.text(
            (_PANEL_W // 2, _PANEL_HEADER_H + 29),
            label,
            font=_font(20, bold=True),
            fill=(255, 255, 255),
            anchor="mm",
        )
    return card


def compose_ladder(
    task: str, frames: list[np.ndarray], states: list[PanelState], seed: int
) -> Image.Image:
    """The five-condition comparison frame.

    All panels share one timeline, so a difference between them at any instant
    is a difference in what their policies were able to do with the same state
    history, not a difference in episode length.
    """
    count = len(states)
    width = _MARGIN * 2 + count * _PANEL_W + (count - 1) * _GAP
    height = _MARGIN + _TITLE_H + _PANEL_HEADER_H + _PANEL_H + _PANEL_FOOTER_H + _MARGIN
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (_MARGIN, 14),
        f"{TASK_TITLES.get(task, task)} — inference-latency ladder",
        font=_font(28, bold=True),
        fill=_INK,
    )
    draw.text(
        (_MARGIN, 50),
        "matched training + evaluation latency per panel  ·  greedy policy  ·  same initial state"
        "  ·  1 frame = 20 ms of simulated time",
        font=_font(18),
        fill=_DIM,
    )
    draw.text(
        (width - _MARGIN, 16),
        f"shared trunk · seed {seed}",
        font=_font(22, bold=True),
        fill=_DIM,
        anchor="ra",
    )

    x = _MARGIN
    top = _MARGIN + _TITLE_H
    for frame, state in zip(frames, states, strict=True):
        canvas.paste(_panel_card(frame, state), (x, top))
        x += _PANEL_W + _GAP
    return canvas


def _ffmpeg() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "no ffmpeg available. Install the bundled static binary without\n"
            "touching the project dependency set:  uv pip install imageio-ffmpeg"
        ) from exc
    return str(get_ffmpeg_exe())


def encode(cards: Iterator[Image.Image], path: Path, size: tuple[int, int], fps: int) -> int:
    """Pipe RGB frames to ffmpeg as H.264.

    Raw frames on stdin rather than a writer library: the encoder is the only
    dependency here that is not already pinned by the project, and a subprocess
    boundary keeps it replaceable if a different encoder is ever wanted.

    Returns:
        The number of frames written.
    """
    width, height = size
    if width % 2 or height % 2:
        raise ValueError(f"frame size {size} must be even for yuv420p")
    cmd = [
        _ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    count = 0
    try:
        for card in cards:
            if proc.stdin is None:  # pragma: no cover - Popen gave us a pipe
                raise RuntimeError("ffmpeg stdin unavailable")
            proc.stdin.write(card.tobytes())
            count += 1
    except BrokenPipeError:  # pragma: no cover - encoder died, reported below
        pass
    finally:
        stream, proc.stdin = proc.stdin, None
        if stream is not None:
            stream.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {stderr[-2000:]}")
    return count


def render_task(
    task: str,
    root: Path,
    out_dir: Path,
    latencies_ms: tuple[int, ...],
    train_seed: int,
    fps: int,
    make_individual: bool,
) -> TaskRender:
    """Render every condition for one task: the ladder plus standalone videos.

    Every panel is a different checkpoint -- the trunk trained at that latency --
    evaluated at that same latency, which is the matched condition the sweep's
    diagonal reports. Every panel is reset with the same evaluation seed, the
    protocol's own pairing, so the five episodes start from identical states.
    """
    dt_env_ns = probe_dt_env_ns(task)
    max_steps = probe_max_episode_steps(task) or 500

    eval_seed = EvaluationConfig().episode_seed(0)

    policies: dict[int, SharedTrunkPolicy] = {}
    checkpoints: dict[int, Path] = {}
    for latency_ms in latencies_ms:
        path = checkpoint_path(root, latency_ms * 1_000_000, train_seed)
        checkpoints[latency_ms] = path
        policies[latency_ms] = load_policy(path)

    generators = [
        rollout(task, latency_ms * 1_000_000, policies[latency_ms], dt_env_ns, eval_seed, max_steps)
        for latency_ms in latencies_ms
    ]

    width = _MARGIN * 2 + len(latencies_ms) * _PANEL_W + (len(latencies_ms) - 1) * _GAP
    height = _MARGIN + _TITLE_H + _PANEL_HEADER_H + _PANEL_H + _PANEL_FOOTER_H + _MARGIN
    ladder_path = out_dir / f"{task}_latency_ladder.mp4"

    final_states: list[PanelState] = []

    def ladder_cards() -> Iterator[Image.Image]:
        for step in range(max_steps):
            frames, states = [], []
            for generator in generators:
                frame, state = next(generator)
                frames.append(frame)
                states.append(state)
            final_states.clear()
            final_states.extend(states)
            if (step + 1) % 100 == 0 or step == max_steps - 1:
                print(f"  {task}: frame {step + 1}/{max_steps}", flush=True)
            yield compose_ladder(task, frames, states, train_seed)

    started = time.perf_counter()
    frames_written = encode(ladder_cards(), ladder_path, (width, height), fps)
    print(
        f"  {task}: ladder {frames_written} frames at {fps} fps "
        f"({frames_written / fps:.1f} s) in {time.perf_counter() - started:.0f}s",
        flush=True,
    )

    summaries = [
        summarise(state, checkpoints[state.latency_ms], eval_seed) for state in final_states
    ]

    individual: dict[int, str] = {}
    if make_individual:
        for latency_ms in latencies_ms:
            path = out_dir / f"{task}_{latency_ms}ms.mp4"
            generator = rollout(
                task,
                latency_ms * 1_000_000,
                policies[latency_ms],
                dt_env_ns,
                eval_seed,
                max_steps,
            )

            def cards(
                generator: Iterator[tuple[np.ndarray, PanelState]] = generator,
            ) -> Iterator[Image.Image]:
                for frame, state in generator:
                    yield _panel_card(frame, state)

            encode(cards(), path, (_PANEL_W, _PANEL_H + _PANEL_HEADER_H + _PANEL_FOOTER_H), fps)
            individual[latency_ms] = str(path)
            print(f"  {task}: {path.name}", flush=True)

    contact = compose_ladder(
        task,
        [state.final_frame for state in final_states],
        final_states,
        train_seed,
    )
    contact_path = out_dir / f"{task}_contact_sheet.png"
    contact.save(contact_path)

    return TaskRender(
        task=task,
        eval_seed=eval_seed,
        latencies_ns=[latency_ms * 1_000_000 for latency_ms in latencies_ms],
        fps=fps,
        frames=frames_written,
        ladder_path=str(ladder_path),
        contact_sheet_path=str(contact_path),
        individual_paths=individual,
        summaries=summaries,
    )


def print_result_table(renders: list[TaskRender]) -> None:
    """What the videos contain, so the outcome is legible without opening them."""
    print()
    print(f"{'task':<13}{'latency':>9}{'steps':>8}{'decisions':>11}{'return':>9}  outcome")
    print("-" * 68)
    for render in renders:
        for summary in render.summaries:
            latency = f"{summary.latency_ns // 1_000_000} ms"
            print(
                f"{summary.task:<13}{latency:>9}{summary.steps:>8}{summary.decisions:>11}"
                f"{summary.return_total:>9.1f}  {summary.outcome or 'running'}"
            )
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/shared-trunk/checkpoints"),
        help="checkpoint tree; '<root>/<latency>ns/seed<n>/policy.pt'",
    )
    parser.add_argument("--out", type=Path, default=Path("results/videos"), help="output directory")
    parser.add_argument(
        "--tasks",
        type=str,
        default="cartpole,lunarlander",
        help="comma-separated task names",
    )
    parser.add_argument(
        "--latencies",
        type=str,
        default=",".join(str(latency) for latency in DEFAULT_LATENCIES_MS),
        help="comma-separated latencies in ms",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="training seed whose checkpoints to render"
    )
    parser.add_argument(
        "--fps", type=int, default=FPS, help="playback rate; 50 is real simulated time"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="skip the per-condition standalone videos",
    )
    args = parser.parse_args(argv)
    args.tasks = tuple(task.strip() for task in args.tasks.split(","))
    args.latencies = tuple(int(latency.strip()) for latency in args.latencies.split(","))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        checkpoint_path(root, latency * 1_000_000, args.seed)
        for latency in args.latencies
        if not checkpoint_path(root, latency * 1_000_000, args.seed).exists()
    ]
    if missing:
        print("missing checkpoints:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(
        f"rendering {list(args.tasks)} from {root} (training seed {args.seed}) -> {out_dir}",
        flush=True,
    )
    renders = [
        render_task(
            task,
            root,
            out_dir,
            args.latencies,
            args.seed,
            args.fps,
            make_individual=not args.no_individual,
        )
        for task in args.tasks
    ]

    manifest = {
        "generator": "scripts/render_videos.py",
        "checkpoint_root": str(root),
        "train_seed": args.seed,
        "fps": args.fps,
        "latencies_ms": list(args.latencies),
        "renders": [render.to_dict() for render in renders],
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2))

    print_result_table(renders)
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
