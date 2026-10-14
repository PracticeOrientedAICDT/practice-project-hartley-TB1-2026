"""Data-driven Pygame renderer. MP4 export also works without a display.
No fictitious hardware timing: budgets are simulated; host benchmarks separate.
"""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import argparse, json, math
from pathlib import Path
import numpy as np
from experiment import load_controller, eval_config, rollout

TILE_W, TILE_H = 640, 720
BG = (8, 12, 28)
PANEL = (17, 25, 45)
WHITE = (230, 238, 255)
CYAN = (70, 210, 255)
GREEN = (40, 230, 150)
RED = (255, 75, 95)
YELLOW = (255, 190, 65)


class AcceleratorRenderer:
    def __init__(self, headless=True):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        self.pg = pygame
        pygame.init()
        self.font = pygame.font.SysFont("consolas", 14)
        self.title = pygame.font.SysFont("consolas", 18, bold=True)

    def label(self, s, text, x, y, color=WHITE, bold=False):
        s.blit(
            (self.title if bold else self.font).render(str(text), True, color), (x, y)
        )

    def panel(self, s, rect, title):
        self.pg.draw.rect(s, PANEL, rect, border_radius=8)
        self.pg.draw.rect(s, (48, 65, 95), rect, 1, border_radius=8)
        self.label(s, title, rect[0] + 10, rect[1] + 8, CYAN, True)

    def lineplot(self, s, rect, values, color, limit=1.2):
        x, y, w, h = rect
        self.pg.draw.line(s, (48, 65, 95), (x, y + h // 2), (x + w, y + h // 2))
        if len(values) < 2:
            return
        pts = [
            (
                x + i * w / (len(values) - 1),
                y + h / 2 - np.clip(v / limit, -1, 1) * h / 2,
            )
            for i, v in enumerate(values)
        ]
        self.pg.draw.lines(s, color, False, pts, 2)

    def tile(self, trace, index, title, cfg, metrics):
        pg = self.pg
        s = pg.Surface((TILE_W, TILE_H))
        s.fill(BG)
        i = min(index, len(trace) - 1)
        info = trace[i]
        history = trace[max(0, i - 127) : i + 1]
        color = (
            RED
            if info["terminated"]
            else GREEN
            if info["amplitude"] < cfg.target_radius
            else YELLOW
        )
        self.label(s, title[:55], 14, 12, CYAN, True)
        self.label(
            s,
            f"Turn {info['step']} | {info['time_us']:.2f} us | seed-synchronized",
            14,
            40,
        )
        self.label(
            s,
            f"Compute {info['inference_ns']:g} ns | action total {info['total_action_ns']:g} ns",
            14,
            61,
        )
        self.label(
            s,
            "SIMULATED latency budget; playback is slowed, not real-time",
            14,
            82,
            YELLOW,
        )
        # Oscilloscope: measured positions versus true centroid and envelope.
        self.panel(s, (12, 108, 616, 176), "BPM OSCILLOSCOPE: normalized x")
        plot = (26, 151, 580, 99)
        self.lineplot(s, plot, [h["x"] for h in history], CYAN)
        self.lineplot(s, plot, [h["observed_x"] for h in history], YELLOW)
        self.lineplot(s, plot, [h["amplitude"] for h in history], GREEN)
        self.label(s, "cyan true x | amber received BPM | green envelope", 26, 260)
        # Ring: schematic trajectory. Offset magnified; no lattice geometry claim.
        self.panel(s, (12, 296, 300, 230), "STORAGE RING (schematic)")
        cx, cy = 162, 413
        rad = 68
        pg.draw.circle(s, (48, 65, 95), (cx, cy), rad, 12)
        pg.draw.circle(s, CYAN, (cx, cy), rad, 1)
        angle = 2 * np.pi * ((info["step"] % 24) / 24)
        offset = float(np.clip(info["x"], -1.2, 1.2)) * 16
        bx = int(cx + (rad + offset) * np.cos(angle))
        by = int(cy + (rad + offset) * np.sin(angle))
        pg.draw.circle(s, color, (bx, by), 7)
        pg.draw.rect(s, GREEN, (cx + rad - 6, cy - 6, 12, 12))
        pg.draw.rect(s, YELLOW, (cx - 6, cy + rad - 6, 12, 12))
        self.label(s, "BPM", cx + rad - 20, cy - 28, GREEN)
        self.label(s, "Kicker", cx - 24, cy + rad + 8, YELLOW)
        self.label(s, "Offset magnified; circulation slowed", 23, 501)
        # Phase space uses the same fixed scale across policies.
        self.panel(s, (324, 296, 304, 230), "PHASE SPACE: x vs v/omega0")
        cx, cy = 476, 416
        scale = 63
        pg.draw.line(s, (48, 65, 95), (cx - 105, cy), (cx + 105, cy))
        pg.draw.line(s, (48, 65, 95), (cx, cy - 74), (cx, cy + 74))
        pg.draw.circle(s, GREEN, (cx, cy), max(2, int(cfg.target_radius * scale)), 1)
        pts = [
            (
                cx + np.clip(h["x"], -1.5, 1.5) * scale,
                cy - np.clip(h["p"], -1.1, 1.1) * scale,
            )
            for h in history
        ]
        if len(pts) > 1:
            pg.draw.lines(s, CYAN, False, pts, 1)
        pg.draw.circle(s, color, (int(pts[-1][0]), int(pts[-1][1])), 4)
        self.label(s, f"Envelope {info['amplitude']:.4f}", 337, 501, color)
        # Queue positions are computed from actual timestamped events.
        self.panel(s, (12, 538, 616, 110), "LATENCY PIPELINE: commands in flight")
        y = 592
        x0 = 40
        x1 = 596
        pg.draw.line(s, (48, 65, 95), (x0, y), (x1, y), 3)
        total = max(info["total_action_ns"], 1e-12)
        split = x0 + (x1 - x0) * min(info["inference_ns"] / total, 1)
        pg.draw.line(s, CYAN, (x0, y), (int(split), y), 4)
        self.label(s, "BPM / MLP", x0, 565)
        self.label(s, "DAC / transport", 350, 565)
        self.label(s, "Kicker", 550, 565)
        delay = info["action_delay_steps"]
        for event in info["pending_events"]:
            issued = event["due_turn"] - delay
            progress = np.clip((info["step"] - issued) / max(delay, 1e-12), 0, 1)
            px = int(x0 + (x1 - x0) * progress)
            pg.draw.circle(s, GREEN if event["command"] >= 0 else RED, (px, y), 5)
        if info["deliveries"]:
            pg.draw.circle(s, YELLOW, (x1, y), 8, 2)
        self.label(
            s,
            f"Queued {info['pending_actions']} | issued {info['issued_action']:+.3f} | applied {info['applied_action']:+.3f}",
            26,
            620,
        )
        status = (
            "BEAM LOSS"
            if info["terminated"]
            else "EPISODE COMPLETE"
            if info["truncated"]
            else "DAMPED"
            if info["amplitude"] < cfg.target_radius
            else "DAMPING OSCILLATION"
        )
        self.label(s, status, 14, 659, color, True)
        self.label(
            s,
            f"Episode RMS {metrics['rms_error']:.4f} | loss {int(metrics['lost'])} | settle {metrics['settle_us']:.1f} us",
            14,
            688,
        )
        return s

    def close(self):
        self.pg.quit()


def render_comparison(
    models,
    budgets,
    output,
    seed=10000,
    fps=60,
    stride=1,
    max_frames=None,
    backend="numpy",
    headless=True,
    preview=None,
    environment="accelerator",
    dynamics_profile="checkpoint",
    case_titles=None,
):
    if len(budgets) == 1:
        budgets = budgets * len(models)
    if len(models) != len(budgets):
        raise ValueError("Supply one budget or one per model")
    if not models or fps <= 0 or stride < 1:
        raise ValueError("Invalid rendering arguments")
    cases = []
    for case_index, (path, budget) in enumerate(zip(models, budgets)):
        policy, cfg, meta = load_controller(
            path, backend, environment if path == "zero" else None
        )
        if environment == "accelerator" and dynamics_profile != "checkpoint":
            from envs.accelerator_profiles import apply_profile

            cfg = apply_profile(cfg, dynamics_profile)
        cfg = eval_config(cfg, budget)
        case_environment = meta.get("environment", "accelerator")
        if case_environment != environment:
            raise ValueError(f"{path} is for {case_environment}, not {environment}")
        metrics, trace, _ = rollout(
            policy, cfg, seed, record=True, environment=environment
        )
        title = (
            case_titles[case_index]
            if case_titles
            else f'{meta.get("name",Path(path).name)} @ {budget:g} ns'
        )
        cases.append((trace, title, cfg, metrics))
    if environment == "laser":
        from render_laser_demo import LaserRenderer

        renderer = LaserRenderer(headless)
    else:
        renderer = AcceleratorRenderer(headless)
    pg = renderer.pg
    columns = min(2, len(cases))
    rows = math.ceil(len(cases) / columns)
    canvas = pg.Surface((columns * TILE_W, rows * TILE_H))
    display = None if headless else pg.display.set_mode(canvas.get_size())
    clock = pg.time.Clock()
    frames = list(range(0, max(len(c[0]) for c in cases), stride))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("No frames requested")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    try:
        with imageio.get_writer(
            str(output), fps=fps, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for frame_no, index in enumerate(frames):
                if any(e.type == pg.QUIT for e in pg.event.get()):
                    break
                canvas.fill(BG)
                for j, (trace, title, cfg, metrics) in enumerate(cases):
                    canvas.blit(
                        renderer.tile(trace, index, title, cfg, metrics),
                        ((j % columns) * TILE_W, (j // columns) * TILE_H),
                    )
                if display is not None:
                    display.blit(canvas, (0, 0))
                    pg.display.flip()
                    clock.tick(fps)
                writer.append_data(pg.surfarray.array3d(canvas).transpose(1, 0, 2))
                if preview and frame_no == min(100, len(frames) - 1):
                    pg.image.save(canvas, str(preview))
        output.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    seed=seed,
                    models=models,
                    inference_budgets_ns=budgets,
                    fps=fps,
                    steps_per_frame=stride,
                    environment=environment,
                    dynamics_profile=dynamics_profile,
                    metrics=[c[3] for c in cases],
                    note=(
                        "Schematic ring"
                        if environment == "accelerator"
                        else "Normalized target plane"
                    )
                    + "; simulation traces; frozen after terminal state.",
                ),
                indent=2,
            )
        )
    finally:
        renderer.close()
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument(
        "--environment", choices=["accelerator", "laser"], default="accelerator"
    )
    p.add_argument(
        "--dynamics-profile",
        choices=["checkpoint", "deterministic", "mild", "uncertain", "stress"],
        default="checkpoint",
    )
    p.add_argument("--budgets-ns", type=float, nargs="+", default=[100])
    p.add_argument("--out", default="videos/comparison.mp4")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--window", action="store_true")
    p.add_argument("--preview")
    p.add_argument("--backend", choices=["numpy", "sb3", "sb3-cuda"], default="numpy")
    a = p.parse_args()
    render_comparison(
        a.models,
        a.budgets_ns,
        a.out,
        a.seed,
        a.fps,
        a.stride,
        a.max_frames,
        a.backend,
        not a.window,
        a.preview,
        a.environment,
        a.dynamics_profile,
    )


if __name__ == "__main__":
    main()
