"""Four-panel renderer for latency-aware 2-D laser targeting."""
import numpy as np
from render_accelerator_demo import (
    AcceleratorRenderer,
    TILE_W,
    TILE_H,
    BG,
    PANEL,
    WHITE,
    CYAN,
    GREEN,
    RED,
    YELLOW,
)


class LaserRenderer(AcceleratorRenderer):
    def tile(self, trace, index, title, cfg, metrics):
        pg = self.pg
        s = pg.Surface((TILE_W, TILE_H))
        s.fill(BG)
        i = min(index, len(trace) - 1)
        info = trace[i]
        history = trace[max(0, i - 149) : i + 1]
        color = RED if info["terminated"] else GREEN if info["on_target"] else YELLOW
        self.label(s, title[:55], 14, 12, CYAN, True)
        self.label(
            s,
            f"Sample {info['step']} | {info['time_us']:.2f} us | synchronized seed",
            14,
            40,
        )
        self.label(
            s,
            f"Compute {info['inference_ns']:g} ns | nominal total {info['total_action_ns']:g} ns",
            14,
            61,
        )
        self.label(s, "SIMULATED normalized optical steering plant", 14, 82, YELLOW)
        # X/Y oscilloscope
        self.panel(s, (12, 108, 616, 176), "BEAM CENTROID OSCILLOSCOPE")
        plot = (26, 151, 580, 99)
        limit = max(3.0, cfg.target_radius * 2)
        self.lineplot(s, plot, [h["x"] for h in history], CYAN, limit)
        self.lineplot(s, plot, [h["y"] for h in history], YELLOW, limit)
        self.label(s, "cyan x | amber y | zero line is target centre", 26, 260)
        # Target plane
        self.panel(s, (12, 296, 300, 230), "TARGET PLANE: x-y")
        cx, cy = 162, 414
        scale = 42 / max(cfg.target_radius, 1e-9)
        target_px = int(cfg.target_radius * scale)
        boundary_px = min(102, int(cfg.boundary * scale))
        pg.draw.circle(s, (42, 19, 34), (cx, cy), boundary_px)
        pg.draw.circle(s, GREEN, (cx, cy), target_px, 2)
        pts = []
        for h in history:
            px = int(cx + np.clip(h["x"] * scale, -105, 105))
            py = int(cy + np.clip(h["y"] * scale, -88, 88))
            pts.append((px, py))
        if len(pts) > 1:
            pg.draw.lines(s, CYAN, False, pts, 2)
        pg.draw.circle(s, color, pts[-1], 7)
        pg.draw.circle(s, WHITE, pts[-1], 2)
        pg.draw.line(s, GREEN, (cx - 10, cy), (cx + 10, cy))
        pg.draw.line(s, GREEN, (cx, cy - 10), (cx, cy + 10))
        self.label(s, f"Distance {info['distance']:.3f}", 25, 501, color)
        # x-vx phase plane
        self.panel(s, (324, 296, 304, 230), "PHASE PLANE: x vs vx/omega")
        cx, cy = 476, 416
        scale = 55 / max(cfg.target_radius, 1e-9)
        pg.draw.line(s, (48, 65, 95), (cx - 105, cy), (cx + 105, cy))
        pg.draw.line(s, (48, 65, 95), (cx, cy - 75), (cx, cy + 75))
        points = [
            (
                cx + np.clip(h["x"] * scale, -100, 100),
                cy
                - np.clip(
                    h["vx"]
                    / max(2 * np.pi * cfg.natural_frequency_hz / cfg.sample_hz, 1e-12)
                    * scale,
                    -70,
                    70,
                ),
            )
            for h in history
        ]
        if len(points) > 1:
            pg.draw.lines(s, CYAN, False, points, 1)
        pg.draw.circle(s, color, (int(points[-1][0]), int(points[-1][1])), 4)
        self.label(s, f"Speed {info['speed']:.3f}", 337, 501, color)
        # Pipeline
        self.panel(s, (12, 538, 616, 110), "LATENCY PIPELINE: 2-D STEERING COMMANDS")
        y = 592
        x0 = 40
        x1 = 596
        pg.draw.line(s, (48, 65, 95), (x0, y), (x1, y), 3)
        total = max(info["total_action_ns"], 1e-12)
        split = x0 + (x1 - x0) * min(info["inference_ns"] / total, 1)
        pg.draw.line(s, CYAN, (x0, y), (int(split), y), 4)
        self.label(s, "Sensor / MLP", x0, 565)
        self.label(s, "Transport", 360, 565)
        self.label(s, "Mirror", 545, 565)
        delay = max(info["action_delay_steps"], 1e-12)
        for event in info["pending_events"]:
            issued = event["due_step"] - delay
            progress = np.clip((info["step"] - issued) / delay, 0, 1)
            command = event["action"]
            strength = min(1, np.hypot(*command))
            px = int(x0 + (x1 - x0) * progress)
            pg.draw.circle(
                s,
                GREEN if command[0] >= 0 else RED,
                (px, y),
                max(3, int(3 + 3 * strength)),
            )
        if info["deliveries"]:
            pg.draw.circle(s, YELLOW, (x1, y), 8, 2)
        self.label(
            s,
            f"Queued {info['pending_actions']} | issued {np.asarray(info['issued_action']).round(2)} | applied {np.asarray(info['applied_action']).round(2)}",
            26,
            620,
        )
        status = (
            "BEAM LOST"
            if info["terminated"]
            else "ON TARGET"
            if info["on_target"]
            else "STEERING TO TARGET"
        )
        self.label(s, status, 14, 659, color, True)
        self.label(
            s,
            f"Episode RMS {metrics['rms_error']:.3f} | tracking {metrics['tracking']:.1%} | loss {int(metrics['lost'])}",
            14,
            688,
        )
        return s
