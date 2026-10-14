"""Latency-aware 2-D laser pointing/steering benchmark.

The plant is a normalized second-order beam-centroid model. It is suitable for
latency/control experiments, but it is not calibrated to a specific optical bench.
Commands are timestamped and may arrive at fractional sample times.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class LaserTargetConfig:
    sample_hz: float = 10e6
    natural_frequency_hz: float = 0.40e6
    damping_ratio: float = 0.02
    control_gain: float = 0.10
    wind_std: float = 0.02
    initial_position: float = 2.0
    initial_velocity: float = 0.20
    boundary: float = 50.0
    target_radius: float = 2.0
    episode_steps: int = 5000
    action_history: int = 8
    inference_ns: float = 100.0
    transport_ns: float = 0.0
    inference_range_ns: Optional[Tuple[float, float]] = None
    inference_jitter_ns: float = 0.0
    latency_scale_ns: float = 5000.0
    latency_features: bool = True
    max_action_delay_steps: float = 128.0
    action_penalty: float = 0.0
    loss_penalty: float = 1000.0
    reward_kind: str = "distance"


class LaserTargetEnv(gym.Env):
    """Two-axis damped oscillator with impulse steering and latency.

    Observation: [x, y, vx/omega, vy/omega, issued action history,
    nominal inference latency, total nominal latency, pending-action fraction].
    Jitter is sampled per action and is not revealed before delivery.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or LaserTargetConfig()
        c = self.cfg
        vals = [
            c.sample_hz,
            c.natural_frequency_hz,
            c.damping_ratio,
            c.control_gain,
            c.wind_std,
            c.boundary,
            c.target_radius,
            c.inference_ns,
            c.transport_ns,
            c.inference_jitter_ns,
            c.latency_scale_ns,
            c.max_action_delay_steps,
        ]
        if (
            not np.isfinite(vals).all()
            or c.sample_hz <= 0
            or c.natural_frequency_hz <= 0
        ):
            raise ValueError("Parameters must be finite and frequencies positive")
        if (
            min(
                c.damping_ratio,
                c.control_gain,
                c.wind_std,
                c.inference_ns,
                c.transport_ns,
                c.inference_jitter_ns,
            )
            < 0
        ):
            raise ValueError("Noise, gain, damping and latency must be nonnegative")
        if c.damping_ratio >= 1 or c.episode_steps < 1 or c.action_history < 0:
            raise ValueError("Requires an underdamped oscillator and valid horizon")
        if c.reward_kind not in ("distance", "squared", "tanhsq"):
            raise ValueError("Unknown reward kind")
        if c.inference_range_ns is not None:
            lo, hi = c.inference_range_ns
            if not 0 <= lo <= hi:
                raise ValueError("Invalid inference range")
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        n = 4 + 2 * c.action_history + (3 if c.latency_features else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.target = np.zeros(2, np.float32)

    @property
    def omega(self):
        return 2 * np.pi * self.cfg.natural_frequency_hz / self.cfg.sample_hz

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.steps = 0
        self.finished = False
        self.x = float(self.np_random.uniform(-c.initial_position, c.initial_position))
        self.y = float(self.np_random.uniform(-c.initial_position, c.initial_position))
        self.vx = float(self.np_random.uniform(-c.initial_velocity, c.initial_velocity))
        self.vy = float(self.np_random.uniform(-c.initial_velocity, c.initial_velocity))
        self.commands = np.zeros((c.action_history, 2), np.float32)
        self.events = []
        self.serial = 0
        self.last_action = np.zeros(2, np.float32)
        self.applied_action = np.zeros(2, np.float32)
        self.last_deliveries = []
        self.inference_ns = (
            c.inference_ns
            if c.inference_range_ns is None
            else float(self.np_random.uniform(*c.inference_range_ns))
        )
        self.nominal_delay_steps = (
            (self.inference_ns + c.transport_ns) * c.sample_hz * 1e-9
        )
        if self.nominal_delay_steps > c.max_action_delay_steps:
            raise ValueError("Nominal latency exceeds queue bound")
        self.history = []
        return self._obs(), self._info()

    def _advance(self, dt):
        # Exact underdamped second-order transition in normalized sample time.
        w0 = self.omega
        z = self.cfg.damping_ratio
        g = z * w0
        wd = w0 * np.sqrt(1 - z * z)
        co, si = np.cos(wd * dt), np.sin(wd * dt)
        decay = np.exp(-g * dt)

        def axis(q, v):
            qn = decay * ((co + g / wd * si) * q + si / wd * v)
            vn = decay * (-w0 * w0 / wd * si * q + (co - g / wd * si) * v)
            return qn, vn

        self.x, self.vx = axis(self.x, self.vx)
        self.y, self.vy = axis(self.y, self.vy)

    def _obs(self):
        base = [self.x, self.y, self.vx / self.omega, self.vy / self.omega]
        base.extend(self.commands.reshape(-1))
        if self.cfg.latency_features:
            scale = self.cfg.latency_scale_ns
            base.extend(
                [
                    self.inference_ns / scale,
                    (self.inference_ns + self.cfg.transport_ns) / scale,
                    len(self.events) / max(1, self.cfg.max_action_delay_steps),
                ]
            )
        return np.asarray(base, np.float32)

    def _info(self):
        distance = float(np.hypot(self.x, self.y))
        speed = float(np.hypot(self.vx, self.vy))
        return dict(
            environment="laser",
            x=float(self.x),
            y=float(self.y),
            vx=float(self.vx),
            vy=float(self.vy),
            distance=distance,
            speed=speed,
            amplitude=distance,
            on_target=distance < self.cfg.target_radius,
            inference_ns=float(self.inference_ns),
            transport_ns=float(self.cfg.transport_ns),
            total_action_ns=float((self.inference_ns + self.cfg.transport_ns)),
            action_delay_steps=float(self.nominal_delay_steps),
            sample_period_ns=1e9 / self.cfg.sample_hz,
            time_us=self.steps / self.cfg.sample_hz * 1e6,
            pending_actions=len(self.events),
            pending_events=[
                dict(due_step=float(t), action=np.asarray(a).tolist())
                for t, _, a in sorted(self.events)
            ],
            deliveries=self.last_deliveries,
            issued_action=self.last_action.tolist(),
            applied_action=self.applied_action.tolist(),
        )

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after termination/truncation")
        a = np.asarray(action, float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Expected finite action of shape (2,)")
        a = np.clip(a, -1, 1).astype(np.float32)
        self.last_action = a.copy()
        if self.cfg.action_history:
            self.commands[1:] = self.commands[:-1].copy()
            self.commands[0] = a
        jitter = self.np_random.normal(0, self.cfg.inference_jitter_ns)
        actual_ns = max(0.0, self.inference_ns + self.cfg.transport_ns + jitter)
        delay = actual_ns * self.cfg.sample_hz * 1e-9
        if delay > self.cfg.max_action_delay_steps:
            delay = self.cfg.max_action_delay_steps
        heapq.heappush(self.events, (self.steps + delay, self.serial, a.copy()))
        self.serial += 1
        cursor = float(self.steps)
        deliveries = []
        self.applied_action = np.zeros(2, np.float32)
        while self.events and self.events[0][0] < self.steps + 1:
            due, _, command = heapq.heappop(self.events)
            self._advance(due - cursor)
            # Positive action steers back toward the origin, preserving original sign.
            self.vx -= self.cfg.control_gain * float(command[0])
            self.vy -= self.cfg.control_gain * float(command[1])
            self.applied_action = command.copy()
            deliveries.append(dict(due_step=float(due), action=command.tolist()))
            cursor = due
        self._advance(self.steps + 1 - cursor)
        # White process disturbance is an unknown velocity impulse once per sample.
        self.vx += float(self.np_random.normal(0, self.cfg.wind_std))
        self.vy += float(self.np_random.normal(0, self.cfg.wind_std))
        self.steps += 1
        self.last_deliveries = deliveries
        self.history.append((self.x, self.y))
        self.history = self.history[-100:]
        distance = float(np.hypot(self.x, self.y))
        reward = {
            "distance": -distance,
            "squared": -(distance**2),
            "tanhsq": -np.tanh(distance**2),
        }[self.cfg.reward_kind]
        reward -= self.cfg.action_penalty * float(a @ a)
        terminated = distance > self.cfg.boundary
        truncated = self.steps >= self.cfg.episode_steps
        if terminated:
            reward -= self.cfg.loss_penalty
        self.finished = terminated or truncated
        return (
            self._obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(),
        )
