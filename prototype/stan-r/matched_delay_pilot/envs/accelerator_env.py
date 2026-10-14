"""Paper-inspired oscillator with optional within-episode uncertainty.

All stochastic-evolution fields default to zero. With them disabled this reduces
to the deterministic timestamped Green-function oscillator used in v4.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import heapq
import numpy as np
from gymnasium import spaces
from .beam_env import BeamConfig, BeamEnv


@dataclass
class AcceleratorConfig(BeamConfig):
    tune: float = 0.24
    tune_jitter: float = 0.005  # sampled once per episode
    obs_delay_range: tuple = (0, 0)
    action_delay_range: tuple = (0.0, 0.0)
    max_action_delay: float = 128.0
    inference_ns: float = 100.0
    transport_ns: float = 0.0
    inference_range_ns: Optional[Tuple[float, float]] = None
    latency_features: bool = True
    latency_scale_ns: float = 5000.0
    loss_penalty: float = 1000.0

    # Optional nondeterministic evolution. All zero means OFF.
    tune_drift_std_per_turn: float = 0.0  # Gaussian innovation in tune/turn
    tune_drift_reversion: float = 0.0  # 0=random walk, (0,1]=mean reversion
    tune_jump_probability: float = 0.0  # Bernoulli probability per turn
    tune_jump_std: float = 0.0  # Gaussian jump in tune
    momentum_process_std: float = 0.0  # unknown p impulse per turn
    inference_jitter_ns: float = 0.0  # Gaussian per-command delivery jitter
    latency_outlier_probability: float = 0.0
    latency_outlier_ns: float = 0.0  # positive extra delay on an outlier
    actuator_gain_std_fraction: float = 0.0  # per-command multiplicative gain std


class AcceleratorEnv(BeamEnv):

    def __init__(self, config=None):

        super().__init__(config or AcceleratorConfig())
        c = self.cfg
        nonnegative = [
            c.inference_ns,
            c.transport_ns,
            c.latency_scale_ns,
            c.loss_penalty,
            c.tune_drift_std_per_turn,
            c.tune_drift_reversion,
            c.tune_jump_probability,
            c.tune_jump_std,
            c.momentum_process_std,
            c.inference_jitter_ns,
            c.latency_outlier_probability,
            c.latency_outlier_ns,
            c.actuator_gain_std_fraction,
        ]
        if (
            not np.isfinite(nonnegative).all()
            or min(nonnegative) < 0
            or c.latency_scale_ns == 0
        ):
            raise ValueError(
                "Latency, noise and penalty values must be finite and nonnegative"
            )
        if not 0 <= c.tune_drift_reversion <= 1:
            raise ValueError("tune_drift_reversion must be in [0,1]")
        if (
            not 0 <= c.tune_jump_probability <= 1
            or not 0 <= c.latency_outlier_probability <= 1
        ):
            raise ValueError("Probabilities must be in [0,1]")
        if c.inference_range_ns is not None:
            lo, hi = c.inference_range_ns
            if not 0 <= lo <= hi or not np.isfinite([lo, hi]).all():
                raise ValueError("Invalid inference range")
        if c.damping >= 2 * np.pi * (c.tune - c.tune_jitter) or c.tune <= c.tune_jitter:
            raise ValueError(
                "Requires an underdamped oscillator with positive frequency"
            )
        n = c.position_history + c.action_history + (3 if c.latency_features else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.inference_ns = c.inference_ns
        self.applied_action = self.last_action = 0.0
        self.last_deliveries = []
        self.current_tune = c.tune
        self.nominal_tune = c.tune
        self.last_process_kick = 0.0
        self.last_delay_jitter_ns = 0.0
        self.last_gain_factor = 1.0
        self.last_outlier = False

    def reset(self, *, seed=None, options=None):

        self.inference_ns = self.cfg.inference_ns
        self.applied_action = self.last_action = 0.0
        self.last_deliveries = []
        self.last_process_kick = 0.0
        self.last_delay_jitter_ns = 0.0
        self.last_gain_factor = 1.0
        self.last_outlier = False
        super().reset(seed=seed, options=options)
        # Base reset samples self.tune. It is the episode mean-reversion centre.
        self.nominal_tune = float(self.tune)
        self.current_tune = float(self.tune)
        if self.cfg.inference_range_ns is not None:
            self.inference_ns = float(
                self.np_random.uniform(*self.cfg.inference_range_ns)
            )
        self.act_delay += (
            (self.inference_ns + self.cfg.transport_ns) * self.cfg.sample_hz * 1e-9
        )
        worst = (
            (
                self.inference_ns
                + self.cfg.transport_ns
                + 6 * self.cfg.inference_jitter_ns
                + self.cfg.latency_outlier_ns
            )
            * self.cfg.sample_hz
            * 1e-9
        )
        if worst > self.cfg.max_action_delay:
            raise ValueError("Configured stochastic delay can exceed max_action_delay")
        return self._obs(), self._info()

    def _obs(self):

        obs = super()._obs()
        if self.cfg.latency_features:
            scale = self.cfg.latency_scale_ns
            obs = np.concatenate(
                [
                    obs,
                    [
                        self.obs_delay / self.cfg.sample_hz * 1e9 / scale,
                        self.inference_ns / scale,
                        self.act_delay / self.cfg.sample_hz * 1e9 / scale,
                    ],
                ]
            )
        return obs.astype(np.float32)

    def _advance(self, turns):

        w0 = 2 * np.pi * self.current_tune
        g = self.cfg.damping
        w = np.sqrt(max(w0 * w0 - g * g, 1e-18))
        co, si = np.cos(w * turns), np.sin(w * turns)
        decay = np.exp(-g * turns)
        x, p = self.x, self.p
        self.x = decay * ((co + g / w * si) * x + w0 / w * si * p)
        self.p = decay * (-w0 / w * si * x + (co - g / w * si) * p)

    def _evolve_tune(self):
        c = self.cfg
        delta = self.current_tune - self.nominal_tune
        self.current_tune += -c.tune_drift_reversion * delta
        if c.tune_drift_std_per_turn:
            self.current_tune += float(
                self.np_random.normal(0, c.tune_drift_std_per_turn)
            )
        jumped = False
        if (
            c.tune_jump_probability
            and self.np_random.random() < c.tune_jump_probability
        ):
            self.current_tune += float(self.np_random.normal(0, c.tune_jump_std))
            jumped = True
        # Fail loudly rather than silently entering an invalid oscillator regime.
        if self.current_tune <= 0 or self.cfg.damping >= 2 * np.pi * self.current_tune:
            raise RuntimeError("Stochastic tune left the valid underdamped regime")
        return jumped

    def _info(self):

        info = super()._info()
        w0 = 2 * np.pi * self.current_tune
        w = np.sqrt(max(w0 * w0 - self.cfg.damping**2, 1e-18))
        amplitude = np.hypot(self.x, (w0 * self.p + self.cfg.damping * self.x) / w)
        pending = []
        for event in sorted(self.events):
            due, serial, effective, *rest = event
            pending.append(
                dict(
                    due_turn=float(due),
                    command=float(effective),
                    gain_factor=float(rest[0]) if rest else 1.0,
                )
            )
        info.update(
            amplitude=float(amplitude),
            inference_ns=self.inference_ns,
            transport_ns=self.cfg.transport_ns,
            total_action_ns=self.act_delay / self.cfg.sample_hz * 1e9,
            issued_action=self.last_action,
            applied_action=self.applied_action,
            deliveries=self.last_deliveries,
            pending_events=pending,
            observed_x=float(self.samples[self.obs_delay]),
            sample_period_ns=1e9 / self.cfg.sample_hz,
            nominal_tune=float(self.nominal_tune),
            current_tune=float(self.current_tune),
            tune_offset=float(self.current_tune - self.nominal_tune),
            process_momentum_kick=float(self.last_process_kick),
            delivery_jitter_ns=float(self.last_delay_jitter_ns),
            actuator_gain_factor=float(self.last_gain_factor),
            latency_outlier=bool(self.last_outlier),
        )
        return info

    def step(self, action):
        
        if self.finished:
            raise RuntimeError("Call reset after termination/truncation")
        a = np.asarray(action, dtype=float)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError("Expected finite action of shape (1,)")
        u = float(np.clip(a[0], -1, 1))
        self.last_action = u
        c = self.cfg
        jitter = (
            float(self.np_random.normal(0, c.inference_jitter_ns))
            if c.inference_jitter_ns
            else 0.0
        )
        outlier = bool(
            c.latency_outlier_probability
            and self.np_random.random() < c.latency_outlier_probability
        )
        actual_ns = max(
            0.0,
            self.inference_ns
            + c.transport_ns
            + jitter
            + (c.latency_outlier_ns if outlier else 0.0),
        )
        delay = actual_ns * c.sample_hz * 1e-9
        gain_factor = (
            float(max(0.0, 1 + self.np_random.normal(0, c.actuator_gain_std_fraction)))
            if c.actuator_gain_std_fraction
            else 1.0
        )
        effective = u * gain_factor
        heapq.heappush(
            self.events, (self.t + delay, self.serial, effective, gain_factor)
        )
        self.serial += 1
        self.last_delay_jitter_ns = jitter
        self.last_outlier = outlier
        self.last_gain_factor = gain_factor
        if c.action_history:
            self.commands[1:] = self.commands[:-1].copy()
            self.commands[0] = u
        if self.t == self.kick_step:
            self.p += self.kick
        jumped = self._evolve_tune()
        cursor = float(self.t)
        deliveries = []
        self.applied_action = 0.0
        while self.events and self.events[0][0] < self.t + 1:
            due, _, command, gain = heapq.heappop(self.events)
            self._advance(due - cursor)
            self.p += c.actuator_gain * command
            cursor = due
            self.applied_action = float(command)
            deliveries.append(
                dict(
                    due_turn=float(due), command=float(command), gain_factor=float(gain)
                )
            )
        self._advance(self.t + 1 - cursor)
        self.last_process_kick = (
            float(self.np_random.normal(0, c.momentum_process_std))
            if c.momentum_process_std
            else 0.0
        )
        self.p += self.last_process_kick
        self.t += 1
        self.last_deliveries = deliveries
        self.samples[1:] = self.samples[:-1].copy()
        self.samples[0] = self.x + self.np_random.normal(0, c.noise_std)
        reward = {
            "l1": -abs(self.x),
            "l2": -self.x**2,
            "tanhsq": -np.tanh(self.x**2),
        }[c.reward_kind] - c.action_penalty * u * u
        terminated = bool(np.hypot(self.x, self.p) > c.loss_radius)
        truncated = bool(self.t >= c.episode_steps)
        if terminated:
            reward -= c.loss_penalty
        self.finished = terminated or truncated
        info = self._info()
        info["tune_jump"] = jumped
        return self._obs(), float(reward), terminated, truncated, info
