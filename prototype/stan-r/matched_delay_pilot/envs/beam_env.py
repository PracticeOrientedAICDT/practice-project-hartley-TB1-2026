"""Paper-inspired simulation, not a calibrated accelerator digital twin."""
from dataclasses import dataclass, asdict
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class BeamConfig:
    sample_hz: float = 2.7e6
    tune: float = 0.76
    tune_jitter: float = 0.005
    damping: float = 0.002  # amplitude decay per turn, illustrative
    actuator_gain: float = 0.08  # momentum impulse per unit action
    noise_std: float = 0.002
    episode_steps: int = 2048
    kick_window: tuple = (32, 128)
    kick_ratio: float = 10.0
    position_history: int = 8
    action_history: int = 16
    max_obs_delay: int = 8
    max_action_delay: float = 16.0
    obs_delay_range: tuple = (0, 2)  # inclusive, sampled once per episode
    action_delay_range: tuple = (0.0, 8.0)  # fractional turns
    target_radius: float = 0.05
    loss_radius: float = 5.0
    reward_kind: str = "l2"
    action_penalty: float = 0.0


class BeamEnv(gym.Env):
    """At t: receive delayed position history, issue impulse due at t+delay.

    step integrates to t+1, processing impulses at exact fractional times.
    Rewards use true post-transition position; no wall-clock sleeping.
    Observation is [latest-first delayed positions, latest-first commands].
    Hidden velocity/delay/queued impulses make this a POMDP. Command history
    helps, but does not guarantee a fully observed Markov state.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or BeamConfig()
        c = self.cfg
        if not (
            c.sample_hz > 0
            and c.episode_steps > 0
            and c.position_history > 0
            and c.action_history >= 0
            and c.max_obs_delay >= 0
            and c.max_action_delay >= 0
            and c.damping >= 0
            and 0 <= c.obs_delay_range[0] <= c.obs_delay_range[1] <= c.max_obs_delay
            and 0
            <= c.action_delay_range[0]
            <= c.action_delay_range[1]
            <= c.max_action_delay
            and 0 <= c.kick_window[0] <= c.kick_window[1] < c.episode_steps
        ):
            raise ValueError("Invalid configuration or delay bounds")
        if c.reward_kind not in ("l1", "l2", "tanhsq"):
            raise ValueError("Unknown reward")
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (c.position_history + c.action_history,), np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.t = 0
        self.x = self.p = 0.0
        self.finished = False
        self.events = []
        self.serial = 0
        self.commands = np.zeros(c.action_history, dtype=np.float32)
        self.samples = np.zeros(c.position_history + c.max_obs_delay, dtype=np.float32)
        self.obs_delay = int(
            self.np_random.integers(c.obs_delay_range[0], c.obs_delay_range[1] + 1)
        )
        self.act_delay = float(self.np_random.uniform(*c.action_delay_range))
        self.tune = c.tune + self.np_random.uniform(-c.tune_jitter, c.tune_jitter)
        self.kick_step = int(
            self.np_random.integers(c.kick_window[0], c.kick_window[1] + 1)
        )
        self.kick = float(
            self.np_random.choice([-1, 1]) * c.kick_ratio * c.actuator_gain
        )
        self.samples[0] = self.np_random.normal(0, c.noise_std)
        return self._obs(), self._info()

    def _obs(self):
        d = self.obs_delay
        return np.concatenate(
            (self.samples[d : d + self.cfg.position_history], self.commands)
        ).astype(np.float32)

    def _advance(self, turns):
        theta = 2 * np.pi * self.tune * turns
        co, si = np.cos(theta), np.sin(theta)
        decay = np.exp(-self.cfg.damping * turns)
        self.x, self.p = decay * (co * self.x + si * self.p), decay * (
            -si * self.x + co * self.p
        )

    def _info(self):
        c = self.cfg
        return dict(
            x=float(self.x),
            p=float(self.p),
            distance=abs(float(self.x)),
            amplitude=float(np.hypot(self.x, self.p)),
            on_target=abs(self.x) < c.target_radius,
            obs_delay_steps=self.obs_delay,
            action_delay_steps=self.act_delay,
            obs_delay_us=self.obs_delay / c.sample_hz * 1e6,
            action_delay_us=self.act_delay / c.sample_hz * 1e6,
            kick_step=self.kick_step,
            kicked=self.t > self.kick_step,
            time_us=self.t / c.sample_hz * 1e6,
            pending_actions=len(self.events),
        )

    def step(self, action):
        
        if self.finished:
            raise RuntimeError("Call reset after termination/truncation")
        a = np.asarray(action, dtype=float)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError("Expected finite action of shape (1,)")
        u = float(np.clip(a[0], -1, 1))
        heapq.heappush(self.events, (self.t + self.act_delay, self.serial, u))
        self.serial += 1
        if self.cfg.action_history:
            self.commands[1:] = self.commands[:-1].copy()
            self.commands[0] = u
        if self.t == self.kick_step:
            self.p += self.kick
        cursor = float(self.t)
        # Boundary events at t+1 are applied at the start of the next step.
        while self.events and self.events[0][0] < self.t + 1:
            due, _, command = heapq.heappop(self.events)
            self._advance(due - cursor)
            self.p += self.cfg.actuator_gain * command
            cursor = due
        self._advance(self.t + 1 - cursor)
        self.t += 1
        self.samples[1:] = self.samples[:-1].copy()
        self.samples[0] = self.x + self.np_random.normal(0, self.cfg.noise_std)
        r = {"l1": -abs(self.x), "l2": -self.x**2, "tanhsq": -np.tanh(self.x**2)}[
            self.cfg.reward_kind
        ]
        r -= self.cfg.action_penalty * u * u
        terminated = bool(np.hypot(self.x, self.p) > self.cfg.loss_radius)
        truncated = bool(self.t >= self.cfg.episode_steps)
        self.finished = terminated or truncated
        return self._obs(), float(r), terminated, truncated, self._info()
