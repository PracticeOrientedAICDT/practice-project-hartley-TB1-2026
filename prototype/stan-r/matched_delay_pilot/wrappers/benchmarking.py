"""Integer-step FIFO ablations. Valid in training too, if used consistently.
Do NOT stack on top of embedded delays unless intentionally adding more delay.
"""
from collections import deque
import gymnasium as gym
import numpy as np


class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay_steps):
        super().__init__(env)
        if not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("Nonnegative integer required")
        self.delay_steps = delay_steps
        self.action_queue = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_queue.clear()
        for _ in range(self.delay_steps):
            self.action_queue.append(
                np.zeros(self.action_space.shape, self.action_space.dtype)
            )
        return obs, info

    def step(self, action):
        self.action_queue.append(np.array(action, copy=True))
        return self.env.step(self.action_queue.popleft())


class ObservationDelayWrapper(gym.Wrapper):
    def __init__(self, env, delay_steps):
        super().__init__(env)
        if not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("Nonnegative integer required")
        self.delay_steps = delay_steps
        self.obs_queue = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_queue.clear()
        for _ in range(self.delay_steps + 1):
            self.obs_queue.append(np.array(obs, copy=True))
        return self.obs_queue.popleft(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.obs_queue.append(np.array(obs, copy=True))
        return self.obs_queue.popleft(), r, term, trunc, info
