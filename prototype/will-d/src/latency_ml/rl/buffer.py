from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor

    def __len__(self) -> int:
        return int(self.obs.shape[0])


class RolloutBuffer:
    """Fixed-shape storage for one PPO rollout with variable hold durations."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape: tuple[int, ...]) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        shape = (num_steps, num_envs)
        self.obs = torch.zeros((*shape, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(shape, dtype=torch.int64)
        self.log_probs = torch.zeros(shape, dtype=torch.float32)
        self.rewards = torch.zeros(shape, dtype=torch.float32)
        self.dones = torch.zeros(shape, dtype=torch.float32)
        self.truncations = torch.zeros(shape, dtype=torch.float32)
        self.trunc_values = torch.zeros(shape, dtype=torch.float32)
        self.values = torch.zeros(shape, dtype=torch.float32)
        self.durations = torch.ones(shape, dtype=torch.float32)
        self.advantages = torch.zeros(shape, dtype=torch.float32)
        self.returns = torch.zeros(shape, dtype=torch.float32)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.num_steps

    @property
    def total_duration(self) -> int:
        return int(self.durations[: self.ptr].sum().item())

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        durations: torch.Tensor,
        truncations: torch.Tensor,
        trunc_values: torch.Tensor,
    ) -> None:
        assert not self.full, "rollout buffer is full"
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.values[i] = values
        self.durations[i] = durations
        self.truncations[i] = truncations
        self.trunc_values[i] = trunc_values
        self.ptr += 1

    def _next_values(self, step: int, last_value: torch.Tensor) -> torch.Tensor:
        truncated = self.truncations[step] > 0
        if bool(truncated.any()):
            fallback = last_value if step == self.ptr - 1 else self.values[step + 1]
            return torch.where(truncated, self.trunc_values[step], fallback)
        return last_value if step == self.ptr - 1 else self.values[step + 1]

    def _same_episode(self, step: int) -> torch.Tensor:
        return 1.0 - torch.clamp(self.dones[step] + self.truncations[step], max=1.0)

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma_phys: float,
        gae_lambda_phys: float,
        duration_aware: bool = True,
    ) -> None:
        gamma: float | torch.Tensor = gamma_phys
        lam: float | torch.Tensor = gae_lambda_phys
        if duration_aware:
            durations = self.durations[: self.ptr]
            gamma = torch.pow(gamma_phys, durations)
            lam = torch.pow(gae_lambda_phys, durations)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for step in reversed(range(self.ptr)):
            next_values = self._next_values(step, last_value)
            not_terminal = 1.0 - self.dones[step]
            gamma_t = gamma[step] if duration_aware else gamma
            lambda_t = lam[step] if duration_aware else lam
            delta = self.rewards[step] + gamma_t * next_values * not_terminal - self.values[step]
            last_gae = delta + gamma_t * lambda_t * self._same_episode(step) * last_gae
            self.advantages[step] = last_gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get(self, advantages_normalised: bool = True) -> Batch:
        flat = slice(0, self.ptr)
        advantages = self.advantages[flat].reshape(-1)
        if advantages_normalised and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        return Batch(
            obs=self.obs[flat].reshape(-1, *self.obs_shape),
            actions=self.actions[flat].reshape(-1),
            log_probs=self.log_probs[flat].reshape(-1),
            advantages=advantages,
            returns=self.returns[flat].reshape(-1),
            values=self.values[flat].reshape(-1),
        )
