from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from latency_ml.rl.task_policy import ParameterGroup

CONTROL_HIDDEN: tuple[int, ...] = (128, 128)
CLIP: float = 2.0


class HartleyBoundedReLU(nn.Module):
    """Capped ReLU: relu with a hard upper clip, the bounded-activation arm of the sweep."""

    def __init__(self, clip: float = CLIP) -> None:
        super().__init__()
        if clip <= 0.0:
            raise ValueError(f"clip must be positive, got {clip!r}")
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.clip)

    def extra_repr(self) -> str:
        return f"clip={self.clip:g}"


def bounded_relu() -> HartleyBoundedReLU:
    return HartleyBoundedReLU()


class MLPPolicy(nn.Module):
    """Dedicated per-task actor-critic; shared by both activation arms."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = CONTROL_HIDDEN,
        activation: nn.Module | None = None,
        use_value_head: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.use_value_head = use_value_head
        act = bounded_relu() if activation is None else activation
        self.activation_name = type(act).__name__
        actor: list[nn.Module] = []
        widths = (obs_dim, *self.hidden)
        for fan_in, fan_out in itertools.pairwise(widths):
            actor += [nn.Linear(fan_in, fan_out), copy.deepcopy(act)]
        actor.append(nn.Linear(widths[-1], n_actions))
        self.actor = nn.Sequential(*actor)
        critic: list[nn.Module] = []
        value_widths = (obs_dim, *self.hidden)
        for fan_in, fan_out in itertools.pairwise(value_widths):
            critic += [nn.Linear(fan_in, fan_out), copy.deepcopy(act)]
        critic.append(nn.Linear(value_widths[-1], 1))
        self.critic = nn.Sequential(*critic) if use_value_head else None

    def parameter_groups(self) -> list[ParameterGroup]:
        from latency_ml.rl.task_policy import ParameterGroup

        return [ParameterGroup("policy", list(self.parameters()), stepped_on_every_task=True)]

    def to_device(self, device: torch.device | str) -> None:
        self.to(device)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.actor(obs)
        value = self.critic(obs).flatten() if self.critic is not None else None
        return (logits, value)

    def action_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        assert value is not None, "action_and_value requires a value head"
        log_probs_all = functional.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs_all.exp(), num_samples=1).squeeze(-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return (actions, log_probs, value)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        assert value is not None, "evaluate_actions requires a value head"
        log_probs_all = functional.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
        return (log_probs, entropy, value, logits)

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        return logits.argmax(dim=-1)


def build_control_policy(task: str, activation: nn.Module | None = None) -> MLPPolicy:
    from latency_ml.envs.factory import probe_env_dims

    obs_dim, n_actions = probe_env_dims(task)
    return MLPPolicy(obs_dim, n_actions, hidden=CONTROL_HIDDEN, activation=activation)


__all__ = ["CONTROL_HIDDEN", "MLPPolicy", "bounded_relu", "build_control_policy"]
