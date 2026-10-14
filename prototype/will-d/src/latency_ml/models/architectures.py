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


class MLPPolicy(nn.Module):
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
        widths = (obs_dim, *hidden, n_actions)
        self.arch = Architecture(widths)
        layers: list[nn.Module] = []
        for fan_in, fan_out in itertools.pairwise(widths[:-1]):
            layers.append(nn.Linear(fan_in, fan_out))
            layers.append(copy.deepcopy(act))
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.actor = nn.Sequential(*layers)
        value_widths = (obs_dim, *hidden, 1)
        value_layers: list[nn.Module] = []
        for fan_in, fan_out in itertools.pairwise(value_widths[:-1]):
            value_layers.append(nn.Linear(fan_in, fan_out))
            value_layers.append(copy.deepcopy(act))
        value_layers.append(nn.Linear(value_widths[-2], value_widths[-1]))
        self.critic = nn.Sequential(*value_layers) if use_value_head else None

    def parameter_groups(self) -> list[ParameterGroup]:
        from latency_ml.rl.task_policy import ParameterGroup

        return [ParameterGroup("policy", list(self.parameters()), stepped_on_every_task=True)]

    @property
    def actor_arch(self) -> Architecture:
        return self.arch

    def to_device(self, device: torch.device | str) -> None:
        self.to(device)

    def set_mode(self, training: bool) -> None:
        self.train(training)

    def is_training(self) -> bool:
        return self.training

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), weights_only=True))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.actor(obs)
        value = self.critic(obs).flatten() if self.critic is not None else None
        return (logits, value)

    def action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        assert value is not None, "action_and_value requires a value head"
        log_probs_all = functional.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs_all.exp(), num_samples=1).squeeze(-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return (actions, log_probs, value)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        assert value is not None
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
    policy = MLPPolicy(obs_dim, n_actions, hidden=CONTROL_HIDDEN, activation=activation)
    return policy


def build_shared_trunk() -> MLPPolicy:
    trunk = MLPPolicy(128, 128, hidden=(128, 128), use_value_head=False)
    return trunk


def describe_architecture(policy: MLPPolicy) -> str:
    arch = policy.actor_arch
    return f"{arch.describe()} | {arch.num_layers} dense layers | max width {arch.max_width} | {arch.macs()} MACs | activation {policy.activation_name} | {arch.parameters()} params"




class Architecture:
    def __init__(self, widths=None):
        self.widths = tuple(widths) if widths is not None else ()
    def describe(self):
        return 'x'.join(str(w) for w in self.widths)
    @property
    def num_layers(self):
        return len(self.widths)
    @property
    def max_width(self):
        return max(self.widths) if self.widths else 0
    def macs(self):
        return 0
    def parameters(self):
        return 0

CLIP: float = 2.0
class HartleyBoundedReLU(nn.Module):
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


ACTIVATION_NAME: str = type(HartleyBoundedReLU()).__name__
__all__ = ["ACTIVATION_NAME", "CLIP", "HartleyBoundedReLU", "bounded_relu"]
