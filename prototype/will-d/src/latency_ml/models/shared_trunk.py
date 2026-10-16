from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from latency_ml.models.architectures import bounded_relu

_SHARED_WIDTH: int = 128
SHARED_HIDDEN: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class TaskShape:
    obs_dim: int
    n_actions: int


class SharedTrunkPolicy(nn.Module):
    """One trunk + per-task adapter/actor/critic heads, trained on both tasks per iteration."""

    def __init__(
        self,
        task_shapes: dict[str, TaskShape],
        shared_width: int = _SHARED_WIDTH,
        hidden: tuple[int, ...] = SHARED_HIDDEN,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not task_shapes:
            raise ValueError("need at least one task")
        self.tasks: tuple[str, ...] = tuple(sorted(task_shapes))
        self.shapes = dict(task_shapes)
        self.shared_width = shared_width
        act = bounded_relu() if activation is None else activation
        self.activation_name = type(act).__name__
        layers: list[nn.Module] = []
        fan_in = shared_width
        for width in hidden:
            layers += [nn.Linear(fan_in, width), copy.deepcopy(act)]
            fan_in = width
        if fan_in != shared_width:
            raise ValueError(f"trunk must end at shared_width={shared_width}, got {fan_in}")
        self.trunk = nn.Sequential(*layers)
        self.adapters = nn.ModuleDict()
        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        for task, shape in task_shapes.items():
            self.adapters[task] = nn.Sequential(nn.Linear(shape.obs_dim, shared_width), copy.deepcopy(act))
            self.actors[task] = nn.Linear(shared_width, shape.n_actions)
            self.critics[task] = nn.Linear(shared_width, 1)

    def trunk_parameters(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters())

    def task_parameters(self, task: str) -> list[nn.Parameter]:
        self._check_task(task)
        return list(self.adapters[task].parameters()) + list(self.actors[task].parameters()) + list(self.critics[task].parameters())

    def _check_task(self, task: str) -> None:
        if task not in self.shapes:
            raise KeyError(f"unknown task {task!r}; known tasks: {self.tasks}")

    def features(self, obs: torch.Tensor, task: str) -> torch.Tensor:
        self._check_task(task)
        return self.trunk(self.adapters[task](obs))

    def forward(self, obs: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(obs, task)
        return (self.actors[task](features), self.critics[task](features).flatten())

    def action_and_value(self, obs: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, task)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (action, dist.log_prob(action), value)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, task)
        dist = torch.distributions.Categorical(logits=logits)
        return (dist.log_prob(actions), dist.entropy(), value, dist.probs)

    def save_shared(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_shared(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))


def probe_task_shapes(tasks: tuple[str, ...]) -> dict[str, TaskShape]:
    from latency_ml.envs.factory import probe_env_dims

    shapes: dict[str, TaskShape] = {}
    for task in tasks:
        obs_dim, n_actions = probe_env_dims(task)
        shapes[task] = TaskShape(obs_dim, n_actions)
    return shapes


def build_shared_trunk_policy(tasks: tuple[str, ...], activation: nn.Module | None = None) -> SharedTrunkPolicy:
    return SharedTrunkPolicy(probe_task_shapes(tasks), activation=activation)


__all__ = ["SHARED_HIDDEN", "SharedTrunkPolicy", "TaskShape", "build_shared_trunk_policy", "probe_task_shapes"]
