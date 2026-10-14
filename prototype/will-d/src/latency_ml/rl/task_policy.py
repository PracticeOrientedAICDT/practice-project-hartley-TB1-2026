from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import nn

if TYPE_CHECKING:
    from latency_ml.models.shared_trunk import SharedTrunkPolicy


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    parameters: list[nn.Parameter]
    stepped_on_every_task: bool


@runtime_checkable
class TaskPolicy(Protocol):
    obs_dim: int
    n_actions: int
    activation_name: str

    def parameter_groups(self) -> list[ParameterGroup]: ...
    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def action_only(self, obs: torch.Tensor) -> torch.Tensor: ...
    def to_device(self, device: torch.device | str) -> None: ...
    def set_mode(self, training: bool) -> None: ...
    def is_training(self) -> bool: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...


class SharedTaskPolicy:
    def __init__(self, policy: SharedTrunkPolicy, task: str) -> None:
        self.policy = policy
        self.task = task
        shape = policy.shapes[task]
        self.obs_dim = shape.obs_dim
        self.n_actions = shape.n_actions
        self.activation_name = policy.activation_name

    def parameter_groups(self) -> list[ParameterGroup]:
        return [
            ParameterGroup("trunk", self.policy.trunk_parameters(), stepped_on_every_task=True),
            ParameterGroup(
                f"{self.task}:adapter_head",
                self.policy.task_parameters(self.task),
                stepped_on_every_task=False,
            ),
        ]

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.policy(obs, self.task)
        return (logits, value)

    def to_device(self, device: torch.device | str) -> None:
        self.policy.to(device)

    def set_mode(self, training: bool) -> None:
        self.policy.train(training)

    def is_training(self) -> bool:
        return self.policy.training

    def action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.action_and_value(obs, self.task)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.evaluate_actions(obs, actions, self.task)

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.policy(obs, self.task)
        action: torch.Tensor = logits.argmax(dim=-1)
        return action

    def describe(self) -> str:
        return f"shared trunk, task {self.task}"

    def save(self, path: str | Path) -> None:
        self.policy.save_shared(path)

    def load(self, path: str | Path) -> None:
        self.policy.load_shared(path)


__all__ = ["ParameterGroup", "SharedTaskPolicy", "TaskPolicy"]
