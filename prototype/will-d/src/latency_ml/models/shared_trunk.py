from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from latency_ml.models.architectures import bounded_relu
class Architecture:
    def __init__(self, widths=None):
        self.widths = tuple(widths) if widths is not None else ()
    def describe(self):
        return "x".join(str(w) for w in self.widths)
@dataclass(frozen=True)
class TaskShape:
    obs_dim: int
    n_actions: int
    @property
    def adapter_architecture(self) -> Architecture:
        return Architecture(widths=(self.obs_dim, _SHARED_WIDTH))
    @property
    def actor_architecture(self) -> Architecture:
        return Architecture(widths=(_SHARED_WIDTH, self.n_actions))
    @property
    def critic_architecture(self) -> Architecture:
        return Architecture(widths=(_SHARED_WIDTH, 1))
_SHARED_WIDTH: int = 128
SHARED_HIDDEN: tuple[int, ...] = (128, 128)
class SharedTrunkPolicy(nn.Module):
    def __init__(self, task_shapes: dict[str, TaskShape], shared_width: int = _SHARED_WIDTH, hidden: tuple[int, ...] = SHARED_HIDDEN, activation: nn.Module | None = None) -> None:
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
            layers.append(nn.Linear(fan_in, width))
            layers.append(copy.deepcopy(act))
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
    def head_parameters(self, task: str) -> list[nn.Parameter]:
        self._check_task(task)
        return list(self.actors[task].parameters()) + list(self.critics[task].parameters())
    def _check_task(self, task: str) -> None:
        if task not in self.shapes:
            raise KeyError(f"unknown task {task!r}; known tasks: {self.tasks}")
    def features(self, obs: torch.Tensor, task: str) -> torch.Tensor:
        self._check_task(task)
        features: torch.Tensor = self.trunk(self.adapters[task](obs))
        return features
    def forward(self, obs: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(obs, task)
        logits: torch.Tensor = self.actors[task](features)
        value: torch.Tensor = self.critics[task](features).flatten()
        return (logits, value)
    def action_and_value(self, obs: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, task)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
        action: torch.Tensor = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(action)
        return (action, log_prob, value)
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, task)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
        log_probs: torch.Tensor = dist.log_prob(actions)
        entropy: torch.Tensor = dist.entropy()
        return (log_probs, entropy, value, dist.probs)
    def adapter_architecture(self, task: str) -> Architecture:
        self._check_task(task)
        return self.shapes[task].adapter_architecture
    def actor_architecture(self, task: str) -> Architecture:
        self._check_task(task)
        return self.shapes[task].actor_architecture
    def trunk_architecture(self) -> Architecture:
        return Architecture(widths=(self.shared_width, *SHARED_HIDDEN))
    def save_shared(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)
    def load_shared(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
    def describe(self) -> str:
        trunk = Architecture(widths=(self.shared_width, *SHARED_HIDDEN))
        return f"shared trunk {trunk.describe()} | " + " | ".join(
            f"{t}: adapter {self.shapes[t].adapter_architecture.describe()}, head {self.shapes[t].actor_architecture.describe()}" for t in self.tasks
        )
def probe_task_shapes(tasks: tuple[str, ...]) -> dict[str, TaskShape]:
    from latency_ml.envs.factory import probe_env_dims
    shapes: dict[str, TaskShape] = {}
    for task in tasks:
        obs_dim, n_actions = probe_env_dims(task)
        shapes[task] = TaskShape(obs_dim=obs_dim, n_actions=n_actions)
    return shapes
def build_shared_trunk_policy(tasks: tuple[str, ...], activation: nn.Module | None = None) -> SharedTrunkPolicy:
    return SharedTrunkPolicy(probe_task_shapes(tasks), activation=activation)
