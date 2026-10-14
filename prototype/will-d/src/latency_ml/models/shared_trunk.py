"""Shared-trunk policy for the Phase 4 generality experiment.

The Phase 4 question is whether a *jointly trained* shared representation retains
per-task control performance relative to independently trained dedicated
networks. The topology therefore matters, and it is the only independent
variable: environment, PPO, budget, latency, evaluation and seeds are all held
identical to the Phase 1/2 dedicated baselines.

Topology, following the PLAN.md 15.2 diagram::

    CartPole obs --- adapter ---\\
                                >-- shared trunk --+-- actor head --> CartPole actions
    LunarLander obs - adapter --/                    \\-- critic head -> CartPole value

                                >-- shared trunk --+-- actor head --> LunarLander actions
                                \\-- critic head -> LunarLander value

    (trunk drawn once, entered twice)

What is shared is the **representation**. Adapters and both heads are
task-specific, and the critics are separate per task. Sharing the critic would
add a representational constraint -- two different reward scales through one
value function -- that has nothing to do with the question being asked, so it is
deliberately not shared.

This is deliberately *not* "two different networks versus one constrained network
everywhere". The dedicated baseline for a task is an MLP whose trunk is exactly
this shared trunk minus one 128x128 layer, with the adapter fused into the input
projection. Comparing against two wholly unrelated networks would confound
capacity with sharing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from latency_ml.hardware.mac_counter import Architecture
from latency_ml.models.activations import bounded_relu


@dataclass(frozen=True)
class TaskShape:
    """Observation and action dimensions for one task.

    Attributes:
        obs_dim: Observation vector length.
        n_actions: Discrete action count.
    """

    obs_dim: int
    n_actions: int

    @property
    def adapter_architecture(self) -> Architecture:
        """The ``obs -> shared_width`` input projection."""
        return Architecture(widths=(self.obs_dim, _SHARED_WIDTH))

    @property
    def actor_architecture(self) -> Architecture:
        """The ``shared_width -> n_actions`` output head."""
        return Architecture(widths=(_SHARED_WIDTH, self.n_actions))

    @property
    def critic_architecture(self) -> Architecture:
        """The ``shared_width -> 1`` value head."""
        return Architecture(widths=(_SHARED_WIDTH, 1))


#: Width of the shared representation, and of every head's input.
_SHARED_WIDTH: int = 128

#: Hidden widths of the shared trunk. PLAN.md 15.2 draws it as 128->128, 128->128.
SHARED_HIDDEN: tuple[int, ...] = (128, 128)


class SharedTrunkPolicy(nn.Module):
    """One shared representation with per-task adapters and heads.

    The parameter split is exposed explicitly because it drives optimiser
    construction: :meth:`trunk_parameters` is shared across tasks, while
    :meth:`task_parameters` is not. See
    :mod:`latency_ml.rl.multi_task` for why that split is load-bearing.

    The activation is the same module
    :class:`~latency_ml.models.architectures.MLPPolicy` uses. The dedicated and
    shared arms are only comparable if their nonlinearity is identical, so this
    takes the module rather than a name and defaults through the same factory.
    """

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

        # Trunk: shared_width -> hidden[0] -> ... -> shared_width. Entered by every
        # task, so it is the only place where gradients legitimately cross tasks.
        layers: list[nn.Module] = []
        fan_in = shared_width
        for width in hidden:
            layers.append(nn.Linear(fan_in, width))
            layers.append(copy.deepcopy(act))
            fan_in = width
        if fan_in != shared_width:
            raise ValueError(f"trunk must end at shared_width={shared_width}, got {fan_in}")
        self.trunk = nn.Sequential(*layers)

        # Task-specific input projection, actor head and critic head.
        self.adapters = nn.ModuleDict()
        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        for task, shape in task_shapes.items():
            self.adapters[task] = nn.Sequential(
                nn.Linear(shape.obs_dim, shared_width), copy.deepcopy(act)
            )
            self.actors[task] = nn.Linear(shared_width, shape.n_actions)
            self.critics[task] = nn.Linear(shared_width, 1)

    # -- parameter groups -------------------------------------------------

    def trunk_parameters(self) -> list[nn.Parameter]:
        """Parameters that legitimately receive gradients from every task."""
        return list(self.trunk.parameters())

    def task_parameters(self, task: str) -> list[nn.Parameter]:
        """Parameters belonging to one task alone.

        These must be updated only on that task's optimiser steps. Stepping them
        on another task's step would move weights on a stale, non-zero Adam
        moment even though the current loss never touched them.
        """
        self._check_task(task)
        return (
            list(self.adapters[task].parameters())
            + list(self.actors[task].parameters())
            + list(self.critics[task].parameters())
        )

    def head_parameters(self, task: str) -> list[nn.Parameter]:
        """Actor and critic heads only, excluding the input adapter."""
        self._check_task(task)
        return list(self.actors[task].parameters()) + list(self.critics[task].parameters())

    def _check_task(self, task: str) -> None:
        if task not in self.shapes:
            raise KeyError(f"unknown task {task!r}; known tasks: {self.tasks}")

    # -- forward paths ----------------------------------------------------

    def features(self, obs: torch.Tensor, task: str) -> torch.Tensor:
        """Shared representation for one task's observations.

        Args:
            obs: ``(batch, obs_dim)`` for ``task``.
            task: Task name.

        Returns:
            ``(batch, shared_width)`` features from the shared trunk.
        """
        self._check_task(task)
        features: torch.Tensor = self.trunk(self.adapters[task](obs))
        return features

    def forward(self, obs: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor logits and value estimate.

        The value is **flattened** to ``(batch,)``, matching
        :meth:`latency_ml.models.architectures.MLPPolicy.forward`. Keeping the
        same convention means the rollout and loss code is byte-identical
        between the dedicated and shared trainers, which is part of what makes
        the Phase 4 comparison a controlled one.

        Returns:
            ``(logits, value)`` with shapes ``(batch, n_actions)`` and
            ``(batch,)``.
        """
        features = self.features(obs, task)
        logits: torch.Tensor = self.actors[task](features)
        value: torch.Tensor = self.critics[task](features).flatten()
        return logits, value

    def action_and_value(
        self, obs: torch.Tensor, task: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return its log-probability and value.

        The sample and its log-probability come from the same distribution, so a
        separate recomputation would score the action under a different
        distribution and quietly corrupt the PPO ratio.
        """
        logits, value = self.forward(obs, task)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
        action: torch.Tensor = dist.sample()  # type: ignore[no-untyped-call]
        log_prob: torch.Tensor = dist.log_prob(action)  # type: ignore[no-untyped-call]
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, task: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score stored actions, for the PPO ratio.

        Returns:
            ``(log_probs, entropy, values, probs)``, matching the dedicated
            trainer's signature so the loss code is identical between the two.
        """
        logits, value = self.forward(obs, task)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(logits=logits)
        log_probs: torch.Tensor = dist.log_prob(actions)  # type: ignore[no-untyped-call]
        entropy: torch.Tensor = dist.entropy()  # type: ignore[no-untyped-call]
        return log_probs, entropy, value, dist.probs

    # -- accounting -------------------------------------------------------

    def adapter_architecture(self, task: str) -> Architecture:
        self._check_task(task)
        return self.shapes[task].adapter_architecture

    def actor_architecture(self, task: str) -> Architecture:
        self._check_task(task)
        return self.shapes[task].actor_architecture

    def trunk_architecture(self) -> Architecture:
        return Architecture(widths=(self.shared_width, *SHARED_HIDDEN))

    def save_shared(self, path: str | Path) -> None:
        """Write the trunk's full state dict.

        One file for the whole trunk, not per task: the weights belong to the
        shared representation, so a per-task checkpoint would be ambiguous about
        what it contains.
        """
        torch.save(self.state_dict(), path)

    def load_shared(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

    def describe(self) -> str:
        trunk = Architecture(widths=(self.shared_width, *SHARED_HIDDEN))
        return f"shared trunk {trunk.describe()} | " + " | ".join(
            f"{t}: adapter {self.shapes[t].adapter_architecture.describe()}, "
            f"head {self.shapes[t].actor_architecture.describe()}"
            for t in self.tasks
        )


def probe_task_shapes(tasks: tuple[str, ...]) -> dict[str, TaskShape]:
    """Build :class:`TaskShape` values by probing each environment.

    Args:
        tasks: Environment names, e.g. ``("cartpole", "lunarlander")``.

    Returns:
        Shapes keyed by task name.
    """
    from latency_ml.envs.factory import probe_env_dims

    shapes: dict[str, TaskShape] = {}
    for task in tasks:
        obs_dim, n_actions = probe_env_dims(task)
        shapes[task] = TaskShape(obs_dim=obs_dim, n_actions=n_actions)
    return shapes


def build_shared_trunk_policy(
    tasks: tuple[str, ...],
    activation: nn.Module | None = None,
) -> SharedTrunkPolicy:
    """Build the shared-trunk policy for the Phase 4 tasks.

    Args:
        tasks: Environment names. Order is irrelevant; parameters are keyed by
            name so task identity never depends on construction order.
        activation: Nonlinearity override. See
            :class:`~latency_ml.models.shared_trunk.SharedTrunkPolicy`.
    """
    return SharedTrunkPolicy(probe_task_shapes(tasks), activation=activation)
