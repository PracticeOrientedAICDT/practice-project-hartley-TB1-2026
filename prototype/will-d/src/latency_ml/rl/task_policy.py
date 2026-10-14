"""The interface the trainer needs from a policy, and the shared-trunk facade.

There is one trainer. It does not know whether it is driving a dedicated network
for a single task or one task's slice of a shared trunk, because that difference
lives entirely behind :class:`TaskPolicy`.

## Why one trainer rather than two

Two trainers existed, with a rollout loop and a PPO update duplicated between
them. The duplication was a liability rather than a convenience: the shared
trunk's comparison to the dedicated controls is only interpretable if the two
arms run *identical* training code, and two copies of that code are identical
only until someone edits one of them. The shared-trunk docstring used to claim
the loss code was "byte-identical between the dedicated and shared trainers";
that was an aspiration about copy-paste discipline, not a property anything
checked. With one trainer it is a structural fact instead.

The price is a small amount of indirection: every policy-specific call goes
through this interface. The benefit is that the arm under comparison cannot
differ from its baseline in any way except by its parameter topology.

## Parameter groups

A policy declares which parameters exist and which of them a given task's update
may move. For a dedicated policy there is one group holding everything. For a
shared trunk there are two: the trunk, which every task's update steps, and the
task's own adapter and heads, which only that task's update may step.

That distinction is load-bearing, not tidiness. Adam carries per-parameter moment
state, and a parameter whose gradient is zero on a step is *not* left alone --
``exp_avg`` decays by ``beta1`` while ``exp_avg_sq`` decays by ``beta2``, so the
ratio stays non-zero and the step still moves the weight. With a single optimiser
over the trunk and every task's heads, each task would nudge the other task's
heads along a decaying momentum, and no gradient inspection would reveal it,
because the gradient really is zero.
"""

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
    """A set of parameters with one stepping rule.

    Attributes:
        name: Identifier, used in diagnostics and error messages.
        parameters: The parameters. Order is preserved and matters: gradient
            clipping sums over them, so a reordering changes the computed norm
            at the last bit.
        stepped_on_every_task: ``True`` when every task's optimiser update
            includes this group. The shared trunk is the only thing that is.
    """

    name: str
    parameters: list[nn.Parameter]
    stepped_on_every_task: bool


@runtime_checkable
class TaskPolicy(Protocol):
    """One task's trainable view of a policy.

    ``MLPPolicy`` satisfies this directly. ``SharedTaskPolicy`` adapts one task's
    slice of a ``SharedTrunkPolicy`` to it. The trainer is written against this
    and nothing else.
    """

    obs_dim: int
    n_actions: int

    activation_name: str

    def parameter_groups(self) -> list[ParameterGroup]:
        """Parameter groups this policy exposes to the optimiser."""
        ...

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Same as :meth:`forward`.

        Declared because the rollout needs a batched value pass, and going through
        ``__call__`` rather than ``forward`` keeps ``nn.Module``'s hook dispatch in
        the path for the dedicated case.
        """
        ...

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(action_logits, value)``, with the value flattened to ``(batch,)``."""
        ...

    def action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(actions, log_probs, values)``, all sampled from one distribution."""
        ...

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(log_probs, entropy, values, _)`` for a PPO update.

        The fourth element differs between implementations -- logits for a
        dedicated policy, action probabilities for a shared one -- and is not
        used by the loss. It is kept because both implementations returned it.
        """
        ...

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy action, for deterministic evaluation."""
        ...

    def to_device(self, device: torch.device | str) -> None:
        """Move the policy's tensors to ``device``.

        Not named ``to``: ``nn.Module.to`` carries a dozen overloads that return
        the module, and a protocol member has to be satisfiable by both
        implementations without fighting them.
        """
        ...

    def set_mode(self, training: bool) -> None:
        """Put the policy in training or evaluation mode.

        Not ``train``/``eval``: those return the module for chaining, so an
        ``nn.Module`` can never structurally satisfy a protocol declaring them to
        return ``None``. The trainer does not chain off them, so it does not need
        them named that way.
        """
        ...

    def is_training(self) -> bool:
        """Whether the policy is currently in training mode."""
        ...

    def save(self, path: str | Path) -> None:
        """Write the policy's trainable weights to ``path`` exactly."""
        ...

    def load(self, path: str | Path) -> None:
        """Read trainable weights from ``path``."""
        ...


class SharedTaskPolicy:
    """One task's slice of a shared trunk, presenting the ``TaskPolicy`` interface.

    The shared trunk stores a single representation entered by several tasks, so
    its methods all need a task name. This facade holds that name so the trainer
    does not have to pass it at every call site.

    This replaces an earlier read-only view of the same name that existed only to
    let evaluation borrow the trunk. The limitation it documented -- training had
    to go through ``SharedTrunkPolicy`` itself -- was precisely what kept the two
    training paths separate.
    """

    def __init__(self, policy: SharedTrunkPolicy, task: str) -> None:
        self.policy = policy
        self.task = task
        shape = policy.shapes[task]
        self.obs_dim = shape.obs_dim
        self.n_actions = shape.n_actions
        self.activation_name = policy.activation_name

    def parameter_groups(self) -> list[ParameterGroup]:
        """The shared trunk, then this task's own adapter and heads.

        The trunk comes first so the parameter order a gradient clip sees is
        stable and independent of how many tasks are sharing.
        """
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
        return logits, value

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
        """Greedy action for this task, for deterministic evaluation."""
        with torch.no_grad():
            logits, _ = self.policy(obs, self.task)
        action: torch.Tensor = logits.argmax(dim=-1)
        return action

    def describe(self) -> str:
        return f"shared trunk, task {self.task}"

    def save(self, path: str | Path) -> None:
        """Write the whole shared trunk, not this task's slice.

        The slice holds no parameters of its own -- the weights are the trunk's --
        so saving "this task's policy" would write a checkpoint that could not be
        loaded back without knowing which task it came from.
        """
        self.policy.save_shared(path)

    def load(self, path: str | Path) -> None:
        self.policy.load_shared(path)


__all__ = ["ParameterGroup", "SharedTaskPolicy", "TaskPolicy"]
