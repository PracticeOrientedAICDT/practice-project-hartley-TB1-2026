"""Policy architectures.

Three architectures are defined by the specification and are reused verbatim
across phases:

- **Control** (`PLAN.md` section 6.2): ``4 -> 128 -> 128 -> 2``, shared by
  CartPole and LunarLander (section 9.2) so Phase 2 tests the *same* chip under
  a richer task rather than searching for a better architecture.
- **HFT Small** (section 11.5): hidden widths 32.
- **HFT Large** (section 11.5): hidden widths 128.

Each policy has a matching critic sharing the hidden trunk, with a scalar value
head. Per section 6.2, inference benchmarking measures the **policy path only**,
since that is the portion needed to select an environment action -- the critic
is a training artefact and must not be counted toward accelerator load.

The trunk width is a chip constraint, not a hyperparameter search result. Phase 1
and 2 deliberately defer architecture search so latency is the only independent
variable (section 3.3).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from latency_ml.hardware.mac_counter import Architecture
from latency_ml.models.activations import bounded_relu
from latency_ml.models.chip_compat import DEFAULT_CONSTRAINTS, ChipConstraints, validate

if TYPE_CHECKING:
    from latency_ml.rl.task_policy import ParameterGroup


CONTROL_HIDDEN: tuple[int, ...] = (128, 128)


HFT_SMALL_HIDDEN: tuple[int, ...] = (32, 32)
HFT_LARGE_HIDDEN: tuple[int, ...] = (128, 128)


class MLPPolicy(nn.Module):
    """Discrete-action actor, optionally with a value head.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        hidden: Hidden layer widths.
        activation: Nonlinearity. Defaults to
            :func:`~latency_ml.models.activations.bounded_relu`. Accepting a
            module rather than a name is deliberate: this repository has one
            activation, and a string would imply a menu of supported options
            that does not exist. The override exists for the parity suite, which
            must reproduce a reference checkpoint trained under an unbounded
            ReLU. Production code leaves it unset.
        use_value_head: Whether to attach the critic head. Always ``True`` for
            training; inference benchmarking builds the policy path alone.

    Attributes:
        arch: The :class:`Architecture` this module implements, used for MAC
            counting and chip-compatibility checks.
        activation_name: Class name of the nonlinearity actually constructed.
            Recorded in manifests so a result identifies its own model rather
            than relying on a config field that could disagree with the code.
    """

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
        """One group holding every parameter.

        Registration order is preserved deliberately. Gradient clipping sums over
        the parameters it is handed, so a different order is a different sum and
        therefore a different clipped norm at the last bit -- which would break
        bit-for-bit reproduction of an earlier run.

        Imported lazily: this module is imported by everything, and the protocol
        lives with the trainer that consumes it.
        """
        from latency_ml.rl.task_policy import ParameterGroup

        return [ParameterGroup("policy", list(self.parameters()), stepped_on_every_task=True)]

    @property
    def actor_arch(self) -> Architecture:
        """Architecture of the inference path only.

        Section 6.2: benchmarking measures the policy path, because that is
        what selects the environment action. The critic exists only during
        training and must not be counted as accelerator load.
        """
        return self.arch

    def to_device(self, device: torch.device | str) -> None:
        """Move the policy's tensors to ``device``; see ``TaskPolicy``."""
        self.to(device)

    def set_mode(self, training: bool) -> None:
        """Put the policy in training or evaluation mode; see ``TaskPolicy``."""
        self.train(training)

    def is_training(self) -> bool:
        """Whether the policy is in training mode; see ``TaskPolicy``."""
        return self.training

    def save(self, path: str | Path) -> None:
        """Write the policy's state dict to ``path``, creating parent directories.

        Only the policy, not the optimiser. Optimiser state would let training
        resume mid-schedule, but evaluation and the latency sweep only need the
        weights, and saving it would make a checkpoint larger for no benefit to
        the experiment.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    def load(self, path: str | Path) -> None:
        """Read weights from a checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        self.load_state_dict(torch.load(Path(path), weights_only=True))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(action_logits, value)``.

        Args:
            obs: ``(batch, obs_dim)`` observations.

        Returns:
            Action logits of shape ``(batch, n_actions)``, and the value
            estimate when a head is present.
        """
        logits = self.actor(obs)
        value = self.critic(obs).flatten() if self.critic is not None else None
        return logits, value

    def action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(actions, log_probs, values)`` for a rollout.

        Log-probabilities come from ``log_softmax`` and actions are drawn with
        ``torch.multinomial`` on CPU, rather than via
        ``torch.distributions.Categorical``.

        Two reasons. ``docs/ENVIRONMENT.md`` records that the MPS
        implementation of that function can return a zero-probability index on
        torch 2.14.0, which would silently corrupt actions on exactly this code
        path. And ``torch.distributions`` ships partial type information, so
        calling it makes correct code fail ``mypy --strict``.
        """
        logits, value = self.forward(obs)
        assert value is not None, "action_and_value requires a value head"
        log_probs_all = functional.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs_all.exp(), num_samples=1).squeeze(-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, log_probs, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions, for a PPO update.

        Returns:
            ``(log_probs, entropy, values, logits)``.
        """
        logits, value = self.forward(obs)
        assert value is not None
        log_probs_all = functional.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
        return log_probs, entropy, value, logits

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic greedy action, for evaluation.

        Evaluation uses the argmax rather than a sample so that paired
        evaluation seeds give reproducible results (section 19).
        """
        logits, _ = self.forward(obs)
        return logits.argmax(dim=-1)


def build_control_policy(
    task: str,
    constraints: ChipConstraints = DEFAULT_CONSTRAINTS,
    activation: nn.Module | None = None,
) -> MLPPolicy:
    """Build the control policy for a task.

    CartPole and LunarLander share the same hidden trunk (section 9.2); only
    input and output dimensions differ.

    Args:
        task: Environment name.
        constraints: The accelerator contract.
        activation: Nonlinearity override. See :class:`MLPPolicy` -- production
            code leaves this unset and the parity suite sets it to reproduce a
            reference checkpoint trained under an unbounded ReLU.

    Raises:
        IncompatibleArchitectureError: If the result exceeds the contract.
    """
    from latency_ml.envs.factory import probe_env_dims

    obs_dim, n_actions = probe_env_dims(task)
    policy = MLPPolicy(obs_dim, n_actions, hidden=CONTROL_HIDDEN, activation=activation)
    validate(policy.actor_arch, constraints)
    return policy


def build_hft_policy(
    model: str,
    n_features: int,
    n_actions: int = 3,
    constraints: ChipConstraints = DEFAULT_CONSTRAINTS,
    activation: nn.Module | None = None,
) -> MLPPolicy:
    """Build an HFT policy by capacity label.

    Args:
        model: ``"small"`` for 2x32 or ``"large"`` for 2x128 (section 11.5).
        n_features: LOB feature dimension.
        n_actions: Output actions; 3 for HOLD/BUY/SELL.
        constraints: The accelerator contract.

    Returns:
        A validated policy.

    Raises:
        ValueError: If the model label is unknown.
        IncompatibleArchitectureError: If the result exceeds the contract.
    """
    match model.lower():
        case "small":
            hidden = HFT_SMALL_HIDDEN
        case "large":
            hidden = HFT_LARGE_HIDDEN
        case _:
            raise ValueError(f"unknown HFT model {model!r}; expected 'small' or 'large'")

    policy = MLPPolicy(
        n_features, n_actions, hidden=hidden, use_value_head=False, activation=activation
    )
    validate(policy.actor_arch, constraints)
    return policy


def build_shared_trunk(
    constraints: ChipConstraints = DEFAULT_CONSTRAINTS,
) -> MLPPolicy:
    """Build the Phase 4 shared trunk.

    A single ``128 -> 128 -> 128`` trunk physically implemented by the
    accelerator, with per-task input adapters and output heads outside it
    (section 15.2). Phase 4 measures whether that reuse is worthwhile.

    Raises:
        IncompatibleArchitectureError: If the trunk exceeds the contract.
    """
    trunk = MLPPolicy(128, 128, hidden=(128, 128), use_value_head=False)
    validate(trunk.actor_arch, constraints)
    return trunk


def describe_architecture(policy: MLPPolicy) -> str:
    """Short human-readable architecture summary.

    Includes the activation actually constructed, not the one a config file
    claims. A manifest carrying this string therefore identifies the model that
    produced a result even if the two could in principle disagree.
    """
    arch = policy.actor_arch
    return (
        f"{arch.describe()} | {arch.num_layers} dense layers | "
        f"max width {arch.max_width} | {arch.macs()} MACs | "
        f"activation {policy.activation_name} | "
        f"{arch.parameters()} params"
    )
