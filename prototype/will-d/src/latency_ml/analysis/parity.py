"""Task-level training protocol fingerprints, for shared/dedicated parity.

Phase 4 compares a shared trunk against dedicated controls. That comparison is
only interpretable if the two arms differ *solely* in representation topology.
Everything else -- discounting, the learning-rate schedule, PPO
hyperparameters, the integration-step budget, the evaluation protocol, the seed
mapping, the activation, the action-hold implementation -- must be identical.

Asserting that in prose is not enough, and asserting it only at runtime leaves no
artifact to audit afterwards. So each task's protocol is reduced to a
:class:`TaskProtocol` record and a short :meth:`~TaskProtocol.fingerprint` hash,
which is written into every manifest. The parity claim is then checkable from the
artifacts after the fact, which is the property that matters: a reader can
confirm the arms matched without trusting the code that produced them.

The fingerprint deliberately covers **discounting basis**, not just the numeric
factors. CartPole's 0.99/0.95 and LunarLander's 0.999/0.98 differ numerically and
also differ in *what they mean* -- per integration step versus per decision. Two
protocols that happened to share a gamma while disagreeing about its clock would
be a different experiment, and the fingerprint makes that difference visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from latency_ml.config import DiscountSpec, ProjectConfig
from latency_ml.models.activations import ACTIVATION_NAME

FINGERPRINT_VERSION = 2


@dataclass(frozen=True, slots=True)
class TaskProtocol:
    """The complete training and evaluation semantics for one task.

    Two runs with equal fingerprints differ only in how the network is wired,
    not in how the task is trained or scored.

    Attributes:
        fingerprint_version: Definition version of this record.
        task: Task name.
        discount_basis: ``physical`` or ``decision``.
        gamma: Discount factor, in the basis above.
        gae_lambda: GAE trace-decay, in the basis above.
        duration_aware: Whether the GAE recursion scales by action-hold length.
        learning_rate: Initial Adam learning rate.
        anneal_lr: Whether the rate is annealed at all.
        lr_schedule_basis: Clock the anneal is denominated in.
        num_envs: Parallel environments.
        steps_per_decision: Decisions per environment per rollout.
        num_minibatches: Minibatches per PPO update.
        update_epochs: PPO epochs per rollout.
        clip_coef: PPO clip range.
        ent_coef: Entropy bonus.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient-norm clip.
        norm_adv: Whether advantages are standardised.
        clip_vloss: Whether the value loss is clipped.
        effective_integration_steps: Budget the run consumes.
        control_episodes_per_seed: Evaluation episodes per seed.
        eval_seed_base: Base seed for the evaluation stream.
        paired_evaluation: Whether evaluation seeds are reused across cells.
        dt_env_ns: Environment integration timestep.
        seed_mapping: Identifier of the seed-derivation scheme.
        action_hold: Identifier of the action-hold implementation.
        activation: Class name of the nonlinearity actually constructed.

            Read from :mod:`latency_ml.models.activations` at fingerprint time
            rather than from a config field. A protocol is a statement about the
            model that will be trained, so it has to name the module the code
            builds; a config value could disagree with that and nothing else in
            the pipeline would notice.
        environment_kwargs: Effective environment kwargs, read from the task
            spec rather than from the config.

            Read from the task spec deliberately. Both trainers build
            environments through ``make_latency_env(task, latency_ns)`` without
            forwarding ``config.environment_kwargs``, so the config field is not
            what determines the environment -- the factory's task spec is.
            Recording the effective value means a change to the factory's
            defaults changes the fingerprint, which it otherwise would not.
    """

    fingerprint_version: int
    task: str
    discount_basis: str
    gamma: float
    gae_lambda: float
    duration_aware: bool
    learning_rate: float
    anneal_lr: bool
    lr_schedule_basis: str
    num_envs: int
    steps_per_decision: int
    num_minibatches: int
    update_epochs: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    norm_adv: bool
    clip_vloss: bool
    effective_integration_steps: int
    control_episodes_per_seed: int
    eval_seed_base: int
    paired_evaluation: bool
    dt_env_ns: int
    seed_mapping: str
    action_hold: str
    activation: str
    environment_kwargs: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Stable short hash of the whole record.

        Any difference in any field changes this, which is the point: there is no
        "close enough" protocol match.
        """
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Manifest-friendly representation, fingerprint included."""
        return {**asdict(self), "fingerprint": self.fingerprint}

    def differs_in(self, other: TaskProtocol) -> list[str]:
        """Field names on which two protocols disagree.

        Used to turn a parity failure into an actionable message rather than
        two opaque hashes.
        """
        mine, theirs = asdict(self), asdict(other)
        return sorted(k for k in mine if mine[k] != theirs.get(k))


SEED_MAPPING_ID = "derive_cell_seeds:train_seed_verbatim+latency_env_mix"
ACTION_HOLD_ID = "icdi_action_steps:semi_markov_duration_aware"


LR_SCHEDULE_BASIS = "simulated_time_fraction_of_effective_budget"


def task_protocol(
    config: ProjectConfig,
    task: str,
    effective_integration_steps: int | None = None,
) -> TaskProtocol:
    """Resolve the protocol a given task trains under.

    Args:
        config: The shared configuration. The same object is passed to the
            dedicated and shared trainers, so this is the single resolution
            point both arms must go through.
        task: Task name, selecting any per-task discount override.
        effective_integration_steps: The budget actually in force, after any
            override. Recorded rather than read from the config, because the
            configured and effective budgets are exactly what must not be
            conflated -- the LR-schedule bug came from conflating them.

    Returns:
        The resolved :class:`TaskProtocol`.
    """
    training = config.training
    discount: DiscountSpec = training.discount_for(task)
    return TaskProtocol(
        fingerprint_version=FINGERPRINT_VERSION,
        task=task,
        discount_basis=discount.basis.value,
        gamma=discount.gamma,
        gae_lambda=discount.gae_lambda,
        duration_aware=discount.duration_aware,
        learning_rate=training.learning_rate,
        anneal_lr=training.anneal_lr,
        lr_schedule_basis=LR_SCHEDULE_BASIS,
        num_envs=training.num_envs,
        steps_per_decision=training.steps_per_decision,
        num_minibatches=training.num_minibatches,
        update_epochs=training.update_epochs,
        clip_coef=training.clip_coef,
        ent_coef=training.ent_coef,
        vf_coef=training.vf_coef,
        max_grad_norm=training.max_grad_norm,
        norm_adv=training.norm_adv,
        clip_vloss=training.clip_vloss,
        effective_integration_steps=(
            effective_integration_steps or training.total_integration_steps
        ),
        control_episodes_per_seed=config.evaluation.control_episodes_per_seed,
        eval_seed_base=config.evaluation.eval_seed_base,
        paired_evaluation=config.evaluation.paired_evaluation,
        dt_env_ns=config.timing.dt_env_ns,
        seed_mapping=SEED_MAPPING_ID,
        action_hold=ACTION_HOLD_ID,
        activation=ACTIVATION_NAME,
        environment_kwargs=_effective_env_kwargs(task),
    )


def _effective_env_kwargs(task: str) -> dict[str, Any]:
    """Environment kwargs the factory will actually apply for this task."""
    from latency_ml.envs.factory import task_spec

    return dict(task_spec(task).get("kwargs") or {})


class ProtocolMismatch(RuntimeError):
    """Raised when two arms' task protocols are not identical.

    **Expected behaviour**, not a bug. Phase 4's shared-versus-dedicated
    comparison means nothing unless the arms match on everything except
    representation topology, so a mismatch must stop the run rather than
    produce a number that looks like a sharing effect.
    """


def assert_same_protocol(
    left: TaskProtocol,
    right: TaskProtocol,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> None:
    """Fail loudly unless two task protocols are identical.

    Raises:
        ProtocolMismatch: With the disagreeing field names, so the fix is
            evident rather than requiring a hash diff to diagnose.
    """
    if left.task != right.task:
        raise ProtocolMismatch(f"{left_label} is {left.task!r} but {right_label} is {right.task!r}")
    if left == right:
        return
    fields = left.differs_in(right)
    raise ProtocolMismatch(
        f"{left_label} and {right_label} protocols differ for task {left.task!r}: "
        f"{fields}\n  {left_label} fingerprint {left.fingerprint}\n"
        f"  {right_label} fingerprint {right.fingerprint}"
    )


__all__ = [
    "ACTION_HOLD_ID",
    "FINGERPRINT_VERSION",
    "LR_SCHEDULE_BASIS",
    "SEED_MAPPING_ID",
    "ProtocolMismatch",
    "TaskProtocol",
    "assert_same_protocol",
    "task_protocol",
]
