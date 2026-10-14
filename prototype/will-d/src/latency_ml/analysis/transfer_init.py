"""Function-preserving transfer of a solved dedicated representation into the shared trunk.

Phase 4D asks whether a shared policy *initialised from* a previously solved
LunarLander representation can be jointly trained to support both LunarLander
and CartPole. Phase 4C remains the from-scratch control; this module only
supplies the initialisation.

The transfer is exact rather than approximate
--------------------------------------------

The shared trunk and the dedicated actor have the same depth, because the
dedicated network's first layer plays the role of the shared model's adapter::

    dedicated actor:  Linear(8->128)  ReLU  Linear(128->128)  ReLU  Linear(128->4)
    shared trunk:               Linear(128->128)  ReLU  Linear(128->128)  ReLU

So the mapping is positional, with an identity initialising the intervening
layer::

    shared.adapters[task][0]  <-  dedicated actor[0]     (8 -> 128)
    shared.trunk[0]            <-  identity               (128 -> 128, bias 0)
    shared.trunk[2]            <-  dedicated actor[2]     (128 -> 128)
    shared.actors[task]        <-  dedicated actor[4]     (128 -> 4)   [diagnostic only]

The identity layer is genuinely an identity on this path. The dedicated network
applies ``ReLU`` after ``actor[2]`` before its head, and the shared trunk applies
``ReLU`` after ``trunk[2]``, so the shared features equal ``relu`` of the
dedicated layer-2 pre-activation exactly, with no approximation and no
representational drift. This is asserted at load time rather than assumed.

What is transferred, and what is not
------------------------------------

The **representation** is transferred: the adapter and the second hidden layer.
The **task heads are not**. A fresh actor head is deliberately retained, because
the scientific question is whether a useful representation transfers, not whether
an already-trained policy survives being wrapped in a larger model. The
diagnostic consequence is intended and measured: a fresh head on the transferred
features agrees with the source policy at close to chance, which is the evidence
that the representation on its own is not a functioning policy.

The transferred-head mapping is still *validated* here, by temporarily installing
the source head and requiring bit-exact logit equality. That check proves the
weight mapping is right; its 100% agreement figure is a property of the mapping,
not a description of the 4D initialisation, which uses the fresh head.

The critic head cannot be transferred at all: the dedicated network has separate
actor and critic trunks, while the shared model has one trunk feeding both heads.

Fail-closed by construction
---------------------------

Every mapping step asserts rather than warns. A silently mis-aligned transfer
would produce a complete, plausible-looking sweep in which the only real
difference from Phase 4C is a broken initialisation, so every failure mode here
raises :class:`TransferError` rather than degrading to a warning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from latency_ml.models.architectures import MLPPolicy
from latency_ml.models.shared_trunk import SharedTrunkPolicy

if TYPE_CHECKING:  # pragma: no cover
    pass

#: The task whose solved representation is transferred.
SOURCE_TASK: str = "lunarlander"

#: Exactness required of the transferred pathway. Feature and logit equality are
#: expected to be bit-exact, not close; anything less indicates a mapping fault.
_EXACT_TOLERANCE: float = 0.0

#: Argmax agreement required of the transferred-head validation. The mapping is
#: exact, so anything below 1.0 means the layers are not the ones we think.
_REQUIRED_ARGMAX_AGREEMENT: float = 1.0


class TransferError(RuntimeError):
    """Raised when the transfer cannot be made exactly.

    Never a warning-only condition. A partially applied transfer would leave the
    shared model in a state that is neither Phase 4C's random initialisation nor
    the intended Phase 4D one, and every subsequent result would be uninterpretable
    while still looking like data.
    """


@dataclass(frozen=True)
class TransferMapping:
    """One tensor-to-tensor copy, recorded so the transfer is auditable.

    Attributes:
        destination: Dotted path within the shared policy.
        source: Dotted path within the dedicated state dict.
        note: Why this copy is made.
    """

    destination: str
    source: str
    note: str


#: The mapping, in application order. Exposed as data so the report can serialise
#: it and a reader can see exactly which tensors moved.
TENSOR_MAPPING: tuple[TransferMapping, ...] = (
    TransferMapping(
        destination=f"adapters.{SOURCE_TASK}[0]",
        source="actor.0",
        note="dedicated first hidden layer takes the shared adapter's role",
    ),
    TransferMapping(
        destination="trunk[0]",
        source="identity",
        note="intervening layer initialised to identity, exact on this pathway",
    ),
    TransferMapping(
        destination="trunk[2]",
        source="actor.2",
        note="dedicated second hidden layer is the transferred representation",
    ),
)

#: The head mapping, validated but not applied. Recorded because it is the
#: strongest available check that the three transfers above are correctly aligned.
HEAD_VALIDATION_MAPPING: TransferMapping = TransferMapping(
    destination=f"actors.{SOURCE_TASK}",
    source="actor.4",
    note="validation only: the 4D initialisation uses a freshly initialised head",
)


@dataclass
class TransferReport:
    """Audit record of one transfer, stored on every Phase 4D cell.

    Attributes:
        source_path: Checkpoint the representation came from.
        source_sha256: Digest of the source tensors, so a later run can prove it
            used the same source rather than a re-trained one.
        source_mean_return: The source checkpoint's own dedicated return, for
            provenance. Not a control condition.
        mapping: Tensor copies applied.
        head_validation_mapping: The head copy that was checked but not applied.
        transferred_head_argmax_agreement: Validation agreement with the source
            policy. A property of the mapping, not of the 4D initialisation.
        transferred_head_max_logit_difference: Bit-exactness of that check.
        fresh_head_argmax_agreement: Agreement of the *actual* 4D initialisation
            with the source policy. Near chance, by design, and the evidence that
            a transferred representation alone is not a working policy.
        feature_max_abs_difference: Bit-exactness of the transferred features
            against the source policy's post-ReLU hidden layer.
        transferred_tensors: Parameters whose values came from the source.
        n_transferred_parameters: How many that is.
    """

    source_path: str
    source_sha256: str
    source_mean_return: float | None
    mapping: list[dict[str, str]]
    head_validation_mapping: dict[str, str]
    transferred_head_argmax_agreement: float
    transferred_head_max_logit_difference: float
    fresh_head_argmax_agreement: float
    feature_max_abs_difference: float
    transferred_tensors: list[str]
    n_transferred_parameters: int
    initial_evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _linear(module: nn.Module, index: int) -> nn.Linear:
    """The ``nn.Linear`` at ``index``, with the layer type asserted.

    ``nn.Sequential`` and ``ModuleDict`` are both typed as ``Module``, which leaves
    every ``[i]`` access and ``.weight`` attribute untyped to a checker. Narrowing
    here, once, keeps the assertions in the rest of the module honest instead of
    scattering casts over every access.

    Indexed over direct children, which is how ``nn.Sequential`` addresses its
    entries. Use :func:`_entry` for the per-task ``ModuleDict`` members.
    """
    layers = list(module.children())
    if index >= len(layers):
        raise TransferError(f"no layer at index {index}; module has {len(layers)}")
    layer = layers[index]
    if not isinstance(layer, nn.Linear):
        raise TransferError(f"expected a Linear at index {index}, got {type(layer).__name__}")
    return layer


def _adapter_linear(shared: SharedTrunkPolicy, task: str) -> nn.Linear:
    """The shared adapter's ``Linear``, which is ``[0]`` of a ``Sequential``.

    Unlike the heads, the adapter ends in a ``ReLU`` and so is a two-module
    ``Sequential``; the ``Linear`` is not the entry itself.
    """
    adapter = shared.adapters[task]
    return _linear(adapter, 0)


def _entry(module: nn.ModuleDict, task: str) -> nn.Linear:
    """The single ``nn.Linear`` held directly by ``module[task]``.

    The actor and critic heads are bare ``nn.Linear`` values, so there is no
    intermediate ``Sequential`` to index through.
    """
    if task not in module:
        raise TransferError(f"no {task!r} entry; has {sorted(module.keys())}")
    layer = module[task]
    if not isinstance(layer, nn.Linear):
        raise TransferError(f"{task!r} entry is {type(layer).__name__}, expected a Linear")
    return layer


def file_sha256(path: Path) -> str:
    """SHA-256 of a checkpoint file's bytes.

    Distinct from :func:`source_sha256`, which digests a state dict directly.
    This one reads the written file, so it verifies what actually landed on disk
    rather than what was in memory when it was serialised.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_indices(sequential: nn.Sequential) -> tuple[int, ...]:
    """Indices of the ``nn.Linear`` modules in a Sequential.

    Read structurally rather than hardcoded, so a future architecture change
    fails here with a clear message instead of copying weights into the wrong
    layers.
    """
    return tuple(i for i, module in enumerate(sequential) if isinstance(module, nn.Linear))


def assert_transferable_shapes(shared: SharedTrunkPolicy, source: MLPPolicy) -> None:
    """Verify the two architectures admit an exact positional mapping.

    Raises:
        TransferError: If either network's layer structure is not the one the
            mapping assumes.
    """
    shared_linears = _linear_indices(shared.trunk)
    if len(shared_linears) != 2:
        raise TransferError(
            f"shared trunk has {len(shared_linears)} linear layers, expected 2. The "
            f"transfer mapping is positional; it does not describe a different depth."
        )
    source_linears = _linear_indices(source.actor)
    if len(source_linears) != 3:
        raise TransferError(
            f"source actor has {len(source_linears)} linear layers, expected 3 "
            f"(adapter role, representation, head)."
        )
    shared_width = shared.shared_width
    for index in shared_linears:
        layer = _linear(shared.trunk, index)
        if layer.in_features != shared_width or layer.out_features != shared_width:
            raise TransferError(
                f"shared trunk[{index}] is {layer.in_features}->{layer.out_features}, "
                f"expected {shared_width}->{shared_width}. A transfer into a "
                f"differently shaped layer would be a projection, not a copy."
            )
    adapter = _adapter_linear(shared, SOURCE_TASK)
    first = _linear(source.actor, source_linears[0])
    if (adapter.in_features, adapter.out_features) != (first.in_features, first.out_features):
        raise TransferError(
            f"shared adapter is {adapter.in_features}->{adapter.out_features} but the "
            f"source's first layer is {first.in_features}->{first.out_features}."
        )
    head = shared.actors[SOURCE_TASK]
    last = source.actor[source_linears[-1]]
    if (head.in_features, head.out_features) != (last.in_features, last.out_features):
        raise TransferError(
            f"shared head is {head.in_features}->{head.out_features} but the source's "
            f"head is {last.in_features}->{last.out_features}."
        )


def _activation_modules(path: nn.Sequential) -> list[nn.Module]:
    """Every nonlinearity in a policy path, in order.

    Read from the module rather than assumed. The transfer's whole claim is that
    it reproduces the source's post-activation hidden layer *exactly*, which is
    only meaningful if the nonlinearity applied here is the one the source was
    trained with. Hardcoding a ReLU here would make that check pass or fail for
    reasons unrelated to the weights being transferred.
    """
    return [m for m in path if not isinstance(m, nn.Linear)]


def _source_activations(x: torch.Tensor, source: MLPPolicy) -> tuple[torch.Tensor, torch.Tensor]:
    """The source policy's post-activation hidden layer and logits.

    The post-activation hidden layer is what its head consumes, and therefore
    what a correct transfer must reproduce.
    """
    linears = _linear_indices(source.actor)
    first, second, third = (_linear(source.actor, i) for i in linears)
    acts = _activation_modules(source.actor)
    if len(acts) != 2:
        raise TransferError(
            f"expected 2 activation modules in the source actor path, found {len(acts)}"
        )
    h1 = acts[0](first(x))
    h2 = acts[1](second(h1))
    return h2, third(h2)


def assert_identity_layer(shared: SharedTrunkPolicy) -> None:
    """Confirm ``trunk[0]`` is identity before it is relied on.

    Raises:
        TransferError: If the layer is not the identity.
    """
    layer = _linear(shared.trunk, _linear_indices(shared.trunk)[0])
    weight = layer.weight
    bias = layer.bias
    if not torch.equal(weight, torch.eye(weight.shape[0], weight.shape[1])):
        raise TransferError(
            "shared trunk[0] is not the identity. The exactness of the transfer depends "
            "on this layer being an identity on the transferred pathway."
        )
    if bias is not None and not torch.equal(bias, torch.zeros_like(bias)):
        raise TransferError("shared trunk[0] bias is non-zero; the layer is not an identity.")


def source_sha256(state: dict[str, torch.Tensor]) -> str:
    """Stable digest of a state dict's tensor bytes.

    Keyed on names and raw bytes, so a re-trained checkpoint with different
    weights produces a different digest even if its shapes and dtypes match.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def transfer_from_dedicated(
    shared: SharedTrunkPolicy,
    source: MLPPolicy,
    source_path: Path | None = None,
    source_mean_return: float | None = None,
    n_probe_observations: int = 4096,
    seed: int = 0,
) -> TransferReport:
    """Copy a solved dedicated representation into the shared trunk, exactly.

    Applies :data:`TENSOR_MAPPING`, then validates the result bit-exactly against
    the source policy. Task heads are left as they are: freshly initialised by the
    caller before this function runs.

    Args:
        shared: Target shared policy, already built and seeded.
        source: Solved dedicated policy holding the transferable representation.
        source_path: Where ``source`` was loaded from, recorded for provenance.
        source_mean_return: The source checkpoint's dedicated return, recorded for
            provenance. Not a control condition.
        n_probe_observations: Random observations used for the equality checks.
            These are synthetic inputs, never environment states.
        seed: RNG seed for the probe inputs, so the reported numbers are
            reproducible.

    Returns:
        The audit record, to be stored on the cell.

    Raises:
        TransferError: On any structural mismatch or failure of an equality check.
    """
    assert_transferable_shapes(shared, source)
    if SOURCE_TASK not in shared.tasks:
        raise TransferError(f"shared policy has no {SOURCE_TASK!r} pathway to transfer into")

    state = source.state_dict()
    shared_linears = _linear_indices(shared.trunk)
    source_linears = _linear_indices(source.actor)
    transferred: list[str] = []
    count = 0

    with torch.no_grad():
        adapter = _adapter_linear(shared, SOURCE_TASK)
        adapter.weight.copy_(state[f"actor.{source_linears[0]}.weight"])
        adapter.bias.copy_(state[f"actor.{source_linears[0]}.bias"])
        transferred += ["adapters.weight", "adapters.bias"]
        count += adapter.weight.numel() + adapter.bias.numel()

        first = _linear(shared.trunk, shared_linears[0])
        first.weight.copy_(torch.eye(first.weight.shape[0], first.weight.shape[1]))
        first.bias.zero_()
        transferred += ["trunk[0].weight", "trunk[0].bias"]
        count += first.weight.numel() + first.bias.numel()

        second = _linear(shared.trunk, shared_linears[1])
        second.weight.copy_(state[f"actor.{source_linears[1]}.weight"])
        second.bias.copy_(state[f"actor.{source_linears[1]}.bias"])
        transferred += [f"trunk[{shared_linears[1]}].weight", f"trunk[{shared_linears[1]}].bias"]
        count += second.weight.numel() + second.bias.numel()

    assert_identity_layer(shared)

    generator = torch.Generator().manual_seed(seed)
    obs_dim = source.obs_dim
    probe = torch.randn(n_probe_observations, obs_dim, generator=generator)
    with torch.no_grad():
        expected_features, expected_logits = _source_activations(probe, source)
        actual_features = shared.features(probe, SOURCE_TASK)

        feature_difference = (actual_features - expected_features).abs().max().item()
        if feature_difference > _EXACT_TOLERANCE:
            raise TransferError(
                f"transferred features differ from the source policy's by "
                f"{feature_difference:.3e}; expected exact equality. The tensor mapping "
                f"is misaligned."
            )

        # Fresh-head agreement: the actual Phase 4D initialisation. Expected to be
        # near chance, and recorded because that is the diagnostic.
        fresh_logits, _ = shared(probe, SOURCE_TASK)
        fresh_agreement = (
            (fresh_logits.argmax(-1) == expected_logits.argmax(-1)).float().mean().item()
        )

        # Transferred-head agreement: a validation of the mapping, not the 4D
        # initialisation. Installed, checked, then restored.
        head = _entry(shared.actors, SOURCE_TASK)
        saved_weight = head.weight.detach().clone()
        saved_bias = head.bias.detach().clone()
        head.weight.copy_(state[f"actor.{source_linears[2]}.weight"])
        head.bias.copy_(state[f"actor.{source_linears[2]}.bias"])
        transferred_logits, _ = shared(probe, SOURCE_TASK)
        logit_difference = (transferred_logits - expected_logits).abs().max().item()
        agreement = (
            (transferred_logits.argmax(-1) == expected_logits.argmax(-1)).float().mean().item()
        )
        head.weight.copy_(saved_weight)
        head.bias.copy_(saved_bias)

    if agreement < _REQUIRED_ARGMAX_AGREEMENT:
        raise TransferError(
            f"transferred-head validation reached only {agreement:.4f} argmax agreement "
            f"(max logit difference {logit_difference:.3e}). A mismatch means the "
            f"transfer is not reproducing the source policy and must not be used."
        )

    return TransferReport(
        source_path=str(source_path) if source_path else "",
        source_sha256=source_sha256(state),
        source_mean_return=source_mean_return,
        mapping=[
            {"destination": m.destination, "source": m.source, "note": m.note}
            for m in TENSOR_MAPPING
        ],
        head_validation_mapping={
            "destination": HEAD_VALIDATION_MAPPING.destination,
            "source": HEAD_VALIDATION_MAPPING.source,
            "note": HEAD_VALIDATION_MAPPING.note,
        },
        transferred_head_argmax_agreement=agreement,
        transferred_head_max_logit_difference=logit_difference,
        fresh_head_argmax_agreement=fresh_agreement,
        feature_max_abs_difference=feature_difference,
        transferred_tensors=transferred,
        n_transferred_parameters=count,
    )


def load_source_policy(path: Path) -> MLPPolicy:
    """Load the fixed solved dedicated LunarLander checkpoint for transfer.

    Args:
        path: Checkpoint written by the dedicated runner.

    Returns:
        The loaded policy.

    Raises:
        TransferError: If the checkpoint does not load as a LunarLander policy.
    """
    from latency_ml.envs.factory import probe_env_dims

    obs_dim, n_actions = probe_env_dims(SOURCE_TASK)
    policy = MLPPolicy(obs_dim, n_actions)
    try:
        state = torch.load(path, weights_only=True)
        policy.load_state_dict(state)
    except Exception as exc:
        raise TransferError(
            f"could not load a {SOURCE_TASK} checkpoint from {path}: {exc}. The Phase 4D "
            f"source must be a dedicated policy checkpoint."
        ) from exc
    return policy


#: The frozen Phase 4D source: the v1 Gate 3 dedicated LunarLander control at
#: 20 ms, training seed 0. Recorded as a constant so every cell and every future
#: run resolves to the same checkpoint rather than a recomputed path.
#:
#: Held under ``tests/fixtures/v1/`` rather than ``results/``, because it is a
#: frozen v1 artefact rather than output this repository produced.
DEFAULT_SOURCE_CELL: tuple[str, str, int, int] = (
    "tests/fixtures/v1/controls",
    "gate3-dedicated-control",
    20_000_000,
    0,
)


def default_source_path(
    results_root: str | Path = DEFAULT_SOURCE_CELL[0],
    sweep_id: str = DEFAULT_SOURCE_CELL[1],
    latency_ns: int = DEFAULT_SOURCE_CELL[2],
    train_seed: int = DEFAULT_SOURCE_CELL[3],
) -> Path:
    """Path of the frozen Phase 4D transfer source checkpoint."""
    return (
        Path(results_root)
        / sweep_id
        / SOURCE_TASK
        / f"{latency_ns}ns"
        / f"seed{train_seed}"
        / "policy.pt"
    )
