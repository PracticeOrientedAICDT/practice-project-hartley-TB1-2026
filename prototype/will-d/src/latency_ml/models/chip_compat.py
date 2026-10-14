"""The hypothetical accelerator's architectural limits and validation.

`PLAN.md` section 3.1 states the contract as "4 layers or 256 neurons", and
section 3.1 then reads it as:

    At most four fully connected layers, with no layer wider than 256 neurons.

The spec itself flags that this interpretation "should be explicitly confirmed
by the supervisor" (section 30, question 1), so it is implemented as the
documented default rather than as a settled fact, and it is a single constant
pair that is trivial to change if the supervisor reads the limit differently.

Validation exists to catch an architecture that silently exceeds the contract
*before* a sweep runs, rather than producing results attributed to a model the
chip could not hold.
"""

from __future__ import annotations

from dataclasses import dataclass

from latency_ml.hardware.mac_counter import Architecture


class IncompatibleArchitectureError(ValueError):
    """Raised when a model exceeds the accelerator's stated limits."""


@dataclass(frozen=True, slots=True)
class ChipConstraints:
    """The architectural contract from `PLAN.md` section 3.1.

    Attributes:
        max_layers: Maximum fully connected layers.
        max_width: Maximum neurons in any layer.
        supports_recurrent: Phase 1-4 exclude recurrent state (section 3.1).
        assumes_weight_residency: Section 3.1 assumes supported networks keep
            weights on-chip, which is why the latency model has no separate
            memory term.
    """

    max_layers: int = 4
    max_width: int = 256
    supports_recurrent: bool = False
    assumes_weight_residency: bool = True

    def __post_init__(self) -> None:
        if self.max_layers < 1:
            raise ValueError(f"max_layers must be >= 1, got {self.max_layers}")
        if self.max_width < 1:
            raise ValueError(f"max_width must be >= 1, got {self.max_width}")


#: The contract as documented, pending supervisor confirmation.
DEFAULT_CONSTRAINTS: ChipConstraints = ChipConstraints()


def validate(arch: Architecture, constraints: ChipConstraints = DEFAULT_CONSTRAINTS) -> None:
    """Check an architecture against the accelerator contract.

    Counts **weight matrices**, not layers including the input projection, so a
    4-128-128-2 policy has 3 dense layers and fits comfortably. This matches
    `PLAN.md` section 6.2's reasoning that the network is "comfortably inside"
    the 4-layer / 256-width limit.

    Raises:
        IncompatibleArchitectureError: If the architecture exceeds a limit.
    """
    if arch.num_layers > constraints.max_layers:
        raise IncompatibleArchitectureError(
            f"architecture {arch.describe()} has {arch.num_layers} dense layers, "
            f"exceeding the accelerator limit of {constraints.max_layers}. "
            "Layers are counted as weight matrices, so a width tuple of n "
            "entries gives n-1 dense layers."
        )
    if arch.max_width > constraints.max_width:
        raise IncompatibleArchitectureError(
            f"architecture {arch.describe()} has a layer of width "
            f"{arch.max_width}, exceeding the accelerator limit of "
            f"{constraints.max_width}."
        )


def is_compatible(arch: Architecture, constraints: ChipConstraints = DEFAULT_CONSTRAINTS) -> bool:
    """Whether an architecture fits the contract, without raising."""
    try:
        validate(arch, constraints)
    except IncompatibleArchitectureError:
        return False
    return True


def validate_many(
    archs: list[Architecture], constraints: ChipConstraints = DEFAULT_CONSTRAINTS
) -> None:
    """Validate several architectures, reporting all violations.

    Raises:
        IncompatibleArchitectureError: If any architecture is incompatible.
    """
    violations = [
        f"{a.describe()}: layers={a.num_layers}, max_width={a.max_width}"
        for a in archs
        if not is_compatible(a, constraints)
    ]
    if violations:
        raise IncompatibleArchitectureError(
            "architectures exceed the accelerator contract ("
            f"max_layers={constraints.max_layers}, max_width={constraints.max_width}): "
            + "; ".join(violations)
        )


def headroom(
    arch: Architecture, constraints: ChipConstraints = DEFAULT_CONSTRAINTS
) -> dict[str, int]:
    """Remaining capacity before the contract is exceeded.

    Useful when choosing a model size, and for reporting how much of the chip a
    given policy consumes.
    """
    return {
        "layers_remaining": constraints.max_layers - arch.num_layers,
        "width_remaining": constraints.max_width - arch.max_width,
    }
