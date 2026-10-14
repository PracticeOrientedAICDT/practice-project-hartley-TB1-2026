"""Exact multiply-accumulate counting for feed-forward MLPs.

Purely arithmetic: no hardware assumptions, no precision, no throughput. This
is why it is implemented separately from :mod:`latency_ml.hardware.latency_model`
-- MAC count is knowable while the accelerator contract is still unresolved.

Definitions used throughout:

- A **MAC** is one multiply-accumulate: ``out += w * x``.
- Each dense layer over a batch of ``B`` vectors performs
  ``B * in_features * out_features`` MACs, counting the accumulation in the
  same operation.
- Bias adds are counted separately as **bias adds**, not folded into the MAC
  total, so the two quantities remain interpretable.

`PLAN.md` section 3.2 defines

    T_inference = T_compute(N_MAC, P_chip) + T_overhead + T_memory

so ``N_MAC`` is the input to the hardware latency model. Computing it
correctly requires no knowledge of the chip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Architecture:
    """A feed-forward MLP described by its layer widths.

    Attributes:
        widths: Input dimension followed by each hidden and output width. So
            ``(4, 128, 128, 2)`` is the `PLAN.md` section 6.2 CartPole policy:
            4 observations, two 128-wide hidden layers, 2 discrete actions.
        batch_size: Batch the MAC count is computed for. ``1`` is the
            per-decision cost relevant to control latency.
        use_bias: Whether dense layers carry biases.
    """

    widths: tuple[int, ...]
    batch_size: int = 1
    use_bias: bool = True

    def __post_init__(self) -> None:
        if len(self.widths) < 2:
            raise ValueError(f"an MLP needs at least input and output widths, got {self.widths!r}")
        if any(w <= 0 for w in self.widths):
            raise ValueError(f"all widths must be positive, got {self.widths!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @property
    def num_layers(self) -> int:
        """Number of dense layers (weight matrices)."""
        return len(self.widths) - 1

    @property
    def max_width(self) -> int:
        """Widest layer, counting the input and output projections."""
        return max(self.widths)

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        """Hidden layer widths only."""
        return self.widths[1:-1]

    def layer_shapes(self) -> list[tuple[int, int]]:
        """``(in_features, out_features)`` per dense layer."""
        return list(zip(self.widths[:-1], self.widths[1:], strict=True))

    def macs_per_sample(self) -> int:
        """MACs for one forward pass through one sample."""
        return sum(fan_in * fan_out for fan_in, fan_out in self.layer_shapes())

    def macs(self) -> int:
        """Total MACs for the configured batch size."""
        return self.batch_size * self.macs_per_sample()

    def bias_adds(self) -> int:
        """Bias additions across the batch: one per output unit per sample."""
        if not self.use_bias:
            return 0
        return self.batch_size * sum(fan_out for _, fan_out in self.layer_shapes())

    def parameters(self) -> int:
        """Trainable parameter count, weights and biases."""
        total = sum(fan_in * fan_out for fan_in, fan_out in self.layer_shapes())
        if self.use_bias:
            total += sum(fan_out for _, fan_out in self.layer_shapes())
        return total

    def weight_count(self) -> int:
        """Weight count, excluding biases."""
        return sum(fan_in * fan_out for fan_in, fan_out in self.layer_shapes())

    def describe(self) -> str:
        """Compact human-readable form, e.g. ``4-128-128-2``."""
        return "-".join(str(w) for w in self.widths)


@dataclass(frozen=True, slots=True)
class MacBreakdown:
    """Per-layer MAC accounting.

    Retained rather than a bare integer so a capacity-vs-latency plot can show
    *where* the MACs are, which is what `PLAN.md` section 20 asks for when it
    asks for a model-MACs-versus-latency figure.
    """

    per_layer_macs: tuple[int, ...]
    batch_size: int

    @property
    def total(self) -> int:
        return sum(self.per_layer_macs)

    def cumulative_fraction(self, layer_index: int) -> float:
        """Fraction of MACs in layers up to and including ``layer_index``."""
        if not 0 <= layer_index < len(self.per_layer_macs):
            raise IndexError(f"layer_index out of range: {layer_index}")
        return sum(self.per_layer_macs[: layer_index + 1]) / self.total


def mac_breakdown(arch: Architecture) -> MacBreakdown:
    """Per-layer MAC counts for an architecture."""
    return MacBreakdown(
        per_layer_macs=tuple(
            arch.batch_size * fan_in * fan_out for fan_in, fan_out in arch.layer_shapes()
        ),
        batch_size=arch.batch_size,
    )


@dataclass(frozen=True, slots=True)
class ResourceAccounting:
    """Phase 4 accounting for a shared trunk plus adapters (`PLAN.md` section 15.4).

    Every field here is derivable without the hardware contract being frozen.
    Memory footprint in *bits* depends on precision, so it is reported per-bit
    and only converted to bytes when precision is known.
    """

    label: str
    trunk: Architecture
    adapters: dict[str, Architecture] = field(default_factory=dict)

    def trunk_macs(self) -> int:
        return self.trunk.macs()

    def trunk_parameters(self) -> int:
        return self.trunk.parameters()

    def adapter_macs(self) -> int:
        return sum(a.macs() for a in self.adapters.values())

    def adapter_parameters(self) -> int:
        return sum(a.parameters() for a in self.adapters.values())

    def total_parameters(self) -> int:
        return self.trunk_parameters() + self.adapter_parameters()

    def parameter_bits(self, bits_per_weight: int | None) -> int | None:
        """Total storage in bits, or ``None`` if precision is unresolved.

        Returning ``None`` rather than guessing is deliberate: `PLAN.md`
        section 15.4 forbids claiming physical silicon area, and by extension a
        byte figure must not appear from an undecided precision.
        """
        if bits_per_weight is None:
            return None
        return self.total_parameters() * bits_per_weight

    def workloads(self) -> int:
        """How many workloads the shared core serves."""
        return len(self.adapters)


BITS_PER_FORMAT: dict[str, int] = {
    "fp32": 32,
    "float32": 32,
    "fp16": 16,
    "float16": 16,
    "bf16": 16,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
}


def bits_per_weight(precision: str | None) -> int | None:
    """Bits per weight for a precision string, or ``None`` if unknown.

    Args:
        precision: Format name, e.g. ``"int8"``. ``None`` means unresolved.

    Returns:
        Bit width, or ``None`` when the format is unrecognised, so an
        unlisted precision never silently becomes a default.
    """
    if precision is None:
        return None
    return BITS_PER_FORMAT.get(precision.strip().lower())


def architectures(width_tuples: Sequence[Sequence[int]], batch_size: int = 1) -> list[Architecture]:
    """Build several architectures at once."""
    return [Architecture(tuple(w), batch_size=batch_size) for w in width_tuples]
