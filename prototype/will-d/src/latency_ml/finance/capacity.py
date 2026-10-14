"""The 2x2 model-capacity x execution-speed experiment.

Two capacities (2x32, 2x128) crossed with two execution conditions (Slow, Fast).

**The Slow axis is measured.** Each Slow latency is the wall-clock time the real
policy takes on this machine, measured by
:mod:`latency_ml.finance.latency_measure`. It is not a MAC-count estimate.

**The Fast axis is deliberately absent.** The Hartley accelerator has no frozen
throughput, clock, memory or precision. PLAN.md section 812 is explicit: do not
invent a numerical Hartley latency until those are fixed. So the Fast cells are
emitted as ``PENDING_HARDWARE_MODEL`` and carry no number.

This is the honest shape of the result. A 2x2 with two invented cells would look
complete and would be worthless, because a Fast latency chosen after seeing the
economic curve is a free parameter with no falsifiable content: any monotonic
result could be obtained by picking the right number. The structurally-complete
2x2 with numerically-pending Fast cells is what the plan anticipated, and it is a
closed result rather than an unfinished one.

What the measured Slow cells *can* show, legitimately, is that inference latency
is not uniform across capacities: a 2x128 forward pass on this machine is slower
than a 2x32 one. Capacity has a cost even before hardware is considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from latency_ml.finance.latency_measure import (
    LatencyMeasurement,
    measure_inference_latency,
)

#: Placeholder for a cell whose execution latency is not yet defined.
PENDING_HARDWARE_MODEL = "PENDING_HARDWARE_MODEL"

Execution = Literal["slow", "fast"]
CellStatus = Literal["measured", "pending"]


@dataclass(frozen=True)
class ExperimentCell:
    """One cell of the 2x2.

    Attributes:
        model: Capacity label.
        execution: Execution condition.
        latency_ns: Measured execution latency, or ``None`` when pending.
        status: ``"measured"`` or ``"pending"``.
        note: Why a cell is pending, recorded so it cannot be mistaken for a zero
            or silently filled in later.
    """

    model: str
    execution: Execution
    latency_ns: float | None
    status: CellStatus
    note: str = ""

    @property
    def is_measured(self) -> bool:
        return self.status == "measured" and self.latency_ns is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "execution": self.execution,
            "latency_ns": self.latency_ns,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class Experiment24:
    """The full 2x2, with provenance for every cell.

    Attributes:
        cells: All four cells, keyed by ``(model, execution)``.
        measurements: The Slow latency measurements behind the measured cells.
    """

    cells: dict[tuple[str, Execution], ExperimentCell] = field(default_factory=dict)
    measurements: dict[str, LatencyMeasurement] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "cells": [c.as_dict() for c in self.cells.values()],
            "measurements": {k: v.as_dict() for k, v in self.measurements.items()},
            "measured_cells": sum(1 for c in self.cells.values() if c.is_measured),
            "total_cells": len(self.cells),
            "fast_cells_pending": all(
                not self.cells[(m, "fast")].is_measured for m in ("small", "large")
            ),
        }


def run_24(n_repeats: int = 2_000, warmup: int = 200) -> Experiment24:
    """Build the 2x2, measuring the Slow axis and leaving Fast pending.

    Args:
        n_repeats: Repetitions per latency measurement.
        warmup: Untimed iterations before timing starts.

    Returns:
        The populated experiment. Fast cells carry ``latency_ns=None``.
    """
    experiment = Experiment24()
    models: tuple[str, ...] = ("small", "large")

    for model in models:
        measurement = measure_inference_latency(
            model, n_repeats=n_repeats, warmup=warmup, device="cpu"
        )
        experiment.measurements[model] = measurement
        experiment.cells[(model, "slow")] = ExperimentCell(
            model=model,
            execution="slow",
            latency_ns=measurement.median_ns,
            status="measured",
            note=(
                f"measured CPU median over {measurement.n_repeats} reps, "
                f"{measurement.threads} threads, {measurement.dtype}"
            ),
        )
        experiment.cells[(model, "fast")] = ExperimentCell(
            model=model,
            execution="fast",
            latency_ns=None,
            status="pending",
            note=(
                "Hartley throughput, clock, memory and precision are not frozen; "
                "PLAN.md 812 forbids inventing a numerical Fast latency"
            ),
        )

    return experiment


def format_table(experiment: Experiment24) -> str:
    """Render the 2x2 as a markdown table with pending cells kept visibly blank."""
    lines = ["| capacity | Slow (measured CPU) | Fast (Hartley) |", "|---|---|---|"]

    def cell(model: str, execution: Execution) -> str:
        c = experiment.cells[(model, execution)]
        if c.latency_ns is not None:
            return f"{c.latency_ns / 1000:.2f} us"
        return PENDING_HARDWARE_MODEL

    for model, label in (("small", "2x32"), ("large", "2x128")):
        lines.append(f"| {label} | {cell(model, 'slow')} | {cell(model, 'fast')} |")
    return "\n".join(lines)


def slow_capacity_gap(experiment: Experiment24) -> float | None:
    """Relative Slow-latency penalty of 2x128 over 2x32, or ``None``.

    The one comparison the measured axis supports: how much slower the larger
    policy is on identical hardware. Derived from the experiment's own
    measurements so the reported ratio cannot drift from the table.
    """
    small = experiment.measurements.get("small")
    large = experiment.measurements.get("large")
    if small is None or large is None or small.median_ns <= 0:
        return None
    return (large.median_ns - small.median_ns) / small.median_ns


def measure_capacity_gap(n_repeats: int = 2_000, warmup: int = 200) -> Experiment24:
    """Run the 2x2. Alias of :func:`run_24`, named for the capacity-gap report."""
    return run_24(n_repeats=n_repeats, warmup=warmup)
