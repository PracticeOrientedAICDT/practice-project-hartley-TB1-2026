"""Measured software reference latency.

This describes *actual code on actual hardware*, so unlike the hypothetical
accelerator model it is legitimately computable. `PLAN.md` section 11.4 lists
measurement as the preferred option for the Slow execution condition, with the
alternative being a documented parameterised reference model.

Measurement protocol, so the number is reproducible rather than a machine
artefact (`CODE_PLAN.md` risk: "CPU reference timing varies run-to-run"):

1. Warm up with discarded iterations, so first-call allocation and lazy
   initialisation are not counted.
2. Time a batch of repeated inferences and divide, rather than timing one
   call, because a single call is dominated by timer and dispatch overhead.
3. Report the median of several batches, which is robust to scheduler noise
   from the parallel runner sharing the machine.

The reported value is a **property of this implementation on this machine**,
not of the hardware hypothesis, and is labelled as such wherever used.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from latency_ml.hardware.mac_counter import Architecture


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    """A measured reference latency, with the protocol that produced it."""

    median_ns: float
    min_ns: float
    max_ns: float
    n_batches: int
    batch_size: int
    macs: int

    def __str__(self) -> str:
        return (
            f"{self.median_ns:.0f} ns "
            f"(min {self.min_ns:.0f}, max {self.max_ns:.0f}, "
            f"{self.n_batches} batches x {self.batch_size}, {self.macs} MACs)"
        )

    def as_dict(self) -> dict[str, float | int]:
        """Manifest-friendly representation."""
        return {
            "median_ns": self.median_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "n_batches": self.n_batches,
            "batch_size": self.batch_size,
            "macs": self.macs,
        }


def measure_inference_latency(
    module: torch.nn.Module,
    arch: Architecture,
    *,
    batch_size: int = 256,
    warmup: int = 20,
    n_batches: int = 7,
) -> LatencyMeasurement:
    """Measure single-sample inference latency for a module.

    Args:
        module: The module to time. Must already be in eval mode; this does
            not change its mode.
        arch: Architecture the module implements, used to report the MAC count
            alongside the timing.
        batch_size: Inferences per timed batch. Amortises timer and dispatch
            overhead across many calls.
        warmup: Discarded batches before timing, to exclude lazy
            initialisation.
        n_batches: Timed batches. The median is reported.

    Returns:
        The measurement, in nanoseconds per single inference.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1, got {n_batches}")

    sample = torch.zeros(arch.widths[0], dtype=torch.float32)
    batch = sample.unsqueeze(0).expand(batch_size, -1).contiguous()

    def run() -> None:
        with torch.no_grad():
            module(batch)

    for _ in range(warmup):
        run()

    per_sample: list[float] = []
    for _ in range(n_batches):
        start = time.perf_counter_ns()
        run()
        elapsed = time.perf_counter_ns() - start
        per_sample.append(elapsed / batch_size)

    return LatencyMeasurement(
        median_ns=statistics.median(per_sample),
        min_ns=min(per_sample),
        max_ns=max(per_sample),
        n_batches=n_batches,
        batch_size=batch_size,
        macs=arch.macs(),
    )


def measure_callable(
    fn: Callable[[], object],
    *,
    batch_size: int = 256,
    warmup: int = 20,
    n_batches: int = 7,
) -> LatencyMeasurement:
    """Measure an arbitrary zero-argument callable's latency.

    For non-MLP execution paths where there is no architecture to report.
    """
    for _ in range(warmup):
        fn()
    per_sample: list[float] = []
    for _ in range(n_batches):
        start = time.perf_counter_ns()
        fn()
        per_sample.append((time.perf_counter_ns() - start) / batch_size)
    return LatencyMeasurement(
        median_ns=statistics.median(per_sample),
        min_ns=min(per_sample),
        max_ns=max(per_sample),
        n_batches=n_batches,
        batch_size=batch_size,
        macs=0,
    )
