"""Measured CPU inference latency for the real HFT policy implementations.

This is the *Slow* axis of the 2x2 model-capacity x execution-speed experiment.

The distinction that matters here: a Slow latency in this project is a **measured
quantity**, not a parameter. It is the wall-clock time the actual deployed module
takes to turn one observation into one action on this machine, including the
framework dispatch overhead that a pure MAC count would omit. A number derived
from FLOPs would be an invented claim about a system that was never run.

Consequently this module refuses to produce a value it cannot support:

* the model is built through the same :func:`build_hft_policy` path used
  everywhere else, so what is timed is the thing that will actually be used;
* the value head is excluded, because inference benchmarks the actor path alone;
* every reported latency carries the machine, thread count, dtype and warm-up
  policy it was measured under, so a number cannot be quoted without its context;
* no cross-platform extrapolation is performed.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from latency_ml.finance.features import FEATURE_NAMES
from latency_ml.models.architectures import build_hft_policy

#: Capacity labels forming the rows of the 2x2, per section 11.5.
CAPACITY_LABELS: tuple[str, str] = ("small", "large")


@dataclass(frozen=True)
class LatencyMeasurement:
    """A measured single-observation inference latency and its conditions.

    Attributes:
        model: Capacity label, ``"small"`` (2x32) or ``"large"`` (2x128).
        median_ns: Median end-to-end latency per observation. The headline value.
        p10_ns / p90_ns: Latency spread across repetitions.
        n_repeats: Repetitions behind the median.
        device: Torch device the measurement ran on.
        dtype: Parameter dtype.
        threads: Intra-op thread count in force during measurement.
        machine: Platform description, so the number is never context-free.
    """

    model: str
    median_ns: float
    p10_ns: float
    p90_ns: float
    n_repeats: int
    device: str
    dtype: str
    threads: int
    machine: str = field(default_factory=platform.machine)

    @property
    def hidden_width(self) -> int:
        return {"small": 32, "large": 128}[self.model]

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form, conditions included."""
        return {
            "model": self.model,
            "hidden_width": self.hidden_width,
            "median_ns": self.median_ns,
            "p10_ns": self.p10_ns,
            "p90_ns": self.p90_ns,
            "n_repeats": self.n_repeats,
            "device": self.device,
            "dtype": self.dtype,
            "threads": self.threads,
            "machine": self.machine,
        }


def measure_inference_latency(
    model: str,
    n_features: int = len(FEATURE_NAMES),
    n_actions: int = 3,
    n_repeats: int = 2_000,
    warmup: int = 200,
    device: str = "cpu",
) -> LatencyMeasurement:
    """Time the real policy forward pass, per observation.

    Times one observation at a time rather than a batch, because the quantity of
    interest is the latency of a single decision. Batching would measure
    throughput and silently answer a different question -- the same
    compute/throughput conflation that invalidates hardware claims elsewhere in
    this project.

    Args:
        model: ``"small"`` or ``"large"``.
        n_features: Observation dimension.
        n_actions: Output action count.
        n_repeats: Timed repetitions. The median is reported, so a few scheduling
            outliers cannot set the headline.
        warmup: Untimed iterations first, so lazy kernel selection and allocator
            growth are not charged to the measurement.
        device: Torch device. ``"cpu"`` is the Slow condition.

    Returns:
        The measured latency and the conditions it was measured under.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    policy = build_hft_policy(model, n_features=n_features, n_actions=n_actions).to(device)
    # Inference benchmarks the actor path alone; the critic head is a training
    # artifact and including it would inflate latency for both models.
    policy.critic = None
    policy.eval()

    observation = torch.zeros(1, n_features, dtype=torch.float32, device=device)

    threads = int(torch.get_num_threads())
    dtype = str(next(policy.parameters()).dtype).replace("torch.", "")

    with torch.inference_mode():
        for _ in range(warmup):
            policy(observation)
        samples = np.empty(n_repeats, dtype=np.float64)
        for i in range(n_repeats):
            start = time.perf_counter_ns()
            policy(observation)
            samples[i] = time.perf_counter_ns() - start

    return LatencyMeasurement(
        model=model,
        median_ns=float(np.median(samples)),
        p10_ns=float(np.percentile(samples, 10)),
        p90_ns=float(np.percentile(samples, 90)),
        n_repeats=n_repeats,
        device=device,
        dtype=dtype,
        threads=threads,
    )
