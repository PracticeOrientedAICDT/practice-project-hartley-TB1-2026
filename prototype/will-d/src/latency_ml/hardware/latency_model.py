"""The parameterised accelerator latency model.

`PLAN.md` section 3.2 defines

    T_inference = T_compute(N_MAC, P_chip) + T_overhead + T_memory

where ``P_chip`` is peak MAC throughput, ``T_overhead`` a fixed pipeline/setup
cost, and ``T_memory`` an optional weight-movement cost.

Those three parameters are **unresolved** (section 30, questions 2, 3 and 10).
This module therefore refuses to produce a latency for the hypothetical
accelerator until they are decided, and raises
:class:`~latency_ml.config.HardwareModelNotFrozen` instead.

That refusal is the point. `PLAN.md` section 3.2 says explicitly: do *not*
invent a single latency number and present it as a property of the chip. A
plausible-looking default would be worse than no number, because it would flow
into results and be indistinguishable from a measured one.

The measured software reference latency is a separate matter: it describes
actual code on actual hardware, so it is legitimately computable and lives in
:mod:`latency_ml.hardware.reference`.
"""

from __future__ import annotations

import math

from latency_ml.config import HardwareModelNotFrozen, HardwareSpec, Unresolved


def _require(value: int | Unresolved, name: str) -> int:
    """Return an integer parameter, raising if unresolved."""
    if isinstance(value, Unresolved):
        raise HardwareModelNotFrozen(
            f"accelerator parameter {name!r} is unresolved "
            f"({value}). PLAN.md section 3.2 forbids inventing hardware "
            "figures; this requires a supervisor decision "
            "(PLAN.md section 30, Q2/Q3/Q10)."
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def compute_time_ns(n_macs: int, peak_macs_per_second: int | Unresolved) -> int:
    """MAC compute time in nanoseconds.

    Args:
        n_macs: Multiply-accumulate count for the operation.
        peak_macs_per_second: The chip's peak throughput.

    Returns:
        Ceil-rounded nanoseconds. Rounds up so the model never under-reports.

    Raises:
        HardwareModelNotFrozen: If throughput is unresolved.
        ValueError: If ``n_macs`` is negative.
    """
    if n_macs < 0:
        raise ValueError(f"n_macs must be non-negative, got {n_macs}")
    throughput = _require(peak_macs_per_second, "peak_macs_per_second")

    return math.ceil(n_macs * 1_000_000_000 / throughput)


def predict_latency_ns(spec: HardwareSpec, n_macs: int) -> int:
    """Predicted accelerator inference latency for a MAC count.

    Applies ``T_compute + T_overhead``. ``T_memory`` is zero because
    `PLAN.md` section 3.1 assumes on-chip weight residency for supported
    networks, so no separate weight-movement term is needed while that
    assumption holds.

    Args:
        spec: The accelerator contract. Must be fully frozen.
        n_macs: Multiply-accumulate count.

    Returns:
        Total predicted latency in nanoseconds.

    Raises:
        HardwareModelNotFrozen: If any required parameter is unresolved. This
            is the expected outcome while the accelerator contract is
            unfrozen, and the reason Phase 3's Fast cells render as
            ``PENDING_HARDWARE_MODEL``.
    """
    spec.require_frozen()
    assert not isinstance(spec.peak_macs_per_second, Unresolved)
    assert not isinstance(spec.fixed_overhead_ns, Unresolved)

    return compute_time_ns(n_macs, spec.peak_macs_per_second) + spec.fixed_overhead_ns


def is_available(spec: HardwareSpec) -> bool:
    """Whether a hardware-derived latency can be produced at all.

    Lets a sweep enumerate Fast conditions and mark them pending, rather than
    crashing halfway through.
    """
    return spec.is_frozen


def explain_pending(spec: HardwareSpec) -> str:
    """Human-readable reason a hardware latency is unavailable."""
    missing = spec.unresolved_parameters()
    if not missing:
        return "hardware model is frozen"
    return (
        "PENDING_HARDWARE_MODEL: unresolved accelerator parameters: "
        + ", ".join(missing)
        + ". See PLAN.md section 30, questions 2, 3 and 10."
    )


def latency_range_ns(spec: HardwareSpec, min_macs: int, max_macs: int) -> tuple[int, int]:
    """Predicted latency bounds across a MAC range, for plotting.

    Args:
        spec: Frozen accelerator contract.
        min_macs: Lower MAC bound.
        max_macs: Upper MAC bound.

    Raises:
        HardwareModelNotFrozen: If the spec is unresolved.
    """
    return predict_latency_ns(spec, min_macs), predict_latency_ns(spec, max_macs)
