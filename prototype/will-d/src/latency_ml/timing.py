"""Latency grid, Inference-Constrained Decision Interval derivation, resolution.

Latency is represented as **integer nanoseconds** throughout. Conversion to
float seconds happens only at the Gymnasium boundary, so that the
sub-timestep comparison which decides whether a condition is resolvable
never depends on float rounding.

The central object is the ICDI (`PLAN.md` section 4): for an environment whose
actions can only change on environment ticks, a requested latency `tau` maps to
a number of held integration steps `N`, and the *effective* decision interval is
`N * dt_env`, which is not generally equal to `tau`.

The distinction between requested and effective is load-bearing. A 30 ms
request against a 20 ms simulator becomes a 40 ms effective interval, so 30 ms
and 40 ms are the *same measurement*. Reporting them as two independent
conditions would fabricate a data point.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

NATIVE_DT_ENV_NS: int = 20_000_000


HIGH_RES_DT_ENV_NS: int = 1_000_000


class Resolution(Enum):
    """How faithfully a requested latency is realised by the simulator.

    A two-state ``resolved`` flag is insufficient: it cannot distinguish a
    condition the simulator represents *exactly* from one it silently
    quantises upward. Those two must be handled differently when aggregating
    and plotting.
    """

    EXACT = "exact"

    QUANTIZED_UP = "quantized_up"

    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ICDI:
    """Inference-Constrained Decision Interval for one latency condition.

    Attributes:
        requested_ns: The nominal latency that was asked for.
        action_steps: ``N``, the number of integration steps the action is
            held for. Always at least 1.
        effective_ns: ``N * dt_env_ns``. The interval actually realised.
        resolution: How ``requested_ns`` relates to ``effective_ns``.
    """

    requested_ns: int
    action_steps: int
    effective_ns: int
    resolution: Resolution

    @property
    def is_independent_measurement(self) -> bool:
        """Whether this condition is a distinct, honestly-measured point.

        ``UNRESOLVED`` conditions collapse onto the sub-timestep behaviour of
        the environment. ``QUANTIZED_UP`` conditions are identical to whatever
        ``EXACT`` condition shares their ``effective_ns``.
        """
        return self.resolution is Resolution.EXACT


def derive_icdi(latency_ns: int, dt_env_ns: int = NATIVE_DT_ENV_NS) -> ICDI:
    """Derive the ICDI for a requested latency against an environment timestep.

    Args:
        latency_ns: Requested nominal latency in nanoseconds. Must be positive.
        dt_env_ns: Environment integration timestep in nanoseconds. Must be
            positive.

    Returns:
        The derived ICDI, including its :class:`Resolution` state.

    Raises:
        ValueError: If either argument is not strictly positive.

    Note:
        Sub-timestep requests are not an error. They resolve to a single held
        step with ``Resolution.UNRESOLVED`` so the condition can be carried
        through configuration and recorded, but they must be excluded from
        independent measurement and from latency-response plots.

    Example:
        >>> derive_icdi(30_000_000).effective_ns
        40000000
        >>> derive_icdi(30_000_000).resolution is Resolution.QUANTIZED_UP
        True
        >>> derive_icdi(100).resolution is Resolution.UNRESOLVED
        True
    """
    if latency_ns <= 0:
        raise ValueError(f"latency_ns must be positive, got {latency_ns}")
    if dt_env_ns <= 0:
        raise ValueError(f"dt_env_ns must be positive, got {dt_env_ns}")

    action_steps = -(-latency_ns // dt_env_ns)
    action_steps = max(1, action_steps)
    effective_ns = action_steps * dt_env_ns

    if latency_ns < dt_env_ns:
        resolution = Resolution.UNRESOLVED
    elif effective_ns != latency_ns:
        resolution = Resolution.QUANTIZED_UP
    else:
        resolution = Resolution.EXACT

    return ICDI(
        requested_ns=latency_ns,
        action_steps=action_steps,
        effective_ns=effective_ns,
        resolution=resolution,
    )


NATIVE_SWEEP_GRID_NS: tuple[int, ...] = (
    20_000_000,
    40_000_000,
    60_000_000,
    80_000_000,
    100_000_000,
)


LATENCY_GRID_NS: tuple[int, ...] = (
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    20_000_000,
    30_000_000,
    50_000_000,
    100_000_000,
)


def latency_grid() -> tuple[int, ...]:
    """The full target latency grid in nanoseconds."""
    return LATENCY_GRID_NS


def resolved_grid(dt_env_ns: int = NATIVE_DT_ENV_NS) -> tuple[int, ...]:
    """Entries of the **native sweep grid** the simulator represents exactly.

    Intended for building control sweeps. Unlike
    :func:`latency_grid`, this grid is constructed from integer multiples of
    the native timestep, so under the default 20 ms timestep every entry is
    exact and no condition collides with another.

    Args:
        dt_env_ns: Environment integration timestep in nanoseconds.

    Note:
        The section 5 target grid does *not* work here: at 20 ms it resolves
        only 20 ms and 100 ms exactly, since 30 ms and 50 ms quantize upward.
    """
    return tuple(
        latency_ns
        for latency_ns in NATIVE_SWEEP_GRID_NS
        if derive_icdi(latency_ns, dt_env_ns).is_independent_measurement
    )


def resolved_target_grid(dt_env_ns: int = NATIVE_DT_ENV_NS) -> tuple[int, ...]:
    """Entries of the **target grid** (section 5) the simulator represents exactly.

    At the native 20 ms timestep this returns only ``(20 ms, 100 ms)`` -- too
    few to characterise a latency-response curve. Use :func:`resolved_grid` for
    control sweeps.

    Args:
        dt_env_ns: Environment integration timestep in nanoseconds.
    """
    return tuple(
        icdi.requested_ns for icdi in iter_icdi(dt_env_ns) if icdi.is_independent_measurement
    )


def iter_icdi(dt_env_ns: int = NATIVE_DT_ENV_NS) -> Iterator[ICDI]:
    """Iterate the ICDI for every entry in the latency grid."""
    for latency_ns in LATENCY_GRID_NS:
        yield derive_icdi(latency_ns, dt_env_ns)


def collapse_by_effective(icdis: Sequence[ICDI]) -> list[ICDI]:
    """Collapse conditions that are identical measurements.

    Distinct requested latencies can realise the same effective interval: at a
    20 ms timestep, 30 ms and 40 ms are both 40 ms. Aggregating and plotting
    must operate on ``effective_ns``, or the shared point is double-counted
    with phantom independence.

    Within each ``effective_ns`` group, an ``EXACT`` condition is preferred as
    the representative. Unresolved conditions are dropped, since they are not
    measurements at all.

    Args:
        icdis: Conditions to collapse.

    Returns:
        One representative per distinct ``effective_ns``, ordered by
        ``effective_ns``.
    """
    by_effective: dict[int, list[ICDI]] = {}
    for icdi in icdis:
        if icdi.resolution is Resolution.UNRESOLVED:
            continue
        by_effective.setdefault(icdi.effective_ns, []).append(icdi)

    representatives: list[ICDI] = []
    for group in by_effective.values():
        exact = [c for c in group if c.resolution is Resolution.EXACT]
        representatives.append(exact[0] if exact else min(group, key=lambda c: c.requested_ns))

    representatives.sort(key=lambda c: c.effective_ns)
    return representatives


def format_ns(latency_ns: int) -> str:
    """Render a nanosecond value with an appropriate unit.

    Useful in logs and manifests, where raw nanosecond integers are hard to
    check by eye.
    """
    if latency_ns < 1_000:
        return f"{latency_ns} ns"
    if latency_ns < 1_000_000:
        return f"{latency_ns // 1_000} us"
    if latency_ns < 1_000_000_000:
        return f"{latency_ns // 1_000_000} ms"
    return f"{latency_ns // 1_000_000_000} s"
