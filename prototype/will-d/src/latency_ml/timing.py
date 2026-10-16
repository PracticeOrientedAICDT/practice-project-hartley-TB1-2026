from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

NATIVE_DT_ENV_NS: int = 20000000
NATIVE_SWEEP_GRID_NS: tuple[int, ...] = (20000000, 40000000, 60000000, 80000000, 100000000)


class Resolution(Enum):
    EXACT = "exact"
    QUANTIZED_UP = "quantized_up"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ICDI:
    """How a requested control latency maps onto whole environment frames."""

    requested_ns: int
    action_steps: int
    effective_ns: int
    resolution: Resolution

    @property
    def is_independent_measurement(self) -> bool:
        return self.resolution is Resolution.EXACT


def derive_icdi(latency_ns: int, dt_env_ns: int = NATIVE_DT_ENV_NS) -> ICDI:
    if latency_ns <= 0:
        raise ValueError(f"latency_ns must be positive, got {latency_ns}")
    if dt_env_ns <= 0:
        raise ValueError(f"dt_env_ns must be positive, got {dt_env_ns}")
    action_steps = max(1, -(-latency_ns // dt_env_ns))
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


def iter_icdi(latencies_ns: tuple[int, ...], dt_env_ns: int = NATIVE_DT_ENV_NS) -> Iterator[ICDI]:
    for latency_ns in latencies_ns:
        yield derive_icdi(latency_ns, dt_env_ns)


def format_ns(latency_ns: int) -> str:
    if latency_ns < 1000:
        return f"{latency_ns} ns"
    if latency_ns < 1000000:
        return f"{latency_ns / 1000:g} us"
    if latency_ns < 1000000000:
        return f"{latency_ns / 1000000:g} ms"
    return f"{latency_ns / 1000000000:g} s"
