from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
NATIVE_DT_ENV_NS: int = 20000000
HIGH_RES_DT_ENV_NS: int = 1000000
class Resolution(Enum):
    EXACT = "exact"
    QUANTIZED_UP = "quantized_up"
    UNRESOLVED = "unresolved"
@dataclass(frozen=True, slots=True)
class ICDI:
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
    action_steps = -(-latency_ns // dt_env_ns)
    action_steps = max(1, action_steps)
    effective_ns = action_steps * dt_env_ns
    if latency_ns < dt_env_ns:
        resolution = Resolution.UNRESOLVED
    elif effective_ns != latency_ns:
        resolution = Resolution.QUANTIZED_UP
    else:
        resolution = Resolution.EXACT
    return ICDI(requested_ns=latency_ns, action_steps=action_steps, effective_ns=effective_ns, resolution=resolution)
NATIVE_SWEEP_GRID_NS: tuple[int, ...] = (20000000, 40000000, 60000000, 80000000, 100000000)
LATENCY_GRID_NS: tuple[int, ...] = (100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 20000000, 30000000, 50000000, 100000000)
def latency_grid() -> tuple[int, ...]:
    return LATENCY_GRID_NS
def resolved_grid(dt_env_ns: int = NATIVE_DT_ENV_NS) -> tuple[int, ...]:
    return tuple(latency_ns for latency_ns in NATIVE_SWEEP_GRID_NS if derive_icdi(latency_ns, dt_env_ns).is_independent_measurement)
def resolved_target_grid(dt_env_ns: int = NATIVE_DT_ENV_NS) -> tuple[int, ...]:
    return tuple(icdi.requested_ns for icdi in iter_icdi(dt_env_ns) if icdi.is_independent_measurement)
def iter_icdi(dt_env_ns: int = NATIVE_DT_ENV_NS) -> Iterator[ICDI]:
    for latency_ns in LATENCY_GRID_NS:
        yield derive_icdi(latency_ns, dt_env_ns)
def collapse_by_effective(icdis: Sequence[ICDI]) -> list[ICDI]:
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
    if latency_ns < 1000:
        return f"{latency_ns} ns"
    if latency_ns < 1000000:
        return f"{latency_ns // 1000} us"
    if latency_ns < 1000000000:
        return f"{latency_ns // 1000000} ms"
    return f"{latency_ns // 1000000000} s"
