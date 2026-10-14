"""Physical-time regimes and the resolution each one can actually support.

A latency grid is only meaningful up to the resolution of the event timeline it
is mapped onto. Both control environments here quantise: CartPole's 20 ms physics
step, and this generator's ``event_spacing_ns``. Reporting a 100 ns condition on
a stream whose events are 100 us apart would be fabricating a measurement, which
is the failure ``PLAN.md`` section 5 exists to prevent.

Regimes differ **only** in ``event_spacing_ns``. Every other generator parameter
is expressed in events, so rescaling the spacing rescales physical time while
leaving the dimensionless relationships between delay, latent persistence and
markout horizon exactly as they were. That is what makes the comparison across
regimes meaningful: it is the same process observed at a different clock rate,
not a different market.

The scale factors are not arbitrary. Each is a round factor applied to the
validated 1 ms regime, chosen so that the reachable delay range moves down by one
decade at a time toward the sub-millisecond and nanosecond regimes the HFT grid
specifies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from latency_ml.finance.market import GeneratorParams


@dataclass(frozen=True)
class Regime:
    """A physical-time realisation of the synthetic process.

    Attributes:
        name: Identifier.
        event_spacing_ns: Gap between consecutive events.
        description: Human-readable note for reports.
    """

    name: str
    event_spacing_ns: int
    description: str

    def apply(self, params: GeneratorParams) -> GeneratorParams:
        """Return ``params`` rescaled to this regime."""
        return replace(params, event_spacing_ns=self.event_spacing_ns)

    @property
    def resolution_ns(self) -> int:
        """Finest delay this regime can distinguish.

        One event spacing. A delay below this cannot move execution to a
        different event, so it is indistinguishable from zero.
        """
        return self.event_spacing_ns


#: Validated millisecond regime. The one Stage A and the mechanism test were run at.
REGIME_MS: Regime = Regime(
    "regime_a_ms",
    1_000_000,
    "1 ms events. Resolves delays >= 1 ms; sub-millisecond conditions collapse onto zero.",
)

#: Sub-millisecond regime. Ten times finer, same dimensionless process.
REGIME_SUBMS: Regime = Regime(
    "regime_b_subms",
    100_000,
    "100 us events. Resolves delays >= 100 us.",
)

#: Nanosecond regime. Fine enough to resolve the full HFT latency grid.
REGIME_NS: Regime = Regime(
    "regime_c_ns",
    100,
    "100 ns events. Resolves the full 100 ns - 100 ms HFT grid.",
)

REGIMES: tuple[Regime, ...] = (REGIME_MS, REGIME_SUBMS, REGIME_NS)


def resolution_of(event_spacing_ns: int) -> int:
    """Finest delay resolvable at this event spacing."""
    return event_spacing_ns


def resolved_delays(latencies_ns: list[int], event_spacing_ns: int) -> list[bool]:
    """Which requested latencies a given regime can actually distinguish.

    A delay is resolved when it is exactly zero, or when it moves execution to a
    *different* event than a slightly smaller delay would. The second condition
    is stricter than ``delay >= spacing`` because of the quantisation described
    in :class:`latency_ml.finance.replay.NanosecondLatency`: the realised delay
    is the first event at or after the target, so two different requested delays
    can land on the same event and be indistinguishable in effect.
    """
    out: list[bool] = []
    for delay in sorted(set(latencies_ns)):
        if delay == 0:
            out.append(True)
            continue
        realised = -(-delay // event_spacing_ns)
        just_below = -(-(delay - 1) // event_spacing_ns)
        out.append(realised != just_below)
    return out


def distinct_executions(latencies_ns: list[int], event_spacing_ns: int) -> int:
    """How many *distinct* execution indices a grid produces in this regime.

    A grid of 14 points collapsing to 3 distinct outcomes is not a 14-point sweep,
    and this is the number that says so.
    """
    seen = set()
    for delay in latencies_ns:
        seen.add(-(-delay // event_spacing_ns) if delay else 0)
    return len(seen)
