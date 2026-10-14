"""Decision-to-execution delay, applied to an already-generated market stream.

This is the **only** module that knows what latency is. The generator does not,
and must not: a market process that varied with latency would make every latency
condition a different market, and the sweep would measure nothing about timing.

The pipeline is:

    market event -> decision -> latency provider -> execution time -> book fill

Keeping the provider behind a small interface is what allows the same replay to
be driven later by measured hardware numbers without touching the financial
experiment. ``FixedEventLatency`` here counts *events*; a hardware-backed
provider counts nanoseconds and converts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream


class LatencyProvider(Protocol):
    """Maps a decision time to an execution time."""

    def execution_index(self, decision_index: int) -> int:
        """Index of the event at which a decision taken at ``decision_index`` executes."""
        ...

    @property
    def nominal_ns(self) -> int:
        """Latency this provider represents, in nanoseconds."""
        ...


@dataclass(frozen=True)
class FixedEventLatency:
    """Execution delayed by a fixed number of events.

    Event-count delay is the right unit for a synthetic stream whose events are
    uniformly spaced. ``PLAN.md` section 11.3` wants event-time delay so that the
    delay is expressible in physical units; :func:`nanosecond_latency` is the
    bridge once a stream carries irregular timestamps.
    """

    delay_events: int
    #: Nanoseconds per event. Defaults to the synthetic generator's 1 ms spacing,
    #: which is exact for it and wrong for real data.
    #:
    #: This used to be a hardcoded constant, which meant every historical sweep
    #: reported a delay that had no relationship to the stream it replayed. On
    #: LOBSTER the median event spacing is roughly 0.5 ms but the median
    #: 10-event shift is about 0.19 ms, so a 200-event delay was reported as
    #: 200 ms when it cost closer to 4 ms -- a 50x overstatement in a latency
    #: study. It is now a field with an honest default, and callers on real data
    #: must supply the measured spacing.
    event_spacing_ns: int = 1_000_000

    def __post_init__(self) -> None:
        if self.delay_events < 0:
            raise ValueError("delay_events must be non-negative")
        if self.event_spacing_ns <= 0:
            raise ValueError("event_spacing_ns must be positive")

    @property
    def nominal_ns(self) -> int:
        """Nanoseconds this delay represents, given the assumed spacing."""
        return self.delay_events * self.event_spacing_ns

    def execution_index(self, decision_index: int) -> int:
        return decision_index + self.delay_events


@dataclass(frozen=True)
class NanosecondLatency:
    """Execution delayed by a fixed nanosecond interval, on a real timeline.

    This is the provider a hardware measurement would feed: the delay is
    physical, and execution lands on whichever event first occurs at or after
    ``decision_time + delay_ns``.
    """

    delay_ns: int

    def __post_init__(self) -> None:
        if self.delay_ns < 0:
            raise ValueError("delay_ns must be non-negative")

    @property
    def nominal_ns(self) -> int:
        return self.delay_ns

    def execution_index(
        self, decision_index: int, times_ns: NDArray[np.int64] | None = None
    ) -> int:
        """First event at or after the delayed decision time.

        Args:
            decision_index: Event at which the decision was taken.
            times_ns: Event timestamps. Required; without them a physical delay
                cannot be resolved onto an event grid.

        Raises:
            ValueError: If no timestamps are supplied.
        """
        if times_ns is None:
            raise ValueError("times_ns is required to resolve a nanosecond delay")
        target = int(times_ns[decision_index]) + self.delay_ns
        return int(np.searchsorted(times_ns, target, side="left"))


def clamp_execution(decision_index: int, execution_index: int, n_events: int) -> int | None:
    """Clamp an execution index into the stream.

    Returns:
        The execution index, or ``None`` if the decision's markout window would
        run past the end of the generated stream. Dropping such samples is the
        honest handling: truncating them would silently substitute a shorter
        opportunity for a delayed one.
    """
    if execution_index >= n_events:
        return None
    return execution_index


def opportunity_pnl(
    stream: MarketStream,
    decision_indices: NDArray[np.int64],
    execution_indices: NDArray[np.int64],
    markout_events: int,
    transaction_cost_bps: float = 0.0,
) -> NDArray[np.float64]:
    """P&L of taking each decision and executing it after a delay.

    Aggressive execution: a buy lifts the ask and is marked out by selling at
    the bid, so the round trip pays the spread exactly once. That is the
    simplest defensible execution model ``PLAN.md` section 11.7 asks for, and it
    is held identical across every latency condition.

    Args:
        stream: Generated market.
        decision_indices: Events at which the decision was taken.
        execution_indices: Matching execution events, from a latency provider.
        markout_events: Events after execution over which the move is marked.
        transaction_cost_bps: Extra cost in basis points of notional, applied
            symmetrically so it does not favour either direction.

    Returns:
        Per-decision P&L in price units, aligned with ``decision_indices``.
        Samples whose markout window runs off the end are dropped.
    """
    n = len(stream)
    keep = (execution_indices >= 0) & (execution_indices + markout_events < n)
    d = decision_indices[keep]
    x = execution_indices[keep]
    if d.size == 0:
        return np.zeros(0, dtype=np.float64)

    mid_exec = stream.mid[x]
    mid_exit = stream.mid[x + markout_events]
    spread_exec = stream.spread[x]

    # Both directions are handled by the same expression: the sign of the
    # position multiplies a long-style round trip.
    position = np.sign(stream.signal[d].astype(np.float64))
    price_move = position * (mid_exit - mid_exec)
    cost = 0.5 * spread_exec + (transaction_cost_bps / 10_000.0) * np.abs(mid_exec)
    return price_move - cost


def stream_digest(stream: MarketStream) -> str:
    """Content hash of a market stream, for run provenance.

    Every latency condition in a sweep is meant to be replayed against the same
    events, with only the decision-availability time moving. That is a claim
    about the artifacts, so it needs to be checkable from them: record this in
    each condition's output and a reader can confirm the arms saw identical data
    without trusting the code that produced it.

    **Provenance only.** The digest is a function of the stream and is never fed
    back into it. Nothing here touches the feature matrix, the policy input, the
    labels or any quantity that could move a reported number -- it is the one
    function in this package whose output is metadata.

    Hashed field by field rather than as one buffer, so the digest changes when
    and only when the *content* changes, and cannot be made to agree by a
    difference in memory layout.
    """
    digest = hashlib.sha256()
    for name in ("times_ns", "mid", "signal", "buy_fraction", "spread", "bid_size", "ask_size"):
        array = np.ascontiguousarray(getattr(stream, name))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        # NaN has many bit patterns; normalise so an unusable touch hashes the
        # same however it was produced.
        if array.dtype.kind == "f":
            payload = np.nan_to_num(array, nan=np.float64("nan"), posinf=0.0, neginf=0.0)
        else:
            payload = array
        digest.update(np.ascontiguousarray(payload).tobytes())
    return digest.hexdigest()
