"""The Phase 3 mechanism test, run with no policy at all.

Question being asked: does the synthetic market actually contain a short-lived
economically meaningful opportunity, such that deciding *correctly* and executing
late captures less of it?

This is deliberately **not** an RL experiment and contains no model. An oracle
that reads the hidden state takes the economically correct action at every
trigger event. If opportunity still decays with delay under an oracle, the decay
comes from the market process. If it did not decay, adding a policy later could
only have muddied the result.

What this module must never do is multiply opportunity by a decay constant. The
entire point is that the decay is a *consequence* of the evolving latent state
and price process, so the plot is a consequence of the generator, not a restated
input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream
from latency_ml.finance.replay import (
    FixedEventLatency,
    LatencyProvider,
    clamp_execution,
    opportunity_pnl,
)


@dataclass(frozen=True)
class OpportunityPoint:
    """Available opportunity at one execution delay.

    Attributes:
        delay_events: Execution delay, in events.
        delay_ns: The same delay in nanoseconds.
        mean_pnl: Mean P&L per triggered opportunity, in price units.
        sem: Standard error of that mean.
        n_samples: Opportunities retained after dropping those whose markout
            window runs past the end of the stream.
        signal_alive_fraction: Fraction of samples where the latent state was
            still non-neutral at execution. This is the mechanism made visible:
            it is the quantity the decay is supposed to track.
    """

    delay_events: int
    delay_ns: int
    mean_pnl: float
    sem: float
    n_samples: int
    signal_alive_fraction: float


def trigger_indices(stream: MarketStream) -> NDArray[np.int64]:
    """Events where the latent state is non-neutral.

    These are the exogenously created opportunities. The oracle acts at each.
    """
    return np.flatnonzero(stream.signal != 0).astype(np.int64)


def _median_event_spacing_ns(stream: MarketStream) -> int:
    """Median gap between consecutive events, in nanoseconds.

    The median of the single-event gaps, not ``span / count``: LOBSTER sessions
    are bursty, with long idle stretches between dense bursts, so the mean spacing
    is dominated by gaps no decision ever waited through. Used only to convert an
    index shift into a reported duration -- the applied delay is the shift itself.
    """
    if len(stream.times_ns) < 2:
        return 1
    gaps = np.diff(stream.times_ns)
    return max(1, int(np.median(gaps)))


def sweep_opportunity(
    stream: MarketStream,
    delays: tuple[int, ...],
    markout_events: int,
    transaction_cost_bps: float = 0.0,
) -> list[OpportunityPoint]:
    """Oracle opportunity as a function of execution delay.

    The decision indices are identical at every delay, so the sweep is paired:
    the same opportunities are being measured, only later.

    Args:
        stream: Generated market.
        delays: Execution delays to measure, in events.
        markout_events: Markout horizon after execution.
        transaction_cost_bps: Extra per-side cost, applied symmetrically.

    Returns:
        One :class:`OpportunityPoint` per delay, in the order given.
    """
    decisions = trigger_indices(stream)
    n = len(stream)
    # Measured from the stream rather than assumed. An index shift needs no clock,
    # but reporting it in nanoseconds does, and a fixed 1 ms/event assumption
    # overstates a real session's elapsed time by more than an order of magnitude.
    spacing_ns = _median_event_spacing_ns(stream)

    points: list[OpportunityPoint] = []

    for delay in delays:
        provider: LatencyProvider = FixedEventLatency(delay, event_spacing_ns=spacing_ns)
        raw_exec = np.array([provider.execution_index(int(i)) for i in decisions], dtype=np.int64)
        exec_idx = np.array(
            [e if (e := clamp_execution(0, int(r), n)) is not None else n for r in raw_exec],
            dtype=np.int64,
        )
        pnl = opportunity_pnl(
            stream,
            decisions,
            exec_idx,
            markout_events,
            transaction_cost_bps,
        )
        kept = (exec_idx < n) & (exec_idx + markout_events < n)
        alive = float(np.mean(stream.signal[exec_idx[kept]] != 0)) if kept.any() else float("nan")
        sem = float(np.std(pnl, ddof=1) / np.sqrt(pnl.size)) if pnl.size > 1 else 0.0
        points.append(
            OpportunityPoint(
                delay_events=delay,
                delay_ns=delay * spacing_ns,
                mean_pnl=float(np.mean(pnl)) if pnl.size else float("nan"),
                sem=sem,
                n_samples=int(pnl.size),
                signal_alive_fraction=alive,
            )
        )
    return points


def decay_fraction(points: list[OpportunityPoint]) -> float:
    """Fraction of the zero-delay opportunity lost at the largest delay.

    Normalised against the zero-delay mean, which is the only scale-free way to
    compare markets with different price volatility.

    Returns:
        The fraction lost, or ``nan`` when the zero-delay reference is not
        positive. A control whose baseline is negative has no opportunity to
        lose, and normalising by its magnitude would report a large positive
        "decay" for a curve that is actually flat and merely drifting upward
        toward zero cost. Those controls are judged on the flatness of their raw
        means instead.
    """
    if not points:
        raise ValueError("no points to summarise")
    baseline = points[0].mean_pnl
    if not np.isfinite(baseline) or baseline <= 0.0:
        return float("nan")
    worst = points[-1].mean_pnl
    return float((baseline - worst) / baseline)
