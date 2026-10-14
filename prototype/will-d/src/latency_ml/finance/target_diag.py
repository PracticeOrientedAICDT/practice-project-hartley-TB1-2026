"""Characterise the trade-direction target on real LOBSTER sessions.

Nothing here trains anything, and nothing here decides anything. The point is to
find out which target definition is even well posed before one is frozen and a
model is built on it.

Why this had to come first
--------------------------
``fit_hold_band`` derives the HOLD threshold as a quantile of absolute forward
mid-price moves. On synthetic data that is unremarkable. On a real book it returns
exactly zero, because a real mid-price is a step function: across a session, a
large fraction of observations have not moved at all over the horizon, and if that
fraction exceeds the quantile then the quantile lands in the point mass at zero.

A band of zero is not a narrow band. ``build_labels`` compares with strict
inequalities, so ``move > 0`` and ``move < 0`` leave every unmoved observation
labelled HOLD: the three-class problem silently collapses to ``sign(move)`` with
HOLD absorbing the entire mass at zero. On one session here that is 95% HOLD, which
is a binary problem wearing a three-class hat.

The existing test asserting ``band > 0`` still passes, because it runs on synthetic
data with a continuous mid. That is the structural reason this went unnoticed: the
one property under test cannot be expressed by the fixture.

Two normalisations, and why both are reported
---------------------------------------------
Moves are reported in **granule units**, where one granule is the GCD of the
observed non-zero mid-price differences -- 50 units, or half a cent, for these
sessions, because a mid is the arithmetic mean of a touch whose spread is usually
an odd number of ticks. No tick constant exists anywhere in ``src/``; the only one
in the repository is a test fixture. Deriving it per session means a symbol whose
tick size differs is not silently measured in the wrong unit.

Granule units are still not comparable *across* symbols: the median half-spread is
one granule on INTC and thirty-eight on AAPL_30. So every economic figure is also
reported in **half-spread multiples**, which is dimensionless. A target that looks
identical in ticks can be an order of magnitude apart in cost terms, and only the
second framing says whether trading it could ever pay.

Cost, and the uncomfortable part
--------------------------------
The execution model is an aggressive fill against the touch, so crossing costs half
the prevailing spread. Whether that is the right execution model for this question
is deliberately *not* settled here -- it is a separate decision, and letting it
choose the prediction target would quietly change the forecasting problem to suit
the economics. What is reported is the consequence of the current model, so the
decision can be made with the number in hand: on these sessions a *perfect*
forecaster would lose money at short horizons, because the mean move conditional
on a correct signal is smaller than the cost of acting on it. That is a fact about
the execution model and the horizon, not about the model's skill.

Latency is a separate axis from horizon, and they are not interchangeable
-----------------------------------------------------------------------
The forecast horizon ``H`` is a look-ahead window; execution latency ``tau`` is
when the resulting action lands. They enter at different places and are swept
independently, which is the property that makes a latency sweep on a fixed
forecasting problem meaningful.

Why the horizon is a duration and not an event count
----------------------------------------------------
The event-count parameterisation is retired for the real-data target. It is not a
clock: the same ``H = 10`` spans a median 8.6 ms on INTC and 345 ms on AMZN, so
"H = 10" names two different forecasting problems forty-fold apart in duration and
a sweep over it compares unlike things. On INTC and MSFT at ``H = 10`` the median
window is *shorter* than the largest delay in the grid, so the delay would consume
the whole prediction and there is nothing left to measure.

The real-data target is therefore defined by duration, with

    t_H = min{ t_i : t_i - t_0 >= H }

and the label being the move from ``t_0`` to ``t_H``. Latency then lands on the
same scale, ``t_exec = t_0 + tau``, so ``H`` and ``tau`` are directly comparable
nanosecond quantities. The event-count view is retained below because it is what
produced the evidence for retiring it.

Value surviving delay, and why that is the quantity to select on
----------------------------------------------------------------
``V(tau)`` is the mean post-cost outcome of a *perfect* forecaster acting on every
signal the band emits, when the action lands at ``t_0 + tau``. The label is fixed
by the move to ``t_H``; the profit is what remains of that move after the delay has
let the price travel further. So ``V(tau)`` is a property of the data, the target
and the execution model -- not of any model, and not of its skill. It is an upper
bound: nothing trained can beat it, and if it is not positive no skill can help.

The derived number that decides whether the latency experiment is identifiable is
``tau_half``: the delay at which ``V`` falls to half its immediate value, which is
how the decay is characterised. Value provably cannot outlive the horizon -- filling
at ``t_H`` earns nothing beyond the spread -- so a zero-crossing would always land
inside ``H`` and would never say whether the whole grid sees decay or a plateau.
Scaled by ``H`` it does: several half-lives per horizon means a real decay, one
means the grid mostly measures a step. Class balance cannot substitute for this,
which the event-count table above demonstrates directly: its most balanced cells
are its least profitable ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream
from latency_ml.finance.target import (
    BUY,
    HOLD,
    SELL,
    TargetError,
    forward_index,
    label_moves,
    price_granule,
)

#: Class-count thresholds for the thinness flags. Deliberately conservative: these
#: flag "too few to learn from and replicate", not "too few to exist".
THIN_CLASS_EVENTS: int = 1_000
DEGENERATE_CLASS_EVENTS: int = 100

#: Quantiles reported for the horizon's wall-clock duration.
DURATION_QUANTILES: tuple[float, ...] = (10.0, 50.0, 90.0)


#: Raised when a session cannot be characterised. An alias, not a subclass: the
#: band, the granule and the horizon search all live in :mod:`finance.target` and
#: raise :class:`~latency_ml.finance.target.TargetError`, so a caller catching a
#: narrower diagnostic-only type would miss the errors raised from inside them.
TargetDiagnosticError = TargetError


@dataclass(frozen=True)
class ClassBalance:
    """How a ``k``-granule band splits a horizon's moves into three classes."""

    k_ticks: int
    buy_fraction: float
    hold_fraction: float
    sell_fraction: float
    buy_events: int
    hold_events: int
    sell_events: int

    @property
    def balanced(self) -> bool:
        """Whether all three classes are large enough to learn from.

        A 30/35/35 split was the original design target. Anything with a class
        below the threshold is reported as unbalanced regardless of how
        attractive the other two look.
        """
        return min(self.buy_events, self.hold_events, self.sell_events) >= THIN_CLASS_EVENTS

    @property
    def thinness(self) -> str:
        smallest = min(self.buy_events, self.hold_events, self.sell_events)
        if smallest < DEGENERATE_CLASS_EVENTS:
            return "degenerate"
        if smallest < THIN_CLASS_EVENTS:
            return "thin"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_ticks": self.k_ticks,
            "buy_fraction": self.buy_fraction,
            "hold_fraction": self.hold_fraction,
            "sell_fraction": self.sell_fraction,
            "buy_events": self.buy_events,
            "hold_events": self.hold_events,
            "sell_events": self.sell_events,
            "balanced": self.balanced,
            "thinness": self.thinness,
        }


@dataclass(frozen=True)
class HorizonStats:
    """Everything measured at one forecast horizon."""

    horizon_events: int
    n_observations: int
    #: Fraction of moves that are exactly zero. The reason a float quantile band
    #: degenerates.
    zero_move_fraction: float
    #: Distribution of |move| in granule units.
    abs_move_median_ticks: float
    abs_move_p95_ticks: float
    #: Signed move distribution, in granule units. Asymmetry here would mean the
    #: BUY and SELL classes are not interchangeable.
    move_mean_ticks: float
    move_skew_proxy: float
    #: Wall-clock duration of the look-ahead window, in nanoseconds.
    duration_ns_p10: float
    duration_ns_p50: float
    duration_ns_p90: float
    #: Median spread over the same observations, in granule units.
    median_half_spread_ticks: float
    #: Median |move| expressed in half-spreads -- the dimensionless scale.
    abs_move_median_half_spreads: float
    balances: tuple[ClassBalance, ...] = field(default_factory=tuple)
    economics: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_events": self.horizon_events,
            "n_observations": self.n_observations,
            "zero_move_fraction": self.zero_move_fraction,
            "abs_move_median_ticks": self.abs_move_median_ticks,
            "abs_move_p95_ticks": self.abs_move_p95_ticks,
            "move_mean_ticks": self.move_mean_ticks,
            "move_skew_proxy": self.move_skew_proxy,
            "duration_ns_p10": self.duration_ns_p10,
            "duration_ns_p50": self.duration_ns_p50,
            "duration_ns_p90": self.duration_ns_p90,
            "median_half_spread_ticks": self.median_half_spread_ticks,
            "abs_move_median_half_spreads": self.abs_move_median_half_spreads,
            "balances": [b.to_dict() for b in self.balances],
            "economics": list(self.economics),
        }


@dataclass(frozen=True)
class SessionDiagnostic:
    """Every horizon characterised on one session."""

    session: str
    n_events: int
    granule_units: int
    granule_note: str
    median_half_spread_ticks: float
    median_event_gap_ns: float
    #: SHA-256 over the stream fields, so two diagnostic runs can be shown to have
    #: read identical data rather than merely similar data.
    stream_digest: str
    horizons: tuple[HorizonStats, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "n_events": self.n_events,
            "granule_units": self.granule_units,
            "granule_note": self.granule_note,
            "median_half_spread_ticks": self.median_half_spread_ticks,
            "median_event_gap_ns": self.median_event_gap_ns,
            "stream_digest": self.stream_digest,
            "horizons": [h.to_dict() for h in self.horizons],
        }


def _quantile(values: NDArray[np.float64], q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def class_balance(moves_ticks: NDArray[np.float64], k: int) -> ClassBalance:
    """Split signed moves into BUY / HOLD / SELL at an integer band of ``k``.

    Inclusive on both sides, unlike ``build_labels``' strict inequalities, so a
    zero band cannot silently degrade to ``sign(move)``. The distinction matters:
    with strict inequalities and ``k = 0`` every unmoved observation is HOLD,
    which is a different target from the one anyone intends to fit.
    """
    labels = label_moves(np.asarray(moves_ticks, dtype=np.float64), k)
    buy = labels == BUY
    sell = labels == SELL
    hold = labels == HOLD
    return ClassBalance(
        k_ticks=k,
        buy_fraction=float(np.mean(buy)),
        hold_fraction=float(np.mean(hold)),
        sell_fraction=float(np.mean(sell)),
        buy_events=int(np.count_nonzero(buy)),
        hold_events=int(np.count_nonzero(hold)),
        sell_events=int(np.count_nonzero(sell)),
    )


def economics_at(
    moves_ticks: NDArray[np.float64],
    half_spread_ticks: NDArray[np.float64],
    k: int,
) -> dict[str, Any]:
    """What a ``k``-band signal is worth against the aggressive-fill cost model.

    ``net_mean`` is the mean outcome of acting on **every** signal the band emits,
    in granule units, after paying half the contemporaneous spread. It is a
    property of the target and the cost model, not of any model: a forecaster
    that is right every time achieves this figure, and one that is wrong every
    time achieves its negative.

    ``surviving_fraction`` is the share of emitted signals whose move is large
    enough to cover the crossing cost even if the sign is right. Where that is a
    few percent, no amount of forecasting skill makes the target worth trading,
    because the cost is paid on every signal rather than only on the wrong ones.
    """
    buy = moves_ticks >= k
    sell = moves_ticks <= -k
    acted = buy | sell
    cost = half_spread_ticks
    if not acted.any():
        return {
            "k_ticks": k,
            "n_signals": 0,
            "surviving_fraction": float("nan"),
            "net_mean_ticks": 0.0,
            "net_mean_half_spreads": 0.0,
            "mean_move_when_right_ticks": float("nan"),
            "cost_multiple": float("nan"),
        }
    # A BUY pays half-spread and gains the move; a SELL pays half-spread and gains
    # its negation. HOLD is free, so it contributes zero either way.
    gross = np.where(buy, moves_ticks, np.where(sell, -moves_ticks, 0.0))
    net = gross - np.where(acted, cost, 0.0)
    right = acted & (gross > cost)
    mean_cost = float(np.mean(cost[acted]))
    return {
        "k_ticks": k,
        "n_signals": int(np.count_nonzero(acted)),
        "surviving_fraction": float(np.count_nonzero(right) / np.count_nonzero(acted)),
        "net_mean_ticks": float(np.mean(net)),
        "net_mean_half_spreads": float(np.mean(net) / mean_cost) if mean_cost > 0 else float("nan"),
        "mean_move_when_right_ticks": float(np.mean(gross[acted & (gross > 0)]))
        if np.any(acted & (gross > 0))
        else float("nan"),
        "cost_multiple": mean_cost,
    }


def characterise_horizon(
    stream: MarketStream,
    horizon: int,
    granule: int,
    ks: Sequence[int],
) -> HorizonStats:
    """Measure one forecast horizon."""
    n = len(stream)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if horizon >= n:
        raise TargetDiagnosticError(
            f"horizon {horizon} needs more than {n} events; the session is too short"
        )

    mid = stream.mid
    moves = mid[horizon:] - mid[:-horizon]
    moves_ticks = moves / granule

    spreads = stream.spread[: moves.size]
    half_spread = spreads / 2.0 / granule

    times = stream.times_ns
    durations = (times[horizon:] - times[:-horizon]).astype(np.float64)
    finite_durations = durations[np.isfinite(durations)]

    abs_moves = np.abs(moves_ticks)
    median_abs = float(np.median(abs_moves))
    median_half = float(np.nanmedian(half_spread))
    nonzero = abs_moves[abs_moves > 0]

    stats = HorizonStats(
        horizon_events=horizon,
        n_observations=int(moves.size),
        zero_move_fraction=float(np.mean(abs_moves == 0.0)),
        abs_move_median_ticks=median_abs,
        abs_move_p95_ticks=_quantile(abs_moves, 95.0),
        move_mean_ticks=float(np.mean(moves_ticks)),
        # Cheap asymmetry proxy: mass above zero against mass below. A large
        # imbalance means BUY and SELL are not interchangeable classes.
        move_skew_proxy=(
            float(np.mean(np.sign(moves_ticks[np.nonzero(moves_ticks)]))) if nonzero.size else 0.0
        ),
        duration_ns_p10=_quantile(finite_durations, DURATION_QUANTILES[0]),
        duration_ns_p50=_quantile(finite_durations, DURATION_QUANTILES[1]),
        duration_ns_p90=_quantile(finite_durations, DURATION_QUANTILES[2]),
        median_half_spread_ticks=median_half,
        abs_move_median_half_spreads=(
            median_abs / median_half if median_half > 0 else float("inf")
        ),
    )
    safe_half_spread = np.nan_to_num(half_spread, nan=0.0)
    return replace(
        stats,
        balances=tuple(class_balance(moves_ticks, k) for k in ks),
        economics=tuple(economics_at(moves_ticks, safe_half_spread, k) for k in ks),
    )


# ---------------------------------------------------------------------------
# Wall-clock horizons: the real-data target definition
# ---------------------------------------------------------------------------
#
# Everything below works in nanoseconds. The label is defined over a duration H;
# the profit is defined over a duration tau. Both index the same clock, so the
# two are comparable without a conversion factor that would differ per symbol.


@dataclass(frozen=True)
class LatencyValue:
    """``V(tau)`` for one band at one execution delay."""

    k_ticks: int
    tau_ns: int
    #: False when tau exceeds H, i.e. the action lands after the forecast has
    #: already resolved. Those entries are reported as unavailable rather than
    #: quietly computed, because acting after t_H is a look-ahead, not a delay.
    applicable: bool
    n_signals: int
    surviving_fraction: float
    net_mean_ticks: float
    net_mean_half_spreads: float
    #: Share of the signalled move still present at t_0 + tau. 1.0 means the delay
    #: cost nothing; lower means the price ran before the action landed.
    capture_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_ticks": self.k_ticks,
            "tau_ns": self.tau_ns,
            "applicable": self.applicable,
            "n_signals": self.n_signals,
            "surviving_fraction": self.surviving_fraction,
            "net_mean_ticks": self.net_mean_ticks,
            "net_mean_half_spreads": self.net_mean_half_spreads,
            "capture_fraction": self.capture_fraction,
        }


@dataclass(frozen=True)
class BandValueCurve:
    """One band's target definition and the value it retains across the sweep."""

    k_ticks: int
    #: The horizon this band was measured at, needed to scale the half-life.
    horizon_ns: int
    balance: ClassBalance
    values: tuple[LatencyValue, ...]
    #: Delay at which ``V`` falls to half its immediate value. Value cannot outlive
    #: the horizon -- filling at t_H leaves nothing to earn -- so this is measured
    #: against H, and it is what says how much of the latency grid will see a
    #: decay rather than a plateau.
    tau_half_ns: float
    #: Value at zero delay, i.e. acting immediately.
    value_at_zero_ns: float
    #: Smallest share of oracle value a policy must capture for the gap between the
    #: grid's extreme delays to clear the MDE. Absent when the grid has one point.
    capture: CaptureFloor | None = None

    @property
    def viable_at_any_delay(self) -> bool:
        """Whether a perfect forecaster makes money acting immediately."""
        return self.value_at_zero_ns > 0.0

    @property
    def half_lives_in_horizon(self) -> float:
        """How many value half-lives fit inside one horizon.

        Two or more means the horizon spans a decay rather than a step. ``inf``
        when the value never halves inside the horizon, which is a plateau: a
        latency sweep there would measure almost nothing.
        """
        if self.tau_half_ns <= 0.0 or not np.isfinite(self.tau_half_ns):
            return float("inf")
        return self.horizon_ns / self.tau_half_ns

    def value_at(self, tau_ns: int) -> LatencyValue | None:
        for v in self.values:
            if v.tau_ns == tau_ns:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_ticks": self.k_ticks,
            "horizon_ns": self.horizon_ns,
            "balance": self.balance.to_dict(),
            "values": [v.to_dict() for v in self.values],
            "tau_half_ns": self.tau_half_ns,
            "value_at_zero_ns": self.value_at_zero_ns,
            "viable_at_any_delay": self.viable_at_any_delay,
            "half_lives_in_horizon": self.half_lives_in_horizon,
            "capture_floor": self.capture.to_dict() if self.capture else None,
        }


@dataclass(frozen=True)
class WallClockHorizon:
    """Everything measured at one wall-clock forecast horizon."""

    horizon_ns: int
    n_observations: int
    #: Realised window, which is at least ``horizon_ns`` because the clock is
    #: irregular; the quantiles show how much overshoot.
    duration_ns_p10: float
    duration_ns_p50: float
    duration_ns_p90: float
    zero_move_fraction: float
    abs_move_median_ticks: float
    abs_move_p95_ticks: float
    move_skew_proxy: float
    median_half_spread_ticks: float
    #: Events spanned by the window, averaged over origins. The bridge between the
    #: two parameterisations, and the reason the horizon must be a duration: on a
    #: bursty book this ratio is both large and wildly unstable, so a duration grid
    #: and an event grid select different windows at different points in the day.
    events_per_horizon: float = 0.0
    curves: tuple[BandValueCurve, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_ns": self.horizon_ns,
            "n_observations": self.n_observations,
            "duration_ns_p10": self.duration_ns_p10,
            "duration_ns_p50": self.duration_ns_p50,
            "duration_ns_p90": self.duration_ns_p90,
            "zero_move_fraction": self.zero_move_fraction,
            "abs_move_median_ticks": self.abs_move_median_ticks,
            "abs_move_p95_ticks": self.abs_move_p95_ticks,
            "move_skew_proxy": self.move_skew_proxy,
            "median_half_spread_ticks": self.median_half_spread_ticks,
            "events_per_horizon": self.events_per_horizon,
            "curves": [c.to_dict() for c in self.curves],
        }


@dataclass(frozen=True)
class WallClockDiagnostic:
    """Every wall-clock horizon characterised on one session."""

    session: str
    role: str
    n_events: int
    granule_units: int
    median_half_spread_ticks: float
    median_event_gap_ns: float
    #: Mean spacing. On a bursty book this is far larger than the median, and it is
    #: the figure that decides how many events a millisecond horizon actually buys.
    mean_event_gap_ns: float
    session_span_ns: int
    stream_digest: str
    horizons: tuple[WallClockHorizon, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "role": self.role,
            "n_events": self.n_events,
            "granule_units": self.granule_units,
            "median_half_spread_ticks": self.median_half_spread_ticks,
            "median_event_gap_ns": self.median_event_gap_ns,
            "mean_event_gap_ns": self.mean_event_gap_ns,
            "session_span_ns": self.session_span_ns,
            "stream_digest": self.stream_digest,
            "horizons": [h.to_dict() for h in self.horizons],
        }


@dataclass(frozen=True)
class _Clean:
    """The finite, in-order part of a session, with original positions kept."""

    idx: NDArray[np.int64]
    times: NDArray[np.int64]
    mid: NDArray[np.float64]
    spread: NDArray[np.float64]


def _clean_session(stream: MarketStream) -> _Clean:
    """Drop rows that cannot be reasoned about: locked books give NaN times.

    Their positions are retained in ``idx`` so anything reported back can be
    traced to the raw session.
    """
    times = np.asarray(stream.times_ns, dtype=np.float64)
    mid = np.asarray(stream.mid, dtype=np.float64)
    spread = np.asarray(stream.spread, dtype=np.float64)
    ok = np.isfinite(times) & np.isfinite(mid) & np.isfinite(spread)
    idx = np.flatnonzero(ok).astype(np.int64)
    if idx.size < 2:  # pragma: no cover - guarded upstream by the loader
        raise TargetDiagnosticError("session has fewer than two usable events")
    return _Clean(
        idx=idx,
        times=times[idx].astype(np.int64),
        mid=mid[idx],
        spread=spread[idx],
    )


#: The horizon search, shared with the trainer. One implementation of
#: ``t_H = min{t_i : t_i - t_0 >= H}``, so the diagnostic that chose the target and
#: the trainer that fits it cannot resolve different windows.
_forward = forward_index


def _executed(
    clean: _Clean,
    origins: NDArray[np.int64],
    targets: NDArray[np.int64],
    tau_ns: int,
    granule: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Move and cost as they stand at ``t_0 + tau``.

    ``targets`` is the horizon destination for each origin, already aligned to
    ``origins`` -- not the raw session-wide array, whose length differs from the
    origin count whenever the session tail has no successor.

    Returns the move in granule units from ``t_0`` to the execution time, the
    half-spread in granule units prevailing *at execution*, and a mask of origins
    that have both an execution event and a horizon event.
    """
    q_t = _forward(clean.times, tau_ns)[origins]
    ok = (q_t >= 0) & (targets >= 0)
    safe = np.where(ok, q_t, 0)
    moves = np.where(ok, clean.mid[safe] - clean.mid[origins], np.nan) / granule
    half = np.where(ok, clean.spread[safe] / 2.0 / granule, np.nan)
    return moves, half, ok


def value_at_latency(
    move_to_horizon: NDArray[np.float64],
    move_to_execution: NDArray[np.float64],
    half_spread: NDArray[np.float64],
    k: int,
    tau_ns: int = 0,
    applicable: bool = True,
) -> LatencyValue:
    """``V(tau)``: what one band's signals are worth once delayed by ``tau``.

    The label comes from the move to ``t_H`` -- that is the prediction being
    evaluated. The payoff comes from the *fill*, which under aggressive touch
    crossing happens against the touch prevailing at ``t_0 + tau``. So what is left
    to earn is the move from the fill to ``t_H``, and that is what shrinks as the
    delay grows::

        V(tau) = sign(label) * (move(t_H) - move(t_0 + tau)) - half_spread

    Splitting those two is the whole point. If the payoff were measured over the
    same window as the label, ``V`` would not depend on ``tau`` at all and a
    latency sweep would be measuring arithmetic rather than markets.

    Value cannot outlive the horizon: filling exactly at ``t_H`` leaves nothing to
    earn beyond the spread. That is a property of the definition, not an empirical
    finding, which is why the decay is characterised by its half-life relative to
    ``H`` rather than by a zero-crossing.

    A perfect forecaster is assumed, so this is an upper bound on what any model
    trained on this target could earn under the same execution model.
    """
    buy = move_to_horizon >= k
    sell = move_to_horizon <= -k
    acted = buy | sell
    n = int(np.count_nonzero(acted))
    if n == 0 or not applicable:
        return LatencyValue(
            k_ticks=k,
            tau_ns=tau_ns,
            applicable=applicable,
            n_signals=n,
            surviving_fraction=float("nan"),
            net_mean_ticks=float("nan"),
            net_mean_half_spreads=float("nan"),
            capture_fraction=float("nan"),
        )
    live = acted & np.isfinite(move_to_execution) & np.isfinite(half_spread)
    if not live.any():
        return LatencyValue(
            k_ticks=k,
            tau_ns=tau_ns,
            applicable=applicable,
            n_signals=n,
            surviving_fraction=float("nan"),
            net_mean_ticks=float("nan"),
            net_mean_half_spreads=float("nan"),
            capture_fraction=float("nan"),
        )
    # The fill lands at t_0 + tau, so the move still to be earned runs from there
    # to t_H. A BUY gains what is left of the rise; a SELL its negation. HOLD is
    # free and contributes zero, which is what makes net comparable across bands.
    remaining = move_to_horizon - move_to_execution
    gross = np.where(buy, remaining, np.where(sell, -remaining, 0.0))
    net = np.where(live, gross - np.where(live, half_spread, 0.0), 0.0)
    mean_cost = float(np.mean(half_spread[live]))
    denom = move_to_horizon[live]
    with np.errstate(divide="ignore", invalid="ignore"):
        capture = np.where(denom != 0.0, remaining[live] / denom, np.nan)
    return LatencyValue(
        k_ticks=k,
        tau_ns=tau_ns,
        applicable=applicable,
        n_signals=n,
        surviving_fraction=float(np.count_nonzero(gross[live] > half_spread[live]) / n),
        net_mean_ticks=float(np.mean(net)),
        net_mean_half_spreads=float(np.mean(net) / mean_cost) if mean_cost > 0 else float("nan"),
        capture_fraction=float(np.nanmean(capture)),
    )


def _tau_half(
    clean: _Clean,
    origins: NDArray[np.int64],
    targets: NDArray[np.int64],
    move_to_horizon: NDArray[np.float64],
    k: int,
    granule: int,
    horizon_ns: int,
    n_grid: int = 256,
) -> float:
    """Interpolate the delay at which ``V`` first halves.

    ``0.0`` when there was never any value to lose, ``inf`` when it never halves
    inside the horizon. Searched on a log grid because the interesting crossing is
    usually a small fraction of the horizon and a linear grid would step over it.

    Half-life rather than a zero-crossing because value provably cannot outlive
    ``H``: filling at ``t_H`` earns nothing beyond the spread, so a ``tau_zero``
    would always land inside the horizon and could never say whether the grid as a
    whole sees decay or a plateau. A half-life scaled by ``H`` can.
    """
    at_zero = _value_at(clean, origins, targets, move_to_horizon, k, granule, 0, horizon_ns)
    if not np.isfinite(at_zero) or at_zero <= 0.0:
        return 0.0
    threshold = at_zero / 2.0
    grid = np.unique(
        np.geomspace(max(1, horizon_ns // 10_000), horizon_ns, n_grid).astype(np.int64)
    )
    prev_tau, prev_val = 0, at_zero
    for tau in grid:
        val = _value_at(clean, origins, targets, move_to_horizon, k, granule, int(tau), horizon_ns)
        if not np.isfinite(val):
            continue
        if val <= threshold:
            # Interpolate in log-tau: the decay is close to exponential over the
            # range that matters, and linear interpolation in tau over-predicts.
            if prev_val > threshold and val != prev_val and prev_tau > 0:
                lo, hi = math.log(prev_tau), math.log(max(int(tau), 1))
                frac = (threshold - prev_val) / (val - prev_val)
                return float(math.exp(lo + frac * (hi - lo)))
            return float(tau)
        prev_tau, prev_val = int(tau), val
    return float("inf")


def _value_at(
    clean: _Clean,
    origins: NDArray[np.int64],
    targets: NDArray[np.int64],
    move_to_horizon: NDArray[np.float64],
    k: int,
    granule: int,
    tau_ns: int,
    horizon_ns: int,
) -> float:
    """``V(tau)`` as a bare float, for the crossing search."""
    if tau_ns > horizon_ns:
        return float("nan")
    moves, half, ok = _executed(clean, origins, targets, tau_ns, granule)
    value = value_at_latency(
        np.where(ok, move_to_horizon, np.nan),
        moves,
        half,
        k,
        tau_ns,
    )
    return value.net_mean_ticks


def characterise_wall_clock_horizon(
    stream: MarketStream,
    horizon_ns: int,
    granule: int,
    ks: Sequence[int],
    tau_grid_ns: Sequence[int],
) -> WallClockHorizon:
    """Measure one duration-defined horizon and its value across the sweep."""
    if horizon_ns <= 0:
        raise ValueError(f"horizon must be positive, got {horizon_ns}")

    clean = _clean_session(stream)
    q_h = _forward(clean.times, horizon_ns)
    origins = np.flatnonzero(q_h >= 0).astype(np.int64)
    if origins.size == 0:
        raise TargetDiagnosticError(
            f"no origin in this session has a {horizon_ns / 1e6:g} ms successor; "
            f"the session spans {int(clean.times[-1] - clean.times[0]) / 1e6:g} ms"
        )

    targets = q_h[origins]
    moves = (clean.mid[targets] - clean.mid[origins]) / granule
    durations = (clean.times[targets] - clean.times[origins]).astype(np.float64)
    abs_moves = np.abs(moves)
    nonzero = abs_moves[abs_moves > 0]

    curves: list[BandValueCurve] = []
    for k in ks:
        values: list[LatencyValue] = []
        for tau in sorted(set(int(t) for t in tau_grid_ns)):
            applicable = tau <= horizon_ns
            moves_t, half_t, _ok = _executed(clean, origins, targets, tau, granule)
            values.append(
                value_at_latency(
                    moves,
                    moves_t,
                    np.nan_to_num(half_t, nan=0.0),
                    k,
                    tau,
                    applicable=applicable,
                )
            )
        taus = sorted({int(t) for t in tau_grid_ns})
        capture = None
        if len(taus) >= 2 and taus[-1] > taus[0]:
            capture = _capture_floor_core(
                origins=origins,
                targets=targets,
                moves=moves,
                clean=clean,
                granule=granule,
                k=k,
                tau_fast_ns=taus[0],
                tau_slow_ns=taus[-1],
                horizon_ns=horizon_ns,
                power=0.80,
                alpha=0.05,
            )
        curves.append(
            BandValueCurve(
                k_ticks=k,
                horizon_ns=horizon_ns,
                balance=class_balance(moves, k),
                values=tuple(values),
                tau_half_ns=_tau_half(clean, origins, targets, moves, k, granule, horizon_ns),
                value_at_zero_ns=values[0].net_mean_ticks if values else float("nan"),
                capture=capture,
            )
        )

    return WallClockHorizon(
        horizon_ns=horizon_ns,
        n_observations=int(origins.size),
        duration_ns_p10=_quantile(durations, DURATION_QUANTILES[0]),
        duration_ns_p50=_quantile(durations, DURATION_QUANTILES[1]),
        duration_ns_p90=_quantile(durations, DURATION_QUANTILES[2]),
        zero_move_fraction=float(np.mean(abs_moves == 0.0)),
        abs_move_median_ticks=float(np.median(abs_moves)),
        abs_move_p95_ticks=_quantile(abs_moves, 95.0),
        move_skew_proxy=(float(np.mean(np.sign(nonzero))) if nonzero.size else 0.0),
        median_half_spread_ticks=float(np.nanmedian(clean.spread) / 2.0 / granule),
        events_per_horizon=float(np.mean(targets - origins)),
        curves=tuple(curves),
    )


def diagnose_wall_clock_session(
    stream: MarketStream,
    session: str,
    horizon_ns_grid: Sequence[int],
    ks: Sequence[int],
    tau_grid_ns: Sequence[int],
    role: str = "primary",
) -> WallClockDiagnostic:
    """Characterise every requested duration horizon on one session."""
    from latency_ml.finance.replay import stream_digest

    granule = price_granule(stream.mid)
    times = np.asarray(stream.times_ns, dtype=np.float64)
    gaps = np.diff(times[np.isfinite(times)])
    span = int(times[np.isfinite(times)].max() - times[np.isfinite(times)].min())
    return WallClockDiagnostic(
        session=session,
        role=role,
        n_events=len(stream),
        granule_units=granule,
        median_half_spread_ticks=float(np.nanmedian(stream.spread) / 2.0 / granule),
        median_event_gap_ns=float(np.median(gaps)),
        mean_event_gap_ns=float(np.mean(gaps)) if gaps.size else 0.0,
        session_span_ns=span,
        stream_digest=stream_digest(stream),
        horizons=tuple(
            characterise_wall_clock_horizon(stream, h, granule, ks, tau_grid_ns)
            for h in horizon_ns_grid
        ),
    )


def diagnose_session(
    stream: MarketStream,
    session: str,
    horizons: Sequence[int],
    ks: Sequence[int],
) -> SessionDiagnostic:
    """Characterise every requested horizon on one session."""
    from latency_ml.finance.replay import stream_digest

    granule = price_granule(stream.mid)
    times = stream.times_ns
    gaps = np.diff(times) if times.size > 1 else np.array([0])
    return SessionDiagnostic(
        session=session,
        n_events=len(stream),
        granule_units=granule,
        granule_note=(
            "derived as GCD of non-zero mid differences; mid is a touch mean so "
            "this is typically half the exchange tick"
        ),
        median_half_spread_ticks=float(np.nanmedian(stream.spread) / 2.0 / granule),
        median_event_gap_ns=float(np.median(gaps)),
        stream_digest=stream_digest(stream),
        horizons=tuple(characterise_horizon(stream, h, granule, ks) for h in horizons),
    )


def sessions_under(root: Path) -> list[Path]:
    """Discover LOBSTER session directories under ``root``.

    Depth is not selected here. It is a loader argument, and it must be passed
    explicitly at the call site: ``load_session`` with no level reads every level
    declared in the filename, which is 1.4 GB for the SPY_50 session, so a
    touch-only diagnostic that forgets would pay for depth it never looks at.
    """
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("*.csv")))


def format_report(
    diagnostics: Sequence[SessionDiagnostic],
    tau_grid_ns: Sequence[int] | None = None,
) -> str:
    """Render the diagnostic as tables.

    Two tables per session. The first is the event-versus-time view, because the
    single most useful thing to see is how long a horizon actually is next to the
    latency grid it will be swept against. The second is the target view: class
    balance and cost survival per band.
    """
    lines: list[str] = []
    max_tau = max(tau_grid_ns) if tau_grid_ns else 0

    for diag in diagnostics:
        lines.append("=" * 100)
        lines.append(f"{diag.session}")
        lines.append(
            f"  {diag.n_events:,} events | price granule {diag.granule_units} units "
            f"| median half-spread {diag.median_half_spread_ticks:.0f} granule(s)"
        )
        lines.append(
            f"  median event gap {diag.median_event_gap_ns / 1e6:.3f} ms | "
            f"digest {diag.stream_digest[:16]}"
        )
        if tau_grid_ns:
            lines.append(
                f"  execution latency grid {min(tau_grid_ns) / 1e6:g}-"
                f"{max_tau / 1e6:g} ms, swept independently of H"
            )
        lines.append("")
        header = (
            f"  {'H':>5} {'duration p10':>13} {'p50':>11} {'p90':>11} |"
            f" {'zero%':>6} {'|move| p50':>10} {'p95':>7} | {'H/tau_max':>9}"
        )
        lines.append("  horizon -> wall-clock duration (ms) and move distribution (granules)")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for h in diag.horizons:
            ratio = (h.duration_ns_p50 / max_tau) if max_tau else float("nan")
            lines.append(
                f"  {h.horizon_events:>5} {h.duration_ns_p10 / 1e6:>13.3f} "
                f"{h.duration_ns_p50 / 1e6:>11.3f} {h.duration_ns_p90 / 1e6:>11.3f} | "
                f"{h.zero_move_fraction * 100:>5.1f}% {h.abs_move_median_ticks:>10.1f} "
                f"{h.abs_move_p95_ticks:>7.1f} | {ratio:>9.1f}"
            )
        lines.append("")
        lines.append("  class balance and cost survival per integer band k (granules)")
        lines.append(
            f"  {'H':>5} {'k':>4} | {'BUY':>6} {'HOLD':>6} {'SELL':>6} | {'min n':>8} {'thin':>9} |"
            f" {'survive':>7} {'net/graspread':>13} {'net ticks':>9}"
        )
        lines.append("  " + "-" * 84)
        for h in diag.horizons:
            for bal, econ in zip(h.balances, h.economics, strict=True):
                smallest = min(bal.buy_events, bal.hold_events, bal.sell_events)
                lines.append(
                    f"  {h.horizon_events:>5} {bal.k_ticks:>4} | "
                    f"{bal.buy_fraction * 100:>5.1f}% {bal.hold_fraction * 100:>5.1f}% "
                    f"{bal.sell_fraction * 100:>5.1f}% | {smallest:>8,} {bal.thinness:>9} | "
                    f"{econ['surviving_fraction'] * 100:>6.2f}% "
                    f"{econ['net_mean_half_spreads']:>13.3f} {econ['net_mean_ticks']:>9.3f}"
                )
        lines.append("")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    """A percentage cell, blank rather than nan when there is nothing to report."""
    if value is None or not np.isfinite(value):
        return "".rjust(5)
    return f"{value * 100:.1f}%"


def format_wall_clock_report(
    diagnostics: Sequence[WallClockDiagnostic],
    tau_grid_ns: Sequence[int],
) -> str:
    """Render the target-selection table.

    Columns are the ones the selection criterion needs: class balance, the number
    of opportunities, post-cost survival, net outcome, the ratio of horizon to the
    largest delay, and the delay at which value crosses zero. The last column is
    the one that decides anything -- it says whether a latency sweep at this
    target could measure decay or would start after the value is already gone.
    """
    lines: list[str] = []
    tau_max = max(tau_grid_ns) if tau_grid_ns else 0
    for diag in diagnostics:
        lines.append("=" * 118)
        lines.append(f"{diag.session}  [{diag.role}]")
        lines.append(
            f"  {diag.n_events:,} events | price granule {diag.granule_units} units "
            f"| median half-spread {diag.median_half_spread_ticks:.0f} granule(s)"
        )
        lines.append(
            f"  median event gap {diag.median_event_gap_ns / 1e6:.3f} ms | "
            f"mean gap {diag.mean_event_gap_ns / 1e6:.2f} ms | "
            f"session span {diag.session_span_ns / 1e6:.0f} ms | "
            f"digest {diag.stream_digest[:16]}"
        )
        lines.append(
            f"  execution assumption: aggressive touch crossing (frozen for the first "
            f"result) | latency grid {min(tau_grid_ns) / 1e6:g}-{tau_max / 1e6:g} ms"
        )
        lines.append("")
        header = (
            f"  {'H (ms)':>7} {'k':>3} | {'BUY':>6} {'HOLD':>6} {'SELL':>6} | "
            f"{'n opp':>9} {'ev/ H':>7} {'zero%':>6} | "
            f"{'surv0':>6} {'net@0':>8} {'net/halfsp':>10} | "
            f"{'H/tau_max':>8} {'tau_half':>9} {'halflives':>9} {'sweep':>7}"
        )
        lines.append(
            "  wall-clock horizon x granule band: target viability under delayed execution"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for h in diag.horizons:
            for curve in h.curves:
                bal = curve.balance
                v0 = curve.values[0] if curve.values else None
                n_opp = v0.n_signals if v0 is not None else 0
                if not curve.viable_at_any_delay:
                    sweep = "no value"
                elif curve.tau_half_ns >= tau_max:
                    sweep = "full"
                elif curve.tau_half_ns >= min(tau_grid_ns):
                    sweep = "partial"
                else:
                    sweep = "step"
                half_ms = (
                    f"{curve.tau_half_ns / 1e6:.2f}" if np.isfinite(curve.tau_half_ns) else "inf"
                )
                half_lives = curve.half_lives_in_horizon
                lines.append(
                    f"  {h.horizon_ns / 1e6:>7.1f} {curve.k_ticks:>3} | "
                    f"{bal.buy_fraction * 100:>5.1f}% {bal.hold_fraction * 100:>5.1f}% "
                    f"{bal.sell_fraction * 100:>5.1f}% | {n_opp:>9,} "
                    f"{h.events_per_horizon:>7.0f} "
                    f"{h.zero_move_fraction * 100:>5.1f}% | "
                    f"{_pct(v0.surviving_fraction if v0 else None):>5} "
                    f"{(v0.net_mean_ticks if v0 else float('nan')):>8.3f} "
                    f"{(v0.net_mean_half_spreads if v0 else float('nan')):>10.3f} | "
                    f"{h.horizon_ns / tau_max if tau_max else float('nan'):>8.1f} "
                    f"{half_ms:>9} {half_lives:>9.2f} {sweep:>7}"
                )
        lines.append("")
        lines.append("  V(tau): net outcome in granules for a perfect forecaster, by delay")
        tau_hdr = (
            "  " + " " * 12 + "".join(f"tau={t / 1e6:g}ms".rjust(11) for t in sorted(tau_grid_ns))
        )
        lines.append(tau_hdr)
        lines.append("  " + "-" * (len(tau_hdr) - 2))
        for h in diag.horizons:
            for curve in h.curves:
                cells = []
                for tau in sorted(tau_grid_ns):
                    v = curve.value_at(tau)
                    if v is None or not v.applicable or not np.isfinite(v.net_mean_ticks):
                        cells.append("".rjust(11))
                    else:
                        cells.append(f"{v.net_mean_ticks:.3f}".rjust(11))
                lines.append(
                    f"  H={h.horizon_ns / 1e6:>6.1f}ms k={curve.k_ticks:<3}" + "".join(cells)
                )
        lines.append("")
        lines.append(
            f"  capture floor: share of oracle value a policy must capture for the "
            f"{min(tau_grid_ns) / 1e6:g}ms-vs-{max(tau_grid_ns) / 1e6:g}ms gap to clear "
            f"the MDE"
        )
        lines.append(
            f"  {'H (ms)':>7} {'k':>3} | {'gap':>7} {'n_eff':>7} {'SE':>7} {'MDE':>7} |"
            f" {'q_floor':>8} {'q=0.2':>7} {'q=0.5':>7} {'q=0.8':>7} {'verdict':>9}"
        )
        lines.append("  " + "-" * 88)
        for h in diag.horizons:
            for curve in h.curves:
                c = curve.capture
                if c is None:
                    continue
                gap = c.oracle_gap
                # q values are fixed, so the closure only closes over the row.
                verdicts = ["seen" if q * abs(gap) >= c.mde else "lost" for q in (0.2, 0.5, 0.8)]
                lines.append(
                    f"  {h.horizon_ns / 1e6:>7.0f} {curve.k_ticks:>3} | {gap:>7.3f} "
                    f"{c.n_effective:>7.0f} {c.standard_error:>7.4f} {c.mde:>7.4f} | "
                    f"{c.capture_floor:>8.3f} "
                    + " ".join(f"{v:>7}" for v in verdicts)
                    + f" {('oracle OK' if c.detectable_at_oracle else 'UNRESOLVABLE'):>9}"
                )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEGENERATE_CLASS_EVENTS",
    "THIN_CLASS_EVENTS",
    "BandValueCurve",
    "CaptureFloor",
    "ClassBalance",
    "HorizonStats",
    "LatencyValue",
    "SessionDiagnostic",
    "TargetDiagnosticError",
    "WallClockDiagnostic",
    "WallClockHorizon",
    "capture_floor",
    "characterise_horizon",
    "characterise_wall_clock_horizon",
    "class_balance",
    "diagnose_session",
    "diagnose_wall_clock_session",
    "economics_at",
    "format_report",
    "format_wall_clock_report",
    "price_granule",
    "sessions_under",
    "value_at_latency",
]


# ---------------------------------------------------------------------------
# How much of the oracle must a policy capture for latency to be observable?
# ---------------------------------------------------------------------------
#
# V(tau) is a ceiling. A trained policy earns some fraction q of it, so the
# latency gap the experiment can hope to see is
#
#     Delta(q) = q * (V(tau_fast) - V(tau_slow))
#
# which shrinks linearly in q. The question is where it crosses the noise floor,
# because a gap nobody can measure is the same as no gap: the training run would
# come back null and be uninterpretable.
#
# The noise floor is not sd/sqrt(n) over the reported observation count. The
# look-ahead windows overlap -- each origin's window covers most of its
# neighbour's -- so consecutive observations share nearly all of their events and
# are not independent draws. A session of 6.5 hours at a 10 s horizon contains
# about 2,300 non-overlapping windows, not 400,000 observations. Using the raw
# count would understate the standard error by a factor of about thirteen and
# would license a design that cannot resolve its own effect.


@dataclass(frozen=True)
class CaptureFloor:
    """The share of oracle value a policy must capture for latency to be visible."""

    k_ticks: int
    horizon_ns: int
    tau_fast_ns: int
    tau_slow_ns: int
    #: Windows the session actually offers.
    n_observations: int
    #: Non-overlapping windows: session span divided by horizon. The count the
    #: standard error is computed on.
    n_effective: float
    oracle_fast: float
    oracle_slow: float
    oracle_gap: float
    #: Spread of the per-window paired difference.
    gap_sd: float
    standard_error: float
    #: Smallest gap detectable at the stated power and alpha.
    mde: float
    #: Smallest q whose gap reaches the MDE. Above 1.0 means not even a perfect
    #: policy clears it on this much data.
    capture_floor: float
    power: float
    alpha: float
    detectable_at_oracle: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_ticks": self.k_ticks,
            "horizon_ns": self.horizon_ns,
            "tau_fast_ns": self.tau_fast_ns,
            "tau_slow_ns": self.tau_slow_ns,
            "n_observations": self.n_observations,
            "n_effective": self.n_effective,
            "oracle_fast": self.oracle_fast,
            "oracle_slow": self.oracle_slow,
            "oracle_gap": self.oracle_gap,
            "gap_sd": self.gap_sd,
            "standard_error": self.standard_error,
            "mde": self.mde,
            "capture_floor": self.capture_floor,
            "power": self.power,
            "alpha": self.alpha,
            "detectable_at_oracle": self.detectable_at_oracle,
        }


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, by bisection.

    A normal reference is the right one here rather than Student's t: the
    reference distribution is built from the session's own windows and the count
    behind ``n_effective`` is in the thousands, so the tail-weighting t would
    apply is negligible.
    """
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def capture_floor(
    stream: MarketStream,
    horizon_ns: int,
    granule: int,
    k: int,
    tau_fast_ns: int,
    tau_slow_ns: int,
    power: float = 0.80,
    alpha: float = 0.05,
) -> CaptureFloor:
    """Smallest share of oracle value that keeps the latency gap measurable.

    Reported per window in granule units, with the standard error taken over
    *non-overlapping* windows. A floor above 1.0 is the important output: it says
    the design cannot resolve its own effect at any model quality, and that is
    worth knowing before spending a training run rather than after.
    """
    if tau_slow_ns <= tau_fast_ns:
        raise ValueError(f"slow delay must exceed fast delay, got {tau_slow_ns} <= {tau_fast_ns}")
    clean = _clean_session(stream)
    q_h = _forward(clean.times, horizon_ns)
    origins = np.flatnonzero(q_h >= 0).astype(np.int64)
    targets = q_h[origins]
    moves = (clean.mid[targets] - clean.mid[origins]) / granule
    return _capture_floor_core(
        origins=origins,
        targets=targets,
        moves=moves,
        clean=clean,
        granule=granule,
        k=k,
        tau_fast_ns=tau_fast_ns,
        tau_slow_ns=tau_slow_ns,
        horizon_ns=horizon_ns,
        power=power,
        alpha=alpha,
    )


def _capture_floor_core(
    *,
    origins: NDArray[np.int64],
    targets: NDArray[np.int64],
    moves: NDArray[np.float64],
    clean: _Clean,
    granule: int,
    k: int,
    tau_fast_ns: int,
    tau_slow_ns: int,
    horizon_ns: int,
    power: float,
    alpha: float,
) -> CaptureFloor:
    """The arithmetic behind :func:`capture_floor`, over already-built arrays."""
    fast_moves, fast_half, fast_ok = _executed(clean, origins, targets, tau_fast_ns, granule)
    slow_moves, slow_half, slow_ok = _executed(clean, origins, targets, tau_slow_ns, granule)
    ok = fast_ok & slow_ok
    if not ok.any():
        raise TargetDiagnosticError("no window has both a fast and a slow execution")

    # The per-window P&L a perfect forecaster earns at each delay. HOLD windows
    # earn zero at both, which is what makes the difference attributable to the
    # signalled subset alone.
    acted = (moves >= k) | (moves <= -k)

    def _payoff(exec_moves: NDArray[np.float64], half: NDArray[np.float64]) -> NDArray[np.float64]:
        remaining = moves - exec_moves  # signed move from the fill to t_H
        signed = np.where(moves >= k, remaining, np.where(moves <= -k, -remaining, 0.0))
        # The spread is paid on every signal that leaves, including one that fills
        # at t_H and banks nothing.
        return np.where(acted, signed - np.nan_to_num(half, nan=0.0), 0.0)

    fast_pnl = np.where(ok, _payoff(fast_moves, fast_half), 0.0)
    slow_pnl = np.where(ok, _payoff(slow_moves, slow_half), 0.0)
    gap = fast_pnl - slow_pnl

    span_ns = int(clean.times[-1] - clean.times[0])
    n_effective = max(1.0, span_ns / horizon_ns)
    sd = float(np.std(gap, ddof=1))
    se = sd / math.sqrt(n_effective)
    mde = (_normal_quantile(1.0 - alpha / 2.0) + _normal_quantile(power)) * se
    oracle_gap = float(np.mean(gap))
    floor = mde / abs(oracle_gap) if oracle_gap != 0.0 else float("inf")
    return CaptureFloor(
        k_ticks=k,
        horizon_ns=horizon_ns,
        tau_fast_ns=tau_fast_ns,
        tau_slow_ns=tau_slow_ns,
        n_observations=int(ok.sum()),
        n_effective=n_effective,
        oracle_fast=float(np.mean(fast_pnl)),
        oracle_slow=float(np.mean(slow_pnl)),
        oracle_gap=oracle_gap,
        gap_sd=sd,
        standard_error=se,
        mde=mde,
        capture_floor=floor,
        power=power,
        alpha=alpha,
        detectable_at_oracle=floor <= 1.0,
    )
