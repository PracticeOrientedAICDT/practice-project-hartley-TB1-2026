"""The prediction target, defined once.

Why this module exists
----------------------
The target was chosen by a diagnostic and then has to be fitted by a trainer. For
that to mean anything, both must compute it the *same* way -- and for a while they
did not. ``target_diag.class_balance`` compares inclusively, ``stage_a.build_labels``
compares strictly, and a quantile-derived band against an explicit one. Two
implementations of one specification is not a specification, and the divergence is
silent: both produce three classes, both pass a shape check, and they disagree only
on the boundaries that decide whether a band is degenerate.

So the band lives here, once, and both callers use it. If the diagnostic that
selected the target and the trainer that fits it ever disagree about a label, the
freeze in ``PROTOCOL.md`` section 12 is void and this file is where that shows up.

The horizon is a duration
-------------------------
``H`` is nanoseconds, not a count of events. Events are an artefact of how the
replay is chunked: AAPL's median event gap is 0.639 ms and its mean is 58.44 ms,
so the same event count names windows of wildly different duration and the same
millisecond horizon buys very different numbers of events on different symbols.
The resolution rule is the one fixed in the protocol::

    t_H = min{ t_i : t_i - t_0 >= H }

Note ``>=``. An origin whose successor lands exactly on the deadline qualifies, so
a horizon that coincides with an integer number of event gaps spans exactly that
many events rather than one fewer.

The band is explicit, not a quantile
------------------------------------
A band of ``k`` granules, inclusive on both sides::

    BUY  if move >= k
    SELL if move <= -k
    HOLD otherwise

Inclusive matters. With strict inequalities a band of zero leaves every unmoved
window labelled HOLD, so the three-class problem silently collapses to
``sign(move)`` with HOLD absorbing the whole mass -- 95% HOLD on one of these
sessions. The earlier quantile formulation reached exactly that, because a
step-function mid puts most of its mass at zero and any quantile above that mass
lands there. See ``target_diag`` for the measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Actions, matching ``stage_a`` and ``policy.hft_output_actions``.
HOLD: Final[int] = 0
BUY: Final[int] = 1
SELL: Final[int] = 2
N_ACTIONS: Final[int] = 3

#: Frozen by PROTOCOL.md section 12, before any training.
#:
#: Primary: AAPL, 10 s, 6 granules. Chosen for enough value, enough observations,
#: a 26% HOLD class, and 14.6% oracle decay across a 0.1-100 ms sweep -- a regime
#: where latency costs something without annihilating the task.
FROZEN_PRIMARY_HORIZON_NS: Final[int] = 10_000_000_000
FROZEN_PRIMARY_BAND_TICKS: Final[int] = 6

#: Secondary: AAPL, 5 s, 6 granules. Same model, execution and latency protocol;
#: a deliberately latency-sensitive arm at 47.7% oracle decay. Better identified
#: than the primary rather than less -- larger gap and twice the windows.
FROZEN_SECONDARY_HORIZON_NS: Final[int] = 5_000_000_000
FROZEN_SECONDARY_BAND_TICKS: Final[int] = 6

#: Held out: AMZN at its own viable horizon, 25 s.
#:
#: Named here rather than constructed at a call site because it is part of the
#: freeze. AMZN is *not* run at AAPL's 10 s: its own viability table does not reach
#: a positive ceiling at 10 s for k=1 or k=2, so matching the horizon would mean
#: reporting a null for a target that was never viable. The comparison is therefore
#: cross-symbol rather than like-for-like, and is labelled as such. Its latency
#: decay is also materially gentler -- 10.0% against AAPL's 14.6% -- so it is the
#: weaker of the two on the question being asked, which is stated in the protocol
#: rather than left for a reader to discover.
FROZEN_HELDOUT_HORIZON_NS: Final[int] = 25_000_000_000
FROZEN_HELDOUT_BAND_TICKS: Final[int] = 6


class TargetError(ValueError):
    """Raised when a target specification is unusable."""


@dataclass(frozen=True)
class TargetSpec:
    """A prediction target: a look-ahead duration and an explicit band.

    Attributes:
        horizon_ns: Look-ahead window in nanoseconds. Must be positive.
        band_ticks: HOLD band in price granules. Must be at least one, because a
            zero band is not a narrow band -- see the module docstring.
    """

    horizon_ns: int
    band_ticks: int

    def __post_init__(self) -> None:
        if self.horizon_ns <= 0:
            raise TargetError(f"horizon must be positive, got {self.horizon_ns} ns")
        if self.band_ticks < 1:
            raise TargetError(
                f"band must be at least one granule, got {self.band_ticks}; a zero band "
                "collapses the three-class problem to sign(move)"
            )

    @property
    def horizon_ms(self) -> float:
        return self.horizon_ns / 1e6

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "horizon_ns": self.horizon_ns,
            "horizon_ms": self.horizon_ms,
            "band_ticks": self.band_ticks,
        }


#: The primary arm, as frozen.
FROZEN_PRIMARY: Final[TargetSpec] = TargetSpec(
    horizon_ns=FROZEN_PRIMARY_HORIZON_NS, band_ticks=FROZEN_PRIMARY_BAND_TICKS
)
#: The secondary arm, as frozen.
FROZEN_SECONDARY: Final[TargetSpec] = TargetSpec(
    horizon_ns=FROZEN_SECONDARY_HORIZON_NS, band_ticks=FROZEN_SECONDARY_BAND_TICKS
)
#: The held-out arm, as frozen. Applied to AMZN, not to AAPL.
FROZEN_HELDOUT: Final[TargetSpec] = TargetSpec(
    horizon_ns=FROZEN_HELDOUT_HORIZON_NS, band_ticks=FROZEN_HELDOUT_BAND_TICKS
)


def price_granule(mid: NDArray[np.float64]) -> int:
    """Smallest non-zero unit the mid-price moves in.

    Derived as the GCD of observed non-zero differences rather than assumed,
    because no price scale is recorded anywhere in the loader. A mid is the mean of
    a touch whose spread is usually an odd number of ticks, so this is typically
    *half* the exchange tick -- which is why hardcoding the tick would halve every
    move and quietly inflate every move-based statistic.
    """
    finite = mid[np.isfinite(mid)]
    diffs = np.diff(finite)
    nonzero = np.unique(diffs[diffs != 0.0])
    if nonzero.size == 0:
        raise TargetError("mid-price never moves; no granule is defined")
    granule = reduce(math.gcd, (abs(round(x)) for x in nonzero), 0)
    if granule <= 0:  # pragma: no cover - unreachable given the check above
        raise TargetError(f"degenerate price granule {granule}")
    return granule


def forward_index(times_ns: NDArray[np.int64], delta_ns: int) -> NDArray[np.int64]:
    """For each position, the first position at least ``delta_ns`` later.

    ``-1`` where the session ends first. Two-pointer rather than a search per
    origin, so a full session costs O(n). The clock is non-decreasing, so
    ``times + delta`` is sorted and ``searchsorted`` can be vectorised.
    """
    n = times_ns.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if delta_ns <= 0:
        return np.arange(n, dtype=np.int64)
    pos = np.searchsorted(times_ns, times_ns + delta_ns, side="left")
    return np.where(pos < n, pos, -1).astype(np.int64)


def label_moves(moves: NDArray[np.float64], band_ticks: int) -> NDArray[np.int64]:
    """Three-class labels from signed moves, inclusive on both sides.

    The one definition of the band in this repository. Inclusive so that a window
    moving exactly ``+k`` granules is BUY rather than HOLD, and a band of one
    granule cannot degenerate into ``sign(move)`` with a huge HOLD class.
    """
    if band_ticks < 1:
        raise TargetError(f"band must be at least one granule, got {band_ticks}")
    buy = moves >= band_ticks
    sell = moves <= -band_ticks
    return np.where(buy, BUY, np.where(sell, SELL, HOLD)).astype(np.int64)


def horizon_moves(
    mid: NDArray[np.float64],
    times_ns: NDArray[np.int64],
    horizon_ns: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Forward moves over a duration horizon, with the origins that have one.

    Returns ``(origins, targets, moves)`` in *original* stream positions, so a
    caller can scatter them back onto a full-length label array. Origins whose
    successor does not exist, or whose clock is unusable, are omitted rather than
    padded: a locked book yields NaN timestamps, and those windows have no
    deadline to hit.
    """
    if horizon_ns <= 0:
        raise TargetError(f"horizon must be positive, got {horizon_ns} ns")
    finite = np.isfinite(np.asarray(times_ns, dtype=np.float64)) & np.isfinite(mid)
    idx = np.flatnonzero(finite).astype(np.int64)
    if idx.size == 0:
        raise TargetError("no usable events: every timestamp or mid is non-finite")

    clock = np.asarray(times_ns, dtype=np.float64)[idx].astype(np.int64)
    targets = forward_index(clock, horizon_ns)
    keep = targets >= 0
    origins = idx[keep]
    resolved = idx[targets[keep]]
    moves = (mid[resolved] - mid[origins]).astype(np.float64)
    return origins, resolved, moves


def label_stream(
    mid: NDArray[np.float64],
    times_ns: NDArray[np.int64],
    spec: TargetSpec,
    granule: int | None = None,
) -> tuple[NDArray[np.int64], dict[str, float | int]]:
    """Full-length labels for a stream, aligned one-to-one with its events.

    Windows with no resolvable successor are HOLD, which is honest: there is no
    evidence about them and they must not be trained on as if there were.

    Returns the labels and a summary, the latter so a caller can record what was
    actually fitted rather than what was requested -- an event-horizon target and a
    duration target ask for the same number of events and get very different ones.
    """
    if granule is None:
        granule = price_granule(mid)
    origins, _targets, moves = horizon_moves(mid, times_ns, spec.horizon_ns)
    labels = np.full(mid.shape[0], HOLD, dtype=np.int64)
    labels[origins] = label_moves(moves / granule, spec.band_ticks)
    resolved_labels = labels[origins]
    summary: dict[str, float | int] = {
        "horizon_ns": spec.horizon_ns,
        "band_ticks": spec.band_ticks,
        "granule_units": granule,
        "n_labelled": int(origins.size),
        "buy_fraction": float(np.mean(resolved_labels == BUY)) if origins.size else 0.0,
        "hold_fraction": float(np.mean(resolved_labels == HOLD)) if origins.size else 0.0,
        "sell_fraction": float(np.mean(resolved_labels == SELL)) if origins.size else 0.0,
    }
    return labels, summary


__all__ = [
    "BUY",
    "FROZEN_HELDOUT",
    "FROZEN_HELDOUT_BAND_TICKS",
    "FROZEN_HELDOUT_HORIZON_NS",
    "FROZEN_PRIMARY",
    "FROZEN_PRIMARY_BAND_TICKS",
    "FROZEN_PRIMARY_HORIZON_NS",
    "FROZEN_SECONDARY",
    "FROZEN_SECONDARY_BAND_TICKS",
    "FROZEN_SECONDARY_HORIZON_NS",
    "HOLD",
    "N_ACTIONS",
    "SELL",
    "TargetError",
    "TargetSpec",
    "forward_index",
    "horizon_moves",
    "label_moves",
    "label_stream",
    "price_granule",
]
