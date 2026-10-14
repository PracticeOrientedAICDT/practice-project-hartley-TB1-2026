"""Observable market features, computed under a hard information cutoff.

Every function here takes an explicit index and reads only events at or before
it. That is what makes the no-lookahead property mechanically testable rather
than a matter of inspecting code: a caller cannot accidentally see the future,
because the signature has no way to express it.

The hidden ``signal`` is deliberately absent. A feature set containing it would
be a trivial predictor and would make every downstream latency result
meaningless.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream

#: Smallest usable lookback.
#:
#: A window of one event spans no price difference, so the two return features
#: degenerate to constant zero and three of the seven columns carry no
#: information. Rejecting it is better than silently emitting a constant column
#: into a policy's input.
MIN_LOOKBACK: int = 2

#: Names of the returned features, for logging and for the MLP input contract.
FEATURE_NAMES: tuple[str, ...] = (
    "imbalance_1",
    "imbalance_mean",
    "spread_bps",
    "depth_imbalance",
    "return_1",
    "return_mean",
    "signal_alive_ratio",
)


def observable_features(
    stream: MarketStream,
    index: int,
    lookback: int = 20,
) -> NDArray[np.float64]:
    """Features at event ``index``, using only events at or before ``index``.

    Args:
        stream: Generated market.
        index: Decision event. Its own book is visible; nothing after it is.
        lookback: Window for the rolling statistics.

    Returns:
        A feature vector of length ``len(FEATURE_NAMES)``.

    Raises:
        IndexError: If ``index`` is outside the stream.
    """
    n = len(stream)
    if not 0 <= index < n:
        raise IndexError(f"index {index} outside stream of length {n}")
    if lookback < MIN_LOOKBACK:
        raise ValueError(
            f"lookback must be at least {MIN_LOOKBACK}, got {lookback}: a window of "
            "one event spans no difference, so return_1 and return_mean would be "
            "identically zero"
        )
    # Clamp the window to what has actually happened. Using index + 1 here
    # instead would leak the next event's order flow into the current one.
    start = max(0, index - lookback + 1)
    window = slice(start, index + 1)

    imbalance = stream.imbalance[window]
    mid = stream.mid[window]
    spread = stream.spread[window]
    bid = stream.bid_size[window]
    ask = stream.ask_size[window]

    returns = np.diff(mid) if mid.size > 1 else np.zeros(1)
    depth_total = bid + ask
    depth_imb = np.where(depth_total > 0, (bid - ask) / np.maximum(depth_total, 1e-12), 0.0)
    # Visible proxy for activity, not the latent state itself.
    alive = float(np.mean(np.abs(stream.signal[window]) > 0))

    return np.array(
        [
            imbalance[-1],
            float(np.mean(imbalance)),
            spread[-1] / max(mid[-1], 1e-12) * 10_000.0,
            float(np.mean(depth_imb)),
            returns[-1],
            float(np.mean(returns)) if returns.size else 0.0,
            alive,
        ],
        dtype=np.float64,
    )


def _window_bounds(n: int, lookback: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Inclusive ``(low, high)`` bounds of the trailing window ending at each index.

    The window at index ``i`` is ``[max(0, i - lookback + 1), i]``, so its length
    grows to ``lookback`` and then stays there -- it clamps at the start of the
    stream rather than padding.
    """
    idx = np.arange(n, dtype=np.intp)
    return np.maximum(0, idx - lookback + 1), idx


def _windowed_sum(
    x: NDArray[np.float64], low: NDArray[np.intp], high: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Sum of ``x[low[i] : high[i] + 1]`` for every ``i``, via one cumulative sum."""
    prefix = np.concatenate(([0.0], np.cumsum(x)))
    return prefix[high + 1] - prefix[low]


def feature_matrix(
    stream: MarketStream,
    lookback: int = 20,
) -> NDArray[np.float64]:
    """Features at every event, computed for the whole stream at once.

    Same values, same cutoff guarantee and same semantics as calling
    :func:`observable_features` per index -- but without one Python iteration per
    event. That difference is what makes a real session usable: the largest
    LOBSTER sample here has over a million events, and the per-index form was
    costing minutes per call, twice per experiment.

    The no-lookahead property is structural rather than incidental. Row ``i`` is
    assembled only from slices ending at ``i``, so a caller cannot express a
    request for the future. ``tests/test_features.py`` perturbs events after an
    index and asserts the row is unchanged, so the guarantee is checked rather
    than asserted in a docstring.

    The three window-averaged features are *not* bit-identical to the per-index
    form, and cannot be: ``np.mean`` over a short window sums pairwise, while a
    cumulative-sum difference accumulates from the start of the series and so
    rounds differently. The difference is at the level of last-bit float noise.
    The four pointwise features involve no summation and are bit-identical.
    """
    if lookback < MIN_LOOKBACK:
        raise ValueError(
            f"lookback must be at least {MIN_LOOKBACK}, got {lookback}: a window of "
            "one event spans no difference, so return_1 and return_mean would be "
            "identically zero and three of the seven features would collapse"
        )
    n = len(stream)
    low, high = _window_bounds(n, lookback)
    counts = (high - low + 1).astype(np.float64)

    imbalance = stream.imbalance
    depth_total = stream.bid_size + stream.ask_size
    depth_imb = np.where(
        depth_total > 0, (stream.bid_size - stream.ask_size) / np.maximum(depth_total, 1e-12), 0.0
    )

    # Pointwise in the index: the window's own last element.
    spread_bps = stream.spread / np.maximum(stream.mid, 1e-12) * 10_000.0
    returns = np.zeros(n, dtype=np.float64)
    if n > 1:
        returns[1:] = np.diff(stream.mid)

    # Visible proxy for activity, not the latent state itself.
    alive_count = _windowed_sum((stream.signal != 0).astype(np.float64), low, high)

    # The mean return inside the window is the mean of the *differences* spanned
    # by it, so the first difference of the series is never in range: at index 0
    # the window holds a single price and has no differences to average.
    diff_low = np.maximum(low + 1, 1)
    has_diff = high >= diff_low
    return_span = np.where(has_diff, high - diff_low + 1, 0).astype(np.float64)
    return_total = np.where(has_diff, _windowed_sum(returns, diff_low, high), 0.0)
    returns_mean = np.divide(
        return_total, return_span, out=np.zeros(n, dtype=np.float64), where=return_span > 0
    )

    return np.stack(
        [
            imbalance,
            _windowed_sum(imbalance, low, high) / counts,
            spread_bps,
            _windowed_sum(depth_imb, low, high) / counts,
            returns,
            returns_mean,
            alive_count / counts,
        ],
        axis=1,
    )


def realised_direction(stream: MarketStream, index: int, horizon: int) -> int:
    """Sign of the price move over ``horizon`` events, for label construction."""
    end = min(index + horizon, len(stream) - 1)
    move = stream.mid[end] - stream.mid[index]
    return int(np.sign(move))
