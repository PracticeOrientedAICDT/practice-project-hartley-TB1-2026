"""LOBSTER historical order-book loader.

The synthetic market in :mod:`latency_ml.finance.market` exists to prove that a
latency-to-value relationship *can* arise from a market mechanism. It cannot
establish that the relationship occurs in a real order book. This module is the
bridge: it converts a LOBSTER session into the same :class:`MarketStream` the
rest of the pipeline consumes, so the fixed policy, the fixed execution model and
the fixed cost model are applied unchanged.

The historical stream therefore plugs into exactly the same latency sweep, and
any difference in the resulting curve is attributable to the market rather than
to the experiment harness.

## The format, as LOBSTER documents it

Every session directory ships ``LOBSTER_SampleFiles_ReadMe.txt``, which is
authoritative. Two things about it are easy to get wrong, and both were wrong in
an earlier version of this module:

**The order-book file has no timestamp column.** It is ``N x 4L`` columns --
price *and* size, interleaved per level:

.. code-block:: text

    ask_price_1, ask_size_1, bid_price_1, bid_size_1,
    ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...

The event timeline lives entirely in the message file. A version of this loader
that expected a leading time column, with prices and sizes in separate blocks,
would reject every real file while passing against its own fixtures.

**Only some message rows move the book.** Event types ``5`` (execution of a
hidden limit order) and ``7`` (trading halt indicator) are written with the
*preceding* book state duplicated, and with price ``-1``. Counting them as fresh
book observations would attribute a transition to the market that did not
happen.

## What this loader does not do

It does not claim LOBSTER data is present. Sample sessions are fetched under
LOBSTER's own terms and must be supplied to this repository; nothing here
downloads or embeds them. It also does not rescale prices, infer a missing
spread, or smooth anything -- a silently repaired order book would not be the
historical record it claims to be.

Format reference: https://lobsterdata.com/ (DataStructure documentation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream

#: Price LOBSTER writes into an unoccupied ask level.
EMPTY_ASK_PRICE: float = 9_999_999_999.0

#: Price LOBSTER writes into an unoccupied bid level.
EMPTY_BID_PRICE: float = -9_999_999_999.0

#: Message types whose book row repeats the previous state instead of advancing
#: it. Type 5 is an execution against a hidden order; type 7 is a halt.
NON_TRANSITION_TYPES: frozenset[int] = frozenset({5, 7})

#: LOBSTER direction codes.
BUY_DIRECTION: int = 1
SELL_DIRECTION: int = -1

#: Columns per level in an order-book file: ask price, ask size, bid price,
#: bid size.
COLUMNS_PER_LEVEL: int = 4

#: Columns in a message file: time, type, order id, size, price, direction.
MESSAGE_COLUMNS: int = 6

#: ``..._orderbook_30.csv`` -> 30
_LEVEL_SUFFIX = re.compile(r"_(\d+)\.csv$")

#: Price levels read by default.
#:
#: :class:`MarketStream` carries the touch only -- best bid, best ask and their
#: sizes -- so deeper levels would be parsed, held in memory, and discarded.
#: Raise this when a depth feature needs them; see :func:`load_order_book` for
#: what that costs.
DEFAULT_LEVELS: int = 1


class LobsterFormatError(ValueError):
    """Raised when a LOBSTER file does not match the documented layout."""


@dataclass(frozen=True)
class OrderBook:
    """Parsed order-book content, one row per message event.

    Unoccupied levels keep LOBSTER's sentinel prices and zero sizes rather than
    being silently rewritten. Masking is a decision about what a *stream* means,
    not something to do to the raw record.

    Attributes:
        ask_prices: ``(n_events, levels)``.
        ask_sizes: ``(n_events, levels)``.
        bid_prices: ``(n_events, levels)``.
        bid_sizes: ``(n_events, levels)``.
    """

    ask_prices: NDArray[np.float64]
    ask_sizes: NDArray[np.float64]
    bid_prices: NDArray[np.float64]
    bid_sizes: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.ask_prices.shape[0])

    @property
    def levels(self) -> int:
        return int(self.ask_prices.shape[1])


@dataclass(frozen=True)
class Messages:
    """Parsed message file: the event timeline.

    Attributes:
        times_s: Event timestamps in seconds after midnight.
        event_types: LOBSTER event type codes.
        directions: Signed direction, clipped into ``{-1, 0, +1}`` where 0 marks
            a row LOBSTER left without a usable side.
        is_transition: Whether this row advanced the book. False for the
            hidden-execution and halt rows.
    """

    times_s: NDArray[np.float64]
    event_types: NDArray[np.int64]
    directions: NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.times_s.shape[0])

    @property
    def is_transition(self) -> NDArray[np.bool_]:
        mask = np.ones(len(self), dtype=np.bool_)
        for code in NON_TRANSITION_TYPES:
            mask &= self.event_types != code
        return mask


def levels_in_filename(path: Path) -> int | None:
    """Number of levels a LOBSTER filename declares, or ``None`` if absent.

    The level is a property of the published file, not a constant of this
    module. Sessions arrive at whatever depth was requested -- the sample data
    here contains levels 10, 30 and 50 -- so the only two files in a directory
    are ``*_orderbook_<L>.csv`` and ``*_message_<L>.csv`` for the *same* ``L``.
    """
    match = _LEVEL_SUFFIX.search(path.name)
    return int(match.group(1)) if match else None


def _require_files(path: Path, available_levels: set[int]) -> tuple[Path, Path]:
    """Locate the order-book and message files for a session directory."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory")

    books: dict[int, Path] = {}
    for candidate in sorted(directory.glob("*_orderbook_*.csv")):
        level = levels_in_filename(candidate)
        if level is not None:
            books[level] = candidate
    messages: dict[int, Path] = {}
    for candidate in sorted(directory.glob("*_message_*.csv")):
        level = levels_in_filename(candidate)
        if level is not None:
            messages[level] = candidate

    if not books:
        raise FileNotFoundError(
            f"no *_orderbook_<levels>.csv in {directory}; expected a LOBSTER "
            "level-N order-book file"
        )
    if not messages:
        raise FileNotFoundError(
            f"no *_message_<levels>.csv in {directory}; expected the message "
            "file, which defines the event timeline"
        )

    shared = sorted(set(books) & set(messages))
    if not shared:
        raise LobsterFormatError(
            f"{directory} has order-book files at levels {sorted(books)} but message "
            f"files at levels {sorted(messages)}; a session needs both at one level"
        )
    level = available_levels.pop() if available_levels else shared[0]
    if level not in books:
        raise FileNotFoundError(f"{directory} has no level-{level} order-book file")
    return books[level], messages[level]


def load_order_book(path: Path, levels: int | None = None) -> OrderBook:
    """Parse a LOBSTER order-book file.

    Layout is ``4 * levels`` columns, interleaved per level, with **no**
    timestamp column -- the timeline comes from :func:`load_messages`.

    Only the requested ``levels`` are read. Parsing every level of a deep file is
    not free: the full level-50 book is four float64 arrays of
    ``n_events x 50``, which for the largest sample session here is over a
    gigabyte. Reading a subset keeps a whole-session load in tens of megabytes.

    Args:
        path: The ``*_orderbook_<L>.csv`` file.
        levels: How many levels to read. ``None`` reads every level the file
            declares.

    Returns:
        The parsed book.

    Raises:
        FileNotFoundError: If the file is absent.
        LobsterFormatError: If a row's width disagrees with the declared depth,
            which in practice means a header row or the wrong file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. LOBSTER data is not bundled with this repository; "
            "fetch sample sessions under LOBSTER's terms and point the loader at "
            "them. See the module docstring."
        )

    declared = levels_in_filename(path)
    available = _declared_columns(path)
    if declared is not None and declared != available // COLUMNS_PER_LEVEL:
        raise LobsterFormatError(
            f"{path.name} declares {declared} levels but has {available} columns, "
            f"which is {available / COLUMNS_PER_LEVEL:g} levels at "
            f"{COLUMNS_PER_LEVEL} columns each"
        )

    wanted = available // COLUMNS_PER_LEVEL if levels is None else int(levels)
    if not 0 < wanted <= available // COLUMNS_PER_LEVEL:
        raise LobsterFormatError(
            f"asked for {wanted} levels of {path.name}, which holds "
            f"{available // COLUMNS_PER_LEVEL}"
        )

    # ask price, ask size, bid price, bid size, per level, in that order.
    usecols = tuple(range(wanted * COLUMNS_PER_LEVEL))
    table = _read_numeric(path, usecols, available, np.dtype(np.int64))

    ask_prices = _as_float(table[:, 0 : 4 * wanted : 4])
    ask_sizes = _as_float(table[:, 1 : 4 * wanted : 4])
    bid_prices = _as_float(table[:, 2 : 4 * wanted : 4])
    bid_sizes = _as_float(table[:, 3 : 4 * wanted : 4])

    return OrderBook(
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
    )


def load_messages(path: Path) -> Messages:
    """Parse a LOBSTER message file: ``time, type, order_id, size, price, direction``.

    Returns:
        Times in seconds, event types, and directions clipped into
        ``{-1, 0, +1}``.

    Raises:
        FileNotFoundError: If the file is absent.
        LobsterFormatError: If a row has the wrong width, which in practice
            means a header row.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; expected the LOBSTER message file")

    table = _read_numeric(path, (0, 1, 5), MESSAGE_COLUMNS, np.dtype(np.float64))
    times_s = _as_float(table[:, 0])
    event_types = table[:, 1].astype(np.int64)
    directions = np.clip(table[:, 2].astype(np.int64), SELL_DIRECTION, BUY_DIRECTION).astype(
        np.int8
    )
    return Messages(times_s=times_s, event_types=event_types, directions=directions)


def _declared_columns(path: Path) -> int:
    """Column count of the first data row.

    Read rather than assumed, because the declared level and the actual width
    are independent claims and a disagreement between them is exactly the kind
    of thing that should stop the load rather than be reinterpreted.
    """
    with path.open() as handle:
        for line in handle:
            if line.strip():
                width = line.count(",") + 1
                if width == 1 and line.strip().isalpha():
                    raise LobsterFormatError(
                        f"{path.name} starts with {line.strip()!r} rather than a "
                        "number; LOBSTER files are headerless"
                    )
                return width
    raise LobsterFormatError(f"{path.name} contains no rows")


def _read_numeric(
    path: Path,
    usecols: tuple[int, ...],
    total_columns: int,
    dtype: np.dtype[Any],
) -> NDArray[np.float64]:
    """Read selected columns of a headerless numeric CSV.

    Order-book cells are published as integers and are read as int64, widening
    only afterwards; message timestamps are fractional seconds and are read as
    float64. Materialising the parsed text as Python objects instead would cost
    one object per cell -- a quarter of a billion for the largest session here --
    for no benefit.
    """
    actual = _declared_columns(path)
    if actual != total_columns:
        raise LobsterFormatError(
            f"{path.name} has {actual} columns, expected {total_columns}; "
            "LOBSTER files are headerless and fixed-width, so a header row or "
            "the wrong file is the usual cause"
        )
    try:
        return np.loadtxt(
            path,
            delimiter=",",
            usecols=usecols,
            dtype=dtype,
            ndmin=2,
        )
    except ValueError as exc:
        raise LobsterFormatError(
            f"{path.name} is not parseable as {total_columns} numeric columns: {exc}"
        ) from exc


def _as_float(table: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(table, dtype=np.float64)


def session_to_stream(
    book: OrderBook,
    messages: Messages,
) -> MarketStream:
    """Convert a LOBSTER session into the pipeline's :class:`MarketStream`.

    The derived quantities are the standard ones and no others:

    * ``mid`` is the arithmetic mid of the touch, not a traded price. LOBSTER's
      raw data has no trade price at this granularity that is free of lookahead.
    * ``spread`` is the full touch spread. A crossed, empty or sentinel-filled
      touch yields NaN rather than a fabricated zero, so a session with unusable
      quotes is visible in the results instead of silently contributing
      zero-cost trades. This matters more than it looks: LOBSTER pads
      unoccupied levels with ``+/-9999999999``, and a spread computed across one
      of those is enormous and positive, which no positivity check would catch.
    * ``buy_fraction`` is the buy-side share of order flow. Rows that did not
      move the book -- hidden executions and halts, which LOBSTER writes with a
      direction but without a state change -- carry the previous row's value
      forward instead of contributing a spurious buy or sell.

    The hidden ``signal`` is set to zero everywhere. Real books do not carry
    ground truth, and inventing one would be the single most damaging thing this
    loader could do: :func:`latency_ml.finance.features.observable_features`
    derives ``signal_alive_ratio`` from it, so a fabricated signal would feed a
    fabricated feature into the policy.

    Args:
        book: Parsed order-book content.
        messages: Parsed message file, one row per book row.

    Returns:
        A stream ready for the fixed-policy latency sweep.

    Raises:
        LobsterFormatError: If the two files disagree on event count.
    """
    n = len(messages)
    if len(book) != n:
        raise LobsterFormatError(
            f"order book has {len(book)} rows but message file has {n}; "
            "LOBSTER pairs these one to one"
        )

    best_ask = book.ask_prices[:, 0]
    best_bid = book.bid_prices[:, 0]

    # A sentinel in either touch means the level is not populated. Treat it as
    # unusable rather than as a price.
    ask_ok = np.isfinite(best_ask) & (best_ask < EMPTY_ASK_PRICE)
    bid_ok = np.isfinite(best_bid) & (best_bid > EMPTY_BID_PRICE)
    mid = np.where(ask_ok & bid_ok, 0.5 * (best_ask + best_bid), np.nan)
    spread = np.where(ask_ok & bid_ok, best_ask - best_bid, np.nan)
    # Do not clamp a non-positive spread to zero: a crossed book is a broken
    # book, and presenting it as a free trade would understate every cost.
    spread = np.where(np.isfinite(spread) & (spread > 0.0), spread, np.nan)

    times_ns = (messages.times_s * 1e9).astype(np.int64)
    # Timestamps must be non-decreasing for the latency replay's event indexing
    # to mean anything; LOBSTER publishes them in order, but two events can
    # share a timestamp and the replay must not go backwards.
    times_ns = np.maximum.accumulate(times_ns)

    return MarketStream(
        times_ns=times_ns,
        mid=mid,
        # No ground truth exists in historical data. Zero means "unknown", and
        # features must not treat it as a real latent state.
        signal=np.zeros(n, dtype=np.int8),
        buy_fraction=_buy_fraction(messages),
        spread=spread,
        bid_size=book.bid_sizes[:, 0].astype(np.float64),
        ask_size=book.ask_sizes[:, 0].astype(np.float64),
    )


def _buy_fraction(messages: Messages) -> NDArray[np.float64]:
    """Buy-side share of order flow, with non-transitions carried forward."""
    n = len(messages)
    out = np.full(n, 0.5, dtype=np.float64)
    transitions = np.flatnonzero(messages.is_transition)
    if transitions.size == 0:
        return out
    out[transitions] = (messages.directions[transitions].astype(np.float64) + 1.0) / 2.0

    gaps = np.flatnonzero(~messages.is_transition)
    if gaps.size:
        # Index of the most recent transition at or before each gap.
        prior = np.searchsorted(transitions, gaps, side="right") - 1
        seen = prior >= 0
        out[gaps[seen]] = out[transitions[prior[seen]]]
    return out


def load_session(directory: Path, levels: int | None = None) -> MarketStream:
    """Load one LOBSTER session directory into a :class:`MarketStream`.

    Args:
        directory: Directory holding ``*_orderbook_<L>.csv`` and
            ``*_message_<L>.csv`` for the same level ``L``.
        levels: Price levels to read. ``None`` reads only what
            :class:`MarketStream` needs, which is the touch.

    Returns:
        The session as a market stream.

    Raises:
        FileNotFoundError: If the directory or the expected files are absent.
        LobsterFormatError: If the two files cannot be reconciled.
    """
    book_path, message_path = _require_files(Path(directory), set())
    declared = levels_in_filename(book_path)
    book = load_order_book(book_path, declared if levels is None else levels)
    messages = load_messages(message_path)
    return session_to_stream(book, messages)


__all__ = [
    "BUY_DIRECTION",
    "COLUMNS_PER_LEVEL",
    "DEFAULT_LEVELS",
    "EMPTY_ASK_PRICE",
    "EMPTY_BID_PRICE",
    "MESSAGE_COLUMNS",
    "NON_TRANSITION_TYPES",
    "SELL_DIRECTION",
    "LobsterFormatError",
    "Messages",
    "OrderBook",
    "levels_in_filename",
    "load_messages",
    "load_order_book",
    "load_session",
    "session_to_stream",
]
