"""Stage A: a fixed MLP replayed under varying inference latency.

The experiment is deliberately boring. One MLP is trained once, its weights are
frozen, and the same event stream is replayed under every latency condition.
Inference latency is the only thing that varies. Nothing here tunes the model,
the features, the transaction cost, or the trading threshold, and no part of
this pipeline is latency-dependent -- the model never sees a delay.

Three quantities are kept distinct throughout, and conflating any two of them is
the easiest way to produce a meaningless latency curve:

* **inference latency** -- supplied by a :class:`LatencyProvider`;
* **execution delay** -- where that lands on the event timeline;
* **economic outcome** -- P&L measured downstream of the fill.

The headline normalisation is *opportunity retained*:

    retained(d) = (V(d) - floor(d)) / (V(0) - floor(0))

where ``floor`` is the measured cost-only outcome from a no-information
baseline on the *same* stream at the *same* delay, and ``V(0)`` is this
policy's own zero-latency reference. The floor is empirical, not a theoretical
constant, and it is delay-matched so that costs cancel out of both the numerator
and the denominator. A delay-dependent floor would otherwise be mistaken for
latency sensitivity.

Absolute ``V(d)`` is reported alongside it. Retained is the better metric when
comparing against the mechanism test, absolute is the better metric when
reporting economic value, and quoting only one of the two would invite the wrong
reading of the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from latency_ml.finance.features import FEATURE_NAMES, feature_matrix
from latency_ml.finance.market import MarketStream
from latency_ml.finance.target import (
    TargetSpec,
    forward_index,
    horizon_moves,
    label_stream,
)
from latency_ml.models.architectures import MLPPolicy, build_hft_policy

#: Actions, matching ``policy.hft_output_actions`` in the config.
HOLD: int = 0
BUY: int = 1
SELL: int = 2
N_ACTIONS: int = 3

#: Seed for the no-information baseline, held fixed rather than varied with the
#: policy.
#:
#: The baseline is a random position stream. Re-drawing it per seed would put
#: baseline noise into every between-arm comparison, so a difference between two arms
#: could no longer be attributed to the policy. It is a control, and controls are held
#: fixed while the thing under test varies. Experiment L1R crosses the policy seed
#: over {0, 1, 2} against this one baseline for exactly that reason.
FROZEN_FLOOR_SEED: int = 991


@dataclass(frozen=True)
class TradingConfig:
    """Frozen Stage A settings. None of these may vary with latency.

    Attributes:
        horizon_events: Forward horizon defining the prediction target.
        hold_quantile: Fraction of the smallest absolute forward moves treated
            as HOLD. Fixed from the *training* split only, then frozen.
        train_fraction: Fraction of the stream used for fitting.
        holdout_events: Events excluded from the trailing end of training so the
            evaluation region is untouched.
    """

    horizon_events: int = 10
    hold_quantile: float = 0.30
    train_fraction: float = 0.60
    holdout_events: int = 20_000
    #: The frozen wall-clock target, when this run uses one.
    #:
    #: ``None`` selects the legacy event-count horizon and quantile band, which is
    #: what the synthetic mechanism work and every Phase G cell were run with. Those
    #: results stay reproducible because that path is untouched; the frozen V2B arms
    #: set this instead. It is opt-in rather than a replacement so that no archived
    #: result silently changes meaning.
    target: TargetSpec | None = None


@dataclass(frozen=True)
class FeatureNormaliser:
    """Feature standardisation fitted on the training split, and nothing else.

    This exists as a value rather than as a function of the stream because
    ``sweep_stage_a`` used to recompute mean and standard deviation from the
    **whole** stream while ``train_policy`` had fitted them on the training
    prefix alone. That was wrong twice over.

    It leaked. The evaluation-side statistics were computed over events the
    model was never meant to see, including the trailing holdout, so the reported
    latency curve depended on data the policy had not been trained on. That is
    the ordinary lookahead defect and it is disqualifying on its own.

    It also mismatched. A policy trained on features standardised one way and
    applied to features standardised another way sees a systematically shifted
    input, so the measured response belonged to a model that had never been
    evaluated. On real data the two fits differ substantially, because feature
    scale drifts across a trading session, which is precisely the regime this
    pipeline is for.

    Holding the fitted values makes the second defect unrepresentable: there is no
    way to sweep without a normaliser, therefore no default that can refit one.
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    #: Row this normaliser was fitted up to, exclusive. Recorded so a downstream
    #: evaluation can *prove* it did not read anything the fit saw, rather than
    #: being trusted to have been called correctly. See :func:`assert_disjoint`.
    fitted_upto: int = 0

    @classmethod
    def fit(cls, features: NDArray[np.float64], n_train: int) -> FeatureNormaliser:
        """Fit on the first ``n_train`` rows. Never on the remainder."""
        if not 0 < n_train <= len(features):
            raise ValueError(
                f"n_train={n_train} must lie in [1, {len(features)}] to fit a normaliser"
            )
        mean = features[:n_train].mean(axis=0)
        std = features[:n_train].std(axis=0)
        # A constant training column is left as-is rather than rescaled: dividing
        # by a zero standard deviation would turn a genuine constant into noise.
        std = np.where(std == 0.0, 1.0, std)
        return cls(mean=mean, std=std, fitted_upto=int(n_train))

    def apply(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardise any number of rows with the fitted values."""
        return (features - self.mean) / self.std


def assert_disjoint(
    normaliser: FeatureNormaliser, eval_start: int, what: str = "evaluation"
) -> None:
    """Refuse an evaluation region that overlaps the region anything was fitted on.

    The holdout exists so the evaluation region is untouched, and for a long time
    nothing enforced it: ``sweep_stage_a`` replayed the whole stream, so 60% of the
    events in the frozen run were fitted. The correction worked, but a convention
    that has already been violated once will be violated again unless violating it
    is impossible.

    So the fitted boundary travels with the fitted values, and any evaluation naming
    a region that starts inside it raises instead of quietly reporting in-sample
    numbers as if they were clean.
    """
    if eval_start < normaliser.fitted_upto:
        raise ValueError(
            f"{what} region starts at {eval_start} but the feature normaliser was "
            f"fitted on rows [0, {normaliser.fitted_upto}). Those events informed the "
            "input scaling, so evaluating there is in-sample. Raise eval_start, or fit "
            "on a strictly earlier prefix."
        )


def build_feature_matrix(stream: MarketStream, lookback: int = 20) -> NDArray[np.float64]:
    """Features for every event, under the same no-lookahead cutoff guarantee.

    Delegates to :func:`latency_ml.finance.features.feature_matrix`, which
    computes the whole matrix in one vectorised pass. This used to be a second
    copy of the per-index loop, which meant the loop had to be fixed in two
    places and there was no way to tell the copies apart.
    """
    return feature_matrix(stream, lookback)


def build_labels(stream: MarketStream, horizon: int, hold_band: float) -> NDArray[np.int64]:
    """Three-class labels over an *event-count* horizon and a quantile band.

    Superseded by :func:`build_labels_wall_clock` for real data, and kept because
    the synthetic mechanism runs depend on it. Two reasons it does not transfer:

    - Strict inequalities. With a band of zero, ``move > 0`` and ``move < 0`` leave
      every unmoved window as HOLD, so the three-class problem collapses to
      ``sign(move)``. On a real book that is most of the session.
    - A quantile band. On a step-function mid the share of unmoved windows exceeds
      any moderate quantile, so the quantile lands in the point mass at zero and
      the band *is* zero. Measured across four level-10 sessions, all four
      degenerate this way.
    """
    move = stream.mid[horizon:] - stream.mid[:-horizon]
    labels = np.where(move > hold_band, BUY, np.where(move < -hold_band, SELL, HOLD))
    return np.concatenate([labels, np.full(horizon, HOLD, dtype=labels.dtype)])


def build_labels_wall_clock(
    stream: MarketStream,
    target: TargetSpec,
    granule: int | None = None,
) -> tuple[NDArray[np.int64], dict[str, float | int]]:
    """Three-class labels over a *duration* horizon and an explicit granule band.

    Delegates the band to :func:`latency_ml.finance.target.label_moves`, the same
    function the diagnostic used to select the band. That sharing is the point: a
    trainer with its own copy of the rule would be fitting a target nobody
    validated, and the two differ exactly at the inclusive boundary.
    """
    return label_stream(stream.mid, stream.times_ns, target, granule=granule)


def fit_hold_band(stream: MarketStream, horizon: int, quantile: float, n_train: int) -> float:
    """Derive a quantile HOLD band from the training split only.

    Retained for the event-count path. Deriving it from the full stream would let
    evaluation data influence a model parameter, which is the same leak the
    mechanism test guards against. It is *not* used for the frozen real-data arms,
    where the band is an explicit granule count; see :func:`build_labels_wall_clock`
    for why a quantile cannot be used on a real mid.
    """
    move = np.abs(stream.mid[horizon:n_train] - stream.mid[: n_train - horizon])
    return float(np.quantile(move, quantile))


def train_policy(
    stream: MarketStream,
    model: str = "small",
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    seed: int = 0,
    lookback: int = 20,
    horizon_events: int | None = None,
    target: TargetSpec | None = None,
) -> tuple[MLPPolicy, TradingConfig, FeatureNormaliser, dict[str, float]]:
    """Fit the MLP once, on the training split, at no latency.

    Supervised rather than reinforcement learning. The Phase 3 question is what
    inference latency does to a fixed decision function, and PPO would add
    latency-dependent training and a second source of variance without changing
    what is being measured.

    Args:
        horizon_events: Forward horizon defining the prediction target. ``None``
            uses :class:`TradingConfig`'s default. This was a hardcoded literal
            here, while ``configs/finance.yaml`` carried a
            ``prediction_horizon_events`` that reached only the mechanism sweep --
            so the number the policy was trained against and the number an
            operator configured were connected by coincidence, both happening to be
            10. A diagnostic that wanted to vary the horizon had no way to move
            the thing it was diagnosing.

    Returns:
        The trained policy, the frozen trading config, the fitted normaliser, and
        fit diagnostics.
    """
    cfg = TradingConfig(
        horizon_events=TradingConfig.horizon_events if horizon_events is None else horizon_events,
        target=target,
    )
    n = len(stream)

    # How many events the horizon actually spans. A 10 s window on a bursty book
    # covers hundreds of events, so the holdout has to be sized in events even when
    # the target is defined in nanoseconds -- otherwise a label window starting in
    # the training region reaches past its end into the evaluation region.
    if cfg.target is not None:
        origins, _, _ = horizon_moves(stream.mid, stream.times_ns, cfg.target.horizon_ns)
        horizon_events_equivalent = max(1, int(n - origins.size))
    else:
        horizon_events_equivalent = cfg.horizon_events

    n_train = int(n * cfg.train_fraction)
    n_train = min(n_train, n - cfg.holdout_events - horizon_events_equivalent)
    if n_train <= 0:
        # Without this the negative index below reaches np.quantile and surfaces
        # as an IndexError about an empty axis, which says nothing about the
        # cause. The real requirement is a stream long enough to hold out an
        # untouched evaluation region after training.
        needed = cfg.holdout_events + horizon_events_equivalent + 1
        raise ValueError(
            f"stream of {n} events is too short to train on: the protocol holds out "
            f"{cfg.holdout_events} trailing events plus a {horizon_events_equivalent}-event "
            f"horizon, so at least {needed} are required. Lower TradingConfig."
            "holdout_events for a shorter session, or use a longer one."
        )

    target_summary: dict[str, float] = {}
    if cfg.target is not None:
        labels, target_summary = build_labels_wall_clock(stream, cfg.target)
        hold_band = float("nan")
    else:
        hold_band = fit_hold_band(stream, cfg.horizon_events, cfg.hold_quantile, n_train)
        labels = build_labels(stream, cfg.horizon_events, hold_band)
    features = build_feature_matrix(stream, lookback)

    # Standardise using training statistics only, and hand those statistics on so
    # that the sweep cannot possibly refit them over the whole stream.
    normaliser = FeatureNormaliser.fit(features, n_train)
    normalised = normaliser.apply(features)

    x = torch.as_tensor(normalised[:n_train], dtype=torch.float32)
    y = torch.as_tensor(labels[:n_train], dtype=torch.int64)

    torch.manual_seed(seed)
    policy = build_hft_policy(model, n_features=len(FEATURE_NAMES), n_actions=N_ACTIONS)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    policy.train()
    generator = torch.Generator().manual_seed(seed)
    n_batches = max(1, x.shape[0] // batch_size)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        for b in range(n_batches):
            idx = order[b * batch_size : (b + 1) * batch_size]
            optimiser.zero_grad()
            loss = loss_fn(policy(x[idx])[0], y[idx])
            loss.backward()
            optimiser.step()
    policy.eval()

    with torch.no_grad():
        logits, _ = policy(x)
        accuracy = float((logits.argmax(dim=1) == y).float().mean())

    return (
        policy,
        cfg,
        normaliser,
        {
            "hold_band": hold_band,
            **target_summary,
            "train_accuracy": accuracy,
            "n_train": float(n_train),
            "train_events": float(x.shape[0]),
        },
    )


@torch.no_grad()
def policy_actions(policy: MLPPolicy, features: NDArray[np.float64]) -> NDArray[np.int64]:
    """Greedy actions for precomputed features. The only place the model is used."""
    logits, _ = policy(torch.as_tensor(features, dtype=torch.float32))
    actions: NDArray[np.int64] = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    return actions


@dataclass(frozen=True)
class StageAPoint:
    """Stage A result at one execution delay.

    Attributes:
        delay_events: Execution delay in events.
        delay_ns: The same delay in nanoseconds, for comparison with the HFT grid.
        mean_pnl: Mean economic value of this policy, per decision.
        floor_pnl: Measured cost-only value from the no-information baseline at
            the same delay.
        retained: Opportunity retained relative to zero latency, on the
            cost-subtracted scale. ``None`` when the reference is degenerate.
        n_trades: Decisions where the policy traded.
    """

    delay_events: int
    delay_ns: int
    mean_pnl: float
    floor_pnl: float
    retained: float | None
    n_trades: int


def _pnl_from_execution(
    stream: MarketStream,
    positions: NDArray[np.int64],
    traded: NDArray[np.int64],
    exec_idx: NDArray[np.int64],
    horizon: int,
    transaction_cost_bps: float,
    target: TargetSpec | None,
    per_event: bool = False,
    n_eligible: int | None = None,
) -> tuple[float, int]:
    """Mean net P&L of ``positions`` filled at ``exec_idx`` and marked out later.

    HOLD contributes nothing and pays nothing, which is why the cost floor is
    established separately by a baseline that always trades. The crossing cost is
    the spread prevailing *at the fill*, not at the observation: that is what makes
    a delayed fill more expensive, and charging the earlier spread would hide the
    effect being measured.

    The markout is anchored at the **decision**, not at the fill. That is the whole
    content of a latency effect: the prediction is about the window
    ``[t_0, t_0 + H]``, and a decision acted on late has less of that window left to
    collect. Anchoring the exit at the fill instead would give every delay a full
    fresh window, on a uniform clock that makes P&L literally independent of the
    delay -- a latency sweep that cannot move. It is also what ``target_diag`` does
    when it computes the oracle ceiling, so anchoring anywhere else would put the
    policy and its ceiling on two different experiments.
    """
    if target is None:
        exit_idx = exec_idx + horizon
        ok = exit_idx < len(stream)
    else:
        exit_idx = forward_index(stream.times_ns, target.horizon_ns)[traded]
        # A fill at or before the window's close is a delayed execution of this
        # prediction. A fill after it is not: the window has already resolved, so
        # the trade is a short position in a forecast that no longer exists. Those
        # are dropped rather than scored, matching target_diag's rule that a delay
        # beyond the horizon is not applicable rather than merely unprofitable.
        ok = (exit_idx >= 0) & (exec_idx <= exit_idx)
    if not ok.any():
        return 0.0, 0
    traded, exec_idx, exit_idx = traded[ok], exec_idx[ok], exit_idx[ok]

    sign = np.sign(positions[traded].astype(np.float64))
    mid_exec = stream.mid[exec_idx]
    mid_exit = stream.mid[exit_idx]
    cost = 0.5 * stream.spread[exec_idx] + (transaction_cost_bps / 10_000.0) * np.abs(mid_exec)
    total = float(np.sum(sign * (mid_exit - mid_exec) - cost))
    if per_event:
        # Divide by every eligible event, not by the number traded.
        #
        # This is the denominator the oracle ceiling uses: V(tau) is the mean over
        # all windows of the value a perfect forecaster earns, with HOLD windows
        # contributing zero. Averaging over trades instead would compare a policy
        # that chooses its moments against a ceiling measured per event, and a
        # policy trading indiscriminately would look better the more often it
        # traded. Whether to trade is part of the policy's performance, so the
        # choice has to be inside the average.
        if n_eligible is None or n_eligible <= 0:
            return 0.0, int(traded.size)
        return total / n_eligible, int(traded.size)
    return total / traded.size, int(traded.size)


def evaluate_delay(
    stream: MarketStream,
    positions: NDArray[np.int64],
    delay: int,
    horizon: int,
    transaction_cost_bps: float,
    target: TargetSpec | None = None,
    per_event: bool = False,
    region: slice | None = None,
) -> tuple[float, int]:
    """Economic value of a position stream executed after ``delay`` *events*.

    The delay is an index shift. That is only the same thing as a duration on a
    uniformly spaced stream; on a bursty book one shift costs a different amount of
    wall time at every decision, so this is the legacy path and
    :func:`evaluate_delay_ns` is the one the frozen protocol uses.
    """
    n = len(stream)
    if region is not None:
        # Restrict to the evaluation region. Everything else is forced to HOLD so it
        # neither trades nor pays, and the denominator is the region's own size --
        # so a per-event figure means "per evaluated event", which is the only
        # denominator the oracle ceiling can be divided by.
        masked = np.zeros_like(positions)
        masked[region] = positions[region]
        positions = masked
        n_eligible = region.stop - region.start if region.stop is not None else len(positions)
    else:
        n_eligible = None
    traded = np.flatnonzero(positions != 0)
    if traded.size == 0:
        return 0.0, 0
    exec_idx = traded + delay
    keep = exec_idx < n
    if target is None:
        keep &= exec_idx + horizon < n
    if not keep.any():
        return 0.0, 0
    return _pnl_from_execution(
        stream,
        positions,
        traded[keep],
        exec_idx[keep],
        horizon,
        transaction_cost_bps,
        target,
        per_event=per_event,
        n_eligible=n_eligible if n_eligible is not None else int(keep.sum()),
    )


def evaluate_delay_ns(
    stream: MarketStream,
    positions: NDArray[np.int64],
    tau_ns: int,
    horizon: int,
    transaction_cost_bps: float,
    target: TargetSpec | None = None,
    per_event: bool = False,
    region: slice | None = None,
) -> tuple[float, int, float]:
    """Economic value of a position stream executed ``tau_ns`` after the decision.

    The latency grid is in nanoseconds, so the delay has to be resolved per
    decision from the clock rather than applied as one index shift. On these
    sessions the two differ enormously: AAPL's median event gap is 0.639 ms and its
    mean is 58.44 ms, so a single shift of 100 events spans 64 ms at the median and
    nearly six seconds at the mean. Applying one shift and calling it 100 ms would
    make the reported latency curve a curve over event count wearing a time label.

    Returns the mean P&L, the trade count, and the median wall time actually
    elapsed by the delayed decisions, so the realised delay can be checked against
    the requested one rather than assumed.
    """
    if tau_ns < 0:
        raise ValueError(f"delay must be non-negative, got {tau_ns} ns")
    if region is not None:
        masked = np.zeros_like(positions)
        masked[region] = positions[region]
        positions = masked
        n_eligible: int | None = (
            region.stop - region.start if region.stop is not None else len(positions)
        )
    else:
        n_eligible = None
    traded = np.flatnonzero(positions != 0)
    if traded.size == 0:
        return 0.0, 0, 0.0
    exec_idx = forward_index(stream.times_ns, tau_ns)[traded]
    ok = exec_idx >= 0
    if not ok.any():
        return 0.0, 0, 0.0
    traded, exec_idx = traded[ok], exec_idx[ok]
    elapsed = (stream.times_ns[exec_idx] - stream.times_ns[traded]).astype(np.float64)
    pnl, n_trades = _pnl_from_execution(
        stream,
        positions,
        traded,
        exec_idx,
        horizon,
        transaction_cost_bps,
        target,
        per_event=per_event,
        n_eligible=n_eligible if n_eligible is not None else int(ok.sum()),
    )
    return pnl, n_trades, float(np.median(elapsed))


def sweep_stage_a(
    stream: MarketStream,
    policy: MLPPolicy,
    cfg: TradingConfig,
    delays: tuple[int, ...],
    transaction_cost_bps: float,
    normaliser: FeatureNormaliser,
    seed: int = FROZEN_FLOOR_SEED,
    delay_grid_ns: tuple[int, ...] | None = None,
    eval_start: int | None = None,
    lookback: int = 20,
) -> list[StageAPoint]:
    """Replay one fixed policy across execution delays.

    The empirical floor is a no-information baseline: a random position at every
    event, which pays the same costs but captures no directional edge. It is
    evaluated at each delay on the same stream, so the floor is delay-matched
    rather than assumed constant.

    ``normaliser`` is required and is the one fitted during training. This
    function previously derived its own mean and standard deviation from the
    whole stream, which leaked the holdout and fed the policy an input
    distribution it had not been trained on; see :class:`FeatureNormaliser`.
    """
    # The same lookback the policy was trained on. It used to be left at the default
    # of 20, which is invisible at L=20 and catastrophic at any other L: the policy
    # would be trained on one window and replayed on another, and the resulting
    # numbers would belong to no model at all. The lookback ablation in PROTOCOL.md
    # section 14 exists to vary exactly this, so it has to be threaded through rather
    # than defaulted.
    normalised = normaliser.apply(build_feature_matrix(stream, lookback))

    if eval_start is not None and not 0 < eval_start < len(stream):
        raise ValueError(
            f"eval_start must leave events on both sides: got {eval_start} of {len(stream)}"
        )
    if eval_start is not None:
        assert_disjoint(normaliser, eval_start, "stage-a evaluation")
    region = None if eval_start is None else slice(eval_start, len(stream))

    actions = policy_actions(policy, normalised)
    positions = np.where(actions == BUY, 1, np.where(actions == SELL, -1, 0)).astype(np.int64)

    rng = np.random.default_rng(seed)
    random_positions = rng.choice(np.array([-1, 1], dtype=np.int64), size=len(stream))

    horizon = cfg.horizon_events
    # (key, n_trades, pnl, floor). The key is the REQUESTED delay -- an event count
    # on the legacy path, nanoseconds on the frozen one -- so results stay keyed by
    # the grid that was asked for. The realised delay is reported separately rather
    # than substituted for the key: on a bursty book the two differ by orders of
    # magnitude, and overwriting the key would make the output impossible to line up
    # against the oracle ceiling, which is indexed by requested delay.
    rows: list[tuple[int, int, float, float]] = []
    realised_ns: dict[int, float] = {}
    if delay_grid_ns is not None:
        # The frozen protocol's grid. Each delay is resolved per decision from the
        # clock, so "100 ms" means 100 ms at every decision rather than a fixed
        # number of events that happens to average 100 ms.
        for tau in delay_grid_ns:
            pnl, n_trades, realised = evaluate_delay_ns(
                stream,
                positions,
                tau,
                horizon,
                transaction_cost_bps,
                target=cfg.target,
                per_event=True,
                region=region,
            )
            floor, _, _ = evaluate_delay_ns(
                stream,
                random_positions,
                tau,
                horizon,
                transaction_cost_bps,
                target=cfg.target,
                per_event=True,
                region=region,
            )
            rows.append((int(tau), n_trades, pnl, floor))
            realised_ns[int(tau)] = realised
    else:
        for delay in delays:
            pnl, n_trades = evaluate_delay(
                stream,
                positions,
                delay,
                horizon,
                transaction_cost_bps,
                target=cfg.target,
                region=region,
            )
            floor, _ = evaluate_delay(
                stream,
                random_positions,
                delay,
                horizon,
                transaction_cost_bps,
                target=cfg.target,
                region=region,
            )
            rows.append((delay, n_trades, pnl, floor))

    # The clock is read from the stream rather than assumed.
    #
    # The obvious shortcut -- take the gap between the first two events and
    # multiply -- is only correct when spacing is uniform. It is, for the
    # synthetic generator, and it was what this code did. It is not correct for
    # historical data: LOBSTER event spacing varies by orders of magnitude within
    # a session, so a single gap reports a physical delay that most decisions did
    # not actually experience. `delay_ns` is therefore the *median* elapsed time
    # over the decisions that were delayed, measured from real timestamps.
    #
    # `delay_events` remains authoritative. A delay is applied as an index shift,
    # so on irregular data the nanosecond figure is a summary of what that shift
    # cost in wall time, not an input to it.
    elapsed = realised_ns if realised_ns else _elapsed_ns_per_delay(stream, delays)

    baseline_edge = rows[0][2] - rows[0][3]
    points: list[StageAPoint] = []
    for delay, n_trades, pnl, floor in rows:
        edge = pnl - floor
        retained = None if abs(baseline_edge) < 1e-12 else float(edge / baseline_edge)
        points.append(
            StageAPoint(
                delay_events=delay,
                delay_ns=int(elapsed.get(delay, 0.0)),
                mean_pnl=pnl,
                floor_pnl=floor,
                retained=retained,
                n_trades=n_trades,
            )
        )
    return points


def _elapsed_ns_per_delay(stream: MarketStream, delays: tuple[int, ...]) -> dict[int, int]:
    """Median wall-clock nanoseconds an index shift of each size actually took.

    Measured over the events that survive the delay, so the figure describes the
    delays the analysis really applied rather than a whole-stream average that
    includes decisions a delay would have pushed past the end.
    """
    n = len(stream)
    out: dict[int, int] = {}
    times = stream.times_ns
    for delay in delays:
        usable = n - delay
        if n == 0 or usable <= 0:
            out[delay] = 0
            continue
        # Decision i executes at event i + delay, so the elapsed time is
        # times[i + delay] - times[i] over the decisions that still land.
        gaps = times[delay:] - times[:usable]
        out[delay] = int(np.median(gaps)) if gaps.size else 0
    return out
