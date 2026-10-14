"""Is the policy's problem direction, or selectivity?

The frozen run showed a policy with directional skill -- accuracy 0.428 against a
0.389 majority class -- that loses money because it acts on 98% of events against
a 15-granule half-spread. Those are different failures and they call for different
interventions, so this module separates them without retraining anything.

What it asks
------------
The policy emits three logits per event. This asks what happens if BUY and SELL are
only taken when the model's own confidence in a *direction* exceeds a threshold, and
HOLD otherwise::

    trade if max(p_BUY, p_SELL) > c

At ``c = 0`` this is exactly the frozen policy -- trade whenever the argmax is a
direction -- so the curve starts at the frozen result and every other row is a
threshold applied to the same trained weights. Nothing is refitted, so this cannot
become the new headline: the frozen number stands regardless of what the curve
shows. This is a diagnostic of one trained policy, not a second experiment.

How to read it
--------------
Two outcomes mean opposite things, and the distinction is the whole point:

- **Selectivity improves with little accuracy cost.** The model already separates
  opportunities; its action rule was simply not selective enough. Worth pursuing.
- **Accuracy collapses as the trade rate falls.** The confidence is not tracking
  anything real. The model does not know when a large move is coming, and the
  honest conclusion is that the representation does not support profitable
  selection -- which sends the next experiment to the input, not the threshold.

Evaluation region
-----------------
Out of sample, and not by default. ``train_policy`` fits on the leading
``train_fraction`` of the stream; ``sweep_stage_a`` replays the whole stream, so a
naive reading of a run mixes fitted events with unseen ones. For a threshold
diagnostic that is fatal rather than merely imprecise: confidence is exactly what
a model is most overconfident about in sample, so an in-sample curve would show
selectivity that evaporates the moment the weights meet new data. Every figure here
is computed on events at or after ``eval_start`` and the region size is reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream
from latency_ml.finance.stage_a import (
    BUY,
    SELL,
    FeatureNormaliser,
    assert_disjoint,
    build_feature_matrix,
    evaluate_delay_ns,
    train_policy,
)
from latency_ml.finance.target import HOLD, TargetSpec


@dataclass(frozen=True)
class SelectivityPoint:
    """One confidence threshold on one trained policy, at one execution delay."""

    threshold: float
    tau_ns: int
    n_events: int
    n_trades: int
    #: Share of evaluated events on which a direction was taken.
    trade_rate: float
    #: Directional accuracy on the events actually traded, against the label.
    accuracy_given_trade: float
    #: Base rate of a correct direction over all evaluated events, for reference: a
    #: threshold that only matches this has not found anything.
    base_rate: float
    #: Net value per evaluated event and per trade, in price units.
    pnl_per_event: float
    pnl_per_trade: float
    #: Share of the frozen oracle ceiling at this delay, or None where the ceiling
    #: is not positive.
    oracle_capture: float | None

    @property
    def lift(self) -> float:
        """Accuracy on traded events over the unconditional base rate.

        The single number that says whether confidence carries information. A
        threshold that trades less but no more accurately has found selectivity
        without skill, which is just a smaller random position.
        """
        if self.base_rate <= 0.0:
            return float("nan")
        return self.accuracy_given_trade - self.base_rate

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "tau_ns": self.tau_ns,
            "n_events": self.n_events,
            "n_trades": self.n_trades,
            "trade_rate": self.trade_rate,
            "accuracy_given_trade": self.accuracy_given_trade,
            "base_rate": self.base_rate,
            "lift": self.lift,
            "pnl_per_event": self.pnl_per_event,
            "pnl_per_trade": self.pnl_per_trade,
            "oracle_capture": self.oracle_capture,
        }


def policy_probabilities(
    policy: torch.nn.Module, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Softmax action probabilities from the model, in the model's own units.

    Read-only with respect to the policy: no parameter is touched, so a threshold
    applied to these probabilities is a property of the frozen weights.
    """
    logits, _ = policy(torch.as_tensor(features, dtype=torch.float32))
    return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)


def confidence_positions(probabilities: NDArray[np.float64], threshold: float) -> NDArray[np.int64]:
    """Positions from a confidence rule: direction when confident, else HOLD.

    ``threshold = 0`` reproduces the frozen policy exactly -- trade whenever the
    argmax is a direction -- which is what makes the curve's first row the frozen
    result rather than a variant of it.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    best_direction = np.where(probabilities[:, BUY] >= probabilities[:, SELL], BUY, SELL)
    # Strictly greater, and that strictness is load-bearing. The argmax is a
    # direction exactly when max(p) > p_HOLD; using >= makes the test vacuous,
    # because max(p) is p_HOLD whenever HOLD wins, so every event would be assigned
    # some direction. The consequence is that threshold 0 would not be the frozen
    # policy, and the whole curve would be measured against a baseline that does not
    # exist. A tie goes to HOLD, matching argmax taking the first maximum.
    confidence = probabilities.max(axis=1)
    confident = (confidence > probabilities[:, HOLD]) & (confidence > threshold)
    return np.where(confident, best_direction, HOLD).astype(np.int64)


def selectivity_curve(
    stream: MarketStream,
    policy: torch.nn.Module,
    normaliser: FeatureNormaliser,
    target: TargetSpec,
    labels: NDArray[np.int64],
    *,
    thresholds: Sequence[float],
    delay_grid_ns: Sequence[int],
    eval_start: int,
    transaction_cost_bps: float = 0.0,
    oracle_capture: dict[int, float] | None = None,
    oracle_granule: int | None = None,
    lookback: int = 20,
) -> list[SelectivityPoint]:
    """Threshold sweep on one trained policy, out of sample.

    ``oracle_capture`` maps requested delay to the frozen ceiling's value at that
    delay, in granules. ``pnl_per_event`` is in price units, so the capture divides
    after converting through ``oracle_granule`` -- skipping that divides half-cents
    by cents and returns a share around -200.
    """
    assert_disjoint(normaliser, eval_start, "selectivity evaluation")
    features = build_feature_matrix(stream, lookback)
    normalised = normaliser.apply(features)
    probabilities = policy_probabilities(policy, normalised)

    # Everything before eval_start is masked out rather than sliced, so the
    # evaluation keeps full-length arrays and stays aligned with the labels.
    n = len(stream)
    if eval_start <= 0 or eval_start >= n:
        raise ValueError(f"eval_start must leave events on both sides: got {eval_start} of {n}")
    n_eval = n - eval_start

    # Accuracy is only meaningful where a direction exists to be right or wrong, so
    # both sides are conditioned on the label being BUY or SELL. Dividing by all
    # evaluated events instead would put a ceiling on accuracy that has nothing to do
    # with the model, and would make lift look like zero at every threshold simply
    # because most events are HOLD.
    predicted = probabilities.argmax(axis=1)
    has_direction = (labels == BUY) | (labels == SELL)
    predicted_direction = (predicted == BUY) | (predicted == SELL)
    correct_direction = has_direction & predicted_direction & (predicted == labels)
    n_directional = int(np.count_nonzero(has_direction[eval_start:]))
    if n_directional == 0:
        raise ValueError(
            f"no BUY or SELL windows in the evaluation region [{eval_start}, {n}); "
            "selectivity cannot be assessed against an empty reference"
        )
    # An independent reference, for the same reason as in oos_directional_lift: a
    # lift column whose two terms share a numerator cannot distinguish skill from
    # trade rate.
    _directional = labels[eval_start:][has_direction[eval_start:]]
    base_rate = (
        max(
            int(np.count_nonzero(_directional == BUY)),
            int(np.count_nonzero(_directional == SELL)),
        )
        / n_directional
    )
    # Granule scale, so a policy P&L in price units can be divided by a ceiling
    # measured in granules.
    granule = oracle_granule or 1.0

    points: list[SelectivityPoint] = []
    for threshold in thresholds:
        actions = confidence_positions(probabilities, threshold)
        for tau in delay_grid_ns:
            pnl, n_trades, _realised = evaluate_delay_ns(
                stream,
                actions,
                int(tau),
                1,
                transaction_cost_bps,
                target=target,
                per_event=True,
                region=slice(eval_start, n),
            )
            traded_directional = (actions[eval_start:] != HOLD) & has_direction[eval_start:]
            traded = np.flatnonzero(traded_directional)
            if traded.size == 0:
                points.append(
                    SelectivityPoint(
                        threshold=threshold,
                        tau_ns=int(tau),
                        n_events=n_eval,
                        n_trades=0,
                        trade_rate=0.0,
                        accuracy_given_trade=float("nan"),
                        base_rate=base_rate,
                        pnl_per_event=0.0,
                        pnl_per_trade=float("nan"),
                        oracle_capture=None,
                    )
                )
                continue
            hits = correct_direction[eval_start:][traded]
            capture = None
            if oracle_capture is not None:
                ceiling = oracle_capture.get(int(tau))
                if ceiling is not None and ceiling > 0.0:
                    capture = (pnl / granule) / ceiling
            points.append(
                SelectivityPoint(
                    threshold=threshold,
                    tau_ns=int(tau),
                    n_events=n_eval,
                    n_trades=int(n_trades),
                    trade_rate=float(np.count_nonzero(actions[eval_start:] != HOLD) / n_eval),
                    accuracy_given_trade=float(np.mean(hits)),
                    base_rate=base_rate,
                    pnl_per_event=float(pnl),
                    pnl_per_trade=float(pnl * n_eval / n_trades),
                    oracle_capture=capture,
                )
            )
    return points


@dataclass(frozen=True)
class PairedDifference:
    """A difference between two arms scored on the same windows.

    The point of pairing. Every arm is evaluated on the same out-of-sample events,
    so the two error sources are strongly positively correlated and differencing
    their independent standard errors would overstate the uncertainty of the
    difference -- potentially by a lot, since both arms are being fooled by the same
    windows.
    """

    label: str
    n_blocks: int
    mean: float
    standard_error: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "label": self.label,
            "n_blocks": self.n_blocks,
            "mean": self.mean,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
        }


#: Two-sided 95% normal quantile. Used rather than Student's t because n_blocks is in
#: the hundreds and the per-block differences are an average over ~140 events each, so
#: the sampling distribution of the mean is close to normal.
Z95: float = 1.959963984540054


def block_lift_series(
    predicted: NDArray[np.int64],
    labels: NDArray[np.int64],
    has_direction: NDArray[np.bool_],
    base_direction: int,
    n_blocks: int,
) -> NDArray[np.float64]:
    """Per-block lift for one arm: mean of (policy right) - (baseline right).

    Contiguous blocks over the out-of-sample region, one per effective window, so the
    series is on the same footing as the standard error already reported with each
    arm's lift. Every arm evaluated on the same region produces a series of the same
    length, block by block, which is what makes the difference between two of them a
    paired quantity.

    Only directional events contribute, since both terms of lift are defined there.
    """
    if n_blocks < 2:
        raise ValueError(f"need at least two blocks to estimate a standard error, got {n_blocks}")
    n = labels.size
    hits = np.where(has_direction, (predicted == labels).astype(np.float64), 0.0)
    baseline = np.where(has_direction, (labels == base_direction).astype(np.float64), 0.0)
    directional = has_direction.astype(np.float64)
    per_block = np.array_split(np.arange(n), n_blocks)
    out = np.empty(n_blocks, dtype=np.float64)
    for i, idx in enumerate(per_block):
        denom = directional[idx].sum()
        out[i] = 0.0 if denom == 0 else float((hits[idx] - baseline[idx]).sum() / denom)
    return out


def paired_difference(
    label: str, series_i: NDArray[np.float64], series_j: NDArray[np.float64]
) -> PairedDifference:
    """Mean and 95% interval for ``series_i - series_j`` on shared blocks."""
    if series_i.shape != series_j.shape:
        raise ValueError(
            f"paired series must be the same length: got {series_i.size} and {series_j.size}. "
            "Arms evaluated on different windows cannot be differenced."
        )
    d = np.asarray(series_i, dtype=np.float64) - np.asarray(series_j, dtype=np.float64)
    n = d.size
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    mean = float(np.mean(d))
    return PairedDifference(
        label=label,
        n_blocks=int(n),
        mean=mean,
        standard_error=se,
        ci_low=mean - Z95 * se,
        ci_high=mean + Z95 * se,
    )


def aggregate_across_seeds(label: str, per_seed: Sequence[PairedDifference]) -> PairedDifference:
    """Combine per-seed paired differences into one interval, stated explicitly.

    The three seeds are evaluated on the **same** windows, so the dominant sampling
    error is *common* to them and does not shrink with the number of seeds.
    Averaging the three estimates and dividing by ``sqrt(3)`` would treat one set of
    windows as three independent samples and understate the interval by roughly a
    factor of 1.7.

    So the aggregate interval combines the typical within-window sampling error with
    the part of the seed-to-seed spread that is genuinely independent:

        se = sqrt(typical_within_se**2 + between_seed_sd**2 / n_seeds)

    The first term is not reduced by the seed count because the windows are shared;
    the second is, because initialisations are not.
    """
    if not per_seed:
        raise ValueError("no per-seed differences to aggregate")
    means = np.array([d.mean for d in per_seed], dtype=np.float64)
    within = float(np.median([d.standard_error for d in per_seed]))
    between = float(np.std(means, ddof=1)) if means.size > 1 else 0.0
    se = math.sqrt(within**2 + (between**2 / means.size))
    mean = float(np.mean(means))
    return PairedDifference(
        label=label,
        n_blocks=int(per_seed[0].n_blocks),
        mean=mean,
        standard_error=se,
        ci_low=mean - Z95 * se,
        ci_high=mean + Z95 * se,
    )


def effective_windows(stream: MarketStream, eval_start: int, horizon_ns: int | None) -> float:
    """Non-overlapping windows in the evaluated region: span divided by horizon.

    The same treatment the pre-registered identifiability check gave the oracle gap.
    Using the raw event count here would shrink the standard error by a factor of
    about 180 and make every arm look significant.
    """
    if horizon_ns is None or horizon_ns <= 0:
        return 0.0
    times = np.asarray(stream.times_ns, dtype=np.float64)
    finite = times[np.isfinite(times)]
    finite = finite[finite >= times[eval_start]] if eval_start < times.size else finite
    if finite.size < 2:
        return 0.0
    return max(1.0, float(finite[-1] - finite[0]) / horizon_ns)


def lift_variance(
    predicted: NDArray[np.int64],
    labels: NDArray[np.int64],
    has_direction: NDArray[np.bool_],
    base_direction: int,
) -> float:
    """Per-event variance of (policy correct) - (constant baseline correct).

    Paired on the same events, so this is the variance of a difference in indicators
    and not of two separately estimated accuracies. Only directional events
    contribute, since that is the set both terms are defined on.
    """
    if not has_direction.any():
        return 0.0
    policy_hit = (predicted[has_direction] == labels[has_direction]).astype(np.float64)
    baseline_hit = (labels[has_direction] == base_direction).astype(np.float64)
    return float(np.var(policy_hit - baseline_hit))


@dataclass(frozen=True)
class LiftSummary:
    """The L1 primary endpoint, on one arm, out of sample."""

    lookback: int
    eval_start: int
    n_evaluated: int
    #: Events in the region whose label is BUY or SELL -- the ones where a
    #: direction exists to be right or wrong about.
    n_directional: int
    trade_rate: float
    accuracy_given_trade: float
    #: Accuracy of always predicting the more common direction among the directional
    #: events. An independent reference: it involves no model.
    base_rate: float
    n_buy: int = 0
    n_sell: int = 0
    #: Non-overlapping windows in the evaluated region. The error bar is computed on
    #: this rather than on ``n_directional``, because consecutive look-ahead windows
    #: share nearly all of their events: on the primary arm that is roughly 650
    #: windows against 118,000 directional events, a factor of about 180 in the
    #: standard error. The pre-registered endpoint said "positive lift is
    #: improvement", which is only a usable rule next to the size of the error.
    n_effective: float = 0.0
    #: Per-event variance of (policy correct) - (constant baseline correct), paired on
    #: the same events. Set by the caller, the only place that sees per-event outcomes.
    lift_variance: float = 0.0

    @property
    def lift(self) -> float:
        """Pre-registered primary endpoint. Positive is improvement.

        Zero means the policy is exactly as good as always betting the more common
        direction, which is the bar a directional model has to clear.
        """
        return self.accuracy_given_trade - self.base_rate

    @property
    def lift_standard_error(self) -> float:
        """One standard error on the lift, over non-overlapping windows."""
        if self.n_effective <= 0:
            return float("nan")
        return math.sqrt(self.lift_variance / self.n_effective)

    @property
    def lift_significant(self) -> bool:
        """Whether the lift clears two standard errors.

        The endpoint is a difference of two paired indicators, so the variance is of
        the difference rather than of two separate accuracies -- differencing
        independent binomial errors would overstate it whenever the policy and the
        baseline agree more often than either succeeds alone.
        """
        se = self.lift_standard_error
        return bool(np.isfinite(se)) and abs(self.lift) > 2.0 * se

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "eval_start": self.eval_start,
            "n_evaluated": self.n_evaluated,
            "n_directional": self.n_directional,
            "n_buy": self.n_buy,
            "n_sell": self.n_sell,
            "trade_rate": self.trade_rate,
            "accuracy_given_trade": self.accuracy_given_trade,
            "base_rate": self.base_rate,
            "lift": self.lift,
            "n_effective": self.n_effective,
            "lift_standard_error": self.lift_standard_error,
            "lift_significant": self.lift_significant,
        }


def oos_directional_lift(
    stream: MarketStream,
    policy: torch.nn.Module,
    normaliser: FeatureNormaliser,
    labels: NDArray[np.int64],
    *,
    eval_start: int,
    lookback: int = 20,
    horizon_ns: int | None = None,
    block_series: list[float] | None = None,
) -> LiftSummary:
    """Out-of-sample directional lift for one trained policy.

    The endpoint registered in PROTOCOL.md section 14. Both terms are restricted to
    events whose label is BUY or SELL, since accuracy is undefined where there is no
    direction to predict, and both are computed on the same held-out events.

    ``block_series``, when given, receives the per-window lift for this arm. It is
    what makes a *paired* comparison against another arm possible: every arm is
    evaluated on the same windows, so two series differenced block by block give a
    far tighter interval than differencing two independently estimated lifts.
    """
    assert_disjoint(normaliser, eval_start, "lift evaluation")
    features = build_feature_matrix(stream, lookback)
    probabilities = policy_probabilities(policy, normaliser.apply(features))
    actions = probabilities.argmax(axis=1)

    n = len(stream)
    has_direction = ((labels == BUY) | (labels == SELL))[eval_start:]
    if not has_direction.any():
        raise ValueError(f"no BUY or SELL windows in [{eval_start}, {n}); lift has no reference")
    predicted = actions[eval_start:]
    correct = (predicted == labels[eval_start:]) & has_direction
    traded = predicted != HOLD
    n_traded = int(np.count_nonzero(traded))
    n_eval = n - eval_start
    n_directional = int(np.count_nonzero(has_direction))
    # Both terms of the endpoint are over the *same* events: those whose label is a
    # direction. Restricting the numerator but not the denominator would compare a
    # directional-event rate against an all-events rate, which for a constant-direction
    # policy is not zero lift.
    traded_directional = traded & has_direction

    # The reference is the best *constant-direction* predictor on these events, not
    # the policy's own accuracy.
    #
    # It was the policy's own accuracy, and that made the endpoint meaningless: both
    # terms shared the numerator `count(correct)`, so lift collapsed to
    # correct * (1/n_traded - 1/n_directional), which is negative whenever the policy
    # trades more events than there are directional ones -- a *perfect* forecaster
    # would have scored strongly negative. The first L1 run was reported against this
    # and its apparent improvement with lookback was an artefact of the reference
    # drifting, not of the policy improving.
    #
    # A constant-direction predictor is the honest no-information baseline for a
    # two-class problem: it is what "always bet the more common direction" achieves,
    # and beating it is what any directional model has to do to be worth anything.
    directional_labels = labels[eval_start:][has_direction]
    n_buy = int(np.count_nonzero(directional_labels == BUY))
    n_sell = int(np.count_nonzero(directional_labels == SELL))
    base_direction = BUY if n_buy >= n_sell else SELL
    base_rate = max(n_buy, n_sell) / n_directional

    return LiftSummary(
        lookback=lookback,
        eval_start=eval_start,
        n_evaluated=n_eval,
        n_directional=n_directional,
        trade_rate=n_traded / n_eval,
        accuracy_given_trade=(
            float(np.count_nonzero(correct) / np.count_nonzero(traded_directional))
            if np.count_nonzero(traded_directional)
            else float("nan")
        ),
        base_rate=base_rate,
        n_buy=n_buy,
        n_sell=n_sell,
        n_effective=effective_windows(stream, eval_start, horizon_ns),
        lift_variance=lift_variance(predicted, labels[eval_start:], has_direction, base_direction),
    )


def emit_block_series(
    stream: MarketStream,
    policy: torch.nn.Module,
    normaliser: FeatureNormaliser,
    labels: NDArray[np.int64],
    *,
    eval_start: int,
    lookback: int,
    horizon_ns: int,
) -> list[float]:
    """Per-window lift for one arm, for paired comparison against another.

    Split out from :func:`oos_directional_lift` so the standard error a run reports
    and the series used for pairing come from the same computation. If they were
    computed separately they could disagree about how many windows there are, and the
    pairing would silently be on a different partition from the marginal interval.
    """
    assert_disjoint(normaliser, eval_start, "block series")
    features = build_feature_matrix(stream, lookback)
    predicted = policy_probabilities(policy, normaliser.apply(features)).argmax(axis=1)
    predicted = predicted[eval_start:]
    region_labels = labels[eval_start:]
    has_direction = (region_labels == BUY) | (region_labels == SELL)
    directional = region_labels[has_direction]
    n_buy = int(np.count_nonzero(directional == BUY))
    n_sell = int(np.count_nonzero(directional == SELL))
    base_direction = BUY if n_buy >= n_sell else SELL
    n_blocks = round(effective_windows(stream, eval_start, horizon_ns))
    return [
        float(v)
        for v in block_lift_series(
            predicted, region_labels, has_direction, base_direction, n_blocks
        )
    ]


def format_selectivity_table(points: list[SelectivityPoint], n_events: int) -> str:
    """The four curves side by side, at the fastest and slowest delays.

    Both ends are shown because the interesting cases differ: a threshold that helps
    only at the fast end is buying selectivity that latency then destroys, and one
    that helps only at the slow end is not buying anything.
    """
    taus = sorted({p.tau_ns for p in points})
    fast, slow = taus[0], taus[-1]
    lines = [
        f"{n_events:,} held-out events, none of them fitted. Accuracy is conditioned on events",
        "whose label is BUY or SELL, so both sides share a reference.",
        f"  {'thresh':>6} | {'trade%':>7} {'acc|trade':>10} {'lift':>7} | "
        f"{'V/event':>9} {'V/trade':>9} {'capture':>8} | {'decay%':>7}",
        f"  {'':>6} | {'--- both delays ---':>26} | "
        f"{('--- fast: ' + str(fast / 1e6) + 'ms ---'):>28} | {'fast->slow':>7}",
        "  " + "-" * 90,
    ]
    by_threshold: dict[float, dict[int, SelectivityPoint]] = {}
    for p in points:
        by_threshold.setdefault(p.threshold, {})[p.tau_ns] = p
    for threshold in sorted(by_threshold):
        row = by_threshold[threshold]
        pf, ps = row.get(fast), row.get(slow)
        if pf is None or ps is None:
            continue
        decay = (
            float("nan")
            if pf.pnl_per_event == 0.0
            else 100.0 * (1.0 - ps.pnl_per_event / pf.pnl_per_event)
        )
        lines.append(
            f"  {threshold:>6.3f} | {100 * pf.trade_rate:>6.1f}% "
            f"{pf.accuracy_given_trade:>10.3f} {pf.lift:>+7.3f} | "
            f"{pf.pnl_per_event:>9.3f} {pf.pnl_per_trade:>9.2f} "
            + (f"{pf.oracle_capture:>8.3f}" if pf.oracle_capture is not None else "     n/a")
            + f" | {decay:>6.1f}%"
        )
    lines.append("")
    lines.append("  threshold 0 is the frozen policy: trade whenever the argmax is a direction.")
    lines.append("  lift is accuracy on traded events over the unconditional base rate; a")
    lines.append("  threshold that trades less without lifting accuracy is a smaller random")
    lines.append("  position, not a selective one.")
    return "\n".join(lines)


__all__ = [
    "Z95",
    "LiftSummary",
    "PairedDifference",
    "SelectivityPoint",
    "aggregate_across_seeds",
    "block_lift_series",
    "confidence_positions",
    "effective_windows",
    "emit_block_series",
    "format_selectivity_table",
    "lift_variance",
    "oos_directional_lift",
    "paired_difference",
    "policy_probabilities",
    "selectivity_curve",
    "train_policy",
]
