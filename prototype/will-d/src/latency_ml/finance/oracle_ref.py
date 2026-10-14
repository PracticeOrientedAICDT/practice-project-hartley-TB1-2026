"""The frozen oracle ceiling a policy result is measured against.

Why this is a separate artifact rather than a call
--------------------------------------------------
``target_diag`` can compute the perfect-forecaster curve for any stream at any time,
and that is exactly why it cannot serve as the reference for a policy result. A
ceiling recomputed at analysis time is a ceiling fitted to whatever the run happened
to do. The whole point of ``V_policy(tau) / V_oracle(tau)`` is to separate two
effects -- how much opportunity the model captured, and how much of that survives
delay -- and the denominator has to be fixed before the numerator exists.

So the ceiling is written once, from a named stream whose content hash is recorded,
and :func:`verify_against` refuses to divide by a curve that was not measured on the
same bytes. The alternative is a ratio that silently compares a policy on one session
against a ceiling from another, which is arithmetically valid and scientifically
worthless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latency_ml.finance.market import MarketStream
from latency_ml.finance.target import TargetSpec


class OracleReferenceError(RuntimeError):
    """Raised when a frozen oracle curve cannot be used as a reference."""


@dataclass(frozen=True)
class OracleCeiling:
    """A frozen perfect-forecaster value curve, tied to one stream by content hash."""

    session: str
    stream_digest: str
    target: TargetSpec
    #: Execution latencies in nanoseconds, ascending.
    tau_grid_ns: tuple[int, ...]
    #: Price units per granule on this stream, 50 for these sessions. Recorded so a
    #: policy result measured in price units can be put on the ceiling's scale
    #: instead of being divided by a number in the wrong unit.
    granule_units: int
    #: ``values[i]`` is the mean post-cost outcome in price granules at
    #: ``tau_grid_ns[i]``, for a perfect forecaster on this exact stream.
    values: tuple[float, ...]
    #: Opportunities available at each delay, so a thin cell is visible.
    n_signals: tuple[int, ...]
    notes: tuple[str, ...] = ()

    @property
    def value_at_zero(self) -> float:
        return self.values[0]

    def ratio(
        self, policy_values: dict[int, float], policy_in_price_units: bool = True
    ) -> dict[int, float]:
        """``V_policy(tau) / V_oracle(tau)`` where the ceiling is positive.

        The policy is measured in price units and the ceiling in granules, so the
        policy side is converted first. Skipping that divides a number in dollars by
        a number in half-cents and returns a share around -200, which is arithmetically
        fine and scientifically meaningless.

        Returns nothing where the ceiling is not positive: a policy cannot capture a
        share of an opportunity that does not exist, and reporting one would turn
        "no opportunity here" into a large negative number that reads like a result.
        """
        scale = float(self.granule_units) if policy_in_price_units else 1.0
        out: dict[int, float] = {}
        for tau, policy in policy_values.items():
            if tau not in self.tau_grid_ns:
                continue
            oracle = self.values[self.tau_grid_ns.index(tau)]
            if oracle > 0.0:
                out[tau] = (policy / scale) / oracle
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "stream_digest": self.stream_digest,
            "granule_units": self.granule_units,
            "target": self.target.to_dict(),
            "tau_grid_ns": list(self.tau_grid_ns),
            "values_ticks": list(self.values),
            "n_signals": list(self.n_signals),
            "notes": list(self.notes),
            "execution_assumption": "aggressive touch crossing",
            "value_model": "perfect forecaster; upper bound, not a prediction",
        }


def freeze_ceiling(
    stream: MarketStream,
    session: str,
    target: TargetSpec,
    tau_grid_ns: tuple[int, ...] | list[int],
) -> OracleCeiling:
    """Measure and return the perfect-forecaster curve for one stream and target."""
    from latency_ml.finance.replay import stream_digest
    from latency_ml.finance.target_diag import characterise_wall_clock_horizon, price_granule

    taus = tuple(sorted({int(t) for t in tau_grid_ns}))
    granule = price_granule(stream.mid)
    stats = characterise_wall_clock_horizon(
        stream, target.horizon_ns, granule, (target.band_ticks,), taus
    )
    curve = stats.curves[0]
    # net_mean_ticks is already in granule units: characterise_wall_clock_horizon
    # divides the moves by the granule before aggregating. Dividing again here
    # silently reported the primary arm's ceiling as 0.126 instead of 6.32.
    values = tuple(
        float(v.net_mean_ticks) if np.isfinite(v.net_mean_ticks) else float("nan")
        for v in curve.values
    )
    n_signals = tuple(int(v.n_signals) for v in curve.values)
    return OracleCeiling(
        session=session,
        stream_digest=stream_digest(stream),
        granule_units=granule,
        target=target,
        tau_grid_ns=taus,
        values=values,
        n_signals=n_signals,
        notes=(
            "Value is in price granules and assumes a perfect forecaster, so it bounds "
            "what any policy trained on this target could earn under aggressive touch "
            "crossing. It is not a prediction of model performance.",
            "Delays beyond the horizon are not applicable and are excluded, matching the trainer.",
        ),
    )


def write_ceiling(ceiling: OracleCeiling, path: Path) -> Path:
    """Write a ceiling as JSON, refusing to clobber a different one silently."""
    payload = ceiling.to_dict()
    if path.exists():
        existing = json.loads(path.read_text())
        same = (
            existing.get("stream_digest") == payload["stream_digest"]
            and existing.get("target") == payload["target"]
            and existing.get("tau_grid_ns") == payload["tau_grid_ns"]
        )
        if not same:
            raise OracleReferenceError(
                f"{path} already holds a different ceiling "
                f"(session {existing.get('session')!r}, digest "
                f"{str(existing.get('stream_digest'))[:16]}, target "
                f"{existing.get('target')}). Refusing to overwrite a frozen reference: "
                "the denominator has to predate the result it is dividing."
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_ceiling(path: Path) -> OracleCeiling:
    payload = json.loads(Path(path).read_text())
    target = payload["target"]
    return OracleCeiling(
        session=payload["session"],
        stream_digest=payload["stream_digest"],
        granule_units=int(payload["granule_units"]),
        target=TargetSpec(
            horizon_ns=int(target["horizon_ns"]), band_ticks=int(target["band_ticks"])
        ),
        tau_grid_ns=tuple(int(t) for t in payload["tau_grid_ns"]),
        values=tuple(float(v) for v in payload["values_ticks"]),
        n_signals=tuple(int(n) for n in payload["n_signals"]),
        notes=tuple(payload.get("notes", ())),
    )


def verify_against(ceiling: OracleCeiling, stream: MarketStream, target: TargetSpec) -> None:
    """Refuse to use a ceiling that was not measured on these bytes and this target.

    Called before any ratio is reported. The digest check is the whole point: two
    sessions can share a symbol, a horizon and a band and still be different data,
    and the resulting ratio would be a comparison of two unrelated experiments
    presented as a share of one.
    """
    from latency_ml.finance.replay import stream_digest

    actual = stream_digest(stream)
    if actual != ceiling.stream_digest:
        raise OracleReferenceError(
            f"oracle ceiling was measured on {ceiling.session!r} with digest "
            f"{ceiling.stream_digest[:16]}, but this stream hashes to {actual[:16]}. "
            "A policy/ceiling ratio across different data is not a capture fraction. "
            "Re-freeze the ceiling on this stream, or evaluate on the frozen one."
        )
    if target != ceiling.target:
        raise OracleReferenceError(
            f"oracle ceiling is for {ceiling.target.to_dict()}, not {target.to_dict()}. "
            "The denominator has to be the target that was frozen."
        )


@dataclass(frozen=True)
class RatioPoint:
    """A policy result beside the ceiling it is to be read against."""

    tau_ns: int
    n_signals: int
    oracle_ticks: float
    policy_ticks: float | None
    #: ``None`` where the ceiling is not positive, rather than a number. A policy
    #: cannot capture a share of an opportunity that does not exist, and a negative
    #: denominator would turn that into a large negative number that reads like a
    #: finding.
    ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_ns": self.tau_ns,
            "n_signals": self.n_signals,
            "oracle_ticks": self.oracle_ticks,
            "policy_ticks": self.policy_ticks,
            "ratio": self.ratio,
        }


def ratio_row(
    ceiling: OracleCeiling,
    policy_values: dict[int, float],
    policy_in_price_units: bool = True,
) -> list[RatioPoint]:
    """One row per delay: the policy value, the ceiling, and the share.

    The policy column is reported in granules alongside the ceiling so the two are
    directly comparable, whatever units the sweep produced.
    """
    scale = float(ceiling.granule_units) if policy_in_price_units else 1.0
    ratios = ceiling.ratio(policy_values, policy_in_price_units=policy_in_price_units)
    return [
        RatioPoint(
            tau_ns=tau,
            n_signals=ceiling.n_signals[i],
            oracle_ticks=ceiling.values[i],
            policy_ticks=(policy_values[tau] / scale if tau in policy_values else None),
            ratio=ratios.get(tau),
        )
        for i, tau in enumerate(ceiling.tau_grid_ns)
    ]


def format_ratio_table(rows: list[RatioPoint]) -> str:
    """Render the two-column comparison the protocol asks for.

    Absolute value and share of ceiling side by side, because either alone is
    ambiguous: absolute value cannot distinguish a bad model from a thin
    opportunity, and the ratio cannot distinguish a good model from an easy one.
    """
    lines = [
        f"  {'tau (ms)':>9} {'n opp':>9} {'oracle':>10} {'policy':>10} {'V_pol/V_or':>11}",
        "  " + "-" * 53,
    ]
    for row in rows:
        policy = row.policy_ticks if row.policy_ticks is not None else float("nan")
        lines.append(
            f"  {row.tau_ns / 1e6:>9g} {row.n_signals:>9,} {row.oracle_ticks:>10.3f} "
            f"{policy:>10.3f} "
            + (f"{row.ratio:>11.3f}" if row.ratio is not None else "n/a".rjust(11))
        )
    return "\n".join(lines)


def load_policy_values(points: Any) -> dict[int, float]:
    """Index a sweep's realised delays to their P&L, for ratio computation.

    Keyed by the *realised* median delay rather than the requested one, because the
    ceiling is indexed by requested delay and the two differ on a bursty book. A
    caller comparing them must do so explicitly rather than by silent zip.
    """
    return {int(p.delay_events): float(p.mean_pnl) for p in points}


def coverage(values: NDArray[np.float64]) -> float:
    """Share of grid points with a usable ceiling, for a sanity line in the report."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(values) & (values > 0.0)) / values.size)


__all__ = [
    "OracleCeiling",
    "OracleReferenceError",
    "RatioPoint",
    "coverage",
    "format_ratio_table",
    "freeze_ceiling",
    "load_ceiling",
    "load_policy_values",
    "ratio_row",
    "verify_against",
    "write_ceiling",
]
