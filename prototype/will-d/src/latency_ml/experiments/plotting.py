"""Figures for `RESULTS.md`.

Two plots, each answering a specific question the tables cannot:

- :func:`plot_latency_response` draws the sweep with the 500-step ceiling
  annotated, so the flat 20-60 ms region reads as "no observable degradation at
  this metric's ceiling" rather than "these policies are equally good". Without
  that annotation the flat line is the most misreadable part of the figure.
- :func:`plot_training_curves` draws return against *environment integration
  steps*, not PPO iteration, so conditions with different iteration counts are
  aligned on the quantity held equal across the sweep.

Conditions whose resolution is not ``exact`` are refused by the caller, never
plotted as independent points. See `PLAN.md` section 5.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from latency_ml.timing import Resolution

CARTPOLE_RETURN_CEILING = 500.0


def _load_aggregates(path: str | Path) -> list[dict[str, Any]]:
    """Read a latency-response aggregate from an evaluation JSON."""
    payload = json.loads(Path(path).read_text())
    return list(payload["latency_response"])


TASK_RETURN_CEILING: dict[str, float] = {
    "cartpole": CARTPOLE_RETURN_CEILING,
    "lunarlander": 200.0,
}


def _task_return_ceiling(task: str) -> float | None:
    """A task's maximum episode return, or ``None`` if unknown.

    Unknown rather than zero: a wrong ceiling line misleads, and a missing one
    does not. A task absent from :data:`TASK_RETURN_CEILING` is simply plotted
    without a reference.
    """
    return TASK_RETURN_CEILING.get(task)


def plot_latency_response(
    aggregates: Sequence[dict[str, Any]],
    path: str | Path,
    title: str = "CartPole: return vs decision interval (matched training)",
) -> Path:
    """Plot mean return against latency, with the metric's ceiling annotated.

    Error bars are the spread across independently trained seeds, not across
    episodes, because `PLAN.md` section 20 makes the trained policy the unit of
    analysis.
    """
    points = sorted(aggregates, key=lambda a: a["latency_ns"])
    xs = [a["latency_ns"] / 1e6 for a in points]
    ys = [a["mean_return"] for a in points]
    es = [a["std_across_seeds"] for a in points]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    ax.axhline(
        CARTPOLE_RETURN_CEILING,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.text(
        xs[0],
        CARTPOLE_RETURN_CEILING + 6,
        f"return ceiling {CARTPOLE_RETURN_CEILING:.0f} (500-step limit) — "
        "points at the ceiling are not distinguishable from one another",
        color="crimson",
        fontsize=8,
        va="bottom",
    )

    seed_counts = {a.get("n_seeds") for a in points if a.get("n_seeds") is not None}
    seed_label = (
        f"mean across {seed_counts.pop()} seeds ±1 sd"
        if len(seed_counts) == 1
        else "mean ±1 sd across seeds"
    )
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        marker="o",
        capsize=4,
        linewidth=1.8,
        color="#1f4e79",
        zorder=3,
        label=seed_label,
    )

    for x, a in zip(xs, points, strict=True):
        n = a["action_steps"]
        ax.annotate(
            f"hold x{n}",
            (x, a["mean_return"]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color="#444444",
        )

    top = max(a["mean_return"] for a in points)
    deviation_floor = 1.0

    def _at_ceiling(a: dict[str, Any]) -> bool:
        band = max(deviation_floor, float(a.get("std_across_seeds", 0.0)))
        return bool(a["mean_return"] >= top - band)

    first_deviant = next((a["latency_ns"] / 1e6 for a in points if not _at_ceiling(a)), None)
    last_clean = max((a["latency_ns"] / 1e6 for a in points if _at_ceiling(a)), default=None)

    if first_deviant is not None and last_clean is not None:
        ax.axvspan(last_clean, max(xs), color="#ff8c00", alpha=0.10, zorder=0)
        ax.text(
            (last_clean + max(xs)) / 2,
            40,
            f"transition region ({last_clean:.0f}, {max(xs):.0f}] ms\n"
            f"first deviation at {first_deviant:.0f} ms;\n"
            "not resolvable finer at 20 ms timestep",
            ha="center",
            fontsize=8,
            color="#8a4b00",
        )
    else:
        ax.text(
            (min(xs) + max(xs)) / 2,
            40,
            f"no deviation from the {top:.0f}-return maximum\nacross the tested latency range",
            ha="center",
            fontsize=8,
            color="#8a4b00",
        )

    ax.set_xlabel("decision interval (ms)  —  requested latency, exact conditions only")
    ax.set_ylabel("mean greedy return")
    ax.set_ylim(0, 560)
    ax.set_xticks(xs)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _load_curves(root: Path, sweep_id: str, task: str) -> dict[int, list[list[dict[str, Any]]]]:
    """Collect per-seed training curves, grouped by latency."""
    grouped: dict[int, list[list[dict[str, Any]]]] = {}
    for path in sorted((root / sweep_id / task).glob("*/seed*/curve.json")):
        payload = json.loads(path.read_text())
        grouped.setdefault(int(payload["latency_ns"]), []).append(payload["points"])
    return grouped


def plot_training_curves(
    root: str | Path,
    sweep_id: str,
    task: str,
    path: str | Path,
    title: str = "Training curves vs environment integration steps",
) -> Path:
    """Plot return against integration steps, one line per latency.

    The x-axis is deliberately *not* PPO iteration. Iterations differ by up to
    5x across the sweep, so an iteration axis would align conditions that have
    seen very different amounts of experience.
    """
    grouped = _load_curves(Path(root), sweep_id, task)
    if not grouped:
        raise ValueError(f"no curves found for sweep {sweep_id!r} under {root}")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for latency in sorted(grouped):
        runs = grouped[latency]
        length = min(len(r) for r in runs)
        steps = [p["integration_steps"] / 1e6 for p in runs[0][:length]]
        for run in runs:
            ax.plot(
                steps,
                [p["mean_return"] for p in run[:length]],
                color="#888888",
                alpha=0.30,
                linewidth=0.8,
                zorder=1,
            )
        mean = [
            sum(p["mean_return"] for p in [r[i] for r in runs]) / len(runs) for i in range(length)
        ]
        ax.plot(
            steps,
            mean,
            linewidth=2.0,
            label=f"{latency / 1e6:.0f} ms (hold x{latency // 20_000_000})",
            zorder=3,
        )

    ceiling = _task_return_ceiling(task)
    if ceiling is not None:
        ax.axhline(ceiling, color="crimson", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlabel("environment integration steps (millions) — equal simulated time")
    ax.set_ylabel("mean greedy return")

    if ceiling is not None:
        observed_max = max(
            point["mean_return"] for runs in grouped.values() for run in runs for point in run
        )
        ax.set_ylim(0, max(observed_max * 1.08, ceiling * 1.05))
    else:
        observed = max(
            point["mean_return"] for runs in grouped.values() for run in runs for point in run
        )
        ax.set_ylim(0, observed * 1.12)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def assert_all_exact(aggregates: Sequence[dict[str, Any]]) -> None:
    """Refuse to plot a condition that is not an independent measurement.

    Guards against plotting a quantised request as its own point, which
    `PLAN.md` section 5 forbids. Called by the CLI before rendering.
    """
    bad = [
        a["latency_ns"]
        for a in aggregates
        if a.get("resolution", Resolution.EXACT.value) != Resolution.EXACT.value
    ]
    if bad:
        raise ValueError(
            "refusing to plot non-independent conditions; these are quantised "
            f"onto another measurement: {bad}"
        )


def plot_opportunity_sweep(results: dict[str, Any], path: str | Path) -> Path:
    """Available opportunity against execution delay, with both controls.

    This is the Phase 3 mechanism figure and it is deliberately a plot of
    *opportunity*, not model accuracy. No policy is involved: the curve is
    produced by an oracle that already takes the correct action, so anything
    that decays here is the market process, not a model.

    The two controls are drawn on the same axes because the figure's claim is
    comparative. The short-lived curve should fall; the persistent ablation
    should be flat; the decoupled control should have nothing to lose. A figure
    showing only the first would not show that the curve came from the intended
    mechanism rather than from the replay layer.
    """
    delays = results["delays_events"]
    millis = [d for d in delays]
    style = {
        "short_lived": ("short-lived signal (main)", "#c0392b", "o"),
        "ablation_persistent": ("ablation: signal never decays", "#7f8c8d", "s"),
        "control_decoupled": ("control: signal decoupled from flow", "#2980b9", "^"),
    }
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )

    for name, (label, colour, marker) in style.items():
        cond = results["conditions"].get(name)
        if cond is None:
            continue
        means = cond["mean_pnl"]
        sems = cond["sem"]
        ax.errorbar(
            millis,
            means,
            yerr=sems,
            label=label,
            color=colour,
            marker=marker,
            markersize=4,
            linewidth=1.6,
            capsize=2,
        )
        if name == "short_lived":
            ax2.plot(millis, cond["signal_alive_fraction"], color=colour, linewidth=1.4)
            ax2.set_ylabel("signal still active\nat execution")

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("oracle opportunity per event (price units)")
    ax.set_title(
        "Available opportunity vs execution delay\n"
        "oracle takes the correct action at $t_0$; only the execution time moves",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    ax2.set_xlabel("execution delay (events, 1 event = 1 ms)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.25)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


__all__ = [
    "CARTPOLE_RETURN_CEILING",
    "assert_all_exact",
    "plot_latency_response",
    "plot_opportunity_sweep",
    "plot_stage_a",
    "plot_training_curves",
]


def plot_stage_a(results: dict[str, Any], path: str | Path) -> Path:
    """Stage A: economic decision value against execution delay.

    Two panels rather than one, because absolute value and *retained* opportunity
    answer different questions and quoting only one invites the wrong reading of
    the other. The upper panel is what a trader would see; the lower is what
    makes the comparison against the latent mechanism test meaningful.

    The cost floor is drawn explicitly so the reader can see that the decay is
    not an artefact of the floor drifting with delay.
    """
    delays = results["delays_events"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)

    value = results["mean_pnl"]
    floor = results["floor_pnl"]
    retained = results["retained"]

    ax.plot(delays, value, "o-", color="#c0392b", markersize=4, label="policy value V(d)")
    ax.plot(
        delays, floor, "s--", color="#7f8c8d", markersize=4, label="cost floor (no-information)"
    )
    ax.plot(
        [delays[0]],
        [value[0]],
        marker="*",
        markersize=14,
        color="black",
        linestyle="none",
        label="zero-latency reference",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("economic value per decision (price units)")
    ax.set_title(
        "Stage A: fixed MLP, fixed weights, latency the only variable",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    ax2.plot(delays, retained, "o-", color="#c0392b", markersize=4)
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_ylabel("opportunity retained\nrelative to delay 0")
    ax2.set_xlabel("execution delay (events, 1 event = 1 ms)")
    ax2.set_ylim(-0.15, 1.15)
    ax2.grid(alpha=0.25)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
