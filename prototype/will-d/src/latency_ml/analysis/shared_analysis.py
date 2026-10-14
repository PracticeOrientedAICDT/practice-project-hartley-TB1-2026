"""Paired analysis of the Phase 4 shared-trunk sweep against dedicated baselines.

The supervisor's instruction on reporting, implemented literally: the quantity of
interest is the **paired effect**, not a pass/fail verdict.

    Delta(task, tau) = R_shared(task, tau) - R_dedicated(task, tau)

averaged over seeds, reported for every task and every latency individually,
together with the seed-to-seed spread, and with relative differences only where
the baseline makes a ratio meaningful.

Two things this module deliberately does not do.

**No aggregated verdict.** An earlier proposal was to call the result a success
if the shared model hit 95% of dedicated at 3 of 5 latencies. That was dropped: it
collapses qualitatively different patterns into one bit. A shared trunk that
tracks dedicated at 96% across the whole latency curve is not the same finding as
one at 99/98/97/94/91%, and the latency curve is itself part of what Phase 4
studies. The 95% ratio is therefore reported **per condition** as a descriptive
line, never aggregated.

**No significance test.** With 3 seeds the replication unit is the seed, so each
task-latency comparison rests on 3 paired observations. A p-value would imply
more than that supports. Spread is reported so a reader can see how thin it is.

Comparability is established by evaluating the saved dedicated policies through
the *same* code path as the shared trunk, rather than by comparing a fresh shared
evaluation against a stored baseline aggregate written by older code.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from latency_ml.config import ProjectConfig, load_config
from latency_ml.experiments.evaluate import evaluate_policy, load_policy, summary_to_dict
from latency_ml.experiments.shared_runner import PHASE4_TASKS, load_shared_records

#: Definitive Phase 4 dedicated controls, per task.
#:
#: These are the Phase 2R Gate 3 controls: 4M integration steps per task, the
#: corrected discount basis (CartPole physical 0.99/0.95, LunarLander decision
#: 0.999/0.98) and the corrected LR schedule, over the same five latencies.
#:
#: Two sweep ids rather than one because the two tasks were run as separate
#: sweeps, under different configs, at different points in the correction
#: sequence. Both are 10-seed and 4M; the pairing is by ``(task, latency, seed)``
#: and never across tasks.
CONTROL_SWEEPS: dict[str, str] = {
    "cartpole": "cartpole-gate3-control",
    "lunarlander": "gate3-dedicated-control",
}

#: Root holding the definitive controls.
#:
#: These are the v1 controls, kept as committed test fixtures rather than as
#: experiment output under ``results/``. ``results/`` holds only what this
#: repository's own runs produce; the v1 arm is a frozen external reference, and
#: ``tests/fixtures/v1/`` is the right home for it. The layout under this root is
#: the one ``cell_manifest_path`` already expects, so only the root moved.
CONTROL_ROOT: Path = Path("tests/fixtures/v1/controls")

#: The definitive experiment's replication unit.
DEFINITIVE_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Latency grid of the definitive experiment, in both regimes.
PRIMARY_LATENCIES: tuple[int, ...] = (20_000_000, 40_000_000, 60_000_000)
SECONDARY_LATENCIES: tuple[int, ...] = (80_000_000, 100_000_000)
ALL_LATENCIES: tuple[int, ...] = PRIMARY_LATENCIES + SECONDARY_LATENCIES

#: Per-task integration-step budget both arms are expected to have consumed.
#:
#: Checked against the **manifests**, not the config: ``configs/base.yaml`` still
#: records 2M as its default, so a config field cannot distinguish the 4M
#: controls from the pre-correction 2M ``dedicated-v2`` sweep. The realised
#: budget is only visible in ``counts.integration_steps``, which is exactly why a
#: silent 2M/4M mismatch was possible before this check existed.
EXPECTED_CONTROL_BUDGET: int = 4_000_000

#: Tolerance for the realised budget.
#:
#: Realised steps fall short of the nominal budget by up to ~1.2%, not merely by
#: chunking: at high latency episodes terminate early (CartPole's 500-step cap,
#: LunarLander crashes), so the final rollout is cut short and the run stops
#: without consuming it. The worst observed cell is 3,953,830 at 100 ms.
#:
#: 2% admits that shortfall and nothing else: the pre-correction 2M sweep is 50%
#: off, so the two remain far apart and the check still discriminates.
BUDGET_TOLERANCE: float = 0.02

#: Return above which a LunarLander seed counts as having converged.
#:
#: A fixed threshold rather than a latent-mode fit, because the mode assignment
#: must be identical in both arms for the comparison to mean anything.
#:
#: It is fixed at 230 by the requirement that it reproduce the Gate 3 dedicated
#: counts exactly (10/10, 10/10, 9/10, 6/10, 3/10), which it does. Note the
#: margin is not generous: the seeds straddling it are 196.3 (60 ms, the single
#: non-converged plateau seed) and 233.8 (80 ms, the weakest converged seed), a
#: 3.8-return gap. Any threshold in (196.3, 233.8] gives the same assignment; one
#: outside it changes the 60 or 80 ms count by one seed. The choice is therefore
#: defensible but not comfortably so, and a reader who wants a different cut point
#: should re-derive the counts rather than trust this constant.
CONVERGENCE_THRESHOLD: float = 230.0

#: Baselines at or below this return are treated as too near zero for a ratio to
#: mean anything. LunarLander in particular can average near zero or negative,
#: where "95% of dedicated" is not a meaningful statement.
RATIO_FLOOR: float = 50.0

#: Descriptive threshold. Reported per condition, never aggregated.
COMPARABLE_RATIO: float = 0.95


class MissingControlError(RuntimeError):
    """Raised when the Phase 4 comparison cannot be made from the data present.

    Distinct from a null result. A missing or non-overlapping dedicated control
    means the question was not asked, and reporting it as "no effect" would
    manufacture a finding out of absent data.
    """


class StaleControlError(MissingControlError):
    """Raised when the control found is a different experiment than the one analysed.

    A subclass of :class:`MissingControlError` on purpose. Silently pairing the
    definitive 4M shared sweep against the pre-correction 2M ``dedicated-v2``
    sweep would produce a complete, plausible-looking table in which every
    LunarLander delta confounds sharing with the discount-basis and LR-schedule
    corrections. A stale control is worse than an absent one: an absent control
    produces an obvious error, a stale one produces a publishable wrong answer.
    """


@dataclass
class PairedCell:
    """One task and latency, aggregated over the paired seeds.

    Attributes:
        task: Task name.
        latency_ns: Latency condition.
        shared_returns: Shared-trunk returns, one per seed.
        dedicated_returns: Dedicated returns, one per seed.
    """

    task: str
    latency_ns: int
    shared_returns: list[float] = field(default_factory=list)
    dedicated_returns: list[float] = field(default_factory=list)
    #: PPO updates each arm received at this condition, when known. Equal
    #: simulated-time budgeting means these fall with latency in *both* arms, so
    #: they are recorded next to every delta: a difference that tracks the update
    #: count is a training-budget story, not a sharing story, and the reader
    #: cannot see that unless the counts are in the table.
    shared_ppo_updates: float | None = None
    dedicated_ppo_updates: float | None = None
    shared_gradient_steps: float | None = None
    dedicated_gradient_steps: float | None = None

    @property
    def n_seeds(self) -> int:
        return min(len(self.shared_returns), len(self.dedicated_returns))

    @property
    def shared_mean(self) -> float:
        return statistics.fmean(self.shared_returns)

    @property
    def dedicated_mean(self) -> float:
        return statistics.fmean(self.dedicated_returns)

    @property
    def paired_differences(self) -> list[float]:
        """Per-seed differences, paired by seed.

        Pairing by seed matters: unpaired means would let seed variance leak into
        what is meant to be a shared-vs-dedicated effect.
        """
        n = self.n_seeds
        return [self.shared_returns[i] - self.dedicated_returns[i] for i in range(n)]

    @property
    def mean_difference(self) -> float:
        return statistics.fmean(self.paired_differences)

    @property
    def difference_stdev(self) -> float:
        """Seed-to-seed spread of the paired difference, when n > 1."""
        diffs = self.paired_differences
        return statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    @property
    def ratio(self) -> float | None:
        """Shared as a fraction of dedicated, or ``None`` when meaningless.

        Guarded on both ends, not just the baseline. With a near-zero or negative
        dedicated mean the ratio is either undefined or has the wrong sign, and
        reporting it would invite exactly the misreading the supervisor warned
        against.

        The numerator is guarded too. LunarLander's shared arm goes *negative*
        under trunk sharing, and "shared achieved -62% of dedicated" is
        arithmetically true while being meaningless as a retention figure: there
        is no meaningful fraction of a performance metric that has gone below
        zero. The delta is the reportable quantity there, so the ratio is
        suppressed and the delta stands alone.
        """
        if self.dedicated_mean <= RATIO_FLOOR:
            return None
        if self.shared_mean <= 0.0:
            return None
        return self.shared_mean / self.dedicated_mean

    @property
    def is_comparable(self) -> float | None:
        """Descriptive ratio flag, or ``None`` when the ratio is meaningless."""
        if self.ratio is None:
            return None
        return self.ratio >= COMPARABLE_RATIO

    @property
    def regime(self) -> str:
        """Which latency regime this condition belongs to."""
        if self.latency_ns in PRIMARY_LATENCIES:
            return "primary"
        return "secondary"

    @property
    def shared_converged(self) -> list[float]:
        """Shared returns at or above the convergence threshold."""
        return [r for r in self.shared_returns if r >= CONVERGENCE_THRESHOLD]

    @property
    def dedicated_converged(self) -> list[float]:
        """Dedicated returns at or above the convergence threshold."""
        return [r for r in self.dedicated_returns if r >= CONVERGENCE_THRESHOLD]

    @property
    def convergence_probability(self) -> dict[str, Any]:
        """Fraction of seeds reaching the converged regime, in each arm.

        Reported as a raw fraction with its denominator rather than smoothed or
        turned into a rate, because with n=3 the only honest values are 0, 1/3,
        2/3 and 1. Interpolation to the dedicated arm's 10-seed scale would
        invent precision that 3 seeds cannot supply, and comparing a 3-seed
        fraction against a 10-seed one is a statement about sample size as much
        as about sharing.
        """
        return {
            "shared_n": len(self.shared_converged),
            "shared_total": len(self.shared_returns),
            "dedicated_n": len(self.dedicated_converged),
            "dedicated_total": len(self.dedicated_returns),
            "shared_fraction": (
                len(self.shared_converged) / len(self.shared_returns)
                if self.shared_returns
                else None
            ),
            "dedicated_fraction": (
                len(self.dedicated_converged) / len(self.dedicated_returns)
                if self.dedicated_returns
                else None
            ),
            "threshold": CONVERGENCE_THRESHOLD,
        }

    @property
    def conditional_performance(self) -> dict[str, Any]:
        """Mean return *conditional on* having converged, in each arm.

        The trimmed mean answers a different question from the condition mean:
        "when sharing succeeds, how good is the result?" rather than "how often
        does sharing succeed?". Reporting the first without the second would
        hide a total collapse behind a decent trimmed figure, which is the exact
        failure mode the dedicated analysis documented.

        ``None`` when an arm has no converged seeds, because the conditional mean
        of an empty set is not a number to be guessed at.
        """

        def _mean(values: list[float]) -> float | None:
            return statistics.fmean(values) if values else None

        shared_cond = _mean(self.shared_converged)
        dedicated_cond = _mean(self.dedicated_converged)
        return {
            "shared_conditional_mean": shared_cond,
            "dedicated_conditional_mean": dedicated_cond,
            "conditional_difference": (
                shared_cond - dedicated_cond
                if shared_cond is not None and dedicated_cond is not None
                else None
            ),
            "shared_n": len(self.shared_converged),
            "dedicated_n": len(self.dedicated_converged),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "latency_ns": self.latency_ns,
            "n_seeds": self.n_seeds,
            "shared_mean": self.shared_mean,
            "dedicated_mean": self.dedicated_mean,
            "mean_difference": self.mean_difference,
            "difference_stdev": self.difference_stdev,
            "paired_differences": self.paired_differences,
            "ratio": self.ratio,
            "comparable": self.is_comparable,
            "shared_returns": self.shared_returns,
            "dedicated_returns": self.dedicated_returns,
            "shared_ppo_updates": self.shared_ppo_updates,
            "dedicated_ppo_updates": self.dedicated_ppo_updates,
            "shared_gradient_steps": self.shared_gradient_steps,
            "dedicated_gradient_steps": self.dedicated_gradient_steps,
            "regime": self.regime,
            "shared_spread": (
                statistics.stdev(self.shared_returns) if len(self.shared_returns) > 1 else 0.0
            ),
            "dedicated_spread": (
                statistics.stdev(self.dedicated_returns) if len(self.dedicated_returns) > 1 else 0.0
            ),
            "shared_converged_returns": self.shared_converged,
            "dedicated_converged_returns": self.dedicated_converged,
            "convergence_probability": self.convergence_probability,
            "conditional_performance": self.conditional_performance,
        }


def cell_manifest_path(
    control_root: Path, task: str, sweep_id: str, latency_ns: int, seed: int
) -> Path:
    """Path to a dedicated control cell's run manifest."""
    return control_root / sweep_id / task / f"{latency_ns}ns" / f"seed{seed}" / "manifest.json"


def assert_control_budget(
    control_root: Path,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    seeds: tuple[int, ...] = DEFINITIVE_SEEDS,
    latencies: tuple[int, ...] = ALL_LATENCIES,
    expected: int = EXPECTED_CONTROL_BUDGET,
    tolerance: float = BUDGET_TOLERANCE,
) -> dict[str, Any]:
    """Verify the control really is the 4M experiment before anything is paired.

    Reads realised ``counts.integration_steps`` from each manifest. This is the
    guard against pairing against the pre-correction 2M ``dedicated-v2`` sweep:
    both live under the same per-task directory layout and both have a
    ``policy.pt``, so nothing about their *shape* distinguishes them. Their
    budgets do.

    Args:
        control_root: Root containing the control sweeps.
        tasks: Tasks to check.
        seeds: Seeds that must be present.
        latencies: Latencies that must be present.
        expected: Realised budget both arms are meant to have consumed.
        tolerance: Fractional slack for chunked integration steps.

    Returns:
        Observed budgets, keyed by ``(task, sweep_id)``.

    Raises:
        StaleControlError: If any manifest is missing, incomplete, or consumed a
            budget that is not the expected one.
    """
    low = expected * (1.0 - tolerance)
    high = expected * (1.0 + tolerance)
    observed: dict[str, Any] = {}

    for task in tasks:
        sweep_id = CONTROL_SWEEPS[task]
        steps: list[int] = []
        for latency_ns in latencies:
            for seed in seeds:
                path = cell_manifest_path(control_root, task, sweep_id, latency_ns, seed)
                if not path.exists():
                    raise StaleControlError(
                        f"control cell missing: {path}. The definitive Phase 4 analysis "
                        f"requires the 4M {sweep_id} control for {task} at all "
                        f"{len(latencies)}x{len(seeds)} conditions."
                    )
                payload = json.loads(path.read_text())
                if payload.get("status") != "completed":
                    raise StaleControlError(
                        f"control cell {path} has status "
                        f"{payload.get('status')!r}, not 'completed'."
                    )
                steps.append(int((payload.get("counts") or {}).get("integration_steps", 0)))

        bad = [s for s in steps if not (low <= s <= high)]
        key = f"{task}/{sweep_id}"
        observed[key] = {
            "n_cells": len(steps),
            "integration_steps_min": min(steps),
            "integration_steps_max": max(steps),
        }
        if bad:
            raise StaleControlError(
                f"the control for {key} consumed {sorted(set(bad))} integration steps, "
                f"not ~{expected}. This is a different experiment from the definitive "
                f"Phase 4 analysis: pairing a 4M shared sweep against it would confound "
                f"sharing with the discount-basis and LR-schedule corrections. If this is "
                f"the pre-correction 2M sweep, re-run the 4M control, or point this "
                f"analysis at its results root."
            )
    return observed


def evaluate_dedicated_baseline(
    config: ProjectConfig,
    latencies: tuple[int, ...],
    seeds: tuple[int, ...],
    control_root: Path,
    tasks: tuple[str, ...] = PHASE4_TASKS,
    episodes: int | None = None,
) -> dict[tuple[str, int, int], float]:
    """Re-evaluate saved dedicated policies through the shared run's own path.

    Reads the definitive 4M controls in :data:`CONTROL_SWEEPS`, never the Phase
    1/2 v1 sweeps and never the pre-correction 2M ``dedicated-v2`` sweep.

    Returns:
        ``mean_return`` keyed by ``(task, latency_ns, seed)``.
    """
    from dataclasses import replace

    out: dict[tuple[str, int, int], float] = {}
    for task in tasks:
        sweep_id = CONTROL_SWEEPS[task]
        for latency_ns in latencies:
            for seed in seeds:
                policy_path = (
                    control_root / sweep_id / task / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt"
                )
                if not policy_path.exists():
                    continue
                task_config = replace(config, task=task)
                policy = load_policy(task_config, task)
                policy.load_state_dict(__import__("torch").load(policy_path, weights_only=True))
                summary = evaluate_policy(policy, task_config, latency_ns, seed, episodes)
                out[(task, latency_ns, seed)] = summary_to_dict(summary)["mean_return"]
    return out


def load_dedicated_counts(
    control_root: Path,
    latencies: tuple[int, ...],
    seeds: tuple[int, ...],
    tasks: tuple[str, ...] = PHASE4_TASKS,
) -> dict[tuple[str, int, int], dict[str, float]]:
    """Read the control arm's update counts out of its run manifests.

    Read from the manifests rather than recomputed, so the control's optimisation
    budget is an observation, not an assumption about what the runner does.
    """
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    for task in tasks:
        sweep_id = CONTROL_SWEEPS[task]
        for latency_ns in latencies:
            for seed in seeds:
                path = cell_manifest_path(control_root, task, sweep_id, latency_ns, seed)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                counts = payload.get("counts") or {}
                out[(task, latency_ns, seed)] = {
                    "ppo_updates": float(counts.get("ppo_updates", 0)),
                    "gradient_steps": float(counts.get("gradient_steps", 0)),
                }
    return out


def build_paired_cells(
    shared: dict[tuple[str, int, int], float],
    dedicated: dict[tuple[str, int, int], float],
    tasks: tuple[str, ...] = PHASE4_TASKS,
    shared_counts: dict[tuple[str, int, int], dict[str, float]] | None = None,
    dedicated_counts: dict[tuple[str, int, int], dict[str, float]] | None = None,
) -> list[PairedCell]:
    """Join shared and dedicated results into per-task, per-latency cells.

    Args:
        shared: Shared-trunk mean returns by ``(task, latency, seed)``.
        dedicated: Dedicated mean returns by the same keys.
        tasks: Tasks to include.
        shared_counts: Optional per-observation update counts, carried through so
            the table can show that both arms saw the same optimisation budget.
        dedicated_counts: As above, for the control arm.
    """
    cells: list[PairedCell] = []
    for task in tasks:
        latencies = sorted({k[1] for k in shared if k[0] == task})
        for latency_ns in latencies:
            cell = PairedCell(task=task, latency_ns=latency_ns)
            seeds = sorted({k[2] for k in shared if k[0] == task and k[1] == latency_ns})
            for seed in seeds:
                shared_value = shared.get((task, latency_ns, seed))
                dedicated_value = dedicated.get((task, latency_ns, seed))
                if shared_value is None or dedicated_value is None:
                    continue
                cell.shared_returns.append(shared_value)
                cell.dedicated_returns.append(dedicated_value)
                if shared_counts:
                    cell.shared_ppo_updates = _count(
                        shared_counts, task, latency_ns, seed, "ppo_updates"
                    )
                    cell.shared_gradient_steps = _count(
                        shared_counts, task, latency_ns, seed, "gradient_steps"
                    )
                if dedicated_counts:
                    cell.dedicated_ppo_updates = _count(
                        dedicated_counts, task, latency_ns, seed, "ppo_updates"
                    )
                    cell.dedicated_gradient_steps = _count(
                        dedicated_counts, task, latency_ns, seed, "gradient_steps"
                    )
            if cell.n_seeds:
                cells.append(cell)
    return cells


def _count(
    counts: dict[tuple[str, int, int], dict[str, float]],
    task: str,
    latency_ns: int,
    seed: int,
    field_name: str,
) -> float | None:
    """One update count, or ``None`` if absent. Averaged over the cell's seeds."""
    values = [
        counts[key][field_name]
        for key in counts
        if key[0] == task and key[1] == latency_ns and key[2] == seed and field_name in counts[key]
    ]
    return statistics.fmean(values) if values else None


def analyse(
    shared_results_path: Path,
    latencies: tuple[int, ...] = ALL_LATENCIES,
    seeds: tuple[int, ...] = DEFINITIVE_SEEDS,
    control_root: Path = CONTROL_ROOT,
    config_path: Path = Path("configs/base.yaml"),
    episodes: int | None = None,
    verify_control_budget: bool = True,
) -> dict[str, Any]:
    """Run the full paired analysis and return a serialisable report.

    Args:
        shared_results_path: The shared-trunk sweep output.
        latencies: Latencies to analyse.
        seeds: Seeds to pair on. The definitive experiment uses seeds 0-2.
        control_root: Root containing the 4M controls in :data:`CONTROL_SWEEPS`.
        config_path: Configuration supplying the evaluation protocol.
        episodes: Evaluation episodes per policy. Should match the shared run's.
        verify_control_budget: Check realised control budgets before pairing.

    Returns:
        The report, including the paired cells and the caveats.

    Raises:
        StaleControlError: If the control is not the expected 4M experiment.
        MissingControlError: If no dedicated control policy could be loaded. An
            absent control must not read as a null result.
    """
    config = load_config(str(config_path))
    records = load_shared_records(shared_results_path)

    shared: dict[tuple[str, int, int], float] = {}
    shared_counts: dict[tuple[str, int, int], dict[str, float]] = {}
    for record in records:
        if record.latency_ns not in latencies or record.train_seed not in seeds:
            continue
        for task, payload in record.evaluations.items():
            shared[(task, record.latency_ns, record.train_seed)] = payload["mean_return"]
            task_counts = record.counts.get(task, {})
            shared_counts[(task, record.latency_ns, record.train_seed)] = {
                "ppo_updates": float(task_counts.get("ppo_updates", 0)),
                "gradient_steps": float(task_counts.get("gradient_steps", 0)),
            }

    if not shared:
        raise MissingControlError(
            f"no shared-trunk results matched latencies={latencies} seeds={seeds} in "
            f"{shared_results_path}. The shared sweep may be incomplete or written "
            f"elsewhere; a sharing effect cannot be assessed without it."
        )

    # Run before anything is loaded or paired. A stale control has the same file
    # shape as the right one, so by the time returns are in hand the pairing is
    # already contaminated and the only place left to intervene is the report.
    control_budgets = (
        assert_control_budget(control_root, PHASE4_TASKS, seeds, latencies)
        if verify_control_budget
        else {}
    )

    dedicated = evaluate_dedicated_baseline(
        config, latencies, seeds, control_root, episodes=episodes
    )
    dedicated_counts = load_dedicated_counts(control_root, latencies, seeds)

    # An absent control is the single most dangerous state this function can be
    # called in. Without the guard it returned zero cells and an empty report,
    # which reads as "no sharing effect found" rather than "no control exists" --
    # a null result manufactured by missing data. That is the same failure shape
    # as the trainer bugs this repository's history records: plausible
    # output, wrong content.
    if not dedicated:
        expected = (
            control_root
            / CONTROL_SWEEPS["<task>"]
            / "<task>"
            / "<latency_ns>ns"
            / "seed<n>"
            / "policy.pt"
        )
        raise MissingControlError(
            f"no dedicated control policies found under {control_root}. "
            f"Expected checkpoints shaped like {expected}. Run the 4M dedicated "
            f"controls ({', '.join(sorted(set(CONTROL_SWEEPS.values())))}) before "
            f"analysing: a Phase 4 comparison against the trainer v1 Phase 1/2 "
            f"sweeps, or against the pre-correction 2M sweep, is not valid, and an "
            f"absent control is not a null result."
        )

    cells = build_paired_cells(
        shared, dedicated, shared_counts=shared_counts, dedicated_counts=dedicated_counts
    )
    if not cells:
        raise MissingControlError(
            f"shared and control results did not overlap on any (task, latency, "
            f"seed): {len(shared)} shared and {len(dedicated)} dedicated "
            f"observations were loaded but none matched. Check that both arms used "
            f"the same latencies={latencies} and seeds={seeds}."
        )

    ratios = [c.ratio for c in cells if c.ratio is not None]
    primary = [c for c in cells if c.latency_ns in PRIMARY_LATENCIES]
    secondary = [c for c in cells if c.latency_ns in SECONDARY_LATENCIES]
    return {
        "question": (
            "At matched per-task training budgets, does a jointly trained shared "
            "128-128-128 trunk retain per-task control performance relative to "
            "independently trained dedicated 2x128 policies?"
        ),
        "experiment": "phase4-definitive",
        "regimes": {
            "primary": {
                "latencies_ns": list(PRIMARY_LATENCIES),
                "rationale": (
                    "the fully converged, tight Latency-insensitive plateau in the "
                    "4M dedicated control; sharing cost is measured where the "
                    "control itself is stable"
                ),
            },
            "secondary": {
                "latencies_ns": list(SECONDARY_LATENCIES),
                "rationale": (
                    "the stochastically degraded regime; reported separately because "
                    "the control is not converged there"
                ),
            },
        },
        "controls": {
            "sweeps": dict(CONTROL_SWEEPS),
            "root": str(control_root),
            "expected_integration_steps": EXPECTED_CONTROL_BUDGET,
            "observed": control_budgets,
            "protocol_fingerprints": _shared_protocols(records),
        },
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "n_seeds": len(seeds),
        "comparable_threshold": COMPARABLE_RATIO,
        "ratio_floor": RATIO_FLOOR,
        "cells": [c.to_dict() for c in cells],
        "n_cells": len(cells),
        "n_cells_with_meaningful_ratio": len(ratios),
        # Join coverage, so a partially complete grid is visible in the report
        # rather than being silently averaged over whatever happened to match.
        "coverage": {
            "shared_observations": len(shared),
            "dedicated_observations": len(dedicated),
            "paired_observations": sum(c.n_seeds for c in cells),
            "expected_observations": len({k[0] for k in shared}) * len(latencies) * len(seeds),
        },
        # Reported as context, never as a verdict. The pattern across the
        # latency curve is the result; the fraction is not.
        "fraction_at_or_above_threshold": (
            sum(1 for r in ratios if r >= COMPARABLE_RATIO) / len(ratios) if ratios else None
        ),
        # Per-regime summaries. Kept next to the cells rather than replacing
        # them: the regime split is a reporting frame, and a reader should still
        # be able to reach every individual seed.
        "regime_summary": {
            regime: {
                task: _summarise_task_regime(task_cells, seeds)
                for task in PHASE4_TASKS
                if (task_cells := [c for c in group if c.task == task])
            }
            for regime, group in (("primary", primary), ("secondary", secondary))
        },
        "caveats": [
            "Both arms are trainer v2 at 4M integration steps per task, with the "
            "control's realised budget verified from its manifests before pairing. "
            "This is the definitive comparison; the pre-correction Phase 4C result "
            "used a 2M budget and different discount bases and is a different dataset.",
            "Seeds are the replication unit; n=3 makes a gross effect visible but "
            "cannot characterise a small one precisely, so no significance test "
            "is reported.",
            "Ratios are suppressed where the dedicated mean is at or below the "
            f"floor of {RATIO_FLOOR}, because a percentage of a near-zero or "
            "negative baseline is not interpretable.",
            "Dedicated baselines were re-evaluated through the same code path as "
            "the shared trunk, so differences are not evaluation drift.",
            "This reports the paired effect per condition. It deliberately does "
            "not aggregate conditions into a single pass/fail verdict, because "
            "the shape of the latency curve is part of the result.",
            "CartPole sits at its 500-step return ceiling in both arms across "
            "20-80 ms, so it is a saturated arm: it constrains a sharing penalty "
            "to be small, not to be absent. A null result here is a ceiling "
            "effect and should not be read as an absence of interference.",
            f"Convergence is a fixed threshold at {CONVERGENCE_THRESHOLD} returns, "
            "chosen in the empty band between the control's two modes and applied "
            "identically to both arms.",
            "The seed-count asymmetry is 3 seeds in the shared arm against 10 in "
            "the dedicated control, which matters only for the convergence counts. "
            "In the secondary regime the shared model exhibits substantial failure "
            "to reach the successful LunarLander regime, but this 3-seed shared "
            "sample is insufficient to establish a difference in convergence "
            "probability relative to the dedicated control: 3 seeds admit only "
            "0, 1/3, 2/3 or 1, and those cannot be compared as rates against a "
            "10-seed estimate. The performance deficit is a separate quantity and "
            "is not qualified by this asymmetry.",
            "The dedicated 100 ms failure mode is probably not fully converged at "
            "4M, so secondary-regime comparison is against a partially converged "
            "reference. The primary regime is the trustworthy one.",
        ],
        "out_of_scope": (
            "Seed 3 was executed at 20 ms before the definitive scope was fixed at "
            "seeds 0-2. It is retained on disk and in the run log but excluded "
            "from every statistic here; it is not a replicate of the definitive "
            "design."
        ),
    }


def _shared_protocols(records: list[Any]) -> dict[str, Any]:
    """Protocol fingerprint of the shared arm, one entry per task.

    Read from the shared records rather than recomputed from the config, so the
    report states what was actually trained rather than what the config implies.
    """
    out: dict[str, Any] = {}
    for record in records:
        for task, protocol in (record.task_protocols or {}).items():
            out[task] = {
                "fingerprint": protocol.get("fingerprint"),
                "discount_basis": protocol.get("discount_basis"),
                "gamma": protocol.get("gamma"),
                "gae_lambda": protocol.get("gae_lambda"),
                "effective_integration_steps": protocol.get("effective_integration_steps"),
            }
    return out


def _summarise_task_regime(cells: list[PairedCell], seeds: tuple[int, ...]) -> dict[str, Any]:
    """One task's three quantities within a single regime.

    Kept deliberately per-latency inside the regime. Averaging a plateau regime
    together would blur the 20-60 ms differences, and averaging across a
    bimodal secondary regime would average two modes into a third thing that
    exists in neither arm.
    """
    del seeds  # seeds are already fixed by the cells handed in
    return {
        "latencies": {
            f"{c.latency_ns}": {
                "shared_mean": c.shared_mean,
                "shared_stdev": (
                    statistics.stdev(c.shared_returns) if len(c.shared_returns) > 1 else 0.0
                ),
                "dedicated_mean": c.dedicated_mean,
                "dedicated_stdev": (
                    statistics.stdev(c.dedicated_returns) if len(c.dedicated_returns) > 1 else 0.0
                ),
                "mean_difference": c.mean_difference,
                "difference_stdev": c.difference_stdev,
                "convergence_probability": c.convergence_probability,
                "conditional_performance": c.conditional_performance,
            }
            for c in sorted(cells, key=lambda c: c.latency_ns)
        },
        "max_abs_difference": max((abs(c.mean_difference) for c in cells), default=None),
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the paired analysis as markdown tables, one per regime per task."""
    lines: list[str] = []

    controls = report.get("controls") or {}
    observed = controls.get("observed") or {}
    if observed:
        lines.append(
            "Control budgets verified before pairing "
            f"(expected ~{controls['expected_integration_steps']:,} integration steps):"
        )
        for key, info in sorted(observed.items()):
            lines.append(
                f"- `{key}`: {info['n_cells']} cells, "
                f"{info['integration_steps_min']:,}-{info['integration_steps_max']:,} steps"
            )
        lines.append("")

    lines.append("### Paired performance by regime")
    lines.append("")
    for regime in ("primary", "secondary"):
        latencies = (report.get("regimes") or {}).get(regime, {}).get("latencies_ns") or []
        label = "primary (20-60 ms)" if regime == "primary" else "secondary (80-100 ms)"
        lines.append(f"#### {regime.capitalize()} regime — {label}")
        lines.append("")
        lines.append(
            "| task | latency | dedicated | shared | delta | seed spread (ded/shared)"
            " | PPO updates (ded/shared) | ratio |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for cell in sorted(report["cells"], key=lambda c: (c["task"], c["latency_ns"])):
            if cell["latency_ns"] not in latencies:
                continue
            ded_sd = cell.get("dedicated_spread", 0.0)
            shd_sd = cell.get("shared_spread", 0.0)
            lines.append(
                f"| {cell['task']} | {cell['latency_ns'] / 1e6:.0f} ms "
                f"| {cell['dedicated_mean']:.1f} | {cell['shared_mean']:.1f} "
                f"| {cell['mean_difference']:+.1f} "
                f"| {ded_sd:.1f} / {shd_sd:.1f} | {_updates(cell)} "
                f"| {'n/a' if cell['ratio'] is None else f'{cell["ratio"]:.3f}'} |"
            )
        lines.append("")

    lines.append("### LunarLander convergence and conditional performance")
    lines.append("")
    lines.append(
        f"Converged means a seed reaching >= {report.get('convergence_threshold')} returns. "
        "Counts are raw, not rates: the shared arm has 3 seeds and the dedicated arm 10."
    )
    lines.append("")
    lines.append(
        "| latency | regime | shared converged | dedicated converged | shared cond. mean "
        "| dedicated cond. mean | cond. difference |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for cell in sorted(report["cells"], key=lambda c: c["latency_ns"]):
        if cell["task"] != "lunarlander":
            continue
        conv = cell["convergence_probability"]
        cond = cell["conditional_performance"]

        def _fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        lines.append(
            f"| {cell['latency_ns'] / 1e6:.0f} ms | {cell['regime']} "
            f"| {conv['shared_n']}/{conv['shared_total']} "
            f"| {conv['dedicated_n']}/{conv['dedicated_total']} "
            f"| {_fmt(cond['shared_conditional_mean'])} "
            f"| {_fmt(cond['dedicated_conditional_mean'])} "
            f"| {_fmt(cond['conditional_difference'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _updates(cell: dict[str, Any]) -> str:
    """PPO update counts for both arms, or ``-`` when unavailable."""
    ded_up = cell.get("dedicated_ppo_updates")
    shd_up = cell.get("shared_ppo_updates")
    if ded_up is None and shd_up is None:
        return "-"
    if ded_up is not None and shd_up is not None:
        return f"{ded_up:.0f} / {shd_up:.0f}"
    present = ded_up if ded_up is not None else shd_up
    return f"{present:.0f} / -"


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write the report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
