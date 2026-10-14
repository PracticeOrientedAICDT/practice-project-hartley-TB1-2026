"""Control-latency command-line interface.

Kept subcommands: sweep (train), evaluate, plot, timescale, verify, show.
Finance and Phase 4 shared-trunk subcommands were removed with their code;
see MIGRATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from latency_ml.experiments.manifest import DEFAULT_RESULTS_ROOT, read_manifest
from latency_ml.experiments.provenance import verify_sweep
from latency_ml.experiments.runner import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_WORKER_THREADS,
    run_sweep,
)


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="latency-ml",
        description="Latency-aware control and inference experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Run a latency x seed PPO sweep.")
    sweep.add_argument("--config", required=True, help="Path to a YAML config.")
    sweep.add_argument("--sweep-id", required=True, help="Identifier for this sweep.")
    sweep.add_argument(
        "--latencies",
        type=int,
        nargs="*",
        default=None,
        help="Latency conditions in ns. Defaults to the config's sweep grid.",
    )
    sweep.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Training seeds. Defaults to the config's replicate count.",
    )
    sweep.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the integration-step budget, for a pilot run.",
    )
    sweep.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    sweep.add_argument("--threads-per-worker", type=int, default=DEFAULT_WORKER_THREADS)
    sweep.add_argument(
        "--curve-every",
        type=int,
        default=0,
        help="Record a training-curve point every N integration steps (0 = off).",
    )
    sweep.add_argument(
        "--curve-episodes", type=int, default=20, help="Episodes per training-curve point."
    )
    sweep.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    sweep.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun cells already recorded in the ledger.",
    )
    sweep.add_argument("--quiet", action="store_true")
    sweep.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="Permit manifests with no protocol fingerprint. Only for artefacts "
        "frozen before the fingerprint existed.",
    )

    verify = sub.add_parser(
        "verify",
        help="Check results on disk for provenance: counts, wall time, "
        "checkpoint, protocol fingerprint, evaluation artifact.",
    )
    verify.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    verify.add_argument(
        "--sweep-id",
        default=None,
        help="Limit to one sweep. Default checks everything under --root.",
    )
    verify.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Do not require evaluation.json. Use when auditing a sweep that has "
        "been trained but not yet evaluated, where its absence is expected.",
    )
    verify.add_argument(
        "--no-checkpoint", action="store_true", help="Do not require checkpoint files."
    )
    verify.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="Permit manifests with no protocol fingerprint. Only for artefacts "
        "frozen before the fingerprint existed, such as tests/fixtures/v1.",
    )
    verify.add_argument("-v", "--verbose", action="store_true", help="List every cell.")
    verify.add_argument("--quiet", action="store_true", help="Suppress detail.")

    show = sub.add_parser("show", help="Print one run's manifest.")
    show.add_argument("run_id", help="Run identifier, e.g. <sweep>/<task>/<ns>ns/seed0")
    show.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)

    evaluate = sub.add_parser(
        "evaluate", help="Evaluate a sweep's checkpoints and print the latency response."
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--sweep-id", required=True)
    evaluate.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    evaluate.add_argument(
        "--episodes", type=int, default=None, help="Override the configured episode count."
    )
    evaluate.add_argument(
        "--cross",
        action="store_true",
        help="Also evaluate each policy at every other training latency.",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write evaluation.json. Defaults to <root>/<sweep-id>/.",
    )
    evaluate.add_argument("--json", action="store_true", help="Print JSON as well as writing it.")
    evaluate.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="Permit manifests with no protocol fingerprint. Only for artefacts "
        "frozen before the fingerprint existed.",
    )

    timescale = sub.add_parser(
        "timescale",
        help="Measure each task's response timescale, independent of any policy.",
    )
    timescale.add_argument(
        "--tasks",
        nargs="+",
        default=["cartpole", "lunarlander"],
        help="Tasks to measure.",
    )
    timescale.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    timescale.add_argument(
        "--out", type=Path, default=None, help="Also write the measurement to this JSON file."
    )

    plot = sub.add_parser(
        "plot",
        help="Regenerate a figure from committed results.",
    )
    plot_sub = plot.add_subparsers(dest="figure", required=True)

    response = plot_sub.add_parser(
        "latency-response",
        help="Mean return against decision interval, from an evaluation.json.",
    )
    response.add_argument("--evaluation", required=True, type=Path)
    response.add_argument("--out", required=True, type=Path)
    response.add_argument("--title", default=None)

    curves = plot_sub.add_parser(
        "training-curves",
        help="Return against integration steps, one line per latency, from a sweep.",
    )
    curves.add_argument("--root", required=True, type=Path)
    curves.add_argument("--sweep-id", required=True)
    curves.add_argument("--task", required=True)
    curves.add_argument("--out", required=True, type=Path)
    curves.add_argument("--title", default=None)

    return parser


def _cmd_plot(args: argparse.Namespace) -> int:
    """Regenerate a figure from committed results.

    A subcommand rather than an ad-hoc script, so every figure in RESULTS.md has
    a documented, reproducible generation path. A one-off command is how a figure
    ends up with no record of how it was made.
    """
    from latency_ml.experiments.plotting import (
        _load_aggregates,
        plot_latency_response,
        plot_training_curves,
    )

    if args.figure == "latency-response":
        if not args.evaluation.exists():
            print(f"no evaluation file at {args.evaluation}", file=sys.stderr)
            return 1
        aggregates = _load_aggregates(args.evaluation)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.title:
            plot_latency_response(aggregates, args.out, title=args.title)
        else:
            plot_latency_response(aggregates, args.out)
        seeds = aggregates[0].get("n_seeds") if aggregates else None
        print(f"wrote {args.out} ({len(aggregates)} latencies, n_seeds={seeds})")
        return 0

    if not args.root.exists():
        print(f"no sweep root at {args.root}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.title:
        plot_training_curves(args.root, args.sweep_id, args.task, args.out, title=args.title)
    else:
        plot_training_curves(args.root, args.sweep_id, args.task, args.out)
    print(f"wrote {args.out} (sweep {args.sweep_id}, task {args.task})")
    return 0


def _cmd_timescale(args: argparse.Namespace) -> int:
    """Dispatch the timescale subcommand.

    The measurement is PPO-free by construction, so it is a separate command
    rather than a flag on sweep: it can be re-run, and its definition re-checked,
    without touching a training run.
    """
    from latency_ml.analysis.timescale import measure_task_timescale, summarise
    from latency_ml.timing import NATIVE_DT_ENV_NS

    per_task: dict[str, dict[str, Any]] = {}
    rows: list[tuple[str, int, float]] = []
    for task in args.tasks:
        measured = measure_task_timescale(task, dt_ns=NATIVE_DT_ENV_NS)
        per_task[task] = {
            "per_action": {
                str(a): {
                    "efold_ms": t.efold_ms,
                    "per_seed_ms": [ns / 1e6 for ns in t.per_seed_ns],
                }
                for a, t in measured.items()
            },
            "summary_ms": summarise(measured),
        }
        rows.extend((task, a, t.efold_ms) for a, t in measured.items())

    payload: dict[str, Any] = {"dt_env_ns": NATIVE_DT_ENV_NS, "tasks": per_task}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{'task':<14}{'action':>7}{'T_task (ms)':>13}")
        for task, action, ms in rows:
            print(f"{task:<14}{action:>7}{ms:>13.1f}")
        print()
        for task in args.tasks:
            s = per_task[task]["summary_ms"]
            print(
                f"{task:<14} min {s['min_ms']:>8.1f}"
                f"  median {s['median_ms']:>8.1f}"
                f"  max {s['max_ms']:>8.1f}"
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Dispatch the sweep subcommand.

    Returns a process exit code: 1 if any cell failed, so a shell script driving
    a sweep does not have to parse stdout to notice.
    """
    outcomes = run_sweep(
        config_path=args.config,
        sweep_id=args.sweep_id,
        latencies_ns=args.latencies,
        train_seeds=args.seeds,
        total_integration_steps=args.steps,
        curve_every=args.curve_every,
        curve_episodes=args.curve_episodes,
        max_workers=args.workers,
        worker_threads=args.threads_per_worker,
        root=args.root,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
    failed = [o for o in outcomes if o.status == "failed"]

    report = verify_sweep(
        args.root,
        args.sweep_id,
        require_evaluation=False,
        allow_missing_provenance=args.allow_missing_provenance,
    )
    print(f"provenance: {len(report.cells)} cells, {'complete' if report.ok else 'INCOMPLETE'}")
    for problem in report.problems:
        print(f"  {problem}", file=sys.stderr)
    bad = [c for c in report.cells if not c.ok]
    for cell in bad[:5]:
        print(f"  {cell.describe()}", file=sys.stderr)
    if len(bad) > 5:
        print(f"  ... and {len(bad) - 5} more", file=sys.stderr)

    return 1 if (failed or bad or not report.ok) else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Dispatch the verify subcommand: provenance of results already on disk."""
    report = verify_sweep(
        args.root,
        args.sweep_id,
        require_evaluation=not args.no_evaluation,
        allow_missing_provenance=args.allow_missing_provenance,
        require_checkpoint=not args.no_checkpoint,
    )
    if args.verbose:
        print(report.describe())
    else:
        print(
            f"{len(report.cells)} cells in {report.root}: "
            f"{'complete' if report.ok else 'INCOMPLETE'}"
        )
        for problem in report.problems:
            print(f"  {problem}", file=sys.stderr)
    if not report.ok and not args.quiet:
        for cell in report.cells:
            if not cell.ok:
                print(cell.describe(), file=sys.stderr)
    return 0 if report.ok else 1


def _cmd_show(args: argparse.Namespace) -> int:
    """Dispatch the show subcommand."""
    import json

    manifest = read_manifest(args.run_id, args.root)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Dispatch the evaluate subcommand.

    Loads every completed checkpoint in a sweep, evaluates it, and prints the
    matched-training latency response. Cross-latency evaluation is opt-in via
    ``--cross`` because it is the secondary design (section 17) and costs
    len(latencies) times as much.
    """
    import json

    from latency_ml.config import load_config
    from latency_ml.experiments.evaluate import (
        aggregate_by_latency,
        cross_latency_matrix,
        evaluate_policy,
        load_policy,
        summarise_matrix,
    )
    from latency_ml.experiments.manifest import read_manifest
    from latency_ml.models.architectures import MLPPolicy

    config = load_config(args.config)
    sweep_dir = args.root / args.sweep_id / config.task
    if not sweep_dir.exists():
        print(f"no runs for sweep {args.sweep_id!r} under {sweep_dir}", file=sys.stderr)
        return 1

    policies: dict[int, dict[int, MLPPolicy]] = {}
    for latency_dir in sorted(sweep_dir.iterdir(), key=lambda p: int(p.name.removesuffix("ns"))):
        latency_ns = int(latency_dir.name.removesuffix("ns"))
        for seed_dir in sorted(
            latency_dir.iterdir(), key=lambda p: int(p.name.removeprefix("seed"))
        ):
            manifest = read_manifest(
                f"{args.sweep_id}/{config.task}/{latency_dir.name}/{seed_dir.name}", args.root
            )
            if manifest.status != "completed" or not manifest.checkpoint_path:
                print(f"skipping {manifest.run_id}: status={manifest.status}", file=sys.stderr)
                continue
            policy = load_policy(config, config.task)
            policy.load_state_dict(
                torch.load(
                    latency_dir / seed_dir.name / manifest.checkpoint_path, weights_only=True
                )
            )
            policies.setdefault(latency_ns, {})[int(seed_dir.name.removeprefix("seed"))] = policy

    if not policies:
        print("no completed runs to evaluate", file=sys.stderr)
        return 1

    summaries = [
        evaluate_policy(policy, config, latency_ns, seed, args.episodes)
        for latency_ns, seeds in sorted(policies.items())
        for seed, policy in sorted(seeds.items())
    ]
    aggregates = aggregate_by_latency(summaries)

    payload: dict[str, object] = {
        "sweep_id": args.sweep_id,
        "n_policies": len(summaries),
        "latency_response": [a.to_dict() for a in aggregates],
    }

    if args.cross:
        latencies = tuple(sorted(policies))
        cells = cross_latency_matrix(policies, config, latencies, args.episodes)
        cross_summary: dict[str, Any] = summarise_matrix(
            cells, args.episodes or config.evaluation.control_episodes_per_seed
        )
        payload["cross_latency"] = cross_summary
        payload["cross_cells"] = [c.to_dict() for c in cells]

    out = args.out or (args.root / args.sweep_id / "evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if out == (args.root / args.sweep_id / "evaluation.json"):
        report = verify_sweep(
            args.root,
            args.sweep_id,
            require_evaluation=True,
            allow_missing_provenance=args.allow_missing_provenance,
        )
        print(f"provenance: {len(report.cells)} cells, {'complete' if report.ok else 'INCOMPLETE'}")
        if not report.ok:
            for problem in report.problems:
                print(f"  {problem}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"sweep {args.sweep_id}: {len(summaries)} policies, matched training")
    print(f"wrote {out}")
    print(f"{'latency':>10}  {'N':>2}  {'mean return':>12}  {'sd(seeds)':>10}  independent")
    for aggregate in aggregates:
        flag = "yes" if aggregate.is_independent_measurement else "no"
        print(
            f"{aggregate.latency_ns / 1e6:>7.0f} ms  {aggregate.action_steps:>2}  "
            f"{aggregate.mean_return:>12.1f}  {aggregate.std_across_seeds:>10.1f}  {flag}"
        )
    if args.cross:
        print(
            f"\ncross-latency (secondary): grand mean "
            f"{cross_summary['grand_mean_return']:.1f}, matched mean "
            f"{cross_summary['matched_mean_return']:.1f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a subcommand.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "sweep":
        return _cmd_sweep(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "timescale":
        return _cmd_timescale(args)
    if args.command == "plot":
        return _cmd_plot(args)
    return _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
