"""Command-line interface.

Subcommands are added as modules land, keeping the entry point stable while
the pipeline grows. The sweep subcommand is the one that runs experiments; the
others are read-only, because inspecting results should never be able to change
them.

``latency-ml sweep`` is deliberately the only command that writes to
``results/``. Everything else reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from latency_ml.analysis.transfer_init import default_source_path, source_sha256
from latency_ml.config import FinanceConfig
from latency_ml.experiments.manifest import DEFAULT_RESULTS_ROOT, read_manifest
from latency_ml.experiments.provenance import verify_sweep
from latency_ml.experiments.runner import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_WORKER_THREADS,
    run_sweep,
)
from latency_ml.finance.market import MarketStream


def _cmd_capacity(args: argparse.Namespace) -> int:
    """Run the 2x2 and report the measured Slow axis with Fast left pending."""
    import platform

    import torch

    from latency_ml.finance.capacity import (
        format_table,
        run_24,
        slow_capacity_gap,
    )

    experiment = run_24(n_repeats=args.repeats, warmup=args.warmup)
    print("capacity x execution (inference latency per observation)")
    print(format_table(experiment))
    print()
    for model, m in experiment.measurements.items():
        print(
            f"  {model:>5} ({m.hidden_width:>3} wide): median {m.median_ns / 1000:6.2f} us"
            f"  [p10 {m.p10_ns / 1000:5.2f}, p90 {m.p90_ns / 1000:6.2f}]"
            f"  {m.threads} threads {m.dtype}"
        )
    gap = slow_capacity_gap(experiment)
    if gap is not None:
        print(f"\n2x128 costs {gap:+.1%} more inference latency than 2x32 on this CPU")
    print("\nFast (Hartley) cells are PENDING_HARDWARE_MODEL and carry no number:")
    print("  throughput, clock, memory and precision are unfrozen (PLAN.md 812).")

    payload = experiment.as_dict()
    payload["capacity_gap_relative"] = gap
    payload["host"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def _cmd_shared_trunk(args: argparse.Namespace) -> int:
    """Run the Phase 4 shared-trunk sweep and report the update accounting."""
    from latency_ml.config import load_config
    from latency_ml.experiments.shared_runner import PHASE4_TASKS, run_shared_sweep

    config = load_config(args.config)
    latencies = tuple(
        args.latencies if args.latencies is not None else config.latency.sweep_grid_ns
    )
    print("Phase 4 shared-trunk generality")
    print(f"  tasks          : {', '.join(PHASE4_TASKS)}")
    print(f"  latencies (ms) : {', '.join(str(lat // 1_000_000) for lat in latencies)}")
    print(f"  seeds          : {', '.join(str(s) for s in args.seeds)}")
    budget = args.budget or config.training.total_integration_steps
    print(f"  per-task budget: {budget:,} integration steps (consumed once per task)")
    print(f"  episodes/task  : {args.episodes or config.evaluation.control_episodes_per_seed}")

    # Resolved once, before any cell runs. The source must not depend on the
    # cell: resolving it per cell from the cell's own latency and seed would
    # quietly give each condition a different starting policy, which is the one
    # thing the pre-registration forbids. "default" expands to the frozen source.
    #
    # B3: bare `--init-source` used to expand to transfer_init.DEFAULT_SOURCE_CELL,
    # which is a frozen *v1* checkpoint trained under an unbounded ReLU. In this
    # repository that is not a source at all.
    #
    # The transfer reproduces a source's post-activation hidden layer, so it is
    # only meaningful when the nonlinearities match. A ReLU-trained source dropped
    # into a bounded trunk does not yield the same representation -- verified
    # directly -- so the arm would have been measuring something that was not the
    # intended intervention, while reporting itself as a clean bit-exact transfer.
    # A silent default that performs a semantically void experiment is the same
    # class of defect as a hold band that quietly collapses to a no-op.
    #
    # So there is no default, and `--init-source` now requires its argument at the
    # parser level. The branch below catches the one remaining way to ask for a
    # default by name -- typing the word -- because that otherwise fails with a bare
    # "checkpoint not found: default", which explains nothing. The frozen cell
    # itself stays available to tests, which is what it is for.
    init_source: Path | None = None
    if args.init_source is not None:
        if args.init_source == "default":
            raise SystemExit(
                "--init-source requires an explicit checkpoint path in v2; no "
                "semantically valid default exists. The frozen default "
                f"({default_source_path()}) is a v1 checkpoint trained under an "
                "unbounded ReLU, and the transfer is only exact between matching "
                "nonlinearities. Use this repository's own solved LunarLander "
                "policy, for example:\n"
                "  --init-source results/latency-lunarlander/lunarlander/"
                "20000000ns/seed0/policy.pt"
            )
        init_source = Path(args.init_source)
        if not init_source.exists():
            raise SystemExit(
                f"--init-source checkpoint not found: {init_source}. Phase 4D needs a "
                "solved dedicated LunarLander policy from this repository; there is "
                "no default."
            )
        source_digest = source_sha256(
            torch.load(init_source, map_location="cpu", weights_only=True)
        )
        print(f"  init source    : {init_source}")
        print(f"  source sha256  : {source_digest}")
        print("                   (fixed for every cell: initialisation is")
        print("                    deliberately latency- and seed-independent)")

    records = run_shared_sweep(
        config,
        latencies,
        tuple(args.seeds),
        args.out,
        total_integration_steps=args.budget,
        episodes=args.episodes,
        log_interval=args.log_interval,
        max_workers=args.workers,
        worker_threads=args.threads_per_worker,
        config_path=args.config,
        init_source=init_source,
        curve_every=args.curve_every,
        curve_episodes=args.curve_episodes,
        save_checkpoints=not args.no_checkpoints,
    )

    print("\nupdate accounting (must match each task's dedicated K exactly):")
    seen: dict[int, dict[str, int]] = {}
    for record in records:
        seen[record.latency_ns] = record.planned_iterations
    for latency_ns, planned in sorted(seen.items()):
        row = ", ".join(f"{t}={planned[t]}" for t in sorted(planned))
        equal = len(set(planned.values())) == 1
        print(f"  {latency_ns / 1e6:5.0f} ms: {row}  {'OK' if equal else 'MISMATCH'}")

    print("\nmean return, shared trunk:")
    header = "  latency |" + "".join(f"{t:>15}" for t in PHASE4_TASKS)
    print(header)
    for latency_ns in latencies:
        cells = [r for r in records if r.latency_ns == latency_ns]
        cells.sort(key=lambda r: r.train_seed)
        if not cells:
            continue
        parts = []
        for task in PHASE4_TASKS:
            values = [r.evaluations[task]["mean_return"] for r in cells]
            parts.append(f"{sum(values) / len(values):15.1f}")
        print(f"  {latency_ns / 1e6:5.0f} ms |" + "".join(parts))

    print(f"\nwrote {args.out / 'shared_trunk.json'}")
    return 0


def _cmd_historical(args: argparse.Namespace) -> int:
    """Run the fixed-policy latency sweep on a LOBSTER session.

    Same policy, same execution model, same cost model, same retained-value
    metric as the synthetic Stage A. The market is the only thing that changes.
    """
    from latency_ml.finance.lobster import load_session
    from latency_ml.finance.replay import stream_digest
    from latency_ml.finance.stage_a import sweep_stage_a, train_policy

    try:
        stream = load_session(args.data)
    except FileNotFoundError as exc:
        print("HISTORICAL        : PENDING_DATA")
        print(f"  {exc}")
        return 2

    print(f"loaded {len(stream)} events from {args.data}")
    policy, cfg, normaliser, diag = train_policy(stream, "small", epochs=12)
    delays = tuple(range(0, args.max_delay + 1, args.step))
    points = sweep_stage_a(stream, policy, cfg, delays, 1.0, normaliser)
    print(f"diagnostics: {diag}")
    print("\n delay_events      delay_ns   mean_pnl    floor_pnl   retained")
    for q in points:
        retained = "n/a" if q.retained is None else f"{q.retained:9.3f}"
        print(
            f" {q.delay_events:>10} {q.delay_ns:>13} {q.mean_pnl:>10.5f}"
            f" {q.floor_pnl:>11.5f} {retained}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "source": str(args.data),
                    "n_events": len(stream),
                    # Provenance: every latency point below is a replay of this
                    # exact event stream, so a reader can confirm that from the
                    # artifact rather than taking it on trust.
                    "stream_sha256": stream_digest(stream),
                    "diagnostics": diag,
                    "points": [
                        {
                            "delay_events": q.delay_events,
                            "delay_ns": q.delay_ns,
                            "mean_pnl": q.mean_pnl,
                            "floor_pnl": q.floor_pnl,
                            "retained": q.retained,
                            "n_trades": q.n_trades,
                        }
                        for q in points
                    ],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


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

    mechanism = sub.add_parser(
        "mechanism",
        help="Phase 3 mechanism test: does the synthetic market contain a decaying opportunity?",
    )
    mechanism.add_argument("--config", required=True)
    mechanism.add_argument(
        "--max-delay",
        type=int,
        default=200,
        help="Largest execution delay to sweep, in events.",
    )
    mechanism.add_argument("--step", type=int, default=10, help="Delay increment in events.")
    mechanism.add_argument("--out", type=Path, default=None, help="Write results as JSON.")
    mechanism.add_argument("--plot", type=Path, default=None, help="Write the sweep figure here.")

    diag = sub.add_parser(
        "diagnose-target",
        help="Characterise the trade-direction target on real LOBSTER sessions. "
        "Read-only; selects no band and trains nothing.",
    )
    diag.add_argument("--root", default="data/lobster", help="Directory of sessions.")
    diag.add_argument("--data", default=None, help="One session directory instead.")
    diag.add_argument(
        "--session",
        action="append",
        default=None,
        help="Substring filter on session names. Repeatable.",
    )
    diag.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Order-book depth to read. Defaults to 1; load_session otherwise "
        "reads every level in the filename, which is 1.4 GB for SPY_50.",
    )
    diag.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200],
        help="Forecast horizons in events. Event-count H is retired for the "
        "real-data target -- see --horizons-ns -- and is kept only for the "
        "evidence that retired it.",
    )
    diag.add_argument(
        "--horizons-ns",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons as durations in ns. Selects the wall-clock target "
        "definition, t_H = min{t_i : t_i - t_0 >= H}, and reports V(tau) across "
        "the latency grid. Overrides --horizons.",
    )
    diag.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Integer HOLD bands, in price granules.",
    )
    diag.add_argument(
        "--latency-grid",
        type=int,
        nargs="+",
        default=None,
        help="Execution latency grid in ns, swept independently of H.",
    )
    diag.add_argument(
        "--roles",
        action="append",
        default=None,
        help="Label sessions as 'primary' or 'reference' in the report. "
        "Repeatable, matched to session order.",
    )
    diag.add_argument("--out", default=None, help="Also write the full report as JSON.")

    stage_a = sub.add_parser(
        "stage-a",
        help="Stage A: one trained MLP replayed across execution delays.",
    )
    stage_a.add_argument("--config", required=True)
    stage_a.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Order-book depth to load when finance.source is a LOBSTER session. "
        "Defaults to 1; load_session otherwise reads every level in the filename.",
    )
    stage_a.add_argument("--model", default="small", help="'small' (2x32) or 'large' (2x128).")
    stage_a.add_argument(
        "--arm",
        default=None,
        choices=["primary", "secondary", "heldout", "legacy"],
        help="Frozen V2B arm, per PROTOCOL.md section 12. 'primary' is AAPL 10s/k=6, "
        "'secondary' AAPL 5s/k=6, 'heldout' AMZN 25s/k=6. 'legacy' uses the "
        "event-count horizon and quantile band the synthetic mechanism runs and the "
        "Phase G cells used.",
    )
    stage_a.add_argument(
        "--latency-grid-ns",
        type=int,
        nargs="+",
        default=None,
        help="Execution latency grid in ns, resolved per decision from the clock. "
        "Required by the frozen arms; --max-delay/--step are the legacy event grid.",
    )
    stage_a.add_argument("--max-delay", type=int, default=200)
    stage_a.add_argument("--step", type=int, default=5)
    stage_a.add_argument(
        "--oracle-ref",
        type=Path,
        default=None,
        help="Frozen perfect-forecaster ceiling to report V_policy/V_oracle against. "
        "The stream digest is checked first and a mismatch is refused.",
    )
    stage_a.add_argument(
        "--emit-block-series",
        action="store_true",
        help="Store the per-window lift series, so this arm can be compared with "
        "another by paired difference rather than by differencing two independent "
        "standard errors.",
    )
    stage_a.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Policy initialisation seed. The no-information baseline is held at a "
        "fixed seed regardless, so a difference between two arms is attributable to "
        "the policy rather than to the control.",
    )
    stage_a.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Historical context in events. This is the independent variable of "
        "experiment L1 (PROTOCOL.md section 14) and must match between training and "
        "replay; it is threaded through both rather than defaulted.",
    )
    stage_a.add_argument(
        "--eval-start",
        type=int,
        default=None,
        help="First event to evaluate. Defaults to the end of the training split, so "
        "the reported region is unseen. TradingConfig.holdout_events exists so the "
        "evaluation region is untouched and this is what honours it; without it the "
        "sweep replays the fitted events too and reports in-sample numbers as if "
        "they were clean.",
    )
    stage_a.add_argument("--out", type=Path, default=None, help="Write results as JSON.")
    stage_a.add_argument("--plot", type=Path, default=None, help="Write the figure here.")

    sel = sub.add_parser(
        "selectivity",
        help="Diagnostic: does the trained policy's own confidence identify when to "
        "act? Retrains nothing; a threshold on frozen weights.",
    )
    sel.add_argument("--config", required=True)
    sel.add_argument("--levels", type=int, default=1)
    sel.add_argument("--model", default="small")
    sel.add_argument(
        "--arm",
        default="primary",
        choices=["primary", "secondary", "heldout"],
        help="Frozen arm whose target is used. See PROTOCOL.md section 12.",
    )
    sel.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.34, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
        help="Confidence thresholds. 0 is the frozen policy: trade whenever the "
        "argmax is a direction.",
    )
    sel.add_argument(
        "--latency-grid-ns",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000, 10_000_000, 30_000_000, 100_000_000],
    )
    sel.add_argument("--oracle-ref", type=Path, default=None)
    sel.add_argument("--out", type=Path, default=None)

    paired = sub.add_parser(
        "paired-lookback",
        help="Paired comparison of two lookback arms on shared windows. Reads "
        "stage-a outputs that were written with --emit-block-series; trains nothing.",
    )
    paired.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="stage-a JSON files carrying a block_series.",
    )
    paired.add_argument(
        "--baseline",
        type=int,
        required=True,
        help="Lookback of the reference arm, subtracted from every other.",
    )
    paired.add_argument("--out", type=Path, default=None)

    capacity = sub.add_parser(
        "capacity",
        help="2x2 model-capacity x execution-speed experiment (Fast pending).",
    )
    capacity.add_argument("--repeats", type=int, default=5000)
    capacity.add_argument("--warmup", type=int, default=500)
    capacity.add_argument("--out", type=Path, default=None)

    shared = sub.add_parser(
        "shared-trunk",
        help="Phase 4: shared-trunk generality sweep across both control tasks.",
    )
    shared.add_argument("--config", required=True)
    shared.add_argument(
        "--latencies",
        type=int,
        nargs="+",
        default=None,
        help="Latencies in ns. Defaults to the config sweep grid.",
    )
    shared.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    shared.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Integration steps PER TASK. Defaults to the config budget.",
    )
    shared.add_argument("--episodes", type=int, default=None)
    shared.add_argument("--out", type=Path, default=Path("results/shared-trunk"))
    shared.add_argument("--log-interval", type=int, default=0)
    shared.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. 1 runs serially; see the dedicated sweep for the"
        " thread-budget flag.",
    )
    shared.add_argument("--threads-per-worker", type=int, default=2)
    shared.add_argument(
        "--curve-every",
        type=int,
        default=0,
        help=(
            "Record a joint training-curve point every this many integration "
            "steps. 0 disables. Added after Phase 4D, whose cells recorded no "
            "curve and so could not distinguish learned-then-plateaued from "
            "still-climbing at the budget."
        ),
    )
    shared.add_argument("--curve-episodes", type=int, default=20)
    shared.add_argument(
        "--no-checkpoints",
        action="store_true",
        help=(
            "Skip writing each cell's final policy. Off by default: a shared cell "
            "without a checkpoint cannot be re-evaluated independently."
        ),
    )
    shared.add_argument(
        "--init-source",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Phase 4D: initialise the shared trunk from a solved dedicated "
            "checkpoint instead of randomly. An explicit path is REQUIRED; there is "
            "no default, because the only frozen source is a v1 checkpoint trained "
            "under an unbounded ReLU and the transfer is exact only between matching "
            "nonlinearities. The source is deliberately independent of each cell's "
            "latency and seed, so one checkpoint is used for the whole sweep."
        ),
    )

    historical = sub.add_parser(
        "historical",
        help="Fixed-policy latency sweep on a LOBSTER session.",
    )
    historical.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Directory holding *_orderbook_10.csv and *_message_10.csv.",
    )
    historical.add_argument("--max-delay", type=int, default=200)
    historical.add_argument("--step", type=int, default=5)
    historical.add_argument("--out", type=Path, default=None)

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


def load_finance_stream(finance: FinanceConfig, levels: int | None = None) -> MarketStream:
    """Build the stream named by ``finance.source`` / ``finance.data_path``.

    These two config fields were previously read by nothing at all: ``stage-a``
    always generated synthetic data and only ``historical --data`` ever touched
    LOBSTER, so a config saying ``source: lobster_sample`` would still train on
    synthetic data without complaint. A configuration field that silently does
    nothing is worse than no field, because it is documentation that lies.

    ``levels`` is passed explicitly because ``load_session`` otherwise reads every
    level declared in the filename -- up to 1.4 GB for the SPY_50 session -- and a
    caller asking for a touch-only diagnostic should not pay for depth it does not
    use.
    """
    if finance.source == "synthetic":
        from latency_ml.finance.market import GeneratorParams, generate

        return generate(GeneratorParams(**finance.synthetic))
    if finance.source in ("lobster", "lobster_sample", "historical"):
        from latency_ml.finance.lobster import load_session

        return load_session(Path(finance.data_path), levels=levels)
    raise SystemExit(
        f"unknown finance.source {finance.source!r}; expected "
        f"'synthetic', 'lobster' or 'lobster_sample'"
    )


def _cmd_diagnose_target(args: argparse.Namespace) -> int:
    """Characterise the trade-direction target on real sessions.

    Read-only and model-free. It reports what the data does with a given horizon
    and band so a target definition can be chosen from evidence, and it decides
    nothing: no band is selected here.

    Two target parameterisations are available. ``--horizons`` measures the
    look-ahead window in events; ``--horizons-ns`` measures it as a duration and
    adds ``V(tau)``, the value a perfect forecaster retains when the action lands
    ``tau`` after the observation. The second is the real-data definition: the
    first is not a clock, spanning a 40x range of durations across these symbols
    at the same event count.
    """
    import json

    from latency_ml.finance.lobster import load_session
    from latency_ml.finance.target_diag import (
        diagnose_session,
        diagnose_wall_clock_session,
        format_report,
        format_wall_clock_report,
        sessions_under,
    )

    tau_grid: tuple[int, ...] | None = None
    if args.latency_grid:
        tau_grid = tuple(args.latency_grid)

    if args.data is not None:
        sessions = [Path(args.data)]
    else:
        root = Path(args.root)
        sessions = sessions_under(root)
        if args.session:
            wanted = set(args.session)
            sessions = [p for p in sessions if any(w in p.name for w in wanted)]
    if not sessions:
        print(f"no LOBSTER sessions found under {args.root}", file=sys.stderr)
        return 2

    roles = tuple(args.roles) if args.roles else ("primary",) * len(sessions)
    if len(roles) != len(sessions):
        print(
            f"--roles has {len(roles)} entries but {len(sessions)} sessions were selected",
            file=sys.stderr,
        )
        return 2

    if args.horizons_ns:
        if not tau_grid:
            print("--horizons-ns needs --latency-grid to report V(tau)", file=sys.stderr)
            return 2
        diagnostics = []
        for session, role in zip(sessions, roles, strict=True):
            # Depth is always explicit. load_session with no level reads every level in
            # the filename -- 1.4 GB for SPY_50 -- and this diagnostic only reads the
            # touch.
            stream = load_session(session, levels=args.levels)
            diagnostics.append(
                diagnose_wall_clock_session(
                    stream,
                    session.name,
                    horizon_ns_grid=tuple(args.horizons_ns),
                    ks=tuple(args.bands),
                    tau_grid_ns=tau_grid,
                    role=role,
                )
            )
        print(format_wall_clock_report(diagnostics, tau_grid_ns=tau_grid))
        print(
            "H is a duration and tau is a delay on the same clock. V(tau) assumes a\n"
            "perfect forecaster under aggressive touch crossing, so it is an upper\n"
            "bound on what training could earn, not a prediction of it. tau_zero is\n"
            "where that bound hits zero: below the grid's smallest delay there is\n"
            "nothing left to measure, so no latency sweep at that target is worth\n"
            "running."
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "target_definition": "wall_clock_ns",
                        "levels": args.levels,
                        "horizons_ns": list(args.horizons_ns),
                        "bands_ticks": list(args.bands),
                        "latency_grid_ns": list(tau_grid),
                        "execution_assumption": "aggressive touch crossing (frozen)",
                        "value_model": "perfect forecaster, upper bound",
                        "sessions": [d.to_dict() for d in diagnostics],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print(f"wrote {out}")
        return 0

    event_diagnostics = []
    for session in sessions:
        # Depth is always explicit. load_session with no level reads every level in
        # the filename -- 1.4 GB for SPY_50 -- and this diagnostic only reads the
        # touch.
        stream = load_session(session, levels=args.levels)
        event_diagnostics.append(
            diagnose_session(
                stream,
                session.name,
                horizons=tuple(args.horizons),
                ks=tuple(args.bands),
            )
        )

    print(format_report(event_diagnostics, tau_grid_ns=tau_grid))
    for diag in event_diagnostics:
        print(f"  {diag.session}: granule {diag.granule_units} units. {diag.granule_note}.")
    print(
        "\nNo band is selected here. Any k chosen from this table must be re-derived "
        "on the\ntraining split before use, and the cost columns describe the current "
        "aggressive-fill\nmodel rather than a view on it."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "levels": args.levels,
                    "horizons": list(args.horizons),
                    "bands_ticks": list(args.bands),
                    "latency_grid_ns": list(tau_grid) if tau_grid else None,
                    "sessions": [d.to_dict() for d in event_diagnostics],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {out}")
    return 0


def _cmd_selectivity(args: argparse.Namespace) -> int:
    """Ask whether the policy's confidence separates opportunities.

    A diagnostic of one trained policy, not a second experiment: the frozen result
    stands whatever this shows, and no threshold chosen here feeds back into it. The
    question is whether the failure is directional skill or selectivity -- the two
    call for completely different next steps, and the frozen run cannot tell them
    apart.
    """
    import json

    from latency_ml.config import load_config
    from latency_ml.finance.selectivity import (
        format_selectivity_table,
        selectivity_curve,
    )
    from latency_ml.finance.stage_a import build_labels_wall_clock, train_policy
    from latency_ml.finance.target import FROZEN_HELDOUT, FROZEN_PRIMARY, FROZEN_SECONDARY

    target = {
        "primary": FROZEN_PRIMARY,
        "secondary": FROZEN_SECONDARY,
        "heldout": FROZEN_HELDOUT,
    }[args.arm]
    cfg = load_config(args.config)
    stream = load_finance_stream(cfg.finance, levels=args.levels)
    policy, _trading_cfg, normaliser, fit = train_policy(stream, args.model, target=target)
    labels, _summary = build_labels_wall_clock(stream, target)
    eval_start = int(fit["n_train"])
    n_eval = len(stream) - eval_start
    print(
        f"arm {args.arm}: H={target.horizon_ms / 1000:g}s k={target.band_ticks}, "
        f"train acc {fit['train_accuracy']:.3f}"
    )
    print(
        f"held-out region: events [{eval_start:,} of {len(stream):,}) -- {n_eval:,} "
        f"events, none of them fitted"
    )

    capture_map: dict[int, float] | None = None
    granule_units: int | None = None
    if args.oracle_ref is not None:
        from latency_ml.finance.oracle_ref import load_ceiling, verify_against

        ceiling = load_ceiling(args.oracle_ref)
        verify_against(ceiling, stream, target)
        capture_map = dict(zip(ceiling.tau_grid_ns, ceiling.values, strict=True))
        granule_units = ceiling.granule_units
    else:
        granule_units = None

    points = selectivity_curve(
        stream,
        policy,
        normaliser,
        target,
        labels,
        thresholds=tuple(args.thresholds),
        delay_grid_ns=tuple(args.latency_grid_ns),
        eval_start=eval_start,
        transaction_cost_bps=cfg.finance.transaction_cost_bps,
        oracle_capture=capture_map,
        oracle_granule=granule_units,
    )
    print()
    print(format_selectivity_table(points, n_eval))
    print(
        "\nThis is a diagnostic of the frozen weights. It does not replace the "
        "frozen result:\nthe policy as trained trades every event the argmax is a "
        "direction, and that\nis what the pre-registered run reports."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "model": args.model,
                    "target": target.to_dict(),
                    "n_events": len(stream),
                    "eval_start": eval_start,
                    "n_evaluated": n_eval,
                    "train_accuracy": fit["train_accuracy"],
                    "thresholds": list(args.thresholds),
                    "latency_grid_ns": list(args.latency_grid_ns),
                    "oracle_ref": str(args.oracle_ref) if args.oracle_ref else None,
                    "points": [p.to_dict() for p in points],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {args.out}")
    return 0


def _cmd_paired_lookback(args: argparse.Namespace) -> int:
    """Compare lookback arms by paired difference on the windows they share.

    Every arm is evaluated on the same out-of-sample events, so the two error sources
    are positively correlated and differencing their independent standard errors
    overstates the uncertainty of the difference. This differs the per-window series
    directly instead.

    The aggregate across seeds is stated rather than assumed: the three seeds share
    the same windows, so the dominant sampling error is common to them and does *not*
    shrink with the seed count. Only initialisation variance does.
    """
    import json

    import numpy as np

    from latency_ml.finance.selectivity import (
        aggregate_across_seeds,
        paired_difference,
    )

    arms: dict[tuple[int, int], list[float]] = {}
    digests: set[str] = set()
    for path in args.results:
        payload = json.loads(Path(path).read_text())
        series = payload.get("block_series")
        if not series:
            print(
                f"{path} has no block_series; re-run stage-a with --emit-block-series",
                file=sys.stderr,
            )
            return 2
        arms[(int(payload["lookback"]), int(payload["seed"]))] = [float(v) for v in series]
        digests.add(str(payload.get("stream_digest")))

    if len(digests) != 1:
        print(f"arms span {len(digests)} stream digests; they are not comparable", file=sys.stderr)
        return 2

    seeds = sorted({s for _lookback, s in arms})
    others = sorted({lb for lb, _s in arms if lb != args.baseline})
    print(f"reference arm: L={args.baseline}; seeds {seeds}; one shared stream")
    print()

    rows: list[dict[str, Any]] = []
    for lookback in others:
        per_seed = []
        for seed in seeds:
            base = arms.get((args.baseline, seed))
            other = arms.get((lookback, seed))
            if base is None or other is None:
                continue
            per_seed.append(
                paired_difference(
                    f"L{lookback}-L{args.baseline}", np.asarray(other), np.asarray(base)
                )
            )
        if not per_seed:
            continue
        agg = aggregate_across_seeds(f"L{lookback}-L{args.baseline}", per_seed)
        print(f"  L={lookback} - L={args.baseline}")
        for d in per_seed:
            print(
                f"    seed-wise  {d.mean:+.4f}  SE {d.standard_error:.4f}  "
                f"95% CI [{d.ci_low:+.4f}, {d.ci_high:+.4f}]"
                f"  {'excludes 0' if d.significant else ''}"
            )
        print(
            f"    aggregate  {agg.mean:+.4f}  SE {agg.standard_error:.4f}  "
            f"95% CI [{agg.ci_low:+.4f}, {agg.ci_high:+.4f}]"
            f"  {'EXCLUDES 0' if agg.significant else 'includes 0'}"
        )
        print("    (aggregate SE keeps the shared-window sampling term undiminished; only the")
        print("     seed term is reduced by the number of seeds)")
        print()
        rows.append(agg.to_dict())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "baseline_lookback": args.baseline,
                    "seeds": seeds,
                    "n_blocks": next(iter(arms.values())).__len__(),
                    "aggregation": (
                        "se = sqrt(median(within-seed paired se)^2 + "
                        "between_seed_sd^2/n_seeds); the within-window term is not "
                        "divided by the seed count because all seeds are scored on the "
                        "same windows"
                    ),
                    "comparisons": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote {args.out}")
    return 0


def _cmd_stage_a(args: argparse.Namespace) -> int:
    """Dispatch the stage-a subcommand.

    One model, trained once at zero latency, then replayed. Nothing in this path
    is latency-dependent, so any variation in the output is attributable to
    execution timing alone.
    """
    from latency_ml.config import load_config
    from latency_ml.finance.replay import stream_digest
    from latency_ml.finance.stage_a import FROZEN_FLOOR_SEED, sweep_stage_a, train_policy
    from latency_ml.finance.target import FROZEN_HELDOUT, FROZEN_PRIMARY, FROZEN_SECONDARY

    cfg = load_config(args.config)
    stream = load_finance_stream(cfg.finance, levels=args.levels)
    print(
        f"stream: {len(stream)} events from "
        f"{'synthetic' if cfg.finance.source == 'synthetic' else cfg.finance.data_path}"
    )

    target = {
        "primary": FROZEN_PRIMARY,
        "secondary": FROZEN_SECONDARY,
        "heldout": FROZEN_HELDOUT,
    }.get(args.arm)
    if args.arm in ("primary", "secondary", "heldout") and not args.latency_grid_ns:
        print(
            f"--arm {args.arm} is defined on the nanosecond latency grid; pass "
            "--latency-grid-ns. The legacy --max-delay/--step grid is an index shift "
            "and does not mean the same thing on a bursty book.",
            file=sys.stderr,
        )
        return 2
    if target is not None:
        print(
            f"frozen arm {args.arm}: H={target.horizon_ms:g} ms, k={target.band_ticks} "
            "granules, aggressive touch crossing"
        )
    # E6: the configured horizon now reaches training rather than coinciding with it.
    policy, trading_cfg, normaliser, fit = train_policy(
        stream,
        model=args.model,
        horizon_events=cfg.finance.prediction_horizon_events,
        target=target,
        lookback=args.lookback,
        seed=args.seed,
    )
    delays = tuple(range(0, args.max_delay + 1, args.step))
    grid_ns = tuple(args.latency_grid_ns) if args.latency_grid_ns else None
    eval_start = args.eval_start if args.eval_start is not None else int(fit["n_train"])
    print(
        f"evaluating on events [{eval_start:,} of {len(stream):,}); "
        f"{len(stream) - eval_start:,} held out, {eval_start:,} excluded as fitted"
    )
    points = sweep_stage_a(
        stream,
        policy,
        trading_cfg,
        delays,
        cfg.finance.transaction_cost_bps,
        normaliser,
        delay_grid_ns=grid_ns,
        eval_start=eval_start,
        lookback=args.lookback,
    )

    print(
        f"model={args.model} L={args.lookback} seed={args.seed} "
        f"hold_band={fit['hold_band']:.4f} train_acc={fit['train_accuracy']:.3f}"
    )
    lift_row = None
    block_series = None
    if target is not None:
        from latency_ml.finance.selectivity import emit_block_series, oos_directional_lift
        from latency_ml.finance.stage_a import build_labels_wall_clock

        labels, _summary = build_labels_wall_clock(stream, target)
        if args.emit_block_series:
            block_series = emit_block_series(
                stream,
                policy,
                normaliser,
                labels,
                eval_start=eval_start,
                lookback=args.lookback,
                horizon_ns=target.horizon_ns,
            )
        summary = oos_directional_lift(
            stream,
            policy,
            normaliser,
            labels,
            eval_start=eval_start,
            lookback=args.lookback,
            horizon_ns=target.horizon_ns,
            block_series=block_series,
        )
        lift_row = summary
        se = summary.lift_standard_error
        print(
            f"PRIMARY endpoint (L1): OOS directional lift = {summary.lift:+.4f} "
            f"+/- {se:.4f} (2 SE) on {summary.n_effective:.0f} non-overlapping "
            f"windows"
        )
        print(
            f"  acc|trade {summary.accuracy_given_trade:.4f} - constant-direction "
            f"baseline {summary.base_rate:.4f}, from {summary.n_directional:,} "
            f"directional events | distinguishable from zero: "
            f"{'YES' if summary.lift_significant else 'no'}"
        )
    print(f"{'delay':>6}{'V(d)':>11}{'floor':>11}{'retained':>10}{'trades':>9}")
    for p in points:
        r = "n/a" if p.retained is None else f"{p.retained:.3f}"
        print(f"{p.delay_events:>6}{p.mean_pnl:>11.5f}{p.floor_pnl:>11.5f}{r:>10}{p.n_trades:>9}")

    ratio_rows = None
    if args.oracle_ref is not None:
        from latency_ml.finance.oracle_ref import (
            format_ratio_table,
            load_ceiling,
            ratio_row,
        )

        ceiling = load_ceiling(args.oracle_ref)
        # Verified before any ratio is printed. A ceiling from different bytes would
        # make every number below meaningless while looking entirely normal.
        if target is not None:
            from latency_ml.finance.oracle_ref import verify_against

            verify_against(ceiling, stream, target)
        # Keyed by requested delay, which is what the ceiling is indexed by.
        policy_values = {int(p.delay_events): float(p.mean_pnl) for p in points}
        ratio_rows = ratio_row(ceiling, policy_values)
        print()
        print(f"oracle ceiling ({ceiling.session}, digest {ceiling.stream_digest[:16]}):")
        print(format_ratio_table(ratio_rows))

    results: dict[str, Any] = {
        "model": args.model,
        "arm": args.arm,
        "n_events": len(stream),
        "eval_start": eval_start,
        "n_evaluated": len(stream) - eval_start,
        # Provenance: the digest the oracle ceiling is checked against, recorded so a
        # result can be tied to the exact bytes it was computed on without re-running.
        "stream_digest": stream_digest(stream),
        "target": target.to_dict() if target is not None else None,
        "latency_grid_ns": list(grid_ns) if grid_ns else None,
        "oracle_ratio": [r.to_dict() for r in ratio_rows] if ratio_rows else None,
        "lookback": args.lookback,
        "seed": args.seed,
        "floor_seed": FROZEN_FLOOR_SEED,
        "oos_lift": lift_row.to_dict() if lift_row else None,
        "block_series": block_series,
        "oracle_ref": str(args.oracle_ref) if args.oracle_ref else None,
        "fit": fit,
        "transaction_cost_bps": cfg.finance.transaction_cost_bps,
        "delays_events": [p.delay_events for p in points],
        "delays_ns": [p.delay_ns for p in points],
        "mean_pnl": [p.mean_pnl for p in points],
        "floor_pnl": [p.floor_pnl for p in points],
        "retained": [p.retained for p in points],
        "n_trades": [p.n_trades for p in points],
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if args.plot is not None:
        from latency_ml.experiments.plotting import plot_stage_a

        plot_stage_a(results, args.plot)
    return 0


def _cmd_mechanism(args: argparse.Namespace) -> int:
    """Dispatch the mechanism subcommand.

    Runs the oracle sweep plus both controls. Exits non-zero if either control
    fails, so the mechanism cannot be silently declared successful on a curve
    that survives removing its cause.
    """
    from latency_ml.config import load_config
    from latency_ml.finance.market import (
        GeneratorParams,
        decoupled_control,
        with_persistence,
    )
    from latency_ml.finance.market import (
        generate as generate_stream,
    )
    from latency_ml.finance.mechanism import decay_fraction, sweep_opportunity

    cfg = load_config(args.config)
    base = GeneratorParams(**cfg.finance.synthetic)
    delays = tuple(range(0, args.max_delay + 1, args.step))

    conditions = {
        "short_lived": base,
        "ablation_persistent": with_persistence(base, "persistent"),
        "control_decoupled": decoupled_control(base),
    }
    results: dict[str, Any] = {"delays_events": list(delays), "conditions": {}}
    print(f"{'condition':>22}" + "".join(f"{d:>9}" for d in delays))
    for name, params in conditions.items():
        points = sweep_opportunity(
            generate_stream(params),
            delays,
            markout_events=cfg.finance.prediction_horizon_events,
            transaction_cost_bps=cfg.finance.transaction_cost_bps,
        )
        results["conditions"][name] = {
            "mean_pnl": [p.mean_pnl for p in points],
            "sem": [p.sem for p in points],
            "signal_alive_fraction": [p.signal_alive_fraction for p in points],
            "decay_fraction": decay_fraction(points),
        }
        print(f"{name:>22}" + "".join(f"{p.mean_pnl:>9.4f}" for p in points))

    # The decoupled control is judged on the flatness of its raw means, not on
    # decay_fraction: its baseline is negative, so there is no opportunity to
    # lose and a normalised ratio would be meaningless.
    decoupled = results["conditions"]["control_decoupled"]["mean_pnl"]
    decoupled_flat = (max(decoupled) - min(decoupled)) < 0.01
    short = abs(results["conditions"]["short_lived"]["decay_fraction"])
    abl = abs(results["conditions"]["ablation_persistent"]["decay_fraction"])
    ok = abl < short / 4 and decoupled_flat
    print()
    print(f"short-lived decay : {short:.1%}")
    print(f"ablation decay    : {abl:.1%}  (must be < 1/4 of short-lived)")
    print(f"decoupled flat    : {'yes' if decoupled_flat else 'NO'}")
    print(f"MECHANISM         : {'PASS' if ok else 'FAIL'}")
    if not ok:
        if not decoupled_flat:
            print("  the decoupled control is not flat, so the curve is not coming")
            print("  from the state-to-flow link -- check the replay layer.")
        if abl >= short / 4:
            print("  the curve survives removing latent persistence, so it is not")
            print("  generated by the market process -- check the generator.")
        return 1
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if args.plot is not None:
        from latency_ml.experiments.plotting import plot_opportunity_sweep

        plot_opportunity_sweep(results, args.plot)
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

    # Per-cell provenance, checked here because this is the last moment the sweep
    # knows what it just produced. The evaluation artifact is deliberately NOT
    # required yet: nothing has been evaluated at this point, and a check that
    # fires on the normal ordering of commands is a check people learn to skip.
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

    import torch

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

    # Written to disk unconditionally, not only under --json.
    #
    # `plot latency-response` reads an evaluation file from disk, and a sweep's
    # aggregate is the artefact a result gets compared against -- including
    # against the previous version. Printing it and letting the operator redirect
    # stdout means the file exists only if someone remembered, and a result that
    # depends on that is not reproducible from the sweep.
    out = args.out or (args.root / args.sweep_id / "evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Now the evaluation artifact exists, so the sweep-level requirement becomes
    # meaningful. Checked here rather than only under `verify` because this is the
    # point at which a sweep first becomes quotable, and a sweep that silently
    # produced nothing evaluable is exactly what this catches.
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
    if args.command == "diagnose-target":
        return _cmd_diagnose_target(args)
    if args.command == "timescale":
        return _cmd_timescale(args)
    if args.command == "mechanism":
        return _cmd_mechanism(args)
    if args.command == "stage-a":
        return _cmd_stage_a(args)
    if args.command == "selectivity":
        return _cmd_selectivity(args)
    if args.command == "paired-lookback":
        return _cmd_paired_lookback(args)
    if args.command == "capacity":
        return _cmd_capacity(args)
    if args.command == "shared-trunk":
        return _cmd_shared_trunk(args)
    if args.command == "historical":
        return _cmd_historical(args)
    if args.command == "plot":
        return _cmd_plot(args)
    return _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
