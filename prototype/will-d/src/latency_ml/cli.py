from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from latency_ml.config import load_config
from latency_ml.experiments.runner import make_activation, run_sweep
from latency_ml.experiments.shared_runner import run_shared_sweep

DEFAULT_RESULTS_ROOT = "results"
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _act(value: str | None) -> str:
    if not value:
        return "capped-relu"
    v = value.strip().lower()
    if v in ("relu", "torch-relu", "unbounded"):
        return "relu"
    if v in ("hartree", "hmlp-ti", "hti"):
        return "hartree"
    return "capped-relu"


def _default_config(phase: str) -> Path:
    name = phase if phase in ("lunarlander", "cartpole") else "base"
    return CONFIGS_DIR / f"{name}.yaml"


def _normalise_latencies(config, values):
    if values is not None:
        return tuple(values)
    return tuple(config.latency.sweep_grid_ns)


def _normalise_seeds(config, values):
    if values is not None:
        return tuple(values)
    return tuple(range(config.training.control_training_seeds))


def _eval_activation(activation: str):
    if activation == "relu":
        return make_activation("relu")
    if activation == "hartree":
        return "hartree"
    return make_activation("capped-relu")


def _evaluate_dedicated(config, sweep_id, root, episodes, activation) -> None:
    import torch

    from latency_ml.experiments.evaluate import aggregate_by_latency, evaluate_policy, load_policy

    base = Path(root) / sweep_id / config.task
    if not base.exists():
        print(f"no runs for sweep {sweep_id}", file=sys.stderr)
        return
    policies: dict[int, dict[int, object]] = {}
    for latdir in sorted(base.iterdir(), key=lambda p: int(p.name.removesuffix("ns"))):
        for seedir in sorted(latdir.iterdir(), key=lambda p: int(p.name.removeprefix("seed"))):
            ckpt = seedir / "policy.pt"
            if not ckpt.exists():
                continue
            policy = load_policy(config, config.task, activation=activation)
            policy.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            policies.setdefault(int(latdir.name.removesuffix("ns")), {})[int(seedir.name.removeprefix("seed"))] = policy
    if not policies:
        print("no completed runs", file=sys.stderr)
        return
    summaries = [evaluate_policy(policy, config, latency_ns, seed, episodes) for latency_ns, seeds in sorted(policies.items()) for seed, policy in sorted(seeds.items())]
    payload = {
        "sweep_id": sweep_id,
        "n_policies": len(summaries),
        "latency_response": aggregate_by_latency(summaries),
    }
    out = Path(root) / sweep_id / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"sweep {sweep_id}: {len(summaries)} policies, wrote {out}")


def _evaluate_shared(configs, sweep_id, root, latencies, seeds, activation, episodes) -> None:
    from latency_ml.experiments.evaluate import aggregate_by_latency, evaluate_policy
    from latency_ml.models.shared_trunk import build_shared_trunk_policy
    from latency_ml.rl.task_policy import SharedTaskPolicy

    model = build_shared_trunk_policy(("cartpole", "lunarlander"), activation=_eval_activation(activation))
    for eval_task in ("cartpole", "lunarlander"):
        cfg = configs[eval_task]
        summaries = []
        for latency_ns in latencies:
            for seed in seeds:
                path = Path(root) / sweep_id / "shared" / f"{latency_ns}ns" / f"seed{seed}" / "policy.pt"
                if not path.exists():
                    continue
                model.load_shared(path)
                summaries.append(evaluate_policy(SharedTaskPolicy(model, eval_task), cfg, latency_ns, seed, episodes))
        if not summaries:
            continue
        out_dir = Path(root) / f"{sweep_id}-shared-{eval_task}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sweep_id": f"{sweep_id}-shared-{eval_task}",
            "n_policies": len(summaries),
            "latency_response": aggregate_by_latency(summaries),
        }
        (out_dir / "evaluation.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out_dir / 'evaluation.json'}")


def _cmd_train(args) -> int:
    activation = _act(args.activation)
    root = Path(args.root)
    phase = args.phase
    if activation == "hartree" and phase != "cartpole":
        print("hartree arm is cartpole-dedicated for now (use --phase cartpole)", file=sys.stderr)
        return 2
    config = load_config(Path(args.config) if args.config else _default_config(phase))
    latencies = _normalise_latencies(config, args.latencies)
    seeds = _normalise_seeds(config, args.seeds)
    if phase in ("cartpole", "lunarlander"):
        outcomes = run_sweep(
            config_path=str(args.config) if args.config else str(_default_config(phase)),
            sweep_id=args.sweep_id,
            latencies_ns=latencies,
            train_seeds=seeds,
            total_integration_steps=args.steps,
            max_workers=args.workers,
            worker_threads=args.threads,
            root=root,
            resume=not args.no_resume,
            verbose=not args.quiet,
            activation=activation,
        )
        failed = [o for o in outcomes if o.status == "failed"]
        _evaluate_dedicated(config, args.sweep_id, root, args.eval_episodes, activation)
        return 1 if failed else 0
    configs = {t: load_config(CONFIGS_DIR / f"{t}.yaml") for t in ("cartpole", "lunarlander")}
    outcomes = run_shared_sweep(
        config_path=str(args.config) if args.config else str(_default_config(phase)),
        sweep_id=args.sweep_id,
        latencies_ns=latencies,
        train_seeds=seeds,
        total_integration_steps=args.steps,
        max_workers=args.workers,
        worker_threads=args.threads,
        root=root,
        resume=not args.no_resume,
        verbose=not args.quiet,
        activation=activation,
        tasks=("cartpole", "lunarlander"),
        episodes=args.eval_episodes,
        log_interval=200,
    )
    failed = [o for o in outcomes if o["status"] == "failed"]
    _evaluate_shared(configs, args.sweep_id, root, latencies, seeds, activation, args.eval_episodes)
    return 1 if failed else 0


def _cmd_push(args) -> int:
    from latency_ml.experiments.push import run_push_sweep

    config = load_config(Path(args.config)) if args.config else load_config(CONFIGS_DIR / f"{args.task}.yaml")
    run_push_sweep(
        sweep_id=args.sweep_id,
        task=args.task,
        kind=args.kind,
        config_path=str(args.config) if args.config else str(CONFIGS_DIR / f"{args.task}.yaml"),
        root=Path(args.root),
        latencies_ns=list(_normalise_latencies(config, args.latencies)),
        train_seeds=list(_normalise_seeds(config, args.seeds)),
        activation=_act(args.activation),
        trials=args.trials,
        horizon=args.horizon,
        verbose=not args.quiet,
    )
    return 0


def _cmd_render(args) -> int:
    from latency_ml.experiments.render import render_missing

    render_missing(
        root=Path(args.root),
        sweep_ids=list(args.sweep_id or ()) or None,
        episode_seed=args.episode_seed,
        fps=args.fps,
        max_frames=args.max_frames,
        modes=("control", "push") if args.mode == "both" else (args.mode,),
        verbose=not args.quiet,
    )
    return 0


def _cmd_view(args) -> int:
    out = Path(args.out)
    subprocess.run(
        [sys.executable, str(CONFIGS_DIR.parent / "scripts" / "build_site.py"), "--results", str(args.results), "--out", str(out / "index.html")],
        check=False,
    )
    if not args.build_only:
        import http.server
        import socketserver

        os.chdir(out)
        with socketserver.TCPServer(("", args.port), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"serving on http://localhost:{args.port}")
            with contextlib.suppress(KeyboardInterrupt):
                httpd.serve_forever()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="latency-ml")
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="Train one phase / activation sweep.")
    tr.add_argument("--phase", choices=["cartpole", "lunarlander", "shared"], default="cartpole")
    tr.add_argument("--activation", choices=["relu", "capped-relu", "hartree"], default="capped-relu")
    tr.add_argument("--config", default=None)
    tr.add_argument("--sweep-id", required=True)
    tr.add_argument("--latencies", type=int, nargs="*", default=None)
    tr.add_argument("--seeds", type=int, nargs="*", default=None)
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--eval-episodes", type=int, default=None)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--threads", type=int, default=2)
    tr.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    tr.add_argument("--no-resume", action="store_true")
    tr.add_argument("--quiet", action="store_true")
    pu = sub.add_parser("push", help="Random-push recovery phase over a trained sweep.")
    pu.add_argument("--sweep-id", required=True)
    pu.add_argument("--task", default="cartpole")
    pu.add_argument("--kind", choices=["auto", "dedicated", "shared"], default="auto")
    pu.add_argument("--config", default=None)
    pu.add_argument("--latencies", type=int, nargs="*", default=None)
    pu.add_argument("--seeds", type=int, nargs="*", default=None)
    pu.add_argument("--activation", choices=["relu", "capped-relu", "hartree"], default="capped-relu")
    pu.add_argument("--trials", type=int, default=20)
    pu.add_argument("--horizon", type=int, default=500)
    pu.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    pu.add_argument("--quiet", action="store_true")
    vi = sub.add_parser("view", help="Build / serve the results site.")
    vi.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS_ROOT))
    vi.add_argument("--out", type=Path, default=Path("site"))
    vi.add_argument("--port", type=int, default=8000)
    vi.add_argument("--build-only", action="store_true")
    re = sub.add_parser("render", help="Record rollout videos for trained sweeps.")
    re.add_argument("--sweep-id", nargs="*", default=None)
    re.add_argument("--episode-seed", type=int, default=7)
    re.add_argument("--fps", type=int, default=30)
    re.add_argument("--max-frames", type=int, default=360)
    re.add_argument("--mode", choices=["control", "push", "both"], default="both")
    re.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    re.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "push":
        return _cmd_push(args)
    if args.command == "render":
        return _cmd_render(args)
    if args.command == "view":
        return _cmd_view(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
