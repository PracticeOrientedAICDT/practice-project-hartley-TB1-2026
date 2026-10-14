from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from latency_ml.config import load_config
from latency_ml.experiments.manifest import DEFAULT_RESULTS_ROOT
from latency_ml.experiments.runner import make_activation, run_sweep
from latency_ml.experiments.shared_runner import train_shared_cell


def _act(value: str | None) -> str:
    if not value:
        return "capped-relu"
    v = value.strip().lower()
    if v in ("relu", "torch-relu", "unbounded"):
        return "relu"
    return "capped-relu"


def _default_config(phase: str) -> Path:
    if phase == "lunarlander":
        return Path("configs/lunarlander.yaml")
    if phase == "cartpole":
        return Path("configs/cartpole.yaml")
    return Path("configs/base.yaml")


def _normalise_latencies(config, values):
    if values is not None:
        return tuple(values)
    return tuple(config.latency.sweep_grid_ns)


def _normalise_seeds(config, values):
    if values is not None:
        return tuple(values)
    return tuple(range(config.training.control_training_seeds))


def _evaluate_dedicated(config, sweep_id: str, root: Path, episodes: int | None, activation: str) -> None:
    import torch

    from latency_ml.experiments.evaluate import aggregate_by_latency, evaluate_policy, load_policy
    from latency_ml.experiments.manifest import read_manifest

    task = config.task
    sweep_dir = root / sweep_id / task
    if not sweep_dir.exists():
        print(f"no runs for sweep {sweep_id!r} under {sweep_dir}", file=sys.stderr)
        return
    policies: dict[int, dict[int, Any]] = {}
    for latency_dir in sorted(sweep_dir.iterdir(), key=lambda p: int(p.name.removesuffix("ns"))):
        latency_ns = int(latency_dir.name.removesuffix("ns"))
        for seed_dir in sorted(latency_dir.iterdir(), key=lambda p: int(p.name.removeprefix("seed"))):
            run_id = f"{sweep_id}/{task}/{latency_dir.name}/{seed_dir.name}"
            try:
                manifest = read_manifest(run_id, root)
            except FileNotFoundError:
                continue
            if manifest.status != "completed" or not manifest.checkpoint_path:
                continue
            policy = load_policy(config, task, activation=activation)
            policy.load_state_dict(torch.load(latency_dir / seed_dir.name / manifest.checkpoint_path, weights_only=True))
            policies.setdefault(latency_ns, {})[int(seed_dir.name.removeprefix("seed"))] = policy
    if not policies:
        print("no completed runs to evaluate", file=sys.stderr)
        return
    summaries = [evaluate_policy(policy, config, latency_ns, seed, episodes) for latency_ns, seeds in sorted(policies.items()) for seed, policy in sorted(seeds.items())]
    aggregates = aggregate_by_latency(summaries)
    payload: dict[str, object] = {"sweep_id": sweep_id, "n_policies": len(summaries), "latency_response": [a.to_dict() for a in aggregates]}
    out = root / sweep_id / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"sweep {sweep_id}: {len(summaries)} policies, wrote {out}")


def _evaluate_shared(configs, sweep_id, root, latencies, seeds, activation, episodes):
    from latency_ml.experiments.evaluate import aggregate_by_latency, evaluate_policy
    from latency_ml.models.shared_trunk import build_shared_trunk_policy
    from latency_ml.rl.task_policy import SharedTaskPolicy

    tasks = ("cartpole", "lunarlander")
    if activation == "relu":
        import torch.nn as nn

        act = nn.ReLU()
    else:
        from latency_ml.models.activations import bounded_relu

        act = bounded_relu()
    model = build_shared_trunk_policy(tasks, activation=act)
    for eval_task in tasks:
        cfg = configs[eval_task]
        summaries = []
        for latency_ns in latencies:
            for seed in seeds:
                path = root / sweep_id / "shared" / (str(latency_ns) + "ns") / ("seed" + str(seed)) / "policy.pt"
                if not path.exists():
                    continue
                model.load_shared(path)
                view = SharedTaskPolicy(model, eval_task)
                summaries.append(evaluate_policy(view, cfg, latency_ns, seed, episodes))
        if not summaries:
            continue
        aggregates = aggregate_by_latency(summaries)
        payload = {"sweep_id": sweep_id + "-shared-" + eval_task, "n_policies": len(summaries), "latency_response": [a.to_dict() for a in aggregates]}
        out_dir = root / (sweep_id + "-shared-" + eval_task)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation.json").write_text(__import__("json").dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out_dir / 'evaluation.json'}")


def _cmd_train(args) -> int:
    activation = _act(args.activation)
    root = Path(args.root)
    phase = args.phase
    config_path = Path(args.config) if args.config else _default_config(phase)
    config = load_config(config_path)
    latencies = _normalise_latencies(config, args.latencies)
    seeds = _normalise_seeds(config, args.seeds)
    if phase in ("cartpole", "lunarlander"):
        outcomes = run_sweep(
            config_path=config_path,
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
    act_mod = make_activation(activation)
    if activation == "relu":
        import torch.nn as nn

        act_mod = nn.ReLU()
    else:
        from latency_ml.models.activations import bounded_relu

        act_mod = bounded_relu()
    tasks = ("cartpole", "lunarlander")
    configs = {t: load_config(Path("configs") / (t + ".yaml")) for t in tasks}
    for latency_ns in latencies:
        for seed in seeds:
            ckpt_dir = root / args.sweep_id / "shared"
            _ = train_shared_cell(config, latency_ns, seed, total_integration_steps=args.steps, tasks=tasks, episodes=args.eval_episodes, checkpoint_dir=ckpt_dir, activation=act_mod)
    _evaluate_shared(configs, args.sweep_id, root, latencies, seeds, activation, args.eval_episodes)
    return 0


def _cmd_evaluate(args) -> int:
    activation = _act(args.activation) if getattr(args, "activation", None) else "capped-relu"
    config_path = Path(args.config)
    config = load_config(config_path)
    root = Path(args.root)
    _evaluate_dedicated(config, args.sweep_id, root, args.episodes if hasattr(args, "episodes") else None, activation)
    return 0


def _cmd_view(args) -> int:
    results = Path(args.results)
    out = Path(args.out)
    subprocess.run([sys.executable, "scripts/build_site.py", "--results", str(results), "--out", str(out)], check=False)
    if getattr(args, "serve", True) and (not args.build_only):
        import http.server
        import socketserver

        os.chdir(out)
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", args.port), handler)
        print(f"serving on http://localhost:{args.port}")
        with contextlib.suppress(KeyboardInterrupt):
            httpd.serve_forever()
        httpd.server_close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="latency-ml")
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="Train one phase / activation.")
    tr.add_argument("--phase", choices=["cartpole", "lunarlander", "shared"], default="cartpole")
    tr.add_argument("--activation", choices=["relu", "capped-relu"], default="capped-relu")
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
    tr.add_argument("--port", type=int, default=8000)
    ev = sub.add_parser("evaluate", help="Evaluate a sweep.")
    ev.add_argument("--config", required=True)
    ev.add_argument("--sweep-id", required=True)
    ev.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ev.add_argument("--episodes", type=int, default=None)
    ev.add_argument("--activation", choices=["relu", "capped-relu"], default="capped-relu")
    vi = sub.add_parser("view", help="Build / serve the site.")
    vi.add_argument("--results", type=Path, default=Path("results"))
    vi.add_argument("--out", type=Path, default=Path("site"))
    vi.add_argument("--port", type=int, default=8000)
    vi.add_argument("--build-only", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "view":
        return _cmd_view(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
