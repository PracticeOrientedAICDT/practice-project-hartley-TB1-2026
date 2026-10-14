"""Cross-evaluate policies across deployed latency budgets using paired seeds."""
import argparse, csv, json
from pathlib import Path
from experiment import (
    BUDGETS,
    load_controller,
    eval_config,
    rollout,
    benchmark,
    host_info,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="Model stems and/or zero")
    p.add_argument(
        "--dynamics-profile",
        choices=["checkpoint", "deterministic", "mild", "uncertain", "stress"],
        default="checkpoint",
        help="Accelerator deployment dynamics; checkpoint preserves training config",
    )
    p.add_argument(
        "--environment",
        choices=["accelerator", "laser"],
        default="accelerator",
        help="Required for zero baseline; checkpoints are auto-validated",
    )
    p.add_argument("--budgets-ns", type=float, nargs="+", default=BUDGETS)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--backend", choices=["numpy", "sb3", "sb3-cuda"], default="numpy")
    p.add_argument(
        "--latency-source", choices=["budget", "measured-p99"], default="budget"
    )
    p.add_argument("--transport-ns", type=float)
    p.add_argument(
        "--inference-jitter-ns",
        type=float,
        default=None,
        help="Override profile jitter; omit to preserve selected/checkpoint profile",
    )
    p.add_argument("--benchmark-calls", type=int, default=2000)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if a.episodes < 1 or (
        a.inference_jitter_ns is not None and a.inference_jitter_ns < 0
    ):
        p.error("Invalid episodes/jitter")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    benchmarks = {}
    for path in a.models:
        policy, base, meta = load_controller(
            path, a.backend, a.environment if path == "zero" else None
        )
        environment = meta.get("environment", "accelerator")
        if environment == "accelerator" and a.dynamics_profile != "checkpoint":
            from envs.accelerator_profiles import apply_profile

            base = apply_profile(base, a.dynamics_profile)
        if path != "zero" and environment != a.environment:
            raise ValueError(
                f"{path} is a {environment} model; selected {a.environment}"
            )
        _, _, obs = rollout(policy, base, a.seed, environment=environment)
        timing = benchmark(policy, obs, a.benchmark_calls)
        benchmarks[path] = dict(
            environment=environment, timing=timing, backend=a.backend, host=host_info()
        )
        budgets = (
            [timing["p99_ns"]] if a.latency_source == "measured-p99" else a.budgets_ns
        )
        for budget in budgets:
            cfg = eval_config(base, budget, a.transport_ns, a.inference_jitter_ns)
            for i in range(a.episodes):
                metrics, _, _ = rollout(
                    policy, cfg, a.seed + i, environment=environment
                )
                rows.append(
                    dict(
                        environment=environment,
                        dynamics_profile=(
                            a.dynamics_profile
                            if a.dynamics_profile != "checkpoint"
                            else meta.get("dynamics_profile", "checkpoint")
                        ),
                        model=path,
                        trained_ns=meta.get("trained_inference_ns", float("nan")),
                        inference_ns=budget,
                        total_action_ns=budget + cfg.transport_ns,
                        transport_ns=cfg.transport_ns,
                        inference_jitter_ns=getattr(cfg, "inference_jitter_ns", 0.0),
                        seed=a.seed + i,
                        backend=a.backend,
                        latency_source=a.latency_source,
                        host_p99_ns=timing["p99_ns"],
                        **metrics,
                    )
                )
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (out / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2))
    print(f"Wrote {len(rows)} paired {a.environment} episodes to {out}")


if __name__ == "__main__":
    main()
