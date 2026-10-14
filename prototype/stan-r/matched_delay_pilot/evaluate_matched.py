"""Evaluate each actor only at its saved training delay and plant configuration."""
import argparse, json, csv
from dataclasses import asdict
from experiment import load_controller, rollout, benchmark, host_info
from matched_models import resolve_models
from pilot_runs import create_run, finish_run


def evaluate(
    inputs,
    episodes=100,
    seed=10000,
    root="runs",
    code=None,
    backend="numpy",
    calls=2000,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    cases = resolve_models(inputs)
    run = create_run(
        root,
        code,
        kind="matched-evaluation",
        environment=cases[0][1].get("environment", "accelerator"),
        episodes_per_model=episodes,
        evaluation_seed_start=seed,
        backend=backend,
        models=[str(p) for p, _ in cases],
    )
    rows = []
    timings = {}
    snapshots = []
    for path, meta in cases:
        policy, cfg, _ = load_controller(str(path), backend)
        budget = float(meta["trained_inference_ns"])
        _, _, observations = rollout(policy, cfg, seed)
        timings[path.name] = dict(
            backend=backend, **benchmark(policy, observations, calls)
        )
        snapshots.append(
            dict(model=path.name, metadata=meta, evaluated_config=asdict(cfg))
        )
        for i in range(episodes):
            metrics, _, _ = rollout(policy, cfg, seed + i)
            rows.append(
                dict(
                    model=path.name,
                    environment=meta.get("environment", "accelerator"),
                    training_seed=meta.get("seed", 0),
                    evaluation_seed=seed + i,
                    trained_ns=budget,
                    inference_ns=budget,
                    total_action_ns=budget + cfg.transport_ns,
                    **metrics,
                )
            )
    with (run / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (run / "models.json").write_text(json.dumps(snapshots, indent=2))
    (run / "benchmarks.json").write_text(
        json.dumps(dict(host=host_info(), models=timings), indent=2)
    )
    from report_generator import generate

    generate(run / "episodes.csv", run / "report")
    finish_run(run, episode_rows=len(rows))
    print(
        f"Saved run: {run.name}\nFolder: {run}\nEvery model was deployed at its own saved delay."
    )
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="Directory, manifest or model stems")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--code")
    p.add_argument("--backend", choices=["numpy", "sb3", "sb3-cuda"], default="numpy")
    p.add_argument("--benchmark-calls", type=int, default=2000)
    a = p.parse_args()
    evaluate(
        a.models, a.episodes, a.seed, a.runs_root, a.code, a.backend, a.benchmark_calls
    )


if __name__ == "__main__":
    main()
