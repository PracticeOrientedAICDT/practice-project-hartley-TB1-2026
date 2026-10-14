"""Train a delay sweep, evaluate its diagonal, save report and matched videos."""
import argparse, subprocess, sys
from pathlib import Path
from pilot_runs import create_run, finish_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--environment", choices=["accelerator", "laser"], default="laser")
    p.add_argument(
        "--dynamics-profile",
        choices=["deterministic", "mild", "uncertain", "stress"],
        default="deterministic",
    )
    p.add_argument(
        "--budgets-ns", type=float, nargs="+", default=[0, 100, 250, 500, 750, 1000]
    )
    p.add_argument("--arch", default="paper16")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=250000)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--code")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--max-frames", type=int)
    a = p.parse_args()
    if a.steps < 1 or a.episodes < 1 or not a.seeds:
        p.error("Positive training steps, episodes and at least one seed required")
    if a.environment == "laser" and a.dynamics_profile != "deterministic":
        p.error("Accelerator dynamics profiles do not apply to laser")
    run = create_run(
        a.runs_root,
        a.code,
        kind="training-and-matched-evaluation",
        environment=a.environment,
        dynamics_profile=a.dynamics_profile,
        budgets_ns=sorted(set(a.budgets_ns)),
        training_seeds=a.seeds,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "train_latency_sweep.py"),
            "--environment",
            a.environment,
            "--dynamics-profile",
            a.dynamics_profile,
            "--arch",
            a.arch,
            "--steps",
            str(a.steps),
            "--seeds",
            *map(str, a.seeds),
            "--budgets-ns",
            *map(str, sorted(set(a.budgets_ns))),
            "--out",
            str(run / "models"),
        ],
        check=True,
    )
    from evaluate_matched import evaluate

    evaluated = evaluate([str(run / "models")], a.episodes, root=run, code="evaluation")
    import shutil

    for name in ["episodes.csv", "models.json", "benchmarks.json", "report"]:
        src = evaluated / name
        if src.is_dir():
            shutil.copytree(src, run / name)
        else:
            shutil.copy2(src, run / name)
    shutil.rmtree(evaluated)
    if not a.no_video:
        from render_matched import render

        render(
            [str(run / "models")],
            run / "videos",
            training_seed=a.seeds[0],
            max_frames=a.max_frames,
        )
    finish_run(run, episodes_per_model=a.episodes)
    print(
        f"Open run {run.name}: streamlit run dashboard.py -- --runs-root {a.runs_root}"
    )


if __name__ == "__main__":
    main()
