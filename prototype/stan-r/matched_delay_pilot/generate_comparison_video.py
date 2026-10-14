"""Batch render all models at all delays plus side-by-side videos per delay."""
import argparse
from pathlib import Path
from experiment import BUDGETS
from render_accelerator_demo import render_comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument(
        "--environment", choices=["accelerator", "laser"], default="accelerator"
    )
    p.add_argument(
        "--dynamics-profile",
        choices=["checkpoint", "deterministic", "mild", "uncertain", "stress"],
        default="checkpoint",
    )
    p.add_argument("--budgets-ns", type=float, nargs="+", default=BUDGETS)
    p.add_argument("--out", default="videos")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--individual", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for budget in a.budgets_ns:
        render_comparison(
            a.models,
            [budget],
            out / f"comparison_{budget:g}ns.mp4",
            seed=a.seed,
            stride=a.stride,
            max_frames=a.max_frames,
            environment=a.environment,
            dynamics_profile=a.dynamics_profile,
        )
        if a.individual:
            for model in a.models:
                render_comparison(
                    [model],
                    [budget],
                    out / f"{Path(model).name}_{budget:g}ns.mp4",
                    seed=a.seed,
                    stride=a.stride,
                    max_frames=a.max_frames,
                    environment=a.environment,
                    dynamics_profile=a.dynamics_profile,
                )


if __name__ == "__main__":
    main()
