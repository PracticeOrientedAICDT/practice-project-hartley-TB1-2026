import argparse, json
from pathlib import Path
from experiment import load_controller, rollout, benchmark, host_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--calls", type=int, default=10000)
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["numpy", "sb3", "sb3-cuda"],
        default=["numpy", "sb3"],
    )
    p.add_argument("--out", default="inference_benchmarks.json")
    a = p.parse_args()
    rows = []
    for path in a.models:
        for backend in a.backends:
            policy, cfg, meta = load_controller(path, backend)
            _, _, obs = rollout(policy, cfg)
            rows.append(
                dict(
                    model=path,
                    backend=backend,
                    architecture=meta.get("architecture"),
                    actor_parameters=meta.get("actor_parameters"),
                    **benchmark(policy, obs, a.calls)
                )
            )
    Path(a.out).write_text(json.dumps(dict(host=host_info(), results=rows), indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
