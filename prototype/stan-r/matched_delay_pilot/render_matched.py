"""Render a numerically ordered grid; every policy uses its own trained delay."""
import argparse, json
from pathlib import Path
from matched_models import resolve_models
from render_accelerator_demo import render_comparison


def render(inputs, out, training_seed=0, episode_seed=10000, stride=1, max_frames=None):
    cases = resolve_models(inputs)
    cases = [(p, m) for p, m in cases if int(m.get("seed", 0)) == training_seed]
    if not cases:
        raise ValueError("No models for selected training seed")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    videos = []
    for start in range(0, len(cases), 4):
        batch = cases[start : start + 4]
        path = out / f"matched_s{training_seed}_part{start//4+1}.mp4"
        render_comparison(
            [str(p) for p, _ in batch],
            [float(m["trained_inference_ns"]) for _, m in batch],
            path,
            seed=episode_seed,
            stride=stride,
            max_frames=max_frames,
            environment=batch[0][1].get("environment", "accelerator"),
            case_titles=[
                f'Train {float(m["trained_inference_ns"]):g} ns | Deploy {float(m["trained_inference_ns"]):g} ns | s{training_seed}'
                for _, m in batch
            ],
        )
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        metadata.update(mode="matched-delay", training_seed=training_seed)
        sidecar.write_text(json.dumps(metadata, indent=2))
        videos.append(str(path))
    return videos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--out", default="videos")
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--episode-seed", type=int, default=10000)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    a = p.parse_args()
    render(a.models, a.out, a.training_seed, a.episode_seed, a.stride, a.max_frames)


if __name__ == "__main__":
    main()
