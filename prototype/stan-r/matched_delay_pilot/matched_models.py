"""Resolve model directories/manifests without ambiguous shell wildcards."""
import json
from pathlib import Path
import numpy as np
from experiment import model_stem


def resolve_models(inputs):
    paths = []
    for value in inputs:
        p = Path(value)
        if p.is_dir():
            candidates = sorted(p.glob("*.json"))
            paths.extend(
                model_stem(x)
                for x in candidates
                if x.name != "manifest.json" and x.with_suffix(".npz").exists()
            )
        elif p.name == "manifest.json":
            for item in json.loads(p.read_text()):
                q = Path(item)
                if not q.with_suffix(".json").exists():
                    q = p.parent / q.name
                paths.append(model_stem(q))
        else:
            paths.append(model_stem(p))
    unique = {str(p.resolve()): p for p in paths}
    cases = []
    for p in unique.values():
        meta = json.loads(p.with_suffix(".json").read_text())
        cfg = meta["config"]
        budget = float(meta["trained_inference_ns"])
        if cfg.get("inference_range_ns") is not None:
            raise ValueError(f"{p}: randomized training has no single matched delay")
        if not np.isfinite(budget) or not np.isclose(
            budget, float(cfg["inference_ns"])
        ):
            raise ValueError(f"{p}: training budget does not match saved config")
        if any(float(x) != 0 for x in cfg.get("action_delay_range", [0, 0])):
            raise ValueError(
                f"{p}: legacy random action delays are not a fixed matched budget"
            )
        cases.append((p, meta))
    if not cases:
        raise ValueError("No fixed-budget models found")
    envs = {m.get("environment", "accelerator") for _, m in cases}
    if len(envs) != 1:
        raise ValueError("Evaluate the two environments in separate runs")
    identities = [
        (float(m["trained_inference_ns"]), int(m.get("seed", 0))) for _, m in cases
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Duplicate latency/training-seed checkpoints; use separate runs"
        )
    # Different physical profiles/architectures must not be silently pooled.
    def signature(m):
        c = dict(m["config"])
        c.pop("inference_ns", None)
        return json.dumps([c, m.get("architecture")], sort_keys=True)

    if len({signature(m) for _, m in cases}) != 1:
        raise ValueError(
            "Models differ in plant/profile/architecture; use separate runs"
        )
    return sorted(
        cases,
        key=lambda x: (
            float(x[1]["trained_inference_ns"]),
            int(x[1].get("seed", 0)),
            x[0].name,
        ),
    )
