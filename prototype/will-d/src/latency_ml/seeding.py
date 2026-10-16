from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Seeds:
    training: int
    env: int
    eval: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def for_latency(self, latency_ns: int) -> Seeds:
        return Seeds(
            training=derive_seed(self.training, latency_ns),
            env=derive_seed(self.env, latency_ns),
            eval=self.eval,
        )

    def eval_episode_seed(self, index: int) -> int:
        return self.eval + index


def derive_seed(base: int, *salts: int) -> int:
    value = base
    for salt in salts:
        value = (value * 6364136223846793005 + salt + 1442695040888963407) % 2**63
    return value % (2**31 - 1)


def seed_everything(seeds: Seeds, deterministic_torch: bool = True) -> None:
    random.seed(seeds.training)
    np.random.seed(seeds.training % 2**32)
    torch.manual_seed(seeds.training)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.training)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_env(env: object, seed: int) -> tuple[Any, dict[str, Any]]:
    obs, info = env.reset(seed=seed)
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        action_space.seed(seed)
    return (obs, info)
