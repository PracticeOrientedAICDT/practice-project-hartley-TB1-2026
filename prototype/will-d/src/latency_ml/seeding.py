"""Deterministic seeding.

`PLAN.md` section 19 requires a fixed seed protocol with environment,
action-space and training seeds recorded separately. Every run's manifest
carries these values, so a result row can be traced to the exact random
stream that produced it.

Two properties matter for the experiment:

- **Evaluation seeds are shared across latency conditions.** Episode ``k`` uses
  ``eval_seed_base + k`` regardless of latency or training seed, which is what
  makes the comparisons paired (section 19).
- **Training seeds are independent per condition.** A policy trained for one
  decision interval is not shared with another, since matched training is the
  primary design (section 17).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Seeds:
    """The three seed roles `PLAN.md` section 19 asks to record.

    Attributes:
        training: Seeds the weight initialisation and all training-time
            sampling.
        env: Base seed for environment resets during training.
        eval: Base seed for evaluation episodes, shared across conditions.
    """

    training: int
    env: int
    eval: int

    def to_dict(self) -> dict[str, int]:
        """Manifest-friendly representation."""
        return asdict(self)

    def for_latency(self, latency_ns: int) -> Seeds:
        """Derive an independent seed set for one latency condition.

        Two conditions must never share a training seed, otherwise differences
        between them could be seed noise rather than latency effects. Mixing the
        latency into the seed guarantees independence while staying
        deterministic, so re-running a condition reproduces it exactly.
        """
        return Seeds(
            training=derive_seed(self.training, latency_ns),
            env=derive_seed(self.env, latency_ns),
            eval=self.eval,
        )

    def eval_episode_seed(self, index: int) -> int:
        """Seed for evaluation episode ``index``, shared across conditions."""
        return self.eval + index


def derive_seed(base: int, *salts: int) -> int:
    """Derive a deterministic child seed from a base and salts.

    Uses a linear congruential mix rather than Python's ``hash``, whose string
    hashing is randomised per process and would break reproducibility.
    """
    value = base
    for salt in salts:
        value = (value * 6364136223846793005 + salt + 1442695040888963407) % (2**63)
    return value % (2**31 - 1)


def seed_everything(seeds: Seeds, deterministic_torch: bool = True) -> None:
    """Seed every random source the training loop touches.

    Args:
        seeds: The seed set to apply.
        deterministic_torch: Request deterministic torch kernels. Trade-off:
            slower, but reproducible across runs, which section 19 requires for
            checkpoint comparison.
    """
    random.seed(seeds.training)
    np.random.seed(seeds.training % (2**32))
    torch.manual_seed(seeds.training)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        torch.cuda.manual_seed_all(seeds.training)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_env(env: object, seed: int) -> tuple[Any, dict[str, Any]]:
    """Reset an environment with a specific seed, including its action space.

    Resetting only the environment leaves the action space unseeded, which
    makes discrete action sampling irreproducible across runs.

    Returns:
        The ``(observation, info)`` pair from the reset, so the caller does not
        have to reset a second time just to obtain it.
    """
    obs, info = env.reset(seed=seed)  # type: ignore[attr-defined]
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        action_space.seed(seed)
    return obs, info
