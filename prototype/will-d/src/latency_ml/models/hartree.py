"""Hartree arm: PPO policy with the private hmlp-ti emulator in the actor loop.

Backend shim: the Hartree emulator (hmlp_ti + hartree-emu/net wheels) is
private and must not be committed to the public repo. A copy lives in
``prototype/will-d/vendor/`` (gitignored). This module adds that vendor dir
to ``sys.path`` and tries the import, exposing a graceful
``HARTREE_AVAILABLE`` flag so public CI without the vendor copy still imports
and only fails with a clear message when the hartree arm is actually built.

Policy: dedicated cartpole ``HartreePolicy``. Only the actor is constrained
through analog physics: obs (4) -> Linear encoder -> sigmoid*255 IN codes
(256) -> frozen hmlp-ti ``DeviceTorch`` (f64, draws=False, inference-only) ->
y volts (256) -> Linear actor head (2). The critic is a normal clean MLP
(obs -> 128 -> 128 -> 1, ReLU, f32) so value learning is never hobbled by the
hardware; it is training-only and never deployed. PPO trains encoder +
actor head + critic; Hartree codes are frozen and never updated.
"""

from __future__ import annotations

import contextlib
import itertools
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from latency_ml.models.architectures import CONTROL_HIDDEN

VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

try:
    import hmlp_ti  # type: ignore[import-not-found]  # private vendored copy

    HARTREE_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # vendor copy missing on public clones
    hmlp_ti = None  # type: ignore[assignment]
    HARTREE_AVAILABLE = False
    _IMPORT_ERROR = exc

HARTREE_WIDTH = 256
HARTREE_ACTIVATION_NAME = "hartree"


def require_hartree() -> None:
    if not HARTREE_AVAILABLE:
        raise ImportError(
            "hartree arm needs the private hmlp-ti emulator in prototype/will-d/vendor/ "
            f"(gitignored, not committed). Original: {_IMPORT_ERROR!r}. "
            "See prototype/will-d/README.md 'Hartree arm (private emulator)'."
        )


def create_device_torch():
    """Fresh deterministic DeviceTorch (draws=False, f64), frozen for inference-only use."""
    require_hartree()
    model = hmlp_ti.DeviceTorch()
    with contextlib.suppress(Exception):
        model.draws = False
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def code_limits(model, name: str) -> tuple[float, float]:
    return model.spec.config[name].limits()


def clamp_codes(model) -> None:
    """Clamp every code back into its valid range (in place, no grad). Kept for tooling; training never updates codes."""
    with torch.no_grad():
        for name, codes in model.config.items():
            try:
                lo, hi = code_limits(model, name)
            except Exception:
                continue
            codes.clamp_(lo, hi)


class HartreePolicy(nn.Module):
    """Cartpole actor through frozen Hartree physics, plus a normal MLP critic.

    Actor (deployed): obs -> encoder -> frozen DeviceTorch -> actor head.
    Critic (training-only): plain MLP from obs, full precision, fully trained.
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        if obs_dim <= 0 or n_actions <= 1:
            raise ValueError(f"hartree arm supports discrete control tasks, got obs_dim={obs_dim} n_actions={n_actions}")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.activation_name = HARTREE_ACTIVATION_NAME
        self.encoder = nn.Linear(self.obs_dim, HARTREE_WIDTH, dtype=torch.float64)
        self.trunk = create_device_torch()
        self.actor_head = nn.Linear(HARTREE_WIDTH, self.n_actions, dtype=torch.float64)
        critic_layers: list[nn.Module] = []
        widths = (self.obs_dim, *CONTROL_HIDDEN)
        for fan_in, fan_out in itertools.pairwise(widths):
            critic_layers += [nn.Linear(fan_in, fan_out), nn.ReLU()]
        critic_layers.append(nn.Linear(widths[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def _actor_logits(self, obs: torch.Tensor) -> torch.Tensor:
        obs_d = obs.to(torch.float64)
        codes = torch.sigmoid(self.encoder(obs_d)) * 255.0
        return self.actor_head(self.trunk(codes))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._actor_logits(obs)
        value = self.critic(obs.to(torch.float32)).flatten()
        return logits, value

    def action_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        log_probs_all = functional.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs_all.exp(), num_samples=1).squeeze(-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return (actions, log_probs, value)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        log_probs_all = functional.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
        return (log_probs, entropy, value, logits)

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._actor_logits(obs).argmax(dim=-1)

    def parameter_groups(self) -> list:
        from latency_ml.rl.task_policy import ParameterGroup

        params = list(self.encoder.parameters()) + list(self.actor_head.parameters()) + list(self.critic.parameters())
        return [ParameterGroup("policy", params, stepped_on_every_task=True)]

    def to_device(self, device: torch.device | str) -> None:
        self.to(device)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.trunk.close()


def build_hartree_policy(task: str) -> HartreePolicy:
    from latency_ml.envs.factory import probe_env_dims

    obs_dim, n_actions = probe_env_dims(task)
    if task != "cartpole":
        raise ValueError(f"hartree arm is cartpole-dedicated for now, got {task!r}")
    return HartreePolicy(obs_dim, n_actions)


def hartree_param_count(policy: HartreePolicy) -> int:
    return sum(p.numel() for p in policy.parameters())


__all__ = [
    "HARTREE_ACTIVATION_NAME",
    "HARTREE_AVAILABLE",
    "HARTREE_WIDTH",
    "VENDOR_DIR",
    "HartreePolicy",
    "build_hartree_policy",
    "clamp_codes",
    "code_limits",
    "create_device_torch",
    "hartree_param_count",
    "require_hartree",
]
