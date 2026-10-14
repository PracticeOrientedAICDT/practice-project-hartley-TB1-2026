"""Deterministic actor-only NumPy export for this project's ReLU PPO models."""
import numpy as np


class NumpyActor:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(self.weights):
            x = w @ x + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
        return np.clip(x, -1, 1).astype(np.float32)

    def save(self, path):
        np.savez(
            path,
            **{
                f"{k}{i}": v
                for i, pair in enumerate(self.weights)
                for k, v in zip(("w", "b"), pair)
            },
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls([(d[f"w{i}"], d[f"b{i}"]) for i in range(len(d.files) // 2)])

    @classmethod
    def from_ppo(cls, model):
        import torch

        layers = list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
        if any(not isinstance(l, (torch.nn.Linear, torch.nn.ReLU)) for l in layers):
            raise ValueError("Exporter only supports Linear/ReLU")
        weights = [
            (
                l.weight.detach().cpu().numpy().copy(),
                l.bias.detach().cpu().numpy().copy(),
            )
            for l in layers
            if isinstance(l, torch.nn.Linear)
        ]
        return cls(weights)
