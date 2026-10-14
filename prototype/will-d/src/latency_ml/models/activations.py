from __future__ import annotations

import torch
from torch import nn

CLIP: float = 2.0


class HartleyBoundedReLU(nn.Module):
    def __init__(self, clip: float = CLIP) -> None:
        super().__init__()
        if clip <= 0.0:
            raise ValueError(f"clip must be positive, got {clip!r}")
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, self.clip)

    def extra_repr(self) -> str:
        return f"clip={self.clip:g}"


def bounded_relu() -> HartleyBoundedReLU:
    return HartleyBoundedReLU()


ACTIVATION_NAME: str = type(HartleyBoundedReLU()).__name__
__all__ = ["ACTIVATION_NAME", "CLIP", "HartleyBoundedReLU", "bounded_relu"]
