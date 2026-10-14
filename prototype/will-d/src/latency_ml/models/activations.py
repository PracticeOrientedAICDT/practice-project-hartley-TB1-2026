"""Activation functions.

The network activation is a fixed property of this repository, not a tunable
knob. It lives here, in one module, so that the single place the nonlinearity
enters the model is obvious to a reader and obvious to change.

## The activation

Hartley's bounded ReLU, with a clip at :data:`CLIP`:

.. code-block:: text

    f(x) = 0    for x <= 0
    f(x) = x    for 0 < x <= CLIP
    f(x) = CLIP for x > CLIP

## Why bounded

Three consequences, all of them reasons to expect this to behave differently
from an unbounded ReLU rather than being a cosmetic variant:

* **Saturation is bounded.** Above the clip the output stops growing, so a
  large pre-activation cannot dominate the next layer's input. Magnitudes
  entering the output head stay inside a fixed range.
* **The gradient vanishes above the clip.** This is the substantive change.
  An unbounded ReLU keeps passing gradient no matter how large the input; this
  one stops entirely once a unit exceeds the clip. Units pushed into
  saturation receive no learning signal, so credit assignment differs in kind
  and not only in magnitude.
* **Dead units are recoverable in one direction only.** A unit driven below
  zero is dead for a standard ReLU too, but here a unit that recovers into
  ``(0, CLIP]`` starts learning again, whereas one that overshoots the clip
  and then falls back does so through a region where it was receiving no
  gradient at all.

The last point bears on the shared-trunk experiment specifically. A shared
representation has to serve two tasks whose activation statistics need not
match; bounding the range removes one mechanism by which one task's large
activities could dominate the other's, which is a reason to expect sharing to
be *helped* rather than hindered, and therefore a hypothesis worth testing
rather than assuming.

## Provenance

The exact clip value is a fixed constant rather than a configuration field.
Changing it changes the experiment, so it should be a deliberate edit to this
file with a corresponding change to the documented protocol -- not something a
sweep can vary while the rest of the design is held still.

## Testing hook

Every model accepts an explicit ``nn.Module`` in place of this activation.
Production code never passes one; the parity suite does, in order to reproduce
a v1 reference checkpoint that was trained under an unbounded ReLU. Keeping the
parameter as a module rather than a string name means that hook does not
require a registry of alternatives that would imply the alternatives are
supported configurations.
"""

from __future__ import annotations

import torch
from torch import nn

CLIP: float = 2.0


class HartleyBoundedReLU(nn.Module):
    """The bounded ReLU defined at the top of this module.

    Args:
        clip: Upper saturation point. Defaults to :data:`CLIP`.

    The lower bound is fixed at zero. A configurable lower bound would let this
    become a general hardtanh, which is not what the hardware-faithful
    nonlinearity being modelled is.
    """

    def __init__(self, clip: float = CLIP) -> None:
        super().__init__()
        if clip <= 0.0:
            raise ValueError(f"clip must be positive, got {clip!r}")
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp ``x`` into ``[0, clip]``.

        Autograd handles the gradient: it is one inside the interval and zero
        outside, which is precisely the vanishing-gradient behaviour described
        in the module docstring.
        """
        return torch.clamp(x, 0.0, self.clip)

    def extra_repr(self) -> str:
        return f"clip={self.clip:g}"


def bounded_relu() -> HartleyBoundedReLU:
    """A fresh default activation.

    A function rather than a shared module instance, so that two policies never
    alias the same activation object. They hold no parameters or buffers today,
    but relying on that is exactly the kind of assumption that breaks silently
    when someone adds state.
    """
    return HartleyBoundedReLU()


ACTIVATION_NAME: str = type(HartleyBoundedReLU()).__name__


__all__ = ["ACTIVATION_NAME", "CLIP", "HartleyBoundedReLU", "bounded_relu"]
