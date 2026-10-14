"""Hardware modelling: exact MAC counting, and the gated accelerator latency.

Two things live here, kept deliberately separate:

- :mod:`mac_counter` is pure arithmetic and always works. MAC count is knowable
  before the accelerator contract is decided.
- :mod:`latency_model` converts a MAC count into a *hypothetical* latency and
  therefore requires a frozen contract, raising rather than inventing numbers
  (``PLAN.md`` section 3.2).

- :mod:`reference` measures the real software implementation, which is a
  different kind of claim from the hardware hypothesis.
"""

from latency_ml.hardware.latency_model import (
    compute_time_ns,
    explain_pending,
    is_available,
    latency_range_ns,
    predict_latency_ns,
)
from latency_ml.hardware.mac_counter import (
    BITS_PER_FORMAT,
    Architecture,
    MacBreakdown,
    ResourceAccounting,
    bits_per_weight,
    mac_breakdown,
)

__all__ = [
    "BITS_PER_FORMAT",
    "Architecture",
    "MacBreakdown",
    "ResourceAccounting",
    "bits_per_weight",
    "compute_time_ns",
    "explain_pending",
    "is_available",
    "latency_range_ns",
    "mac_breakdown",
    "predict_latency_ns",
]
