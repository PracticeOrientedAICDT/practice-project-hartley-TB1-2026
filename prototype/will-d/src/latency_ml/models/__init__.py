"""Policy architectures and the accelerator's chip-compatibility limits."""

from latency_ml.models.architectures import (
    CONTROL_HIDDEN,
    HFT_LARGE_HIDDEN,
    HFT_SMALL_HIDDEN,
    MLPPolicy,
    build_control_policy,
    build_hft_policy,
    build_shared_trunk,
    describe_architecture,
)
from latency_ml.models.chip_compat import (
    DEFAULT_CONSTRAINTS,
    ChipConstraints,
    IncompatibleArchitectureError,
    is_compatible,
    validate,
)

__all__ = [
    "CONTROL_HIDDEN",
    "DEFAULT_CONSTRAINTS",
    "HFT_LARGE_HIDDEN",
    "HFT_SMALL_HIDDEN",
    "ChipConstraints",
    "IncompatibleArchitectureError",
    "MLPPolicy",
    "SharedTaskPolicy",
    "SharedTrunkPolicy",
    "TaskShape",
    "build_control_policy",
    "build_hft_policy",
    "build_shared_trunk",
    "build_shared_trunk_policy",
    "describe_architecture",
    "is_compatible",
    "validate",
]
