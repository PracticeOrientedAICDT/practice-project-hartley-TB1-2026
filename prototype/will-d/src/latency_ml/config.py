"""Typed experiment configuration.

Configuration is YAML on disk, validated into frozen dataclasses on load. A
run's configuration is hashed into its manifest (`PLAN.md` section 19), so a
result row can be tied to exactly the settings that produced it.

The important subtlety here is the **TBD sentinel**. `PLAN.md` section 3.1
leaves the accelerator's numerical precision, peak throughput, fixed overhead
and memory as unresolved pending supervisor review (section 30, questions 2,
3 and 10). Those must not be filled with plausible-looking numbers, because
`PLAN.md` section 3.2 forbids inventing hardware figures.

So ``TBD`` parses to :class:`Unresolved`, a first-class value rather than the
string ``"TBD"``. Software-only experiments, which treat latency as the free
experimental variable, run unaffected. Anything that genuinely needs a
hardware number raises :class:`HardwareModelNotFrozen` rather than producing a
plausible fabrication.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from latency_ml.timing import NATIVE_DT_ENV_NS, NATIVE_SWEEP_GRID_NS

#: The sentinel token recognised in YAML for an unresolved hardware parameter.
TBD_TOKEN: Final = "TBD"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


class HardwareModelNotFrozen(RuntimeError):
    """Raised when a calculation requires a hardware parameter that is unresolved.

    This is **expected behaviour**, not a bug. `PLAN.md` section 3.2 requires
    that hardware assumptions stay parameterised rather than invented, so any
    code path needing an unresolved parameter must refuse to produce a number.

    A software-only experiment must never raise this.
    """


class Unresolved:
    """Sentinel for a hardware parameter that has not been decided yet.

    Distinct from ``None``, which would be ambiguous, and from the string
    ``"TBD"``, which would silently propagate as a string.

    Deliberately **not** a dataclass: ``dataclasses.asdict`` recurses into
    nested dataclasses, which would flatten this sentinel into an anonymous
    dict and lose its meaning on serialisation.
    """

    __slots__ = ("parameter",)

    def __init__(self, parameter: str) -> None:
        self.parameter = parameter

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{TBD_TOKEN}({self.parameter})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unresolved) and other.parameter == self.parameter

    def __hash__(self) -> int:
        return hash((Unresolved, self.parameter))

    def __repr__(self) -> str:
        return f"Unresolved({self.parameter!r})"


def _parse_tbd_or_int(value: object, parameter: str) -> int | Unresolved:
    """Parse a value that may be the TBD sentinel."""
    if isinstance(value, str) and value.strip().upper() == TBD_TOKEN:
        return Unresolved(parameter)
    if isinstance(value, bool):
        raise ConfigError(f"{parameter} must be an integer or {TBD_TOKEN}, got a bool")
    if isinstance(value, int):
        return value
    raise ConfigError(f"{parameter} must be an integer or {TBD_TOKEN}, got {type(value).__name__}")


def _parse_tbd_or_str(value: object, parameter: str) -> str | Unresolved:
    """Parse a value that may be the TBD sentinel."""
    if isinstance(value, str) and value.strip().upper() == TBD_TOKEN:
        return Unresolved(parameter)
    if isinstance(value, str):
        return value
    raise ConfigError(f"{parameter} must be a string or {TBD_TOKEN}, got {type(value).__name__}")


def _require_mapping(data: object, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """The hypothetical accelerator contract (`PLAN.md` section 3.1).

    Unresolved parameters stay :class:`Unresolved`. They are *not* defaults,
    so an unfrozen model cannot be mistaken for a frozen one.
    """

    max_layers: int
    max_width: int
    precision: str | Unresolved
    peak_macs_per_second: int | Unresolved
    fixed_overhead_ns: int | Unresolved
    memory_bytes: int | Unresolved
    fast_execution_source: str
    slow_execution_source: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareSpec:
        """Build from a parsed YAML mapping."""
        return cls(
            max_layers=int(data.get("max_layers", 4)),
            max_width=int(data.get("max_width", 256)),
            precision=_parse_tbd_or_str(data.get("precision", TBD_TOKEN), "precision"),
            peak_macs_per_second=_parse_tbd_or_int(
                data.get("peak_macs_per_second", TBD_TOKEN), "peak_macs_per_second"
            ),
            fixed_overhead_ns=_parse_tbd_or_int(
                data.get("fixed_overhead_ns", TBD_TOKEN), "fixed_overhead_ns"
            ),
            memory_bytes=_parse_tbd_or_int(data.get("memory_bytes", TBD_TOKEN), "memory_bytes"),
            fast_execution_source=str(data.get("fast_execution_source", "hartley_parameterised")),
            slow_execution_source=str(data.get("slow_execution_source", "conventional_reference")),
        )

    def unresolved_parameters(self) -> list[str]:
        """Names of parameters still awaiting supervisor decision."""
        return [
            name
            for name, value in (
                ("precision", self.precision),
                ("peak_macs_per_second", self.peak_macs_per_second),
                ("fixed_overhead_ns", self.fixed_overhead_ns),
                ("memory_bytes", self.memory_bytes),
            )
            if isinstance(value, Unresolved)
        ]

    @property
    def is_frozen(self) -> bool:
        """Whether every hardware parameter has been decided."""
        return not self.unresolved_parameters()

    def require_frozen(self) -> None:
        """Raise unless every hardware parameter is resolved.

        Raises:
            HardwareModelNotFrozen: If any parameter is still unresolved.
        """
        missing = self.unresolved_parameters()
        if missing:
            raise HardwareModelNotFrozen(
                "accelerator parameters unresolved: "
                + ", ".join(missing)
                + ". PLAN.md section 3.2 forbids inventing hardware figures; "
                "these require supervisor decisions (PLAN.md section 30, Q2/Q3/Q10)."
            )

    def frozen_copy(
        self,
        *,
        precision: str,
        peak_macs_per_second: int,
        fixed_overhead_ns: int,
        memory_bytes: int,
    ) -> HardwareSpec:
        """Return a copy with all hardware parameters explicitly resolved.

        Used to define hypothetical-but-explicit operating points in analysis
        without mutating the on-disk configuration. All four parameters are
        required rather than optional, so a caller cannot produce a
        "half-frozen" model that still raises later.
        """
        return replace(
            self,
            precision=precision,
            peak_macs_per_second=peak_macs_per_second,
            fixed_overhead_ns=fixed_overhead_ns,
            memory_bytes=memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class TimingConfig:
    """Environment timing (`PLAN.md` section 5.1)."""

    dt_env_ns: int = NATIVE_DT_ENV_NS
    max_episode_steps: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingConfig:
        return cls(
            dt_env_ns=int(data.get("environment_timestep_ns", NATIVE_DT_ENV_NS)),
            max_episode_steps=data.get("max_episode_steps"),
        )


class DiscountBasis(Enum):
    """What the discount factor is measured *in*.

    This is not a tuning knob. It records which clock the discount runs on, and
    the two choices give materially different RL problems under action-hold
    latency.

    ``PHYSICAL``
        ``gamma`` and ``gae_lambda`` are per **integration step**, so a
        transition holding an action for ``N`` steps is discounted as
        ``gamma ** N``. The effective credit horizon in *simulated time* is
        then held constant across latency conditions, which was the original
        intent.

    ``DECISION``
        ``gamma`` and ``gae_lambda`` are per **decision**, applied unscaled, so
        a held transition is discounted exactly like any other. The credit
        horizon in *decisions* is then held constant across latency conditions.

    Why LunarLander needs ``DECISION``: under ``PHYSICAL`` the per-decision
    discount collapses as latency rises -- ``0.99 ** N`` -- so the GAE horizon
    falls from ~17 decisions at 20 ms to ~4 at 100 ms. The latency manipulation
    would then be changing the credit-assignment horizon at the same time as
    the control bandwidth, which confounds the measurement with a second
    unrequested change. The half-life also lands far too short for a task whose
    decisive signal is terminal shaping at the end of an episode.
    """

    PHYSICAL = "physical"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class DiscountSpec:
    """Resolved discount settings for one task.

    Attributes:
        basis: Which clock the factors run on.
        gamma: Discount factor, interpreted per :attr:`basis`.
        gae_lambda: GAE trace-decay, interpreted per :attr:`basis`.
    """

    basis: DiscountBasis
    gamma: float
    gae_lambda: float

    @property
    def duration_aware(self) -> bool:
        """Whether the GAE recursion must scale by the action-hold length.

        ``True`` for :attr:`DiscountBasis.PHYSICAL`, where a transition spanning
        ``N`` integration steps is discounted as ``gamma ** N``. ``False`` for
        :attr:`DiscountBasis.DECISION`, where the factor is per decision and
        the hold length is already accounted for by the reward being a sum over
        the held steps.
        """
        return self.basis is DiscountBasis.PHYSICAL

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscountSpec:
        """Parse a ``{basis, gamma, gae_lambda}`` mapping.

        Raises:
            ConfigError: If the basis is unknown or a factor is missing.
        """
        missing = {"basis", "gamma", "gae_lambda"} - set(data)
        if missing:
            raise ConfigError(f"training.discount_overrides entry missing {sorted(missing)}")
        try:
            basis = DiscountBasis(data["basis"])
        except ValueError as exc:
            raise ConfigError(
                f"unknown discount basis {data['basis']!r}; "
                f"expected one of {[b.value for b in DiscountBasis]}"
            ) from exc
        return cls(
            basis=basis,
            gamma=float(data["gamma"]),
            gae_lambda=float(data["gae_lambda"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Manifest-friendly representation."""
        return {"basis": self.basis.value, "gamma": self.gamma, "gae_lambda": self.gae_lambda}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """PPO hyperparameters (`PLAN.md` section 6.3).

    Defaults are the frozen CleanRL CartPole baseline. These are held fixed
    across every latency condition; tuning them per latency would confound the
    experiment.

    ``discount_overrides`` carries the one deliberate per-task departure,
    LunarLander's decision-level discounting, and lives here rather than in a
    single task config so the shared-trunk trainer and the dedicated controls
    resolve discounting from the *same* source and cannot silently drift apart.
    """

    algorithm: str = "ppo"
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 4
    steps_per_decision: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    gae_lambda_phys: float = 0.95
    gamma_phys: float = 0.99
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    total_integration_steps: int = 2_000_000
    control_training_seeds: int = 10
    #: Per-task discounting, overriding the physical defaults above. See
    #: :class:`DiscountBasis` for why LunarLander overrides them.
    discount_overrides: dict[str, DiscountSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "learning_rate" in kwargs:
            kwargs["learning_rate"] = float(kwargs["learning_rate"])
        if "discount_overrides" in kwargs:
            raw = kwargs["discount_overrides"]
            if not isinstance(raw, dict):
                raise ConfigError(
                    f"training.discount_overrides must be a mapping, got {type(raw).__name__}"
                )
            kwargs["discount_overrides"] = {
                str(task): DiscountSpec.from_dict(spec) for task, spec in raw.items()
            }
        return cls(**kwargs)

    def discount_for(self, task: str) -> DiscountSpec:
        """Resolve the discounting a given task trains under.

        Falls back to the physical defaults, which is what every task except
        those named in ``discount_overrides`` trains under. CartPole is
        therefore unchanged by construction rather than by remembering not to
        configure it.
        """
        override = self.discount_overrides.get(task)
        if override is not None:
            return override
        return DiscountSpec(
            basis=DiscountBasis.PHYSICAL,
            gamma=self.gamma_phys,
            gae_lambda=self.gae_lambda_phys,
        )

    def iterations(self, action_steps: int) -> int:
        """PPO iterations for a given action-hold count.

        With ``steps_per_decision`` and ``num_minibatches`` fixed across
        conditions (`CODE_PLAN.md` section 3.3), the rollout's integration
        work per iteration is ``num_envs * steps_per_decision * action_steps``,
        so iterations scale inversely with action-hold length. This is what
        keeps simulated-time exposure equal without shrinking the minibatch.
        """
        if action_steps < 1:
            raise ConfigError(f"action_steps must be >= 1, got {action_steps}")
        per_iter = self.num_envs * self.steps_per_decision * action_steps
        return self.total_integration_steps // per_iter

    def minibatch_size(self, action_steps: int) -> int:
        """Minibatch size for a given action-hold count.

        Invariant across latency conditions by construction; asserted in tests
        because a regression here would silently reintroduce the confound
        documented in `CODE_PLAN.md` section 3.3.
        """
        transitions = self.num_envs * self.steps_per_decision
        return transitions // self.num_minibatches


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation protocol (`PLAN.md` sections 6.7, 7, 19)."""

    control_episodes_per_seed: int = 200
    eval_seed_base: int = 10_000
    #: Evaluation seeds are reused across every latency condition and training
    #: seed, which makes the comparisons paired (`PLAN.md` section 19).
    paired_evaluation: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def episode_seed(self, index: int) -> int:
        """Seed for evaluation episode ``index``, shared across all conditions."""
        return self.eval_seed_base + index


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Policy architecture (`PROTOCOL.md`).

    There is deliberately no ``activation`` field. The nonlinearity is fixed by
    :mod:`latency_ml.models.activations` and recorded in each manifest from the
    module that was actually constructed. A config field would be a second,
    weaker source of truth that a reader could reasonably trust over the code,
    and which a typo could make disagree with it silently.
    """

    control_hidden_layers: tuple[int, ...] = (128, 128)
    hft_small: tuple[int, ...] = (32, 32)
    hft_large: tuple[int, ...] = (128, 128)
    hft_output_actions: tuple[int, ...] = (0, 1, 2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyConfig:
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(
                f"unknown policy config key(s): {sorted(unknown)}; accepted keys are "
                f"{sorted(cls.__dataclass_fields__)}"
            )
        return cls(
            control_hidden_layers=tuple(data.get("control_hidden_layers", (128, 128))),
            hft_small=tuple(data.get("hft_small", (32, 32))),
            hft_large=tuple(data.get("hft_large", (128, 128))),
            hft_output_actions=tuple(data.get("hft_output_actions", (0, 1, 2))),
        )


#: The HFT fixed-model sweep from `PLAN.md` section 11.6. Unlike the control
#: sweep this reaches the sub-millisecond regime, because in event-time replay
#: latency is a timestamp shift rather than a held action, so the market stream
#: resolves it.
HFT_SWEEP_GRID_NS: tuple[int, ...] = (
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
    100_000_000,
)


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    """Latency sweep configuration (`PLAN.md` section 5)."""

    #: The section 5.3 sweep: integer timestep multiples giving five
    #: independent native conditions. The section 5 target grid is in
    #: :mod:`latency_ml.timing` and resolves to only two exact conditions
    #: natively, so it is not used for control sweeps.
    sweep_grid_ns: tuple[int, ...] = NATIVE_SWEEP_GRID_NS
    #: The HFT fixed-model sweep, which can resolve sub-millisecond latencies
    #: because it shifts a timestamp rather than holding an action.
    hft_grid_ns: tuple[int, ...] = HFT_SWEEP_GRID_NS

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyConfig:
        sweep = data.get("sweep_grid_ns")
        hft = data.get("hft_grid_ns")
        return cls(
            sweep_grid_ns=tuple(int(x) for x in sweep) if sweep else NATIVE_SWEEP_GRID_NS,
            hft_grid_ns=tuple(int(x) for x in hft) if hft else HFT_SWEEP_GRID_NS,
        )


@dataclass(frozen=True, slots=True)
class FinanceConfig:
    """Phase 3 replay settings (`PLAN.md` section 11).

    Execution assumptions are held constant across every latency condition;
    the only intervention is inference delay (`PLAN.md` section 11.7).
    """

    source: str = "synthetic"
    data_path: str = "data/synthetic"
    execution_model: str = "aggressive"
    transaction_cost_bps: float = 0.0
    prediction_horizon_events: int = 10
    synthetic: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinanceConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        if "transaction_cost_bps" in kwargs:
            kwargs["transaction_cost_bps"] = float(kwargs["transaction_cost_bps"])
        if "synthetic" in kwargs:
            kwargs["synthetic"] = dict(_require_mapping(kwargs["synthetic"], "finance.synthetic"))
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """A complete experiment configuration."""

    name: str
    seed: int
    task: str
    timing: TimingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    policy: PolicyConfig
    latency: LatencyConfig
    hardware: HardwareSpec
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    finance: FinanceConfig = field(default_factory=FinanceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        """Validate a parsed YAML mapping into a frozen config."""
        project = _require_mapping(data.get("project", {}), "project")
        for key in ("name", "seed", "task"):
            if key not in project:
                raise ConfigError(f"project.{key} is required")

        return cls(
            name=str(project["name"]),
            seed=int(project["seed"]),
            task=str(project["task"]),
            timing=TimingConfig.from_dict(_require_mapping(data.get("timing", {}), "timing")),
            training=TrainingConfig.from_dict(
                _require_mapping(data.get("training", {}), "training")
            ),
            evaluation=EvaluationConfig.from_dict(
                _require_mapping(data.get("evaluation", {}), "evaluation")
            ),
            policy=PolicyConfig.from_dict(_require_mapping(data.get("policy", {}), "policy")),
            latency=LatencyConfig.from_dict(_require_mapping(data.get("latency", {}), "latency")),
            hardware=HardwareSpec.from_dict(_require_mapping(data.get("hardware", {}), "hardware")),
            environment_kwargs=dict(
                _require_mapping(data.get("environment_kwargs", {}), "environment_kwargs")
            ),
            finance=FinanceConfig.from_dict(_require_mapping(data.get("finance", {}), "finance")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the same nested shape :meth:`from_dict` accepts.

        Must round-trip: dumping a config and reloading it yields an identical
        config hash, so a stored manifest configuration is verifiably the one
        that produced the run.
        """
        flat = _jsonable(asdict(self))
        assert isinstance(flat, dict)
        scalar_keys = ("name", "seed", "task")
        nested: dict[str, Any] = {"project": {k: flat.pop(k) for k in scalar_keys}}
        nested.update(flat)
        return nested

    def config_hash(self) -> str:
        """Stable content hash, recorded in run manifests.

        Excludes nothing: the hash identifies the exact configuration that
        produced a result row.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_hardware(self, hardware: HardwareSpec) -> ProjectConfig:
        """Return a copy using a different :class:`HardwareSpec`."""
        return replace(self, hardware=hardware)


def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON/YAML-serialisable types.

    :class:`Unresolved` must survive the round trip so a sentinel written to
    disk reloads as a sentinel rather than as an opaque string.

    Enums are reduced to their ``value``, so a config carrying a
    :class:`DiscountBasis` still hashes and still round-trips through YAML.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Unresolved):
        return TBD_TOKEN
    if isinstance(value, Enum):
        return value.value
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a configuration file.

    A config may declare ``extends: <path>`` to inherit from another, with its
    own keys deep-merged over the parent's. This keeps the shared frozen
    hyperparameters in one place, so every task sweep runs the identical PPO
    configuration and a per-task divergence is visible as a diff.

    Args:
        path: Path to the YAML configuration.

    Raises:
        ConfigError: If the file is missing, empty, malformed, or has an
            inheritance cycle.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError(f"configuration file is empty: {path}")
    data = _require_mapping(data, str(path))

    parent_ref = data.get("extends")
    if parent_ref is not None:
        seen = {path.resolve()}
        parent_path = (path.parent / str(parent_ref)).resolve()
        while True:
            if parent_path in seen:
                raise ConfigError(f"configuration inheritance cycle at {parent_path}")
            seen.add(parent_path)
            if not parent_path.exists():
                raise ConfigError(f"inherited config not found: {parent_path}")
            with parent_path.open() as handle:
                parent_data = yaml.safe_load(handle) or {}
            grandparent = parent_data.get("extends") if isinstance(parent_data, dict) else None
            if grandparent is None:
                break
            parent_path = (parent_path.parent / str(grandparent)).resolve()
        with parent_path.open() as handle:
            parent_data = yaml.safe_load(handle) or {}
        data = _deep_merge(_require_mapping(parent_data, str(parent_path)), data)

    return ProjectConfig.from_dict(data)


def dump_config(config: ProjectConfig, path: str | Path) -> None:
    """Write a configuration back out, preserving TBD sentinels."""
    with Path(path).open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, default_flow_style=False)
