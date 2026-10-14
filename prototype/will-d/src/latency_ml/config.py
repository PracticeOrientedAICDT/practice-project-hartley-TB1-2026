from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from latency_ml.timing import NATIVE_SWEEP_GRID_NS

TBD_TOKEN: Final = "TBD"


class ConfigError(ValueError):
    pass


class HardwareModelNotFrozen(RuntimeError):
    pass


class Unresolved:
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


def _require_mapping(data: object, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(data).__name__}")
    return data


class DiscountBasis(Enum):
    PHYSICAL = "physical"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    algorithm: str = "ppo"
    learning_rate: float = 0.00025
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
    total_integration_steps: int = 2000000
    control_training_seeds: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "learning_rate" in kwargs:
            kwargs["learning_rate"] = float(kwargs["learning_rate"])
        return cls(**kwargs)

    def iterations(self, action_steps: int) -> int:
        if action_steps < 1:
            raise ConfigError(f"action_steps must be >= 1, got {action_steps}")
        per_iter = self.num_envs * self.steps_per_decision * action_steps
        return self.total_integration_steps // per_iter

    def minibatch_size(self, action_steps: int) -> int:
        transitions = self.num_envs * self.steps_per_decision
        return transitions // self.num_minibatches


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    control_episodes_per_seed: int = 200
    eval_seed_base: int = 10000
    paired_evaluation: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def episode_seed(self, index: int) -> int:
        return self.eval_seed_base + index


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    sweep_grid_ns: tuple[int, ...] = NATIVE_SWEEP_GRID_NS

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyConfig:
        sweep = data.get("sweep_grid_ns")
        return cls(sweep_grid_ns=tuple(int(x) for x in sweep) if sweep else NATIVE_SWEEP_GRID_NS)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str
    seed: int
    task: str
    training: TrainingConfig
    evaluation: EvaluationConfig
    latency: LatencyConfig
    environment_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        project = _require_mapping(data.get("project", {}), "project")
        for key in ("name", "seed", "task"):
            if key not in project:
                raise ConfigError(f"project.{key} is required")
        return cls(
            name=str(project["name"]),
            seed=int(project["seed"]),
            task=str(project["task"]),
            training=TrainingConfig.from_dict(
                _require_mapping(data.get("training", {}), "training")
            ),
            evaluation=EvaluationConfig.from_dict(
                _require_mapping(data.get("evaluation", {}), "evaluation")
            ),
            latency=LatencyConfig.from_dict(_require_mapping(data.get("latency", {}), "latency")),
            environment_kwargs=dict(
                _require_mapping(data.get("environment_kwargs", {}), "environment_kwargs")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        flat = _jsonable(asdict(self))
        assert isinstance(flat, dict)
        scalar_keys = ("name", "seed", "task")
        nested: dict[str, Any] = {"project": {k: flat.pop(k) for k in scalar_keys}}
        nested.update(flat)
        return nested

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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
    with Path(path).open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, default_flow_style=False)
