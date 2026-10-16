from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latency_ml.timing import NATIVE_SWEEP_GRID_NS


class ConfigError(ValueError):
    pass


def _require_mapping(data: object, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True, slots=True)
class TrainingConfig:
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
    total_integration_steps: int = 2000000
    control_training_seeds: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "learning_rate" in kwargs:
            kwargs["learning_rate"] = float(kwargs["learning_rate"])
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    control_episodes_per_seed: int = 200
    eval_seed_base: int = 10000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

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
            training=TrainingConfig.from_dict(_require_mapping(data.get("training", {}), "training")),
            evaluation=EvaluationConfig.from_dict(_require_mapping(data.get("evaluation", {}), "evaluation")),
            latency=LatencyConfig.from_dict(_require_mapping(data.get("latency", {}), "latency")),
        )


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


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    return data


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config, following the ``extends`` chain from root down."""
    chain: list[dict[str, Any]] = []
    seen: set[Path] = set()
    current = Path(path)
    while True:
        current = current.resolve()
        if current in seen:
            raise ConfigError(f"configuration inheritance cycle at {current}")
        seen.add(current)
        data = _read_config(current)
        chain.append(data)
        parent = data.get("extends")
        if parent is None:
            break
        current = current.parent / str(parent)
    merged: dict[str, Any] = {}
    for data in reversed(chain):
        merged = _deep_merge(merged, data)
    return ProjectConfig.from_dict(merged)
