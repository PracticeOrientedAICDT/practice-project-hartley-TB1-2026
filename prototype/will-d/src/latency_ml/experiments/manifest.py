from __future__ import annotations

import json
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latency_ml.config import ProjectConfig
from latency_ml.rl.ppo import RunCounts
from latency_ml.seeding import Seeds
from latency_ml.timing import ICDI

MANIFEST_VERSION: int = 1
DEFAULT_RESULTS_ROOT: Path = Path("results")


class ManifestError(RuntimeError):
    pass


@dataclass(slots=True)
class RunManifest:
    run_id: str
    sweep_id: str
    task: str
    latency_ns: int
    train_seed: int
    config: ProjectConfig
    seeds: Seeds
    icdi: ICDI
    dt_env_ns: int
    counts: RunCounts | None = None
    checkpoint_path: str | None = None
    status: str = "running"
    started_at: str = ""
    finished_at: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_id,
            "sweep_id": self.sweep_id,
            "task": self.task,
            "latency_ns": self.latency_ns,
            "train_seed": self.train_seed,
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "seeds": self.seeds.to_dict(),
            "icdi": {
                "requested_latency_ns": self.icdi.requested_ns,
                "effective_latency_ns": self.icdi.effective_ns,
                "action_steps": self.icdi.action_steps,
                "resolution": self.icdi.resolution.value,
                "is_independent_measurement": self.icdi.is_independent_measurement,
            },
            "dt_env_ns": self.dt_env_ns,
            "counts": self.counts.to_dict() if self.counts is not None else None,
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "software": software_versions(),
            **({"extra": self.extra} if self.extra else {}),
        }


def software_versions() -> dict[str, str]:
    import gymnasium
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gymnasium": gymnasium.__version__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_id(sweep_id: str, task: str, latency_ns: int, train_seed: int) -> str:
    return f"{sweep_id}/{task}/{latency_ns}ns/seed{train_seed}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def manifest_path(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return root / manifest.run_id / "manifest.json"


def result_path(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return root / manifest.run_id / "result.json"


def write_manifest(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    if not manifest.started_at:
        raise ManifestError(f"manifest for {manifest.run_id} has no started_at")
    if not manifest.seeds.eval:
        raise ManifestError(f"manifest for {manifest.run_id} has no eval seed")
    path = manifest_path(manifest, root)
    write_json_atomic(path, manifest.to_dict())
    return path


def mark_completed(manifest: RunManifest, counts: RunCounts, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    manifest.counts = counts
    manifest.status = "completed"
    manifest.finished_at = utc_now()
    manifest.error = None
    return write_manifest(manifest, root)


def mark_failed(manifest: RunManifest, error: BaseException, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    manifest.status = "failed"
    manifest.finished_at = utc_now()
    manifest.error = f"{type(error).__name__}: {error}"
    return write_manifest(manifest, root)


def read_manifest(run_id: str, root: Path = DEFAULT_RESULTS_ROOT) -> RunManifest:
    from latency_ml.timing import Resolution

    path = root / run_id / "manifest.json"
    payload = json.loads(path.read_text())
    icdi_payload = payload["icdi"]
    counts_payload = payload.get("counts")
    return RunManifest(
        run_id=payload["run_id"],
        sweep_id=payload["sweep_id"],
        task=payload["task"],
        latency_ns=int(payload["latency_ns"]),
        train_seed=int(payload["train_seed"]),
        config=ProjectConfig.from_dict(payload["config"]),
        seeds=Seeds(**payload["seeds"]),
        icdi=ICDI(
            requested_ns=int(icdi_payload["requested_latency_ns"]),
            action_steps=int(icdi_payload["action_steps"]),
            effective_ns=int(icdi_payload["effective_latency_ns"]),
            resolution=Resolution(icdi_payload["resolution"]),
        ),
        dt_env_ns=int(payload["dt_env_ns"]),
        counts=RunCounts(**counts_payload) if counts_payload else None,
        checkpoint_path=payload.get("checkpoint_path"),
        status=payload["status"],
        started_at=payload["started_at"],
        finished_at=payload.get("finished_at", ""),
        error=payload.get("error"),
        extra=dict(payload.get("extra", {})),
    )


@dataclass(frozen=True, slots=True)
class SweepCell:
    latency_ns: int
    train_seed: int


def build_cells(latencies_ns: tuple[int, ...], train_seeds: tuple[int, ...]) -> list[SweepCell]:
    return [SweepCell(latency_ns=ns, train_seed=seed) for ns in latencies_ns for seed in train_seeds]


def cells_for_seed(latencies_ns: tuple[int, ...], train_seed: int) -> list[SweepCell]:
    return [SweepCell(latency_ns=ns, train_seed=train_seed) for ns in latencies_ns]


@dataclass(frozen=True, slots=True)
class SweepPlan:
    sweep_id: str
    task: str
    latencies_ns: tuple[int, ...]
    train_seeds: tuple[int, ...]
    config_hash: str
    configured_integration_steps: int
    effective_integration_steps: int
    cells: tuple[SweepCell, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "task": self.task,
            "latencies_ns": list(self.latencies_ns),
            "train_seeds": list(self.train_seeds),
            "config_hash": self.config_hash,
            "configured_integration_steps": self.configured_integration_steps,
            "effective_integration_steps": self.effective_integration_steps,
            "n_cells": len(self.cells),
            "cells": [asdict(cell) for cell in self.cells],
            "created_at": self.created_at,
        }

    def run_ids(self) -> tuple[str, ...]:
        return tuple(run_id(self.sweep_id, self.task, cell.latency_ns, cell.train_seed) for cell in self.cells)


def make_sweep_plan(
    *, sweep_id: str, task: str, latencies_ns: tuple[int, ...], train_seeds: tuple[int, ...], config: ProjectConfig, total_integration_steps: int | None = None, root: Path = DEFAULT_RESULTS_ROOT
) -> SweepPlan:
    if not latencies_ns:
        raise ManifestError(f"sweep {sweep_id} has no latency conditions")
    if not train_seeds:
        raise ManifestError(f"sweep {sweep_id} has no training seeds")
    bad = [ns for ns in latencies_ns if ns <= 0]
    if bad:
        raise ManifestError(f"sweep {sweep_id} has non-positive latencies: {bad}")
    plan = SweepPlan(
        sweep_id=sweep_id,
        task=task,
        latencies_ns=tuple(latencies_ns),
        train_seeds=tuple(train_seeds),
        config_hash=config.config_hash(),
        configured_integration_steps=config.training.total_integration_steps,
        effective_integration_steps=total_integration_steps or config.training.total_integration_steps,
        cells=tuple(build_cells(latencies_ns, train_seeds)),
        created_at=utc_now(),
    )
    write_json_atomic(root / sweep_id / "plan.json", plan.to_dict())
    return plan


class SweepLedger:
    pass

    def __init__(self, root: Path = DEFAULT_RESULTS_ROOT, sweep_id: str = "") -> None:
        self.root = root
        self.sweep_id = sweep_id
        self.entries: dict[str, dict[str, Any]] = {}
        if sweep_id:
            existing = self.path
            if existing.exists():
                payload = json.loads(existing.read_text())
                for entry in payload.get("entries", []):
                    self.entries[entry["run_id"]] = entry

    @property
    def path(self) -> Path:
        if not self.sweep_id:
            raise ManifestError("SweepLedger needs a sweep_id to resolve a path")
        return self.root / self.sweep_id / "ledger.json"

    def record(self, manifest: RunManifest, root: Path | None = None) -> None:
        entry = {
            "run_id": manifest.run_id,
            "task": manifest.task,
            "latency_ns": manifest.latency_ns,
            "train_seed": manifest.train_seed,
            "status": manifest.status,
            "error": manifest.error,
            "wall_time_s": manifest.counts.wall_time_s if manifest.counts else None,
            "integration_steps": manifest.counts.integration_steps if manifest.counts else None,
        }
        self.entries[manifest.run_id] = entry
        self._flush(root or self.root)

    def _flush(self, root: Path) -> None:
        write_json_atomic(root / self.sweep_id / "ledger.json", {"sweep_id": self.sweep_id, "n_completed": len(self.entries), "entries": [self.entries[k] for k in sorted(self.entries)]})

    def completed(self) -> list[str]:
        return [k for k, v in self.entries.items() if v["status"] in ("completed", "failed")]

    def pending(self, plan: SweepPlan) -> list[str]:
        done = set(self.completed())
        return [rid for rid in plan.run_ids() if rid not in done]


__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "MANIFEST_VERSION",
    "ManifestError",
    "RunManifest",
    "SweepCell",
    "SweepLedger",
    "SweepPlan",
    "build_cells",
    "cells_for_seed",
    "make_sweep_plan",
    "manifest_path",
    "mark_completed",
    "mark_failed",
    "read_manifest",
    "result_path",
    "run_id",
    "software_versions",
    "utc_now",
    "write_json_atomic",
    "write_manifest",
]
