"""Run manifests and result records (`PLAN.md` section 19).

Every training run writes one JSON manifest plus one JSON result, both under
``results/<run_id>/``. The manifest is written **before** training starts and
marked complete only when it finishes, so an interrupted sweep leaves
``status: "running"`` rows rather than silently vanishing. That distinction
matters for a 50-cell sweep: a missing file is ambiguous, a half-written one is
not.

What goes in and why:

- ``config`` and ``config_hash`` tie a result to the exact settings. The hash
  is content-derived, so two runs with the same hash really did run the same
  configuration.
- ``seeds`` records the three roles section 19 requires separately, and the
  eval base seed is *shared* across latency conditions -- that sharing is what
  makes the comparison paired rather than merely repeated.
- ``icdi`` records both the requested and effective latency. Recording only the
  requested value would mislabel a quantised condition, which is the error
  `PLAN.md` section 5 is built to prevent.
- ``counts`` records the four quantities that must be inspectable per run. They
  are consequences of equal simulated-time exposure, not tuning knobs, and a
  reader should not have to rerun anything to see them.
- ``software`` records the interpreter and package versions, since a
  Gymnasium or torch change can move a result and the sweep is not rerun
  lightly.

Results are appended to a per-sweep ``summary.json`` as they complete, so a
long sweep can be monitored without a second pass.
"""

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

#: Manifest version, bumped if the schema changes incompatibly.
MANIFEST_VERSION: int = 1

#: Default root for run directories, overridable per sweep.
DEFAULT_RESULTS_ROOT: Path = Path("results")


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be written or is internally inconsistent."""


@dataclass(slots=True)
class RunManifest:
    """Everything needed to reproduce or audit one training run.

    Attributes:
        run_id: Unique identifier, ``<sweep>/<task>/<latency_ns>ns/seed<k>``.
        sweep_id: The sweep this run belongs to.
        task: Task name.
        latency_ns: Requested latency condition.
        train_seed: The training seed for this cell, recorded separately from
            the eval seeds so the two roles cannot be confused.
        config: The full validated configuration.
        config_hash: Content hash of ``config``.
        seeds: The three seed roles actually used.
        icdi: Derived condition metadata.
        dt_env_ns: The probed integration timestep.
        counts: Run accounting, filled in when training finishes.
        status: The run's lifecycle state. ``"running"`` is written before
            training begins, so an interrupted run is distinguishable from one
            that never started.
        checkpoint_path: File name of the policy weights, relative to this run's
            own directory. Relative to the run directory rather than the results
            root so it cannot duplicate the run id already in the path.
        error: Failure reason, present only when ``status`` is ``"failed"``.
    """

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
        """Manifest-friendly representation, JSON-serialisable throughout."""
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
    """Interpreter and key package versions.

    ``PLAN.md`` section 19 requires the environment be reproducible, and a
    Gymnasium or torch upgrade is exactly the kind of change that can move a
    result without anyone noticing.
    """
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
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_id(sweep_id: str, task: str, latency_ns: int, train_seed: int) -> str:
    """Canonical identifier for one sweep cell."""
    return f"{sweep_id}/{task}/{latency_ns}ns/seed{train_seed}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file and rename, so readers never see a partial file.

    A sweep can be inspected while it runs. A half-written manifest that parses
    as invalid JSON is worse than a slightly stale one, so the write is
    atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def manifest_path(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Where a manifest is stored."""
    return root / manifest.run_id / "manifest.json"


def result_path(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Where a result is stored."""
    return root / manifest.run_id / "result.json"


def write_manifest(manifest: RunManifest, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Write a manifest and return its path.

    Raises:
        ManifestError: If the manifest is missing a field needed to identify the
            run.
    """
    if not manifest.started_at:
        raise ManifestError(f"manifest for {manifest.run_id} has no started_at")
    if not manifest.seeds.eval:
        raise ManifestError(f"manifest for {manifest.run_id} has no eval seed")
    path = manifest_path(manifest, root)
    write_json_atomic(path, manifest.to_dict())
    return path


def mark_completed(
    manifest: RunManifest, counts: RunCounts, root: Path = DEFAULT_RESULTS_ROOT
) -> Path:
    """Finalise a manifest after a successful run."""
    manifest.counts = counts
    manifest.status = "completed"
    manifest.finished_at = utc_now()
    manifest.error = None
    return write_manifest(manifest, root)


def mark_failed(
    manifest: RunManifest, error: BaseException, root: Path = DEFAULT_RESULTS_ROOT
) -> Path:
    """Finalise a manifest after a failure, preserving the reason.

    A failed cell is recorded rather than dropped. In a 50-cell sweep a silently
    missing cell changes every downstream mean.
    """
    manifest.status = "failed"
    manifest.finished_at = utc_now()
    manifest.error = f"{type(error).__name__}: {error}"
    return write_manifest(manifest, root)


def read_manifest(run_id: str, root: Path = DEFAULT_RESULTS_ROOT) -> RunManifest:
    """Read a manifest back from disk.

    Round-trips the stored config through validation, so ``config_hash`` can be
    re-derived and compared against the recorded one. A manifest edited after
    the fact then fails that comparison instead of being trusted, which is the
    only reason to keep the full config in the file rather than just its hash.

    Args:
        run_id: A run identifier, as built by :func:`run_id`.
        root: Results root.

    Returns:
        The reconstructed manifest.

    Raises:
        FileNotFoundError: If the run has no manifest on disk.
    """
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
    """One (latency, seed) cell of a sweep.

    Attributes:
        latency_ns: The latency condition.
        train_seed: This cell's training seed.
    """

    latency_ns: int
    train_seed: int


def build_cells(latencies_ns: tuple[int, ...], train_seeds: tuple[int, ...]) -> list[SweepCell]:
    """Expand the Cartesian product of latencies and training seeds.

    Ordering is latency-major so a sweep can be interrupted part-way and still
    have complete seeds for the low-latency conditions, which are the reference
    points. A truncated sweep is then interpretable rather than a diagonal.
    """
    return [
        SweepCell(latency_ns=ns, train_seed=seed) for ns in latencies_ns for seed in train_seeds
    ]


def cells_for_seed(latencies_ns: tuple[int, ...], train_seed: int) -> list[SweepCell]:
    """All latency conditions for a single training seed.

    This is the unit of a paired comparison: one policy architecture and seed,
    evaluated across every latency.
    """
    return [SweepCell(latency_ns=ns, train_seed=train_seed) for ns in latencies_ns]


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """The full description of a sweep, written before any run starts.

    Written to ``results/<sweep_id>/plan.json`` so the intended design is
    recorded even if the sweep is abandoned, and so a resumed sweep can be
    checked against what was originally intended.
    """

    sweep_id: str
    task: str
    latencies_ns: tuple[int, ...]
    train_seeds: tuple[int, ...]
    config_hash: str
    #: Budget declared in the config file. Recorded separately from
    #: :attr:`effective_integration_steps` because a ``--steps`` override makes
    #: them differ, and the two are exactly what must not be conflated: the LR
    #: schedule bug came from annealing against one while training against the
    #: other.
    configured_integration_steps: int
    #: Budget the runs will actually consume, after any override.
    effective_integration_steps: int
    cells: tuple[SweepCell, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Plan-friendly representation."""
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
        """Canonical run ids for every cell, in plan order."""
        return tuple(
            run_id(self.sweep_id, self.task, cell.latency_ns, cell.train_seed)
            for cell in self.cells
        )


def make_sweep_plan(
    *,
    sweep_id: str,
    task: str,
    latencies_ns: tuple[int, ...],
    train_seeds: tuple[int, ...],
    config: ProjectConfig,
    total_integration_steps: int | None = None,
    root: Path = DEFAULT_RESULTS_ROOT,
) -> SweepPlan:
    """Build and write the sweep plan.

    Args:
        sweep_id: Identifier for this sweep.
        task: Task name.
        latencies_ns: Latency conditions.
        train_seeds: Training seeds, one replicate each.
        config: The configuration, hashed into the plan.
        total_integration_steps: Budget override, if any. Recorded as the
            effective budget so the plan states what the runs will consume
            rather than requiring it to be reconstructed from the config file
            and the command line.
        root: Results root. Taken as a parameter rather than assumed, so a
            caller with a custom root does not get a plan written to the
            default one.

    Raises:
        ManifestError: If the grid is empty or contains a non-positive latency.
    """
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
        effective_integration_steps=(
            total_integration_steps or config.training.total_integration_steps
        ),
        cells=tuple(build_cells(latencies_ns, train_seeds)),
        created_at=utc_now(),
    )
    write_json_atomic(root / sweep_id / "plan.json", plan.to_dict())
    return plan


class SweepLedger:
    """Append-only record of completed cells, for monitoring and resume.

    Kept as a single JSON file rewritten atomically per cell. A sweep is small
    (tens of cells), so an append-only log format would add parsing complexity
    for no benefit; the whole ledger is cheap to load and trivially inspectable.
    """

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
        """Where the ledger is stored."""
        if not self.sweep_id:
            raise ManifestError("SweepLedger needs a sweep_id to resolve a path")
        return self.root / self.sweep_id / "ledger.json"

    def record(self, manifest: RunManifest, root: Path | None = None) -> None:
        """Add or update a cell's outcome."""
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
        """Rewrite the ledger atomically, ordered by run id for stable diffs."""
        write_json_atomic(
            root / self.sweep_id / "ledger.json",
            {
                "sweep_id": self.sweep_id,
                "n_completed": len(self.entries),
                "entries": [self.entries[k] for k in sorted(self.entries)],
            },
        )

    def completed(self) -> list[str]:
        """Run ids with a terminal status."""
        return [k for k, v in self.entries.items() if v["status"] in ("completed", "failed")]

    def pending(self, plan: SweepPlan) -> list[str]:
        """Plan run ids not yet terminal, for resuming an interrupted sweep."""
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
