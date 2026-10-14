"""Provenance checks on completed runs: is a result operationally complete?

This exists because of a specific failure. The trainer consolidation computed
wall time and put it on the ``TrainResult``, but never onto each task's
``RunCounts`` -- and ``RunCounts`` is the object that reaches the manifest. Every
cell in a 100-cell sweep therefore reported ``in 0.0s`` and every manifest
recorded a wall time of exactly zero.

The number was wrong and nothing objected. The bit-identity parity gate stayed
green throughout, correctly: it compares weights, and a logging field is not a
weight. Both gates were doing their jobs and neither was scoped to catch this.

That is the general lesson, and it is what this module encodes. **Numerical
parity and provenance completeness are different invariants.** A run can be
bit-perfect and operationally empty at the same time, so "all tests pass" is not
evidence that a result is usable. A zero where a measurement belongs is the worst
case available: it reads as a measurement rather than as a missing one, and no
downstream consumer has any way to tell it apart from a genuinely instantaneous
run. This validator therefore treats a zero count as a failure rather than as a
value, and requires the rest of the evidence a result needs to be interpreted.

Two levels, because they can be checked at different times
---------------------------------------------------------
:func:`validate_completed_cell` checks one cell and is safe to run the moment a
cell finishes: the manifest, the counts, the protocol fingerprint and the
checkpoint. :func:`verify_sweep` additionally requires the sweep-level
evaluation artifact, which legitimately does not exist until ``evaluate`` has
been run -- so requiring it at sweep end would flag every sweep that had not yet
been evaluated, which is a false alarm and would train people to ignore the
validator.

Historical artefacts are exempt by explicit request, never by default
--------------------------------------------------------------------
``tests/fixtures/v1/`` holds frozen manifests from a previous version that
predate the protocol fingerprint. Re-verifying those requires
``allow_missing_provenance=True``, passed on purpose by a caller who knows why.
The default is strict, because a default that quietly tolerates the defect is
worth nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Counts that must be strictly positive on a completed cell. A zero in any of
#: these is treated as "absent", never as "measured as zero", except for
#: ``wall_time_s`` which is handled separately because it shares the failure mode
#: with the rest but has a different justification (see module docstring).
REQUIRED_POSITIVE_COUNTS = (
    "integration_steps",
    "decision_transitions",
    "ppo_updates",
    "gradient_steps",
    "action_steps",
)


class ProvenanceError(RuntimeError):
    """Raised when a run is numerically fine but operationally incomplete.

    Deliberately distinct from :class:`ManifestError`. A manifest error means the
    file cannot be written or read; this means the file is readable, internally
    consistent, and still insufficient to support a claim.
    """


@dataclass(frozen=True)
class CellReport:
    """What one cell's provenance looks like."""

    run_id: str
    ok: bool
    problems: tuple[str, ...] = ()
    #: Accepted-by-request absences. Recorded so a caller can see what was waived,
    #: but never counted as failures -- otherwise asking for tolerance would still
    #: report the cell as broken and the flag would be useless.
    notes: tuple[str, ...] = ()
    fingerprint: str | None = None
    wall_time_s: float | None = None

    def describe(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        detail = "; ".join(self.problems)
        if self.notes:
            detail = (detail + "; " if detail else "") + f"[{'; '.join(self.notes)}]"
        return f"{mark} {self.run_id}" + (f": {detail}" if detail else "")


@dataclass(frozen=True)
class SweepReport:
    """Provenance across a whole sweep directory."""

    root: Path
    cells: tuple[CellReport, ...] = field(default_factory=tuple)
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems and all(c.ok for c in self.cells)

    @property
    def fingerprints(self) -> tuple[str, ...]:
        seen = {c.fingerprint for c in self.cells if c.fingerprint}
        return tuple(sorted(seen))

    def describe(self) -> str:
        lines = [c.describe() for c in self.cells]
        lines.extend(f"FAIL {p}" for p in self.problems)
        if self.ok:
            lines.append(f"ok   {len(self.cells)} cells, all complete")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ProvenanceError(self.describe())


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def _check_count(name: str, value: Any, run_id: str, problems: list[str]) -> None:
    if value is None:
        problems.append(f"{name} is absent")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        problems.append(f"{name} is not numeric ({value!r})")
        return
    if value <= 0:
        problems.append(
            f"{name} is {value!r}, which reads as a measurement of zero rather "
            f"than as a missing value"
        )


def validate_completed_cell(
    run_dir: Path,
    *,
    allow_missing_provenance: bool = False,
    require_checkpoint: bool = True,
) -> CellReport:
    """Check one run directory for the evidence a result needs.

    Args:
        run_dir: A directory containing ``manifest.json``.
        allow_missing_provenance: Permit a manifest with no
            ``extra.task_protocol.fingerprint``. Only for artefacts frozen before
            the fingerprint existed.
        require_checkpoint: Check that the recorded checkpoint exists and is
            non-empty.

    Returns:
        A :class:`CellReport`. Does not raise; use
        :meth:`CellReport.describe` or ``raise_for_status``-style handling at the
        call site. Validation reports on partial damage without stopping at the
        first problem, which matters when a sweep is being audited after the fact.
    """
    problems: list[str] = []
    notes: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return CellReport(
            run_id=run_dir.name, ok=False, problems=(f"no manifest.json in {run_dir}",)
        )

    try:
        payload = _load(manifest_path)
    except json.JSONDecodeError as exc:
        return CellReport(
            run_id=run_dir.name, ok=False, problems=(f"manifest is not valid JSON: {exc}",)
        )

    run_id = str(payload.get("run_id", run_dir.name))
    status = payload.get("status")

    if status != "completed":
        detail = payload.get("error")
        suffix = f" ({detail})" if detail else ""
        problems.append(f"status is {status!r}, not 'completed'{suffix}")

    if not payload.get("started_at"):
        problems.append("started_at is absent")
    if not payload.get("finished_at"):
        problems.append("finished_at is absent")
    if not payload.get("config_hash"):
        problems.append("config_hash is absent")

    counts = payload.get("counts")
    wall_time: float | None = None
    if counts is None:
        problems.append("counts is absent")
    elif not isinstance(counts, dict):
        # A malformed manifest is a finding, not an exception. This validator runs
        # over whole directories during an audit, and one unparseable field must not
        # abort the sweep it is checking.
        problems.append(f"counts is {type(counts).__name__}, not an object")
        counts = None
    if counts is not None:
        wall = counts.get("wall_time_s")
        wall_time = wall if isinstance(wall, (int, float)) and not isinstance(wall, bool) else None
        if wall is None:
            problems.append("wall_time_s is absent")
        elif wall <= 0:
            problems.append(
                f"wall_time_s is {wall!r}: a completed cell cannot have taken no time, "
                f"and 0.0 reads as a measurement rather than as a missing one"
            )
        for name in REQUIRED_POSITIVE_COUNTS:
            _check_count(name, counts.get(name), run_id, problems)

    fingerprint: str | None = None
    protocol = (payload.get("extra") or {}).get("task_protocol") or {}
    fingerprint = protocol.get("fingerprint")
    activation = protocol.get("activation")
    if not fingerprint:
        if allow_missing_provenance:
            notes.append("protocol fingerprint absent (tolerated)")
        else:
            problems.append(
                "protocol fingerprint is absent: without it the manifest cannot "
                "be shown to have run a specific protocol"
            )
    if not activation and not allow_missing_provenance:
        problems.append("recorded activation is absent")

    if require_checkpoint:
        checkpoint = payload.get("checkpoint_path")
        if not checkpoint:
            problems.append("checkpoint_path is absent")
        else:
            checkpoint_path = run_dir / checkpoint
            if not checkpoint_path.exists():
                problems.append(f"checkpoint {checkpoint} does not exist")
            elif checkpoint_path.stat().st_size == 0:
                problems.append(f"checkpoint {checkpoint} is empty")

    return CellReport(
        run_id=run_id,
        ok=not problems,
        problems=tuple(problems),
        notes=tuple(notes),
        fingerprint=fingerprint,
        wall_time_s=wall_time,
    )


def find_run_dirs(root: Path, sweep_id: str | None = None) -> list[Path]:
    """Every run directory under ``root``, optionally scoped to one sweep.

    A sweep directory itself holds ``plan.json`` and ``ledger.json`` rather than a
    manifest, so recursing finds exactly the per-cell directories.
    """
    base = root / sweep_id if sweep_id else root
    if not base.exists():
        return []
    return sorted({path.parent for path in base.rglob("manifest.json")})


def verify_sweep(
    root: Path,
    sweep_id: str | None = None,
    *,
    require_evaluation: bool = True,
    allow_missing_provenance: bool = False,
    require_checkpoint: bool = True,
) -> SweepReport:
    """Check a sweep directory end to end.

    Args:
        root: Results root, e.g. ``results``.
        sweep_id: Sweep subdirectory. ``None`` checks everything under ``root``.
        require_evaluation: Require ``<root>/<sweep_id>/evaluation.json``. Pass
            ``False`` at the end of a sweep, before ``evaluate`` has run, where
            its absence is expected rather than wrong.
        allow_missing_provenance: See :func:`validate_completed_cell`.
        require_checkpoint: See :func:`validate_completed_cell`.
    """
    base = root / sweep_id if sweep_id else root
    sweep_problems: list[str] = []
    if not base.exists():
        return SweepReport(root=base, problems=(f"{base} does not exist",))

    run_dirs = find_run_dirs(root, sweep_id)
    cells = tuple(
        validate_completed_cell(
            path,
            allow_missing_provenance=allow_missing_provenance,
            require_checkpoint=require_checkpoint,
        )
        for path in run_dirs
    )

    if not cells:
        return SweepReport(
            root=base,
            problems=(
                f"{base} contains no run directories",
                *((f"no evaluation.json in {base}",) if require_evaluation else ()),
            ),
        )

    failed = [c for c in cells if not c.ok]
    if failed:
        sweep_problems.append(
            f"{len(failed)} of {len(cells)} cells are incomplete; first: "
            f"{failed[0].run_id}: {failed[0].problems[0] if failed[0].problems else ''}"
        )

    # Fingerprints are compared within a task, never across tasks.
    #
    # A run id is <sweep>/<task>/<ns>ns/seed<N> and TaskProtocol is task-specific,
    # so two different tasks legitimately carry different fingerprints. Comparing
    # them as though they ought to match would report every multi-task results
    # directory as broken.
    by_task: dict[str, set[str]] = {}
    for cell in cells:
        parts = cell.run_id.split("/")
        task = parts[1] if len(parts) > 1 else "?"
        if cell.fingerprint:
            by_task.setdefault(task, set()).add(cell.fingerprint)
    for task, prints in sorted(by_task.items()):
        if len(prints) > 1:
            sweep_problems.append(
                f"{task} cells disagree on the protocol fingerprint: {sorted(prints)}; "
                f"cells that ran different protocols are not one comparison"
            )

    # The evaluation artifact lives at the sweep level, so it is checked per sweep
    # rather than at the root. Sweep ids are the first component of a run id.
    if require_evaluation:
        sweeps = {cell.run_id.split("/")[0] for cell in cells}
        for sweep in sorted(sweeps):
            evaluation = root / sweep / "evaluation.json"
            if not evaluation.exists():
                sweep_problems.append(
                    f"no evaluation.json for sweep {sweep!r}; the checkpoints exist "
                    f"but nothing has been evaluated, so no latency response can be claimed"
                )

    return SweepReport(root=base, cells=cells, problems=tuple(sweep_problems))


__all__ = [
    "REQUIRED_POSITIVE_COUNTS",
    "CellReport",
    "ProvenanceError",
    "SweepReport",
    "find_run_dirs",
    "validate_completed_cell",
    "verify_sweep",
]
