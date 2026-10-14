"""Phase 4 reuse accounting: task-specific accelerator vs shared trunk.

PLAN.md section 15 asks whether the specialised neural-compute portion of the
hypothetical chip can be reused across workloads. This module answers the
**architecture-level** half of that question, which is fully determined by layer
widths and therefore needs no hardware specification at all.

The half it deliberately does *not* answer is latency, which requires
``peak_macs_per_second`` and ``fixed_overhead_ns`` -- both still ``TBD``. Those
columns are emitted as ``PENDING_HARDWARE_MODEL`` for the same reason the Phase 3
Fast cells are: a modelled latency derived from an unfrozen throughput would be
an invented number wearing a derived costume.

**The finding this accounting exists to state precisely.** Under the frozen
section 15.2 whole-trunk architecture, sharing reduces distinct accelerator MACs by
6.6% while increasing per-decision MACs by 93%, because the shared trunk contains
an additional 128x128 layer relative to the dedicated task networks.

Whether that trade-off is desirable depends on the hardware cost model, which is
exactly what the unresolved hardware model is meant to determine. So the result is
**not** "sharing is bad", and this module must never be summarised that way. What
it establishes is that **parameter/silicon sharing and execution efficiency are not
the same objective**: an architecture can win on one and lose badly on the other.

Reporting either side alone would mislead. Total MACs alone hides that distinct
silicon fell; distinct silicon alone hides that per-decision work rose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from latency_ml.config import HardwareSpec
from latency_ml.hardware.latency_model import explain_pending, is_available, predict_latency_ns
from latency_ml.hardware.mac_counter import Architecture

#: Tasks in the Phase 4 generality comparison, per PLAN.md section 15.2.
TASKS: dict[str, dict[str, int]] = {
    "cartpole": {"obs_dim": 4, "n_actions": 2},
    "lunarlander": {"obs_dim": 8, "n_actions": 4},
}

#: Shared trunk width, fixed by PLAN.md section 15.2.
SHARED_WIDTH: int = 128

#: The frozen Phase 4 architecture, answering PLAN.md section 30 question 13.
#:
#: Decided: **whole-trunk sharing**. The entire ``128-128-128`` trunk is shared;
#: per-task input projections and output heads stay outside it as adapters.
#:
#: The alternative considered and *not* taken was sharing only a matrix-
#: multiplication engine while each task keeps its own depth. It was rejected on
#: methodological grounds, not aesthetic ones: the result below is only
#: interpretable as a statement about one specific architecture, and switching
#: architectures because a measurement came out unfavourably would convert an
#: experiment into a search. It is recorded here as a candidate for a later
#: sensitivity experiment rather than quietly becoming the primary design.
PRIMARY_ARCHITECTURE: str = "whole_trunk"

#: Recorded alternative, not measured in this phase.
ALTERNATE_ARCHITECTURE: str = "matmul_engine_only"

#: Phase 4A is architecture-exact and needs no hardware model.
#: Phase 4B converts those quantities into latency/resource estimates and is
#: blocked until the Hartley model is frozen.
PHASE_4A_EXACT: tuple[str, ...] = (
    "parameter counts",
    "distinct accelerator MACs",
    "per-decision execution MACs",
    "shared vs dedicated silicon",
    "workloads supported",
    "adapter/core decomposition",
)

#: Hardware-model parameters Phase 4B requires before it can produce a number.
PHASE_4B_REQUIRES: tuple[str, ...] = (
    "precision",
    "throughput / MAC rate",
    "latency",
    "memory / weight storage assumptions",
    "fixed overhead",
    "analogue / noise assumptions, if they enter the calculation",
)


def _latency_or_pending(spec: HardwareSpec, macs: int) -> int | str:
    """Modelled latency, or an explicit pending marker."""
    if not is_available(spec):
        return "PENDING_HARDWARE_MODEL"
    return predict_latency_ns(spec, macs)


@dataclass(frozen=True)
class TaskDesign:
    """One task's cost under one of the two topologies.

    Attributes:
        task: Task name.
        topology: ``"task_specific"`` or ``"shared_trunk"``.
        accel_macs: MACs executed inside the shared/reusable accelerator.
        adapter_macs: MACs executed outside it, in software adapters.
        total_macs: ``accel_macs + adapter_macs``.
        accel_params / adapter_params: Parameter counts in each region.
        accel_latency_ns: Modelled accelerator latency, or pending.
    """

    task: str
    topology: str
    accel_macs: int
    adapter_macs: int
    accel_params: int
    adapter_params: int
    accel_latency_ns: int | str

    @property
    def total_macs(self) -> int:
        return self.accel_macs + self.adapter_macs

    @property
    def total_params(self) -> int:
        return self.accel_params + self.adapter_params

    @property
    def adapter_overhead_fraction(self) -> float:
        """Share of arithmetic spent outside the reused core."""
        return self.adapter_macs / self.total_macs if self.total_macs else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "topology": self.topology,
            "accel_macs": self.accel_macs,
            "adapter_macs": self.adapter_macs,
            "total_macs": self.total_macs,
            "accel_params": self.accel_params,
            "adapter_params": self.adapter_params,
            "total_params": self.total_params,
            "adapter_overhead_fraction": self.adapter_overhead_fraction,
            "accel_latency_ns": self.accel_latency_ns,
        }


@dataclass
class ReuseReport:
    """Both topologies, side by side, with the physical accounting PLAN.md 15.4 asks for."""

    task_specific: list[TaskDesign] = field(default_factory=list)
    shared: list[TaskDesign] = field(default_factory=list)
    spec: HardwareSpec | None = None

    @property
    def n_workloads(self) -> int:
        return len(self.shared)

    @property
    def total_accel_macs(self) -> int:
        """Accelerator MACs summed over every design that must be built.

        For the shared topology this counts the core **once**, not once per task.
        That single distinction is the entire hardware argument for reuse.
        """
        return sum(d.accel_macs for d in self.task_specific)

    @property
    def shared_accel_macs(self) -> int:
        return sum(d.accel_macs for d in self.shared[:1]) if self.shared else 0

    @property
    def distinct_stored_params(self) -> dict[str, int]:
        """Parameter memory that must actually exist, per topology.

        The shared core is stored **once**, so summing per-task parameter counts
        would double-count it and overstate shared memory by nearly 2x. What is
        stored is the core plus every task's adapters.
        """
        dedicated = sum(d.total_params for d in self.task_specific)
        shared = 0
        if self.shared:
            shared = self.shared[0].accel_params + sum(d.adapter_params for d in self.shared)
        return {"task_specific": dedicated, "shared_trunk": shared}

    def as_dict(self) -> dict[str, object]:
        return {
            "architecture": PRIMARY_ARCHITECTURE,
            "alternate_architecture_not_run": ALTERNATE_ARCHITECTURE,
            "phase_4a_exact": list(PHASE_4A_EXACT),
            "phase_4b_requires": list(PHASE_4B_REQUIRES),
            "n_workloads": self.n_workloads,
            "task_specific": [d.as_dict() for d in self.task_specific],
            "shared_trunk": [d.as_dict() for d in self.shared],
            "distinct_accelerator_macs": {
                "task_specific": self.total_accel_macs,
                "shared_trunk": self.shared_accel_macs,
            },
            "distinct_stored_params": self.distinct_stored_params,
            "total_execution_macs": {
                "task_specific": sum(d.total_macs for d in self.task_specific),
                "shared_trunk": sum(d.total_macs for d in self.shared),
            },
            "latency_status": (
                "measured" if self.spec and is_available(self.spec) else "PENDING_HARDWARE_MODEL"
            ),
        }


def build_report(spec: HardwareSpec | None = None) -> ReuseReport:
    """Compute the reuse accounting for both topologies.

    Args:
        spec: Hardware contract, used only for the modelled-latency column. When
            unresolved, that column is a pending marker and every other column is
            still exact, because MAC and parameter counts are architecture-level
            quantities that need no hardware assumption.

    Returns:
        The populated report.
    """
    report = ReuseReport(spec=spec)
    w = SHARED_WIDTH

    for task, shape in TASKS.items():
        obs, acts = shape["obs_dim"], shape["n_actions"]

        # Topology A: one dedicated 2x128 unit per task. Nothing is reused.
        dedicated = Architecture(widths=(obs, w, w, acts))
        report.task_specific.append(
            TaskDesign(
                task=task,
                topology="task_specific",
                accel_macs=dedicated.macs(),
                adapter_macs=0,
                accel_params=dedicated.parameters(),
                adapter_params=0,
                accel_latency_ns=_latency_or_pending(spec, dedicated.macs())
                if spec
                else "PENDING_HARDWARE_MODEL",
            )
        )

        # Topology B: the two 128x128 matrices live in the shared accelerator,
        # built once; the input projection and output head stay in software.
        #
        # The adapter cost is computed from its own two projections, NOT by
        # subtracting the trunk from the full network. Those are not nested: in
        # the dedicated design the first matrix is obs->128, whereas the shared
        # core's first matrix is 128->128, so the trunk is not a subset of the
        # full net and subtracting it yields a negative adapter cost.
        trunk = Architecture(widths=(w, w, w))
        in_adapter = Architecture(widths=(obs, w))
        out_adapter = Architecture(widths=(w, acts))
        accel_macs = trunk.macs()
        adapter_macs = in_adapter.macs() + out_adapter.macs()
        adapter_params = in_adapter.parameters() + out_adapter.parameters()
        report.shared.append(
            TaskDesign(
                task=task,
                topology="shared_trunk",
                accel_macs=accel_macs,
                adapter_macs=adapter_macs,
                accel_params=trunk.parameters(),
                adapter_params=adapter_params,
                accel_latency_ns=_latency_or_pending(spec, accel_macs)
                if spec
                else "PENDING_HARDWARE_MODEL",
            )
        )

    return report


def format_report(report: ReuseReport) -> str:
    """Render the PLAN.md 15.4 accounting, split into Phase 4A and 4B.

    Phase 4A rows are exact and need no hardware model. Phase 4B rows are
    emitted as pending markers, with the parameters that would unlock them, so a
    reader can see what is missing without reading the source.
    """
    lines: list[str] = ["### Phase 4A — architecture accounting (exact, no hardware model)", ""]
    lines.append("| | task-specific | shared trunk |")
    lines.append("|---|---|---|")

    dedicated_work = sum(d.total_macs for d in report.task_specific)
    shared_work = sum(d.total_macs for d in report.shared)

    lines.append(
        f"| distinct accelerator MACs (silicon built once) | {report.total_accel_macs} | "
        f"{report.shared_accel_macs} |"
    )
    lines.append(f"| total execution MACs (all decisions) | {dedicated_work} | {shared_work} |")
    stored = report.distinct_stored_params
    lines.append(
        f"| distinct stored parameters | {stored['task_specific']} | {stored['shared_trunk']} |"
    )
    lines.append(f"| distinct accelerator instances | {len(report.task_specific)} | 1 |")
    lines.append(f"| workloads supported | {len(report.task_specific)} | {report.n_workloads} |")
    lines.append("")
    lines.append(
        "| task | accelerator MACs | adapter MACs | adapter share | accel params | adapter params |"
    )
    lines.append("|---|---|---|---|---|---|")
    for design in report.shared:
        lines.append(
            f"| {design.task} | {design.accel_macs} | {design.adapter_macs} | "
            f"{design.adapter_overhead_fraction:.1%} | {design.accel_params} | "
            f"{design.adapter_params} |"
        )

    silicon_delta = (report.shared_accel_macs - report.total_accel_macs) / report.total_accel_macs
    work_delta = (shared_work - dedicated_work) / dedicated_work
    lines.append("")
    lines.append(
        f"Under the frozen `{PRIMARY_ARCHITECTURE}` architecture: silicon "
        f"{silicon_delta:+.1%}, per-decision work {work_delta:+.1%}. The shared "
        "trunk holds an additional 128x128 layer relative to each dedicated "
        "network, and the cheap adapters do not offset it."
    )

    lines.append("")
    lines.append("### Phase 4B — hardware performance (blocked)")
    lines.append("")
    if report.spec and not is_available(report.spec):
        lines.append(
            f"Latency: `PENDING_HARDWARE_MODEL` — {explain_pending(report.spec).splitlines()[0]}"
        )
    else:
        lines.append("Latency: `PENDING_HARDWARE_MODEL` — no hardware spec supplied.")
    lines.append("")
    lines.append("Requires an explicitly frozen model for:")
    for item in PHASE_4B_REQUIRES:
        lines.append(f"- {item}")

    lines.append("")
    lines.append(
        f"Recorded alternative architecture, not run: `{ALTERNATE_ARCHITECTURE}`. "
        "It was not selected because a measurement came out unfavourably — that "
        "would be a search, not an experiment — and is held for a later "
        "sensitivity study."
    )
    return "\n".join(lines)
