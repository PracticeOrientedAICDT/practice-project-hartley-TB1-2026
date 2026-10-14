"""PPO over the action-hold wrapper.

Vendored from CleanRL's single-file reference. The upstream file is frozen (last
touched 2023-11) and predates the Gymnasium 1.0 autoreset rework, so it cannot
run against the pinned 1.3.0. Its *hyperparameters* are used as a frozen
baseline (see ``docs/ENVIRONMENT.md``); the environment handling here is written
against the current API.

Two deviations from a textbook per-step PPO, both forced by the action-hold
wrapper rather than chosen:

1. **One rollout step is one decision, not one integration step.** The wrapper
   collapses a held action and its ``N`` integration steps into one semi-Markov
   transition, so the rollout covers ``steps_per_decision * N`` integration
   steps per environment.

2. **The budget is measured in integration steps.** Equal simulated task time
   across latency conditions is what makes the comparison fair. At a fixed
   budget, a slower policy simply gets fewer decisions, which is the phenomenon
   under study rather than a confound.

The four quantities the protocol requires be logged per run --
``integration_steps``, ``decision_transitions``, ``ppo_updates`` and
``gradient_steps`` -- are accumulated in :class:`RunCounts` and reported at the
end of training, so a reader can see exactly what changed when latency changed
rather than reconstructing it.

## One trainer, two topologies

:class:`PPOTrainer` drives a mapping of task name to :class:`TaskPolicy`. With
one entry it trains a dedicated network. With several it alternates whole
minibatches between one shared trunk's per-task views.

The rollout and the update are written once. The dedicated/shared comparison is
only interpretable if both arms run identical training code, and two copies of
that code are identical only until someone edits one of them -- so this used to
be an aspiration about copy-paste discipline rather than a property anything
checked. With one implementation it is structural.

See :mod:`latency_ml.rl.task_policy` for the interface and for why the shared
trunk's optimiser split is load-bearing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from latency_ml.config import ProjectConfig
from latency_ml.envs.latency_env import HOLD_STEPS_KEY
from latency_ml.rl.buffer import RolloutBuffer
from latency_ml.rl.task_policy import TaskPolicy
from latency_ml.seeding import Seeds, seed_env, seed_everything


@dataclass(slots=True)
class RunCounts:
    """The quantities the protocol requires be logged per run.

    Held fixed across latency conditions:
        ``integration_steps``, ``minibatch_size``, ``optimizer hyperparameters``.

    Vary with latency, as consequences of equal simulated-time exposure:
        ``decision_transitions``, ``ppo_updates``, ``gradient_steps``.
    """

    integration_steps: int = 0
    decision_transitions: int = 0
    ppo_updates: int = 0
    gradient_steps: int = 0
    minibatch_size: int = 0
    num_envs: int = 0
    steps_per_decision: int = 0
    action_steps: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Manifest-friendly representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One point on a training curve.

    Recorded against **environment integration steps**, not PPO iteration count.
    The distinction is the whole point: a higher-latency policy completes far
    fewer iterations within the same simulated time, so plotting against
    iteration index would align the curves at a point where they have seen
    different amounts of experience. Plotting against integration steps aligns
    them on the quantity that is actually held equal across conditions.

    Attributes:
        integration_steps: Simulated environment steps consumed so far, per task.
            A mapping rather than a scalar because a shared run tracks each task
            against its own budget.
        gradient_steps: Optimiser updates applied so far, per task.
        mean_return: Greedy evaluation return, per task.
        episodes: Episodes averaged per task, for the noise floor.
    """

    integration_steps: dict[str, int]
    gradient_steps: dict[str, int]
    mean_return: dict[str, float]
    episodes: int

    def to_dict(self) -> dict[str, object]:
        """Manifest-friendly representation.

        A single-task run emits scalars and a shared run emits one entry per
        task. The special case is deliberate: a one-task run has exactly one
        task's value, and keeping the scalar shape means a dedicated cell's curve
        file is directly comparable with one produced before the trainers were
        consolidated. Only the shared case has something to disambiguate.
        """
        if len(self.integration_steps) == 1:
            (only,) = self.integration_steps
            return {
                "integration_steps": self.integration_steps[only],
                "gradient_steps": self.gradient_steps[only],
                "mean_return": self.mean_return[only],
                "episodes": self.episodes,
            }
        return {
            "integration_steps": dict(self.integration_steps),
            "gradient_steps": dict(self.gradient_steps),
            "mean_return": dict(self.mean_return),
            "episodes": self.episodes,
        }


@dataclass
class TrainResult:
    """Outcome of one training run."""

    counts: dict[str, RunCounts] = field(default_factory=dict)
    curve: list[CurvePoint] = field(default_factory=list)
    wall_time_s: float = 0.0
    planned_iterations: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": {t: c.to_dict() for t, c in self.counts.items()},
            "planned_iterations": dict(self.planned_iterations),
            "wall_time_s": self.wall_time_s,
        }


@dataclass
class TaskState:
    """Per-task training state, including its own budget accounting.

    Attributes:
        task: Task name.
        envs: Environment replicas.
        obs_shape: Observation shape.
        counts: This task's own run counts. Tracked separately per task precisely
            so the update-accounting invariant can be asserted rather than assumed.
        optimisers: One optimiser per parameter group this task's update steps.
        optimiser_by_group: Group name to its optimiser, for LR annealing.
    """

    task: str
    envs: list[Any]
    obs_shape: tuple[int, ...]
    counts: RunCounts
    optimisers: list[torch.optim.Optimizer]
    optimiser_by_group: dict[str, torch.optim.Optimizer]


class PPOTrainer:
    """PPO over one dedicated policy, or over several tasks sharing a trunk.

    Args:
        config: The experiment configuration.
        policies: Task name to the policy that task trains through. One entry is
            a dedicated run; several is a shared-trunk run, and they are the same
            code path.
        latency_ns: The latency condition being trained.
        device: Torch device. CPU only in practice; see ``docs/ENVIRONMENT.md``
            for why MPS is excluded.
        seed_offset: Added to the derived training seed, so a pilot and a full
            run do not collide.
        seeds: Explicit seeds, overriding the derived ones. A sweep cell supplies
            these so its streams depend on the cell's identity (latency and
            replicate index) rather than on the order work happened to be scheduled
            in. Omit for standalone training. Keys must cover ``policies``.
    """

    def __init__(
        self,
        config: ProjectConfig,
        policies: dict[str, TaskPolicy],
        latency_ns: int,
        device: str = "cpu",
        seed_offset: int = 0,
        seeds: dict[str, Seeds] | None = None,
    ) -> None:
        if not policies:
            raise ValueError("need at least one task")
        self.config = config
        self.tasks: tuple[str, ...] = tuple(sorted(policies))
        self.policies = dict(policies)
        self.latency_ns = latency_ns
        self.device = torch.device(device)
        self.seed_offset = seed_offset

        if seeds is not None:
            self.seeds = dict(seeds)
            missing = set(self.tasks) - set(self.seeds)
            if missing:
                raise ValueError(f"no seeds supplied for {sorted(missing)}")
        else:
            base = Seeds(
                training=config.seed + seed_offset,
                env=config.seed + 10_000 + seed_offset,
                eval=config.evaluation.eval_seed_base,
            )
            self.seeds = {t: base.for_latency(latency_ns) for t in self.tasks}

        seed_everything(self.seeds[self.tasks[0]])

        self.initial_lr = config.training.learning_rate
        self.states: dict[str, TaskState] = {}
        for task in self.tasks:
            policy = self.policies[task]
            policy.to_device(self.device)
            self.states[task] = self._make_state(task, policy)

    def _make_state(self, task: str, policy: TaskPolicy) -> TaskState:
        from latency_ml.envs.factory import make_latency_env

        cfg = self.config.training
        envs = [make_latency_env(task, self.latency_ns) for _ in range(cfg.num_envs)]
        shape = envs[0].observation_space.shape
        if shape is None:
            raise ValueError(f"{task} environment reported no observation shape")
        counts = RunCounts(num_envs=cfg.num_envs, steps_per_decision=cfg.steps_per_decision)
        counts.action_steps = envs[0].action_steps
        counts.minibatch_size = cfg.minibatch_size(counts.action_steps)

        optimisers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(group.parameters, lr=self.initial_lr, eps=1e-5)
            for group in policy.parameter_groups()
        ]
        return TaskState(
            task=task,
            envs=envs,
            obs_shape=(int(shape[0]),),
            counts=counts,
            optimisers=optimisers,
            optimiser_by_group={
                group.name: opt
                for group, opt in zip(policy.parameter_groups(), optimisers, strict=True)
            },
        )

    def planned_iterations(self, budget: int | None = None) -> dict[str, int]:
        """Iterations each task will run, from its own budget.

        Latency-dependent -- more integration steps are spent per decision as the
        action hold lengthens, so fewer iterations fit in the same budget. That is
        the phenomenon under study, not an artefact to be smoothed away.
        """
        cfg = self.config.training
        total = budget or cfg.total_integration_steps
        out: dict[str, int] = {}
        for task in self.tasks:
            action_steps = self.states[task].counts.action_steps
            per_iter = cfg.num_envs * cfg.steps_per_decision * action_steps
            out[task] = max(1, total // per_iter)
        return out

    def learn(
        self,
        total_integration_steps: int | None = None,
        log_interval: int = 50,
        curve_every: int = 0,
        curve_episodes: int = 20,
    ) -> TrainResult:
        """Train for the configured budget.

        Args:
            total_integration_steps: Override the configured budget. Used by
                pilots, which need a short run before committing to a sweep.
            log_interval: Iterations between progress lines; 0 disables.
            curve_every: Record a curve point every this many integration steps.
                0 disables curve recording.
            curve_episodes: Episodes per curve point. Small by design: the curve
                is for shape, not for a precise endpoint, and the final numbers
                come from the full evaluation.

        Returns:
            Per-task counts and the planned iteration counts, so the caller can
            verify the update-accounting invariant from the result alone.
        """
        cfg = self.config.training
        budget = total_integration_steps or cfg.total_integration_steps
        planned = self.planned_iterations(budget)
        iterations = max(planned.values())

        seed_everything(self.seeds[self.tasks[0]])

        current = {t: self._initial_obs(t) for t in self.tasks}
        curve: list[CurvePoint] = []
        next_curve_at = curve_every

        start = time.perf_counter()
        for iteration in range(1, iterations + 1):
            for task in self.tasks:
                if iteration > planned[task]:
                    continue
                buffer, current[task] = self._collect(task, current[task])

                if cfg.anneal_lr:
                    self._anneal(task, budget)

                transitions = self._update(task, buffer)

                state = self.states[task]
                state.counts.integration_steps += int(buffer.total_duration)
                state.counts.decision_transitions += transitions
                state.counts.ppo_updates += 1

            if curve_every and self._curve_progress() >= next_curve_at:
                curve.append(self._curve_point(curve_episodes))
                next_curve_at = (self._curve_progress() // curve_every + 1) * curve_every

            if log_interval and iteration % log_interval == 0:
                progress = " ".join(
                    f"{t}={self.states[t].counts.ppo_updates}/{planned[t]}" for t in self.tasks
                )
                print(f"iter {iteration:5d}/{iterations} | updates {progress}", flush=True)

        for task in self.tasks:
            for env in self.states[task].envs:
                env.close()
        wall_time = time.perf_counter() - start

        for task in self.tasks:
            self.states[task].counts.wall_time_s = wall_time

        result = TrainResult(
            counts={t: self.states[t].counts for t in self.tasks},
            curve=curve,
            wall_time_s=wall_time,
            planned_iterations=planned,
        )
        self._assert_invariant(result, budget)
        return result

    def _initial_obs(self, task: str) -> torch.Tensor:
        """Seeded reset observations, one distinct draw per environment.

        One reset per environment, with a distinct seed, seeding the action space
        too. Resetting twice would discard the first draw and make the initial
        states depend on reset ordering.
        """
        envs = self.states[task].envs
        return torch.as_tensor(
            np.stack([seed_env(env, self.seeds[task].env + i)[0] for i, env in enumerate(envs)]),
            dtype=torch.float32,
            device=self.device,
        )

    def _anneal(self, task: str, budget: int) -> None:
        """Anneal on simulated-time progress, against the *effective* budget.

        Annealed on progress through the budget in integration steps rather than
        on update count, so the schedule is identical across latency conditions
        at any point in simulated time.

        The earlier form was inverted: ``frac = 1 - (iter-1)/N`` feeding
        ``lr = initial*(1-frac)`` started every condition at exactly zero and
        ramped *up*, so each spent its whole budget traversing a different part
        of the ramp.

        Denominated in the effective budget, not the configured one. Using
        ``cfg.total_integration_steps`` made any run with an overridden budget
        reach ``frac = 1.0`` halfway and then train at exactly ``lr = 0`` for the
        remainder, silently: a 4M pilot looked like a 4M run while being a 2M run
        with a frozen tail.
        """
        frac = self.states[task].counts.integration_steps / max(1, budget)
        for optimiser in self.states[task].optimisers:
            anneal_learning_rate(optimiser, frac, self.initial_lr)

    def _collect(self, task: str, current_obs: torch.Tensor) -> tuple[RolloutBuffer, torch.Tensor]:
        """Roll out one iteration for one task.

        Returns:
            The filled buffer and the next observation tensor.
        """
        cfg = self.config.training
        policy = self.policies[task]
        state = self.states[task]
        envs = state.envs
        num_envs = cfg.num_envs

        buffer = RolloutBuffer(cfg.steps_per_decision, num_envs, state.obs_shape)
        for _ in range(cfg.steps_per_decision):
            with torch.no_grad():
                actions, log_probs, value = policy.action_and_value(current_obs)

            next_obs_list: list[np.ndarray] = []
            final_obs_list: list[np.ndarray] = []
            rewards = torch.zeros(num_envs)
            terminateds = torch.zeros(num_envs)
            truncateds = torch.zeros(num_envs)
            durations = torch.zeros(num_envs)

            for i, env in enumerate(envs):
                obs_i, reward, terminated, truncated, info = env.step(int(actions[i].item()))
                rewards[i] = reward

                terminateds[i] = float(terminated)
                truncateds[i] = float(truncated)
                durations[i] = float(info.get(HOLD_STEPS_KEY, state.counts.action_steps))

                final_obs_list.append(np.asarray(obs_i, dtype=np.float32).copy())
                next_obs_list.append(obs_i)

            with torch.no_grad():
                _, final_values = policy(
                    torch.as_tensor(
                        np.stack(final_obs_list), dtype=torch.float32, device=self.device
                    )
                )

            for i, env in enumerate(envs):
                if terminateds[i] > 0.0 or truncateds[i] > 0.0:
                    env.release_terminal()
                    obs_i, _ = env.reset()
                    next_obs_list[i] = obs_i

            buffer.add(
                obs=current_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=terminateds,
                values=value,
                durations=durations,
                truncations=truncateds,
                trunc_values=final_values,
            )
            current_obs = torch.as_tensor(
                np.stack(next_obs_list), dtype=torch.float32, device=self.device
            )

        with torch.no_grad():
            _, last_value = policy(current_obs)

        discount = cfg.discount_for(task)
        buffer.compute_returns(
            last_value=value_tensor(last_value),
            gamma_phys=discount.gamma,
            gae_lambda_phys=discount.gae_lambda,
            duration_aware=discount.duration_aware,
        )
        return buffer, current_obs

    def _update(self, task: str, buffer: RolloutBuffer) -> int:
        """One PPO update for one task.

        Returns:
            Number of decision transitions in the batch.
        """
        cfg = self.config.training
        policy = self.policies[task]
        state = self.states[task]
        batch = buffer.get(advantages_normalised=cfg.norm_adv)
        n = len(batch)
        minibatch_size = max(1, n // cfg.num_minibatches)

        clipped = [p for group in policy.parameter_groups() for p in group.parameters]

        for _ in range(cfg.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb = indices[start : start + minibatch_size]
                new_log_probs, entropy, new_values, _ = policy.evaluate_actions(
                    batch.obs[mb], batch.actions[mb]
                )

                log_ratio = new_log_probs - batch.log_probs[mb]

                logratio2 = log_ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                loss1 = -log_ratio * batch.advantages[mb]
                loss2 = -logratio2 * batch.advantages[mb]
                pg_loss = torch.max(loss1, loss2).mean()

                new_values_clipped = batch.values[mb] + (new_values - batch.values[mb]).clamp(
                    -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_unclipped = (new_values - batch.returns[mb]) ** 2
                v_loss_clipped = (new_values_clipped - batch.returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                for optimiser in state.optimisers:
                    optimiser.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(clipped, cfg.max_grad_norm)
                for optimiser in state.optimisers:
                    optimiser.step()
                state.counts.gradient_steps += 1

        return n

    def _curve_progress(self) -> int:
        """Simulated steps used to place curve points.

        The minimum across tasks, not the mean or the maximum. At high latency one
        task exhausts its planned iterations before another does, and sampling on
        the maximum would place late points past the slower task's budget.
        """
        return min(self.states[t].counts.integration_steps for t in self.tasks)

    def _curve_point(self, episodes: int) -> CurvePoint:
        """Evaluate every task for one curve point.

        All tasks are evaluated against the same weights at the same moment, which
        is the only way a shared-trunk curve can show interference: a point where
        one task's return has risen while the other's has not.

        Evaluation does not touch the environments' training state, and the
        rollout observations are re-seeded from the evaluation seed base, so
        recording a point does not perturb what the training run then sees.
        """
        from dataclasses import replace

        from latency_ml.experiments.evaluate import evaluate_policy

        was_training = self.policies[self.tasks[0]].is_training()
        for policy in self.policies.values():
            policy.set_mode(False)
        returns: dict[str, float] = {}
        try:
            for task in self.tasks:
                task_config = replace(self.config, task=task)
                summary = evaluate_policy(
                    self.policies[task], task_config, self.latency_ns, 0, episodes
                )
                returns[task] = summary.mean_return
        finally:
            if was_training:
                for policy in self.policies.values():
                    policy.set_mode(True)
        return CurvePoint(
            integration_steps={t: int(self.states[t].counts.integration_steps) for t in self.tasks},
            gradient_steps={t: int(self.states[t].counts.gradient_steps) for t in self.tasks},
            mean_return=returns,
            episodes=episodes,
        )

    def _assert_invariant(self, result: TrainResult, budget: int) -> None:
        """Fail loudly if any task's budget or update count drifted.

        The requirement, turned into a runtime check rather than a code comment:
        for each task, a shared run must expose that task to the same
        integration-step budget and the same number of PPO updates as its
        dedicated baseline. Joint training adds cross-task gradient exposure to
        the trunk; it does not halve any task's optimisation budget.

        A future change to the alternation that silently halved a task's budget
        would otherwise produce a shared arm that looks like a sharing effect.
        """
        cfg = self.config.training
        for task in self.tasks:
            counts = result.counts[task]
            if counts.ppo_updates != result.planned_iterations[task]:
                raise AssertionError(
                    f"{task}: ran {counts.ppo_updates} PPO updates but "
                    f"{result.planned_iterations[task]} were planned; the "
                    "per-task optimisation budget must match the dedicated "
                    "baseline exactly"
                )
            expected = cfg.total_integration_steps if budget is None else budget
            allowance = cfg.num_envs * cfg.steps_per_decision * counts.action_steps
            if counts.integration_steps > expected + allowance:
                raise AssertionError(
                    f"{task}: consumed {counts.integration_steps} integration "
                    f"steps, exceeding its {expected}-step budget"
                )


def value_tensor(value: torch.Tensor | None) -> torch.Tensor:
    """Coerce an optional value tensor to a concrete one."""
    assert value is not None, "rollout requires a value head"
    return value


def anneal_learning_rate(optimizer: torch.optim.Optimizer, frac: float, initial_lr: float) -> float:
    """Set the learning rate to ``initial_lr * (1 - frac)``.

    ``frac`` is training progress in [0, 1]: 0 leaves the rate at its initial
    value, 1 drives it to zero. Clamped so a rounding overshoot cannot produce a
    negative rate.

    Not hyperparameter tuning -- this is a convention fix. The caller passes
    simulated-time progress, so the rate decays over the same span of simulated
    time in every latency condition regardless of how many updates that condition
    gets.
    """
    new_lr = initial_lr * (1.0 - min(max(frac, 0.0), 1.0))
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    return new_lr


__all__ = [
    "CurvePoint",
    "PPOTrainer",
    "RunCounts",
    "TaskState",
    "TrainResult",
    "anneal_learning_rate",
    "value_tensor",
]
