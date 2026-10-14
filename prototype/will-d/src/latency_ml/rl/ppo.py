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
        return asdict(self)


@dataclass
class TrainResult:
    counts: dict[str, RunCounts] = field(default_factory=dict)
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
    task: str
    envs: list[Any]
    obs_shape: tuple[int, ...]
    counts: RunCounts
    optimisers: list[torch.optim.Optimizer]
    optimiser_by_group: dict[str, torch.optim.Optimizer]


class PPOTrainer:
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
                env=config.seed + 10000 + seed_offset,
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
            torch.optim.Adam(group.parameters, lr=self.initial_lr, eps=1e-05)
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
        cfg = self.config.training
        total = budget or cfg.total_integration_steps
        out: dict[str, int] = {}
        for task in self.tasks:
            action_steps = self.states[task].counts.action_steps
            per_iter = cfg.num_envs * cfg.steps_per_decision * action_steps
            out[task] = max(1, total // per_iter)
        return out

    def learn(
        self, total_integration_steps: int | None = None, log_interval: int = 50
    ) -> TrainResult:
        cfg = self.config.training
        budget = total_integration_steps or cfg.total_integration_steps
        planned = self.planned_iterations(budget)
        iterations = max(planned.values())
        seed_everything(self.seeds[self.tasks[0]])
        current = {t: self._initial_obs(t) for t in self.tasks}
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
            wall_time_s=wall_time,
            planned_iterations=planned,
        )
        self._assert_invariant(result, budget)
        return result

    def _initial_obs(self, task: str) -> torch.Tensor:
        envs = self.states[task].envs
        return torch.as_tensor(
            np.stack([seed_env(env, self.seeds[task].env + i)[0] for i, env in enumerate(envs)]),
            dtype=torch.float32,
            device=self.device,
        )

    def _anneal(self, task: str, budget: int) -> None:
        frac = self.states[task].counts.integration_steps / max(1, budget)
        for optimiser in self.states[task].optimisers:
            anneal_learning_rate(optimiser, frac, self.initial_lr)

    def _collect(self, task: str, current_obs: torch.Tensor) -> tuple[RolloutBuffer, torch.Tensor]:
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
        if task == "lunarlander":
            gamma, lam, duration_aware = (0.999, 0.98, False)
        else:
            gamma, lam, duration_aware = (cfg.gamma_phys, cfg.gae_lambda_phys, True)
        buffer.compute_returns(
            last_value=value_tensor(last_value),
            gamma_phys=gamma,
            gae_lambda_phys=lam,
            duration_aware=duration_aware,
        )
        return (buffer, current_obs)

    def _update(self, task: str, buffer: RolloutBuffer) -> int:
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
                loss.backward()
                nn.utils.clip_grad_norm_(clipped, cfg.max_grad_norm)
                for optimiser in state.optimisers:
                    optimiser.step()
                state.counts.gradient_steps += 1
        return n

    def _assert_invariant(self, result: TrainResult, budget: int) -> None:
        cfg = self.config.training
        for task in self.tasks:
            counts = result.counts[task]
            if counts.ppo_updates != result.planned_iterations[task]:
                raise AssertionError(
                    f"{task}: ran {counts.ppo_updates} PPO updates but {result.planned_iterations[task]} were planned; the per-task optimisation budget must match the dedicated baseline exactly"
                )
            expected = cfg.total_integration_steps if budget is None else budget
            allowance = cfg.num_envs * cfg.steps_per_decision * counts.action_steps
            if counts.integration_steps > expected + allowance:
                raise AssertionError(
                    f"{task}: consumed {counts.integration_steps} integration steps, exceeding its {expected}-step budget"
                )


def value_tensor(value: torch.Tensor | None) -> torch.Tensor:
    assert value is not None, "rollout requires a value head"
    return value


def anneal_learning_rate(optimizer: torch.optim.Optimizer, frac: float, initial_lr: float) -> float:
    new_lr = initial_lr * (1.0 - min(max(frac, 0.0), 1.0))
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    return new_lr


__all__ = [
    "PPOTrainer",
    "RunCounts",
    "TaskState",
    "TrainResult",
    "anneal_learning_rate",
    "value_tensor",
]
