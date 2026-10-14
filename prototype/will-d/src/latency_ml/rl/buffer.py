"""Rollout storage.

Each transition in this buffer corresponds to one **action-hold interval**, not
one integration step. The wrapper at :mod:`latency_ml.envs.latency_env`
collapses a held action and its ``N`` integration steps into a single
semi-Markov transition, so ``num_steps`` here counts decisions, and the
simulated time a rollout covers is ``sum(durations) * dt_env_ns``.

``durations`` is stored per transition. At the standard duration of 1 it is
uninformative, which is why commit 8 can ship a correct standard-GAE PPO before
commit 9 makes the advantage estimate duration-aware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Batch:
    """A flattened set of transitions ready for a PPO update.

    All leading dimensions are collapsed, so ``len(batch)`` is the total
    transition count.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    durations: torch.Tensor

    def __len__(self) -> int:
        return int(self.obs.shape[0])


class RolloutBuffer:
    """Fixed-size storage for one PPO rollout.

    Args:
        num_steps: Transitions per environment per rollout.
        num_envs: Parallel environments.
        obs_shape: Observation shape, excluding the batch dimension.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shape: tuple[int, ...]) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape

        shape = (num_steps, num_envs)
        self.obs = torch.zeros((*shape, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(shape, dtype=torch.int64)
        self.log_probs = torch.zeros(shape, dtype=torch.float32)
        self.rewards = torch.zeros(shape, dtype=torch.float32)
        self.dones = torch.zeros(shape, dtype=torch.float32)
        #: True *termination* only. Truncation is recorded separately, because a
        #: time limit must still bootstrap while a terminal state must not.
        self.dones = torch.zeros(shape, dtype=torch.float32)
        #: 1.0 where the episode hit a time limit rather than a terminal state.
        self.truncations = torch.zeros(shape, dtype=torch.float32)
        #: V(s_final) for truncated transitions, captured before the reset. This
        #: is the value the time-limit bootstrap needs; without it the rollout
        #: would bootstrap from the *new* episode's first observation.
        self.trunc_values = torch.zeros(shape, dtype=torch.float32)
        self.values = torch.zeros(shape, dtype=torch.float32)
        #: Integration steps covered by each transition. Always >= 1.
        self.durations = torch.ones(shape, dtype=torch.float32)

        self.advantages = torch.zeros(shape, dtype=torch.float32)
        self.returns = torch.zeros(shape, dtype=torch.float32)

        self.ptr = 0

    def reset(self) -> None:
        """Empty the buffer."""
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.num_steps

    @property
    def size(self) -> int:
        """Transitions stored so far."""
        return self.ptr * self.num_envs

    @property
    def total_duration(self) -> int:
        """Integration steps covered by the stored transitions."""
        return int(self.durations[: self.ptr].sum().item())

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        durations: torch.Tensor,
        truncations: torch.Tensor | None = None,
        trunc_values: torch.Tensor | None = None,
    ) -> None:
        """Append one transition per environment.

        Args:
            obs: Observations at the decision point.
            actions: Actions taken.
            log_probs: Log-probabilities under the sampling distribution.
            rewards: Reward earned over the hold.
            dones: 1.0 where the episode *terminated*. Truncation must not be
                included here, or the time-limit bootstrap is lost.
            values: V(obs) at the decision point.
            durations: Integration steps covered by each transition.
            truncations: 1.0 where the episode hit a time limit.
            trunc_values: V(s_final) at truncated transitions, pre-reset.

        Raises:
            AssertionError: If the buffer is already full, which would silently
                overwrite earlier transitions.
        """
        assert not self.full, "rollout buffer is full"
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.values[i] = values
        self.durations[i] = durations
        if truncations is not None:
            self.truncations[i] = truncations
        if trunc_values is not None:
            self.trunc_values[i] = trunc_values
        self.ptr += 1

    def _next_values(self, step: int, last_value: torch.Tensor) -> torch.Tensor:
        """Value of the state the transition at `step` leads into.

        At a truncated transition this is V(s_final) captured before the reset.
        Using ``values[step + 1]`` there would instead read the *new* episode's
        first observation, which is what the first version of this trainer did
        by discarding truncation entirely.
        """
        truncated = self.truncations[step] > 0
        if bool(truncated.any()):
            fallback = last_value if step == self.ptr - 1 else self.values[step + 1]
            return torch.where(truncated, self.trunc_values[step], fallback)
        return last_value if step == self.ptr - 1 else self.values[step + 1]

    def _same_episode(self, step: int) -> torch.Tensor:
        """1.0 where the transition at `step + 1` continues the same episode."""
        return 1.0 - torch.clamp(self.dones[step] + self.truncations[step], max=1.0)

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma_phys: float,
        gae_lambda_phys: float,
        duration_aware: bool = True,
    ) -> None:
        """Fill ``advantages`` and ``returns`` by GAE.

        With ``duration_aware`` (the default) the recursion is **semi-Markov**
        (`PLAN.md` section 6.6), and the factors are read as *per integration
        step*. Each transition spans ``N_i`` integration steps, so:

            delta_t = r_t + gamma_phys^(N_t) * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t     = delta_t + (gamma_phys^(N_t) * lambda_phys^(N_t))
                                 * (1 - done) * A_{t+1}

        With ``duration_aware=False`` the factors are read as *per decision* and
        applied unscaled:

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t     = delta_t + (gamma * lambda) * (1 - done) * A_{t+1}

        The two branches are the two :class:`~latency_ml.config.DiscountBasis`
        conventions, not a correct/incorrect pair. They differ *only* in the
        clock the factors run on, and the caller's configured basis picks one.
        Decision-level discounting is what keeps the credit horizon in decisions
        constant as action-hold length rises; physical-level keeps it constant
        in simulated time but lets the per-decision horizon shrink with latency.

        ``N_i`` is the **actual** hold length, not the requested one, because an
        episode boundary commonly cuts the final hold short. Discounting a
        truncated hold as though it were full would under-credit it. That only
        applies on the duration-aware branch; a decision-level factor is
        indifferent to the hold length either way.
        """
        if not duration_aware:
            self._compute_returns_standard(last_value, gamma_phys, gae_lambda_phys)
            return

        # Per-transition discounts, so a short final hold gets its own value.
        durations = self.durations[: self.ptr]
        gamma_base = torch.as_tensor(gamma_phys, dtype=torch.float32)
        lambda_base = torch.as_tensor(gae_lambda_phys, dtype=torch.float32)
        gammas = torch.pow(gamma_base, durations)
        lambdas = torch.pow(lambda_base, durations)

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for step in reversed(range(self.ptr)):
            next_values = self._next_values(step, last_value)
            not_terminal = 1.0 - self.dones[step]
            gamma_t = gammas[step]
            lambda_t = lambdas[step]

            delta = self.rewards[step] + gamma_t * next_values * not_terminal - self.values[step]
            # The recursion stops at any episode boundary, terminated or
            # truncated: the next stored transition belongs to a new episode.
            # The bootstrap above is gated only on termination, so a time limit
            # still contributes V(s_final).
            last_gae = delta + gamma_t * lambda_t * self._same_episode(step) * last_gae
            self.advantages[step] = last_gae

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def _compute_returns_standard(
        self, last_value: torch.Tensor, gamma_phys: float, gae_lambda_phys: float
    ) -> None:
        """GAE with factors read per decision, ignoring transition durations.

        This is the :attr:`~latency_ml.config.DiscountBasis.DECISION` recursion.
        The parameter names retain ``_phys`` for signature stability, but the
        factors are used unscaled -- see :meth:`compute_returns`.
        """
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for step in reversed(range(self.ptr)):
            next_values = self._next_values(step, last_value)
            not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma_phys * next_values * not_terminal - self.values[step]
            last_gae = delta + gamma_phys * gae_lambda_phys * self._same_episode(step) * last_gae
            self.advantages[step] = last_gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get(self, advantages_normalised: bool = True) -> Batch:
        """Flatten the buffer into a batch.

        Args:
            advantages_normalised: Whether to standardise advantages. PPO's
                clipping assumes roughly unit-scale advantages; done per
                rollout, across all environments.
        """
        flat = slice(0, self.ptr)
        advantages = self.advantages[flat].reshape(-1)
        if advantages_normalised and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return Batch(
            obs=self.obs[flat].reshape(-1, *self.obs_shape),
            actions=self.actions[flat].reshape(-1),
            log_probs=self.log_probs[flat].reshape(-1),
            advantages=advantages,
            returns=self.returns[flat].reshape(-1),
            values=self.values[flat].reshape(-1),
            durations=self.durations[flat].reshape(-1),
        )
