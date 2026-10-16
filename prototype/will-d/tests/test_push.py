from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from latency_ml.config import load_config
from latency_ml.experiments.push import (
    PushResult,
    PushTrial,
    aggregate_push,
    run_push_trials,
)
from latency_ml.rl.task_policy import ParameterGroup

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "cartpole.yaml")


class _AlwaysLeftPolicy:
    """Deterministic stub: always push left, no value head needed for push."""

    obs_dim = 4
    n_actions = 2
    activation_name = "stub"

    def parameter_groups(self) -> list[ParameterGroup]:
        return []

    def action_only(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0, dtype=torch.int64)

    def save(self, path) -> None:  # pragma: no cover - unused by push
        raise AssertionError("not used")


def _result(latency_ns: int, train_seed: int, trials: list[PushTrial]) -> PushResult:
    return PushResult(
        task="cartpole",
        latency_ns=latency_ns,
        train_seed=train_seed,
        action_steps=1,
        dt_env_ns=20000000,
        trials=trials,
    )


def test_summary_counts_successes_and_censors_failures():
    trials = [
        PushTrial(seed=1, tts_frames=10, decisions_to_stable=10, fell=False),
        PushTrial(seed=2, tts_frames=40, decisions_to_stable=40, fell=False),
        PushTrial(seed=3, tts_frames=None, decisions_to_stable=None, fell=True),
        PushTrial(seed=4, tts_frames=None, decisions_to_stable=None, fell=False),
    ]
    summary = _result(20000000, 0, trials).summary()
    assert summary["n_trials"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["mean_tts_frames"] == 25.0
    assert summary["median_tts_frames"] == 25.0
    assert summary["mean_tts_decisions"] == 25.0
    assert summary["mean_tts_ms"] == pytest.approx(500.0)


def test_aggregate_groups_by_latency_and_averages_seeds():
    low_a = _result(20000000, 0, [PushTrial(1, 10, 10, False), PushTrial(2, 30, 30, False)])
    low_b = _result(20000000, 1, [PushTrial(3, 20, 20, False), PushTrial(4, None, None, True)])
    high = _result(100000000, 0, [PushTrial(5, None, None, True)])
    rows = aggregate_push([high, low_a, low_b])
    assert [r["latency_ns"] for r in rows] == [20000000, 100000000]
    assert rows[0]["n_seeds"] == 2
    assert rows[0]["success_rate"] == 0.75
    assert rows[0]["mean_tts_frames"] == 20.0
    assert rows[1]["mean_tts_frames"] is None
    assert rows[1]["n_trials"] == 1


def test_push_trials_records_valid_outcomes():
    result = run_push_trials(_AlwaysLeftPolicy(), CONFIG, 20000000, 0, trials=3, horizon=60)
    assert result.task == "cartpole"
    assert result.action_steps == 1
    assert len(result.trials) == 3
    for trial in result.trials:
        if trial.tts_frames is not None:
            assert 0 <= trial.tts_frames <= 200
            assert not trial.fell
        else:
            assert trial.fell or trial.decisions_to_stable is not None


def test_stable_band_sits_outside_nominal_hover():
    """A well-trained policy must hover inside the band but be pushed out by the kick."""
    from latency_ml.experiments.push import STABLE_THETA_DOT, STABLE_THETA_RAD

    assert STABLE_THETA_RAD > 0.105, "band must contain the nominal hover envelope"
    assert STABLE_THETA_DOT > 0.94
    assert STABLE_THETA_RAD < 0.2095, "band must sit inside the termination envelope"


def test_trial_kicks_are_external_force_pulses():
    """Kick magnitude/duration are fixed; only the direction is randomised per trial."""
    from latency_ml.experiments.push import KICK_FORCE_N, KICK_STEPS, kick_force_for_trial

    assert KICK_FORCE_N == 15.0
    assert KICK_STEPS == 6
    forces = {kick_force_for_trial(CONFIG, 20000000, 0, trial)[1] for trial in range(20)}
    assert forces <= {KICK_FORCE_N, -KICK_FORCE_N}
    assert forces == {KICK_FORCE_N, -KICK_FORCE_N}, "both shove directions must occur"


class _HoverPolicy:
    """Tiny PD controller that hovers from a centered reset (test stub, not trained)."""

    def action_only(self, obs):
        import torch

        theta, theta_dot = float(obs[2]), float(obs[3])
        return torch.tensor(1 if theta + 0.3 * theta_dot > 0 else 0)


def test_recovery_is_measured_post_kick():
    """A stable preamble must not count as recovery: only post-kick holds qualify."""
    from latency_ml.envs.factory import make_base_env
    from latency_ml.experiments.push import run_push_rollout
    from latency_ml.seeding import seed_env

    env = make_base_env("cartpole")
    try:
        obs, _ = seed_env(env, 7)
        roll = run_push_rollout(_HoverPolicy(), env, np.asarray(obs, dtype=np.float32), 1, 61, lambda frame: 0.0, disturb_from=60)
        assert roll.survived == 61 and not roll.knocked_out
        assert not roll.recovered
        assert roll.tts_frames is None and roll.decisions_to_stable is None
    finally:
        env.close()


def test_fall_rollout_runs_until_pole_is_down():
    """Video rule: a fall plays past the 12-degree termination until the pole is horizontal."""
    from latency_ml.envs.factory import make_base_env
    from latency_ml.experiments.push import POLE_DOWN_THETA_RAD, run_push_rollout

    def fallen_rollout(extend_fall):
        env = make_base_env("cartpole")
        try:
            env.reset(seed=7)
            env.unwrapped.state = np.array([0.0, 0.0, 0.25, 2.0])
            return run_push_rollout(
                _AlwaysLeftPolicy(),
                env,
                np.asarray(env.unwrapped.state, dtype=np.float32),
                1,
                500,
                lambda frame: 0.0,
                extend_fall=extend_fall,
            )
        finally:
            env.close()

    short = fallen_rollout(False)
    assert short.survived == 1 and short.fell
    assert abs(short.final_theta) < POLE_DOWN_THETA_RAD
    long = fallen_rollout(True)
    assert long.fell and long.survived > short.survived
    assert abs(long.final_theta) >= POLE_DOWN_THETA_RAD
    assert not any(long.flags)


def test_fall_extension_skips_upright_cart_exit():
    """A cart-track exit with the pole upright gets no pole-down extension."""
    from latency_ml.envs.factory import make_base_env
    from latency_ml.experiments.push import run_push_rollout

    env = make_base_env("cartpole")
    try:
        env.reset(seed=7)
        env.unwrapped.state = np.array([2.39, 5.0, 0.0, 0.0])
        roll = run_push_rollout(_AlwaysLeftPolicy(), env, np.asarray(env.unwrapped.state, dtype=np.float32), 1, 500, lambda frame: 0.0, extend_fall=True)
        assert roll.fell and roll.survived == 1
    finally:
        env.close()


def test_external_force_preserves_pose():
    """One forced step must integrate positions from pre-step velocities: no teleport."""
    from latency_ml.envs.factory import make_base_env
    from latency_ml.experiments.push import step_with_external_force
    from latency_ml.seeding import seed_env

    env = make_base_env("cartpole")
    try:
        _obs, _ = seed_env(env, 7)
        before = np.asarray(env.unwrapped.state, dtype=np.float64)
        tau = float(env.unwrapped.tau)
        mag = float(env.unwrapped.force_mag)
        step_with_external_force(env, 1, 15.0)
        after = np.asarray(env.unwrapped.state, dtype=np.float64)
        # Euler integration moves positions from pre-step velocities only.
        assert after[0] == pytest.approx(before[0] + tau * before[1])
        assert after[2] == pytest.approx(before[2] + tau * before[3])
        # Velocities must feel the extra 15 N on top of the policy's +10 N.
        assert after[1] != pytest.approx(before[1])
        assert after[3] != pytest.approx(before[3])
        assert float(env.unwrapped.force_mag) == pytest.approx(mag), "force_mag must be restored"
    finally:
        env.close()


def test_trials_are_deterministic_for_a_fixed_seed():
    a = run_push_trials(_AlwaysLeftPolicy(), CONFIG, 40000000, 3, trials=2, horizon=40)
    b = run_push_trials(_AlwaysLeftPolicy(), CONFIG, 40000000, 3, trials=2, horizon=40)
    assert [t.to_dict() for t in a.trials] == [t.to_dict() for t in b.trials]
    assert a.action_steps == 2
    assert all(np.isfinite(t.seed) for t in a.trials)


def test_json_round_trip(tmp_path):
    result = _result(20000000, 0, [PushTrial(1, 10, 10, False)])
    payload = json.dumps(result.to_dict())
    assert json.loads(payload)["mean_tts_frames"] == 10.0
