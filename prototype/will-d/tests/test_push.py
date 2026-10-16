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
            assert 1 <= trial.tts_frames <= 200
            assert not trial.fell
        else:
            assert trial.fell or trial.decisions_to_stable is not None


def test_stable_band_sits_outside_nominal_hover():
    """A well-trained policy must hover inside the band but be pushed out by the kick."""
    from latency_ml.experiments.push import STABLE_THETA_DOT, STABLE_THETA_RAD

    assert STABLE_THETA_RAD > 0.105, "band must contain the nominal hover envelope"
    assert STABLE_THETA_DOT > 0.94
    assert STABLE_THETA_RAD < 0.2095, "band must sit inside the termination envelope"


def test_trial_kicks_leave_the_stable_band():
    """Kick bounds must exceed the band so recovery time is measurable."""
    from latency_ml.experiments.push import KICK_THETA_DOT, KICK_THETA_RAD, STABLE_THETA_DOT, STABLE_THETA_RAD

    assert KICK_THETA_RAD > STABLE_THETA_RAD
    assert KICK_THETA_DOT > STABLE_THETA_DOT


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
