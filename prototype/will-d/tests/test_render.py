import numpy as np
import pytest
from latency_ml.experiments.render import rollout_frames, rollout_push_frames, select_median_seed, write_video


class _FakeEnv:
    def __init__(self, episode_len: int = 7, hold: int = 1):
        self.episode_len = episode_len
        self.steps = 0
        self.hold = hold
        self.state = None

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self.steps = 0
        return (np.zeros(4, dtype=np.float32), {})

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.episode_len
        obs = np.asarray(self.state if self.state is not None else np.zeros(4), dtype=np.float32)
        return (obs, 1.0, done, False, {})

    def render(self):
        return np.full((16, 16, 3), self.steps % 256, dtype=np.uint8)

    def close(self):
        pass


class _Policy:
    def __init__(self):
        self.calls = 0

    def action_only(self, obs):
        self.calls += 1
        import torch

        return torch.tensor(0)


def test_select_median_seed_picks_middle():
    evaluation = {"latency_response": [{"latency_ns": 20000000, "per_seed_returns": [10.0, 30.0, 20.0]}]}
    assert select_median_seed(evaluation, 20000000) == 2


def test_select_median_seed_missing_latency():
    with pytest.raises(KeyError):
        select_median_seed({"latency_response": []}, 20000000)


def test_rollout_frames_respects_hold_and_cap():
    policy = _Policy()
    env = _FakeEnv(episode_len=100)
    frames = rollout_frames(policy, "cartpole", 60000000, max_frames=10, dt_env_ns=20000000, env=env)
    assert len(frames) == 10
    # 60ms / 20ms dt = 3 frames per decision (plus the initial reset frame).
    assert policy.calls == 3


def test_rollout_frames_stops_at_termination():
    policy = _Policy()
    env = _FakeEnv(episode_len=4)
    frames = rollout_frames(policy, "cartpole", 20000000, max_frames=360, dt_env_ns=20000000, env=env)
    assert len(frames) == 1 + 4


def test_write_video_encodes_mp4(tmp_path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
    out = write_video(frames, tmp_path / "clip.mp4", fps=10, label="relu - cartpole - 20ms")
    assert out.exists() and out.stat().st_size > 0


def test_write_video_flash_banner(tmp_path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
    out = write_video(frames, tmp_path / "push.mp4", fps=10, label="relu - cartpole - 20ms", flash="PUSH")
    assert out.exists() and out.stat().st_size > 0


def test_rollout_push_frames_applies_bounded_kick():
    from types import SimpleNamespace

    from latency_ml.experiments.push import KICK_THETA_RAD, KICK_X, _stable

    config = SimpleNamespace(task="cartpole", seed=12345)
    env = _FakeEnv(episode_len=10000)
    frames, flags, kick_index = rollout_push_frames(_Policy(), config, 20000000, train_seed=3, dt_env_ns=20000000, env=env)
    assert kick_index == 60
    assert len(frames) == len(flags) == 420
    assert all(flags[:kick_index])  # hover preamble inside the stable band
    assert flags[kick_index] == _stable(np.asarray(env.state))
    x, _, theta, _ = env.state
    assert abs(x) <= KICK_X and abs(theta) <= KICK_THETA_RAD
    assert any(abs(v) > 0 for v in env.state)  # a real perturbation, not zeros


def test_rollout_push_frames_deterministic_per_seed():
    from types import SimpleNamespace

    config = SimpleNamespace(task="cartpole", seed=12345)
    env_a = _FakeEnv(episode_len=100)
    rollout_push_frames(_Policy(), config, 20000000, train_seed=3, dt_env_ns=20000000, env=env_a)
    env_b = _FakeEnv(episode_len=100)
    rollout_push_frames(_Policy(), config, 20000000, train_seed=3, dt_env_ns=20000000, env=env_b)
    assert np.array_equal(np.asarray(env_a.state), np.asarray(env_b.state))
    env_c = _FakeEnv(episode_len=100)
    rollout_push_frames(_Policy(), config, 20000000, train_seed=4, dt_env_ns=20000000, env=env_c)
    assert not np.array_equal(np.asarray(env_a.state), np.asarray(env_c.state))


def test_pick_push_trial_prefers_recovery():
    from latency_ml.experiments.render import _pick_push_trial

    assert _pick_push_trial(lambda t: (10, False, True)) == 0
    # Kick-stable recovery loses to a later kick-unstable recovery.
    assert _pick_push_trial(lambda t: [(5, False, True), (50, True, True), (30, True, False), (99, True, False)][t]) == 2
    # No recovery anywhere: longest survival wins.
    assert _pick_push_trial(lambda t: (t * 10, False, False)) == 9


def test_simulate_push_trial_runs():
    from types import SimpleNamespace

    from latency_ml.experiments.render import _simulate_push_trial

    config = SimpleNamespace(task="cartpole", seed=12345)
    survived, recovered, kick_stable = _simulate_push_trial(_Policy(), config, 20000000, 0, 0, 50)
    assert survived > 0
    assert isinstance(recovered, bool)
    assert isinstance(kick_stable, bool)
