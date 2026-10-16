import numpy as np
import pytest
from latency_ml.experiments.render import rollout_frames, select_median_seed, write_video


class _FakeEnv:
    def __init__(self, episode_len: int = 7, hold: int = 1):
        self.episode_len = episode_len
        self.steps = 0
        self.hold = hold

    def reset(self, **kwargs):
        self.steps = 0
        return (np.zeros(4, dtype=np.float32), {})

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.episode_len
        return (np.zeros(4, dtype=np.float32), 1.0, done, False, {})

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
