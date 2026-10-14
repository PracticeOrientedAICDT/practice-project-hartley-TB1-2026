import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from envs.laser_target_env import LaserTargetConfig, LaserTargetEnv


def cfg(**kw):
    return LaserTargetConfig(wind_std=0, initial_position=0, initial_velocity=0, **kw)


def test_api():
    check_env(LaserTargetEnv(), skip_render_check=True)


def test_zero_latency_action_arrives_same_step():
    env = LaserTargetEnv(cfg(inference_ns=0))
    env.reset(seed=1)
    info = env.step([1, 0])[4]
    assert info["vx"] < 0 and info["deliveries"]


def test_fractional_latency():
    env = LaserTargetEnv(cfg(inference_ns=50, sample_hz=10e6))
    env.reset(seed=1)
    info = env.step([1, 0])[4]
    assert np.isclose(info["action_delay_steps"], .5)
    assert info["x"] < 0


def test_one_step_boundary_delivery():
    env = LaserTargetEnv(cfg(inference_ns=100, sample_hz=10e6))
    env.reset(seed=1)
    assert not env.step([1, 0])[4]["deliveries"]
    assert env.step([0, 0])[4]["deliveries"]


def test_latency_features_and_shape():
    env = LaserTargetEnv(cfg(inference_ns=100, transport_ns=50))
    obs, info = env.reset(seed=1)
    assert obs.shape == (23,)
    np.testing.assert_allclose(obs[-3:-1], [.02, .03])
    assert info["environment"] == "laser"


def test_seeded_jitter_reproducible():
    c = cfg(inference_ns=100, inference_jitter_ns=25)
    a, b = LaserTargetEnv(c), LaserTargetEnv(c)
    a.reset(seed=42); b.reset(seed=42)
    for action in ([.2,-.1],[0,0],[.5,.5]):
        aa, bb = a.step(action), b.step(action)
        np.testing.assert_array_equal(aa[0], bb[0])
        assert aa[1:] == bb[1:]


def test_reset_clears_queue():
    env = LaserTargetEnv(cfg(inference_ns=500))
    env.reset(seed=1); env.step([1,1]); assert env.events
    env.reset(seed=1); assert not env.events and not env.commands.any()


def test_invalid_action():
    env = LaserTargetEnv(cfg()); env.reset(seed=1)
    with pytest.raises(ValueError): env.step([1])
