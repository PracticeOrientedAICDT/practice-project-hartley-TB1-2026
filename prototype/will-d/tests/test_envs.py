from __future__ import annotations

from pathlib import Path

import pytest
from latency_ml.config import load_config
from latency_ml.envs.factory import make_latency_env
from latency_ml.timing import Resolution, derive_icdi, format_ns

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_derive_icdi_exact_grid():
    icdi = derive_icdi(40000000, 20000000)
    assert icdi.action_steps == 2
    assert icdi.effective_ns == 40000000
    assert icdi.resolution is Resolution.EXACT
    assert icdi.is_independent_measurement


def test_derive_icdi_quantises_up_and_rejects_sub_frame():
    assert derive_icdi(30000000, 20000000).resolution is Resolution.QUANTIZED_UP
    assert derive_icdi(30000000, 20000000).action_steps == 2
    assert derive_icdi(1000000, 20000000).resolution is Resolution.UNRESOLVED
    assert derive_icdi(1000000, 20000000).action_steps == 1


def test_derive_icdi_rejects_non_positive():
    for bad in (0, -5):
        with pytest.raises(ValueError, match="must be positive"):
            derive_icdi(bad)


def test_format_ns_units():
    assert format_ns(500) == "500 ns"
    assert format_ns(20000) == "20 us"
    assert format_ns(20000000) == "20 ms"
    assert format_ns(2000000000) == "2 s"


def test_latency_env_holds_action_for_quantised_frames():
    env = make_latency_env("cartpole", 40000000, dt_env_ns=20000000)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["hold_steps"] == 2
    assert set(info) == {"hold_steps"}
    env.close()


def test_latency_env_flags_end_of_episode_mid_hold():
    env = make_latency_env("cartpole", 100000000, dt_env_ns=20000000)
    env.reset(seed=0)
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            assert info["hold_steps"] <= 5
            env.release_terminal()
            env.reset()
            break
    env.close()


def test_config_extends_chain_merges():
    cartpole = load_config(CONFIGS / "cartpole.yaml")
    base = load_config(CONFIGS / "base.yaml")
    assert cartpole.task == "cartpole"
    assert base.task == "cartpole"
    assert cartpole.training.total_integration_steps == base.training.total_integration_steps
    assert cartpole.latency.sweep_grid_ns == (20000000, 40000000, 60000000, 80000000, 100000000)
    lunarlander = load_config(CONFIGS / "lunarlander.yaml")
    assert lunarlander.task == "lunarlander"
    assert lunarlander.training.control_training_seeds == 10


def test_config_ignores_unknown_keys():
    assert load_config(CONFIGS / "base.yaml").training.learning_rate == 2.5e-4


def test_missing_config_raises():
    with pytest.raises(Exception, match="not found"):
        load_config(CONFIGS / "nope.yaml")
