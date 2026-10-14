from dataclasses import replace
import numpy as np
from gymnasium.utils.env_checker import check_env
from envs.beam_env import BeamEnv, BeamConfig

def quiet(delay=0,obs=0):
    return BeamEnv(BeamConfig(tune_jitter=0,noise_std=0,kick_ratio=0,
        action_delay_range=(delay,delay),obs_delay_range=(obs,obs)))

def test_api():
    check_env(BeamEnv(),skip_render_check=True)

def test_zero_delay():
    e=quiet(); e.reset(seed=1); _,_,_,_,i=e.step([1])
    assert np.isclose(i['x'],e.cfg.actuator_gain*np.sin(2*np.pi*e.cfg.tune)*np.exp(-e.cfg.damping))

def test_integer_delay():
    e=quiet(2); e.reset(seed=1)
    assert e.step([1])[4]['amplitude']==0
    assert e.step([0])[4]['amplitude']==0
    assert e.step([0])[4]['amplitude']>0

def test_fractional_delay():
    e=quiet(.5); e.reset(seed=1); i=e.step([1])[4]
    assert np.isclose(i['x'],e.cfg.actuator_gain*np.sin(np.pi*e.cfg.tune)*np.exp(-e.cfg.damping*.5))

def test_observation_delay():
    e=quiet(0,1); e.reset(seed=1)
    obs,*_=e.step([1]); assert obs[0]==0
    obs,*_=e.step([0]); assert obs[0]!=0

def test_seed_and_reset():
    e=BeamEnv(); f=BeamEnv()
    np.testing.assert_array_equal(e.reset(seed=42)[0],f.reset(seed=42)[0])
    for _ in range(200):
        a=e.step([.1]); b=f.step([.1]); np.testing.assert_array_equal(a[0],b[0]); assert a[1:]==b[1:]
    e.reset(seed=42); assert not e.events and not e.commands.any()

def test_truncation():
    e=BeamEnv(BeamConfig(episode_steps=3,kick_window=(0,0)))
    e.reset(seed=1)
    for _ in range(3): result=e.step([0])
    assert result[3] and not result[2]
