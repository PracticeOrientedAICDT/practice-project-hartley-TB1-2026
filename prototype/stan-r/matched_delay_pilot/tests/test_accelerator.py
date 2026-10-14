from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from envs.accelerator_env import AcceleratorConfig,AcceleratorEnv
from experiment import rollout,eval_config,ARCHS
from wrappers.benchmarking import ActionDelayWrapper,ObservationDelayWrapper

def config(**kw):
    return AcceleratorConfig(tune_jitter=0,noise_std=0,kick_ratio=0,**kw)

def test_api():check_env(AcceleratorEnv(),skip_render_check=True)

@pytest.mark.parametrize('ns',[0,10,100,250,500,1000,5000])
def test_green_convolution(ns):
    c=config(inference_ns=ns);e=AcceleratorEnv(c);e.reset(seed=1)
    actions=np.sin(np.arange(40)*.7)*.4
    w0=2*np.pi*c.tune;g=c.damping;w=np.sqrt(w0*w0-g*g)
    for k,a in enumerate(actions):
        info=e.step([a])[4];t=k+1
        expected=0.
        for j,u in enumerate(actions[:k+1]):
            dt=t-j-ns*c.sample_hz*1e-9
            if dt>0:expected+=c.actuator_gain*u*w0/w*np.exp(-g*dt)*np.sin(w*dt)
        assert np.isclose(info['x'],expected,atol=1e-12)

def test_latency_features():
    e=AcceleratorEnv(config(inference_ns=100,transport_ns=50))
    obs,info=e.reset(seed=1)
    assert obs.shape==(27,)
    np.testing.assert_allclose(obs[-3:],[0,.02,.03])
    assert np.isclose(info['action_delay_steps'],.405)

def test_random_latency_seed():
    c=config(inference_range_ns=(10,1000));e=AcceleratorEnv(c)
    a=e.reset(seed=123);b=e.reset(seed=123)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]

def test_pending_reset():
    e=AcceleratorEnv(config(inference_ns=1000));e.reset(seed=1)
    info=e.step([1])[4];assert info['pending_actions']==1
    e.reset(seed=1);assert not e.events and not e.commands.any()

def test_paired_disturbances():
    c=AcceleratorConfig()
    _,a,_=rollout(lambda obs:np.zeros(1),eval_config(c,100),123,True)
    _,b,_=rollout(lambda obs:np.zeros(1),eval_config(c,5000),123,True)
    np.testing.assert_allclose([x['x'] for x in a],[x['x'] for x in b],atol=1e-12,rtol=1e-10)

def test_integer_wrapper_equivalence():
    c=config(inference_ns=0)
    embedded=AcceleratorEnv(replace(c,inference_ns=2/c.sample_hz*1e9))
    wrapped=ActionDelayWrapper(AcceleratorEnv(c),2)
    embedded.reset(seed=1);wrapped.reset(seed=1)
    for a in [.1,.2,-.1,0,.3]:
        assert np.isclose(embedded.step([a])[4]['x'],wrapped.step([a])[4]['x'])

def test_observation_wrapper_copy():
    e=ObservationDelayWrapper(AcceleratorEnv(config(inference_ns=0)),1)
    obs,_=e.reset(seed=1);obs[:]=999
    assert e.step([1])[0][0]==0

def test_max_architecture():assert ARCHS['max']==[256,256,256]

def test_invalid_latency():
    with pytest.raises(ValueError):AcceleratorEnv(config(inference_ns=-1))

def test_envelope_free_decay():
    e=AcceleratorEnv(config(inference_ns=0));e.reset(seed=1);e.p=1
    amp=e._info()['amplitude'];e._advance(.7)
    assert np.isclose(e._info()['amplitude'],amp*np.exp(-e.cfg.damping*.7))


def test_all_nondeterministic_features_default_off():
    c=AcceleratorConfig()
    assert c.tune_drift_std_per_turn==0
    assert c.tune_jump_probability==0
    assert c.momentum_process_std==0
    assert c.inference_jitter_ns==0
    assert c.latency_outlier_probability==0
    assert c.actuator_gain_std_fraction==0


def test_stochastic_rollout_seed_reproducibility():
    from envs.accelerator_profiles import apply_profile
    c=apply_profile(AcceleratorConfig(episode_steps=50,kick_window=(5,5)),'uncertain')
    a,b=AcceleratorEnv(c),AcceleratorEnv(c)
    oa,_=a.reset(seed=77);ob,_=b.reset(seed=77)
    np.testing.assert_array_equal(oa,ob)
    for _ in range(50):
        ra=a.step([.2]);rb=b.step([.2])
        np.testing.assert_array_equal(ra[0],rb[0]);assert ra[1:]==rb[1:]
        if ra[2] or ra[3]:break


def test_stochastic_process_changes_trajectory():
    base=AcceleratorConfig(episode_steps=40,kick_window=(5,5),noise_std=0,tune_jitter=0)
    stochastic=replace(base,momentum_process_std=.002,tune_drift_std_per_turn=1e-5,
                       inference_jitter_ns=10,actuator_gain_std_fraction=.01)
    a,b=AcceleratorEnv(base),AcceleratorEnv(stochastic)
    a.reset(seed=9);b.reset(seed=9)
    xa=[];xb=[]
    for _ in range(40):
        xa.append(a.step([.1])[4]['x']);xb.append(b.step([.1])[4]['x'])
    assert not np.allclose(xa,xb)


def test_forced_outlier_and_gain_are_reported():
    c=AcceleratorConfig(inference_ns=100,latency_outlier_probability=1,
        latency_outlier_ns=500,actuator_gain_std_fraction=.01,max_action_delay=128)
    e=AcceleratorEnv(c);e.reset(seed=1);info=e.step([.2])[4]
    assert info['latency_outlier']
    assert info['pending_actions']>=1
    assert np.isfinite(info['actuator_gain_factor'])


def test_profiles_leave_deterministic_unchanged():
    from envs.accelerator_profiles import apply_profile
    c=AcceleratorConfig()
    assert apply_profile(c,'deterministic')==c
    stress=apply_profile(c,'stress')
    assert stress.momentum_process_std>0
    assert apply_profile(stress,'deterministic').momentum_process_std==0
