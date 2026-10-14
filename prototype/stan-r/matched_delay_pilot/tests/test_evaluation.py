import numpy as np
from envs.beam_env import BeamEnv, BeamConfig
from evaluate import rollout
from actor import NumpyActor

def test_uncontrolled_rollout():
    r=rollout(BeamEnv(),lambda obs:np.zeros(1),42)
    assert r['survival']==2048 and r['lost']==0
    assert np.isfinite(r['settle_us'])

def test_pre_kick_loss():
    c=BeamConfig(loss_radius=0.001,action_delay_range=(0,0))
    r=rollout(BeamEnv(c),lambda obs:np.ones(1),42)
    assert r['lost']==1 and np.isnan(r['mean_error'])

def test_numpy_actor_roundtrip(tmp_path):
    actor=NumpyActor([(np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32)),
                      (np.ones((1,2),dtype=np.float32),np.zeros(1,dtype=np.float32))])
    path=tmp_path/'actor.npz'; actor.save(path)
    np.testing.assert_array_equal(NumpyActor.load(path)([-1,.25]),[.25])
