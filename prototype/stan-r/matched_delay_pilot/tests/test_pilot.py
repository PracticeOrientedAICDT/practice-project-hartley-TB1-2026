import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from actor import NumpyActor
from envs.laser_target_env import LaserTargetConfig
from matched_models import resolve_models
from pilot_runs import create_run
from report_generator import diagonal,summarize,aggregate


def fixture_models(path):
    # Zero-weight actors are test fixtures, not trained policies or claimed results.
    for budget in [1000,100,500]:
        cfg=LaserTargetConfig(inference_ns=budget,episode_steps=20)
        stem=path/f'fixture_{budget}ns'
        NumpyActor([(np.zeros((2,23),np.float32),np.zeros(2,np.float32))]).save(stem.with_suffix('.npz'))
        stem.with_suffix('.json').write_text(json.dumps(dict(name=stem.name,environment='laser',
            trained_inference_ns=budget,seed=0,architecture=[],config=asdict(cfg))))


def test_numeric_model_order(tmp_path):
    fixture_models(tmp_path)
    assert [m['trained_inference_ns'] for _,m in resolve_models([str(tmp_path)])]==[100,500,1000]


def test_code_cannot_overwrite(tmp_path):
    create_run(tmp_path,'same')
    with pytest.raises(FileExistsError):create_run(tmp_path,'same')
    with pytest.raises(ValueError):create_run(tmp_path,'../escape')


def test_diagonal_excludes_mismatch():
    # Hand-written unit-test values, not experimental data.
    df=pd.DataFrame(dict(model=['a','a','b'],trained_ns=[100,100,500],inference_ns=[100,500,500],rms_error=[1,99,2]))
    assert len(diagonal(df))==2
    summary=aggregate(summarize(df))
    assert summary.latency_ns.tolist()==[100,500]
    assert summary.rms_error.tolist()==[1,2]


def test_randomized_model_rejected(tmp_path):
    fixture_models(tmp_path)
    p=tmp_path/'fixture_100ns.json';meta=json.loads(p.read_text());meta['config']['inference_range_ns']=[0,500];p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='randomized'):resolve_models([str(tmp_path)])


def test_matched_evaluation(tmp_path):
    from evaluate_matched import evaluate
    models=tmp_path/'models';models.mkdir();fixture_models(models)
    run=evaluate([str(models)],episodes=2,root=tmp_path/'runs',code='fixture-test',calls=2)
    df=pd.read_csv(run/'episodes.csv')
    assert len(df)==6
    assert np.allclose(df.trained_ns,df.inference_ns)
    assert pd.read_csv(run/'report'/'diagonal.csv').latency_ns.tolist()==[100,500,1000]


def test_demo_orchestration_without_training(tmp_path,monkeypatch):
    # Stub only the expensive trainer; real evaluation/report/run saving executes.
    import sys,run_demo
    def fake_train(args,check):
        out=tmp_path/'runs'/'demo-test'/'models';out.mkdir();fixture_models(out)
    monkeypatch.setattr(run_demo.subprocess,'run',fake_train)
    monkeypatch.setattr(sys,'argv',['run_demo.py','--code','demo-test','--runs-root',str(tmp_path/'runs'),
        '--episodes','2','--no-video','--seeds','0'])
    run_demo.main()
    run=tmp_path/'runs'/'demo-test'
    assert json.loads((run/'run.json').read_text())['status']=='complete'
    assert (run/'report'/'diagonal.csv').exists()
    assert not (run/'evaluation').exists()
