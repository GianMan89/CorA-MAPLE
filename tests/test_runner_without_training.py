"""Orchestration integration tests. Training is replaced by mocks; optimizers forbidden."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from cora_maple.common import atomic_json
from cora_maple.config import ROOT, load_config
from cora_maple.protocol import prepare
from cora_maple.report import report
from cora_maple.runner import run_study


def fake_config(tmp_path):
    cfg=load_config(ROOT/'configs/repeated_cv.yaml')
    X=np.zeros((100,2,60),dtype=np.uint8)
    for i in range(100): X[i,0,:8]=[(i>>bit)&1 for bit in range(8)]
    path=tmp_path/'synthetic.npz'
    np.savez_compressed(path,X=X,y=np.repeat(np.arange(5),20),ids=np.array([f'synthetic_{i}' for i in range(100)]),tags=np.array(['a','b']))
    cfg['data'].update(path=str(path),expected_recordings=100,expected_channels=2)
    cfg['cv'].update(n_repeats=2,n_splits=2)
    cfg['test'].update(severities=[0.,.3],draws=1)
    cfg['study'].update(name='mock_only',kind='diagnostic',methods=['scratch','scratch_aug','cora_maple_excerpt'],label_budgets=[1])
    return cfg


def test_prepare_and_report_fail_closed_without_training(tmp_path,monkeypatch):
    import cora_maple.learning as learn
    def forbidden(*args,**kwargs): raise AssertionError('Preparation must not train')
    monkeypatch.setattr(learn,'pretrain_model',forbidden)
    monkeypatch.setattr(learn,'fit_classifier',forbidden)
    cfg=fake_config(tmp_path); work=tmp_path/'work'
    plan=prepare(cfg,work)
    assert plan['counts']=={'pretrain':4,'fit':12,'baseline':0}
    assert not (work/'tasks').exists()
    assert not (work/'cache').exists()
    with pytest.raises(RuntimeError,match='incomplete'): report(work,'mock_only')
    assert not (work/'reports').exists()


def test_resume_cache_and_aggregation_with_mocked_training(tmp_path,monkeypatch):
    import cora_maple.learning as learn
    def forbidden_optimizer(*a,**k): raise AssertionError('No optimizer allowed in this test')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden_optimizer)
    calls={'pretrain':0,'fit':0}
    def pretrain(X,cfg,task,directory,device,threads):
        calls['pretrain']+=1
        torch.save({'state_dict':{}},directory/'encoder.pt')
        atomic_json(directory/'training.json',{'MOCK_ONLY':True})
    def fit(X,y,nclasses,cfg,task,directory,encoder_path,device,threads):
        calls['fit']+=1
        if encoder_path is not None: assert encoder_path.exists()
        torch.save({'MOCK_ONLY':True},directory/'classifier.pt')
        atomic_json(directory/'training.json',{'MOCK_ONLY':True})
        return object()
    monkeypatch.setattr(learn,'pretrain_model',pretrain)
    monkeypatch.setattr(learn,'fit_classifier',fit)
    monkeypatch.setattr(learn,'predict_neural',lambda model,X,device,batch_size:np.zeros(len(X),dtype=np.int16))
    cfg=fake_config(tmp_path);work=tmp_path/'work'
    outcome=run_study(cfg,work)
    assert outcome['complete']
    assert calls=={'pretrain':4,'fit':12}
    assert len(list((work/'cache/test').glob('*.npz')))==4
    previous=dict(calls)
    assert run_study(cfg,work)['newly_executed_tasks']==0
    assert calls==previous
    result=report(work,'mock_only')
    assert result['metric_rows']==12*7
    assert (work/'reports/mock_only/per_repeat.csv').exists()


def test_new_workspace_required_after_configuration_change(tmp_path):
    cfg=fake_config(tmp_path);work=tmp_path/'work'
    prepare(cfg,work)
    cfg['training']['pretrain_steps']+=1
    with pytest.raises(ValueError,match='different source/data/protocol'): prepare(cfg,work)
