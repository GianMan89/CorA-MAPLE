"""Public experiment invariants and numerically identical Scratch controls."""
from pathlib import Path
import re
import numpy as np
import torch
from cora_maple.config import ROOT, METHODS, load_config, training_settings
from cora_maple.learning import fit_classifier, predict_neural


def test_default_schedules_equal_across_all_neural_conditions():
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    settings=[training_settings(c,name) for name in METHODS]
    assert all(v==settings[0] for v in settings)
    assert settings[0]['pretrain_steps']==4000
    assert settings[0]['finetune_steps']==2400
    assert settings[0]['supervised_objective']=='clean_cross_entropy'


def test_paired_scratch_scopes_are_not_independent_models(tmp_path):
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    c['model'].update(d_model=8,n_layers=1)
    c['training'].update(log_interval=1,finetune_warmup=1)
    c['training']['schedules']={s:{'pretrain_steps':2,'finetune_steps':3} for s in ['excerpt','full']}
    rng=np.random.default_rng(112)
    x=rng.integers(0,2,size=(6,3,12)).astype(np.float32)
    y=np.repeat(np.arange(3),2)
    seeds={'initialization_seed':71,'head_seed':73,'batch_seed':79,'dropout_seed':83,'corruption_seed':89}
    models=[]
    for scope in ['excerpt','full']:
        model=fit_classifier(x,y,3,c,{'method':'scratch_'+scope,**seeds},tmp_path/scope,None,'cpu',1)
        models.append(model)
    for name,value in models[0].state_dict().items():
        assert torch.equal(value,models[1].state_dict()[name]),name
    test=rng.integers(0,2,size=(9,3,18)).astype(np.float32)
    np.testing.assert_array_equal(predict_neural(models[0],test,'cpu',4),predict_neural(models[1],test,'cpu',4))
