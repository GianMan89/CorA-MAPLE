"""Clean-only supervised training is enforced, not merely a default flag."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from afc_excerpt_ssl.models import PatchEncoder
from cora_maple.config import load_config,ROOT,METHODS,encoder_kwargs
from cora_maple.learning import fit_classifier


@pytest.mark.parametrize('method',list(METHODS))
def test_every_classifier_uses_clean_input_only(tmp_path,monkeypatch,method):
    import cora_maple.learning as learning
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    c['model'].update(d_model=8,n_layers=1)
    c['training']['schedules']={s:{'pretrain_steps':2,'finetune_steps':2} for s in ['excerpt','full']}
    c['training']['log_interval']=1
    x=np.random.default_rng(8).integers(0,2,size=(6,3,12)).astype(np.float32)
    y=np.repeat(np.arange(3),2)
    enc=None
    if METHODS[method]['pretrain']:
        torch.manual_seed(7)
        enc=tmp_path/'encoder.pt'
        torch.save({'protocol':c['protocol'],'state_dict':PatchEncoder(3,**encoder_kwargs(c)).state_dict()},enc)
    def forbidden(*args,**kwargs):
        raise AssertionError('Supervised fitting must never instantiate or request a corruptor')
    monkeypatch.setattr(learning,'Corruptor',forbidden)
    t={'method':method,'initialization_seed':7,'head_seed':11,'batch_seed':13,'corruption_seed':17,'dropout_seed':19}
    out=tmp_path/'fit'
    original=x.copy()
    fit_classifier(x,y,3,c,t,out,enc,'cpu',1)
    np.testing.assert_array_equal(x,original)
    stats=json.loads((out/'training.json').read_text())
    assert stats['augmentation']=='none' and stats['operator_exposures']=={}
    assert stats['max_training_severity_observed']==0 and stats['corrupted_sample_steps']==0
    h=pd.read_csv(out/'finetuning.csv')
    assert (h.corrupted_ce==0).all()
    np.testing.assert_allclose(h.minibatch_loss,h.clean_ce,rtol=0,atol=0)
