"""Frozen clean-only scientific comparison, not a search over the test dataset."""
from copy import deepcopy
import pytest
from cora_maple.config import (load_config, ROOT, METHODS, BASELINES, PROPOSED,
    training_settings, pretraining_method, validate_config, operator_groups)


def test_exactly_three_pretraining_conditions_per_scope():
    cfg=load_config(ROOT/'configs/repeated_cv.yaml')
    assert len(cfg['study']['methods'])==11
    assert set(cfg['study']['methods'])==set(METHODS)|set(BASELINES)
    for scope in ['excerpt','full']:
        cells=[m for m in METHODS.values() if m['scope']==scope]
        assert {(m['pretraining_state'],m['augmentation']) for m in cells}=={
            ('none','none'),('clean','none'),('corrupted','none')}
        assert len(cells)==3


def test_canonical_pretraining_names_match_current_methods():
    for scope in ['excerpt','full']:
        assert pretraining_method('maple_'+scope)=='maple_'+scope
        assert pretraining_method('cora_maple_'+scope)=='cora_maple_'+scope
        with pytest.raises(ValueError):pretraining_method('scratch_'+scope)


def test_within_scope_schedule_is_matched():
    cfg=load_config(ROOT/'configs/repeated_cv.yaml')
    for scope,expected in [('excerpt',(4000,2400)),('full',(4000,2400))]:
        sets=[training_settings(cfg,k) for k,m in METHODS.items() if m['scope']==scope]
        assert all(s==sets[0] for s in sets)
        assert (sets[0]['pretrain_steps'],sets[0]['finetune_steps'])==expected
        assert sets[0]['supervised_objective']=='clean_cross_entropy'
    assert cfg['model']['d_model']==96 and cfg['model']['n_layers']==3
    assert cfg['objective']=={'ema':.995,'raw_weight':.50,'var_weight':.10,'gamma':.50,'epsilon':1e-4}


def test_exposure_groups_include_delay_in_test_only():
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    g=operator_groups(c)
    assert g['pretraining_only']==['false_full','missing_full','timing']
    assert g['trained']==g['pretraining_only']
    assert g['test_only']==['late_start','missing','spurious']
    assert 'supervised' not in c['corruption']
    assert c['test']['severities']==[0.,.1,.2,.3,.4,.5]
    assert c['corruption']['event_initial_state']=='known_snapshot'


@pytest.mark.parametrize('operator',['missing','spurious','late_start'])
def test_test_only_operators_cannot_silently_enter_pretraining(operator):
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    c['corruption']['pretraining']['operators']=[operator]
    c['corruption']['pretraining']['probabilities']=[1.]
    with pytest.raises(ValueError):validate_config(c)


def test_30_percent_pretraining_ceiling_enforced():
    c=load_config(ROOT/'configs/repeated_cv.yaml');c['corruption']['pretraining']['severity_max']=.31
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('bad_change',['supervised_distribution','paired_weight','paired_objective'])
def test_supervised_corruption_cannot_be_silently_reenabled(bad_change):
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    if bad_change=='supervised_distribution':
        c['corruption']['supervised']={'operators':['false_full'],'probabilities':[1.]}
    elif bad_change=='paired_weight':c['training']['paired_clean_weight']=.5
    else:c['training']['supervised_objective']='paired_cross_entropy'
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('unknown_method',['unsupported_model','unknown_excerpt','unknown_full'])
def test_unknown_method_names_are_rejected(unknown_method):
    c=load_config(ROOT/'configs/repeated_cv.yaml');c['study']['methods']=[unknown_method]
    with pytest.raises(ValueError):validate_config(c)


def test_snapshot_is_required():
    c=load_config(ROOT/'configs/repeated_cv.yaml')
    c['corruption']['event_initial_state']='unknown_policy'
    with pytest.raises(ValueError):validate_config(c)


def test_optional_short_schedule_is_equal_and_separate():
    c=load_config(ROOT/'configs/diagnostics/short_schedule.yaml')
    assert c['study']['name']=='short_schedule'
    assert c['training']['schedules']=={s:{'pretrain_steps':1600,'finetune_steps':900} for s in ['excerpt','full']}


def test_complete_pool_counts_not_per_class():
    c=load_config(ROOT/'configs/pool_sizes.yaml')
    assert c['study']['methods']==list(PROPOSED)
    assert c['study']['pool_sizes']==[1,2,5,10,20,50,100,200,400,'full']
    assert c['study']['pool_draws']==[0,1,2]
