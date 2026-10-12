"""Physical input semantics, padding invariance, gradients and reference isolation."""
from copy import deepcopy
import numpy as np
import torch
import pytest
from afc_robustness.domain import AlarmEpisode,AlarmEvent,EventType
from afc_robustness.representations import series_to_episode
from afc_excerpt_ssl.models import PatchEncoder,Classifier
from cora_maple.transforms import reconstruct_snapshot,Corruptor,input_effects
from cora_maple.config import load_config,ROOT
from cora_maple.learning import ControlledPretrainer,make_mask
from cora_maple.baselines import predict_baseline_valid


def cfg():return load_config(ROOT/'configs/repeated_cv.yaml')


def test_lost_act_does_not_create_past_activity():
    ep=AlarmEpisode([AlarmEvent(0,'RTN',7,1)],horizon=11,n_tags=1)
    x=reconstruct_snapshot(ep,np.array([0]),12,1.)
    assert x.sum()==0


def test_initially_active_excerpt_accepts_leading_rtn():
    ep=AlarmEpisode([AlarmEvent(0,'RTN',4,0)],horizon=11,n_tags=1)
    x=reconstruct_snapshot(ep,np.array([1]),12,1.)
    assert x[0,:4].all() and not x[0,4:].any()


def test_lost_rtn_can_extend_activity_without_inventing_prefix():
    ep=AlarmEpisode([AlarmEvent(0,'ACT',5,0)],horizon=11,n_tags=1)
    x=reconstruct_snapshot(ep,np.array([0]),12,1.)
    assert not x[0,:5].any() and x[0,5:].all()


def test_snapshot_not_counted_as_event_or_overwritten_by_t0_shift():
    x=np.ones((1,12),dtype=np.uint8);x[:,5:]=0
    ep=series_to_episode(x,initial_active_policy='ignore')
    assert len(ep.events)==1 and ep.events[0].event_type==EventType.RTN
    changed=ep.with_events([ep.events[0].with_timestamp(0)])
    y=reconstruct_snapshot(changed,x[:,0],12,1.)
    assert y[0,0]==1 and not y[0,1:].any()


def test_known_snapshot_round_trip_arbitrary_start_states():
    rng=np.random.default_rng(21)
    for _ in range(15):
        x=rng.integers(0,2,size=(5,17),dtype=np.uint8)
        ep=series_to_episode(x,initial_active_policy='ignore')
        np.testing.assert_array_equal(reconstruct_snapshot(ep,x[:,0],17,1.),x)


def test_persistent_full_corruption_has_exact_channel_budget():
    c=Corruptor(cfg());x=np.zeros((50,60),dtype=np.float32)
    for s in [.1,.3,.5]:
        y=c.apply(x,'false_full',s,np.random.default_rng(12))
        assert y.sum()==int(np.floor(s*50))*60
        assert not x.any()
        y=c.apply(1-x,'missing_full',s,np.random.default_rng(12))
        assert y.sum()==(50-int(np.floor(s*50)))*60


def test_delay_removes_prefix_and_reports_invalid_suffix():
    c=Corruptor(cfg());x=np.zeros((2,12),dtype=np.float32);x[0,1:3]=1;x[1,7:10]=1
    obs=c.apply_with_length(x,'late_start',.5,np.random.default_rng(1))
    assert obs.valid_length==6
    np.testing.assert_array_equal(obs.values[:,:6],x[:,6:])
    assert not obs.values[:,6:].any()
    ef=input_effects(x,obs)
    assert ef['lost_samples']==6 and ef['changed_state_cells']==0 and ef['unchanged_observation']==0


@pytest.mark.parametrize('n',[1,2,5,7,9,12,18])
def test_neural_logits_ignore_padding_values_and_padding_amount(n):
    torch.manual_seed(101);torch.set_num_threads(1)
    model=Classifier(PatchEncoder(3,d_model=12,n_layers=1,dropout=0),5).eval()
    short=torch.rand(2,3,n)
    padded=torch.cat([short,torch.rand(2,3,30-n)*100],dim=-1)
    with torch.no_grad():
        a=model(short)
        b=model(padded,lengths=torch.tensor([n,n]))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)


def test_mixed_length_minibatch_equals_individual_observations():
    torch.manual_seed(77);m=Classifier(PatchEncoder(3,d_model=12,n_layers=1,dropout=0),5).eval()
    x=torch.randn(3,3,21);lengths=torch.tensor([4,12,17])
    with torch.no_grad():
        batch=m(x,lengths=lengths)
        individual=torch.cat([m(x[i:i+1,:,:n]) for i,n in enumerate(lengths)])
    torch.testing.assert_close(batch,individual,rtol=1e-5,atol=2e-6)


def test_no_gradient_from_invalid_samples():
    torch.manual_seed(23);m=Classifier(PatchEncoder(2,d_model=8,n_layers=1,dropout=0),3)
    x=torch.randn(2,2,12,requires_grad=True);m(x,lengths=torch.tensor([4,8])).sum().backward()
    assert x.grad[0,:,4:].abs().sum()==0 and x.grad[1,:,8:].abs().sum()==0


def test_binary_and_latent_losses_ignore_padding():
    torch.manual_seed(23);p=ControlledPretrainer(PatchEncoder(2,d_model=8,n_layers=1,dropout=0),cfg()['objective']).eval()
    x=torch.randint(0,2,(4,2,12)).float();x2=x.clone();x2[:,:,6:]=1-x2[:,:,6:]
    mask=torch.zeros((4,4),dtype=torch.bool);mask[:,0]=True
    with torch.no_grad():
        a,_=p.loss(x,x,mask,lengths=torch.full((4,),6))
        b,_=p.loss(x2,x2,mask,lengths=torch.full((4,),6))
    torch.testing.assert_close(a,b,rtol=0,atol=1e-7)


def test_classical_grouping_excludes_padding(monkeypatch):
    seen=[]
    import cora_maple.baselines as b
    def fake(model,x):
        seen.append(x.shape[-1]);assert not np.any(x==99)
        return np.full(len(x),x.shape[-1],dtype=int)
    monkeypatch.setattr(b,'predict_baseline',fake)
    x=np.zeros((3,2,12));lengths=np.array([4,9,4])
    for i,n in enumerate(lengths):x[i,:,n:]=99
    result=predict_baseline_valid(None,x,lengths)
    np.testing.assert_array_equal(result,lengths)
    assert seen==[4,9]


def test_pretraining_cannot_sample_heldout_families_or_unaligned_delay():
    c=Corruptor(cfg());x=np.zeros((20,15));x[:,5:10]=1;rng=np.random.default_rng(42)
    seen=set()
    for _ in range(150):
        v=c.training_view(x,rng,'pretraining')
        assert v.severity<=.3 and v.operator not in ['missing','spurious','late_start']
        assert v.valid_length==15
        seen.add(v.operator)
    assert seen=={'clean','false_full','missing_full','timing'}


def test_supervised_view_request_is_rejected():
    c=Corruptor(cfg());x=np.zeros((20,15));rng=np.random.default_rng(42)
    with pytest.raises(ValueError,match='Supervised corruption is disabled'):
        c.training_view(x,rng,'supervised')
