"""Fixed tensors and gradients only. No optimizer updates or research runs."""
import numpy as np
import pytest
import torch
from afc_excerpt_ssl.models import PatchEncoder
from cora_maple.learning import ControlledPretrainer, make_mask
from cora_maple.transforms import Corruptor, supervised_batch, test_conditions as scenario_conditions
from cora_maple.config import load_config, ROOT


def configuration():
    return load_config(ROOT / 'configs/repeated_cv.yaml')


def test_loss_controls_remove_exact_terms():
    torch.set_num_threads(1); torch.manual_seed(3)
    cfg = configuration()
    model = ControlledPretrainer(PatchEncoder(3), cfg['objective']).eval()
    x = torch.randint(0,2,(4,3,60)).float()
    mask = torch.from_numpy(make_mask(4,20,.3,np.random.default_rng(2)))
    with torch.no_grad():
        full, a = model.loss(x,x,mask,'latent')
        rec, b = model.loss(x,x,mask,'reconstruction')
        no_latent, c = model.loss(x,x,mask,'reconstruction_variance')
    assert rec.item() == pytest.approx(b['reconstruction'])
    assert no_latent.item() == pytest.approx(.25*c['reconstruction']+.1*c['variance'])
    assert full.item() - no_latent.item() == pytest.approx(a['latent'], abs=1e-6)


def test_stopped_teacher_and_zero_gradients_for_removed_predictor():
    torch.set_num_threads(1); torch.manual_seed(2)
    model = ControlledPretrainer(PatchEncoder(3), configuration()['objective']).train()
    x = torch.zeros(4,3,60); x[:,0,10:20] = 1
    mask = torch.from_numpy(make_mask(4,20,.3,np.random.default_rng(3)))
    loss,_ = model.loss(x,x,mask,'reconstruction_variance')
    loss.backward()  # Check differentiation; no optimizer is constructed or stepped.
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())
    assert model.student.embed.weight.grad is not None
    assert not model.teacher.training


def test_empty_reference_cells_have_finite_losses():
    torch.set_num_threads(1)
    model = ControlledPretrainer(PatchEncoder(2), configuration()['objective']).eval()
    mask = torch.ones(4,5,dtype=torch.bool)
    for val in [0.,1.]:
        x = torch.full((4,2,15),val)
        with torch.no_grad():
            for mode in ['latent','reconstruction','reconstruction_variance']:
                loss,_ = model.loss(x,x,mask,mode)
                assert torch.isfinite(loss)


def test_variable_lengths_and_mask_count():
    torch.set_num_threads(1)
    enc = PatchEncoder(2).eval()
    for length in [15,30,45,60,390]:
        mask = make_mask(2,length//3,.3,np.random.default_rng(0))
        assert (mask.sum(1) == max(1,round(.3*(length//3)))).all()
        with torch.no_grad():
            assert enc(torch.zeros(2,2,length),torch.from_numpy(mask)).shape == (2,length//3,48)
            assert enc.pooled(torch.zeros(2,2,length)).shape == (2,96)


def test_fresh_supervised_views_are_called_per_sample_every_step():
    class Spy:
        def __init__(self): self.calls = 0
        def training(self,x,rng):
            self.calls += 1
            return x + rng.random()
    X = np.zeros((5,2,15),dtype=np.float32)
    spy, rng = Spy(), np.random.default_rng(8)
    a = supervised_batch(X,np.arange(5),'online',spy,rng)
    b = supervised_batch(X,np.arange(5),'online',spy,rng)
    assert spy.calls == 10 and not np.array_equal(a,b)
    assert np.array_equal(X,np.zeros_like(X))


def test_supervised_source_order_is_independent_of_augmentation_rng():
    cfg = configuration(); cor = Corruptor(cfg)
    X = np.zeros((5,2,15),dtype=np.float32);X[:,0,2:7]=1
    a = np.random.default_rng(2); b = np.random.default_rng(2); noise = np.random.default_rng(3)
    for _ in range(5):
        indices_a = a.choice(5,3,replace=False)
        indices_b = b.choice(5,3,replace=False)
        assert np.array_equal(indices_a,indices_b)
        supervised_batch(X,indices_b,'online',cor,noise)


def test_zero_severity_and_unchanged_reference():
    cfg = configuration(); cor = Corruptor(cfg)
    x = np.zeros((3,60),dtype=np.float32);x[0,:20]=1;x[1,8:30]=1
    original=x.copy()
    for op in cor.ops:
        assert np.array_equal(x,cor.apply(x,op,0,np.random.default_rng(1)))
        y=cor.apply(x,op,.3,np.random.default_rng(1))
        assert y.shape==x.shape and np.isin(y,[0,1]).all()
        assert np.array_equal(x,original)
    assert len(scenario_conditions(cfg))==55


def test_no_teacher_forward_for_reconstruction_controls(monkeypatch):
    torch.set_num_threads(1)
    model=ControlledPretrainer(PatchEncoder(2),configuration()['objective']).eval()
    def forbidden(*args,**kwargs): raise AssertionError('Teacher must not be evaluated')
    monkeypatch.setattr(model.teacher,'forward',forbidden)
    x=torch.zeros(2,2,15);mask=torch.zeros(2,5,dtype=torch.bool)
    with torch.no_grad():
        model.loss(x,x,mask,'reconstruction')
        model.loss(x,x,mask,'reconstruction_variance')
