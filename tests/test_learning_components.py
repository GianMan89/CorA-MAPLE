"""Fixed tensors and gradients only. No optimizer updates or research runs."""
import numpy as np
import pytest
import torch
from afc_excerpt_ssl.models import PatchEncoder
from cora_maple.learning import ControlledPretrainer, make_mask
from cora_maple.transforms import Corruptor, test_conditions as scenario_conditions
from cora_maple.config import load_config, ROOT


def configuration():
    return load_config(ROOT / 'configs/repeated_cv.yaml')


def test_compound_objective_matches_its_components():
    torch.set_num_threads(1); torch.manual_seed(3)
    cfg = configuration()
    model = ControlledPretrainer(PatchEncoder(3), cfg['objective']).eval()
    x = torch.randint(0,2,(4,3,60)).float()
    mask = torch.from_numpy(make_mask(4,20,.3,np.random.default_rng(2)))
    with torch.no_grad():
        total, parts = model.loss(x,x,mask,'latent')
    assert total.item() == pytest.approx(parts['latent'] + cfg['objective']['raw_weight']*parts['reconstruction'] + .1*parts['variance'], abs=1e-6)


def test_teacher_targets_stop_gradients():
    torch.set_num_threads(1); torch.manual_seed(2)
    model = ControlledPretrainer(PatchEncoder(3), configuration()['objective']).train()
    x = torch.zeros(4,3,60); x[:,0,10:20] = 1
    mask = torch.from_numpy(make_mask(4,20,.3,np.random.default_rng(3)))
    loss,_ = model.loss(x,x,mask,'latent')
    loss.backward()  # Differentiation only, without optimizer updates.
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert model.student.embed.weight.grad is not None
    assert not model.teacher.training


def test_empty_reference_cells_have_finite_losses():
    torch.set_num_threads(1)
    model = ControlledPretrainer(PatchEncoder(2), configuration()['objective']).eval()
    mask = torch.ones(4,5,dtype=torch.bool)
    for val in [0.,1.]:
        x = torch.full((4,2,15),val)
        with torch.no_grad():
            loss,_ = model.loss(x,x,mask,'latent')
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


def test_zero_severity_and_unchanged_reference():
    cfg = configuration(); cor = Corruptor(cfg)
    x = np.zeros((3,60),dtype=np.float32);x[0,:20]=1;x[1,8:30]=1
    original=x.copy()
    for op in cor.ops:
        assert np.array_equal(x,cor.apply(x,op,0,np.random.default_rng(1)))
        y=cor.apply(x,op,.3,np.random.default_rng(1))
        assert y.shape==x.shape and np.isin(y,[0,1]).all()
        assert np.array_equal(x,original)
    assert len(list(scenario_conditions(cfg)))==91


