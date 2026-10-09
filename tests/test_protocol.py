"""Software checks only: synthetic inputs, no fitting, no benchmark scores."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from afc_excerpt_ssl.data import Dataset
from cora_maple.config import load_config, METHODS, ROOT, validate_config
from cora_maple.protocol import partitions, build_plan, pool_indices


def synthetic_dataset(n=1000):
    X = np.zeros((n, 2, 60), dtype=np.float32)
    for i in range(n):
        X[i, 0, :16] = [(i >> bit) & 1 for bit in range(16)]
    y = np.repeat(np.arange(5), n // 5)
    hashes = np.asarray([hashlib.sha256(x.astype('uint8').tobytes()).hexdigest() for x in X])
    return Dataset(X, y, np.asarray([f"synthetic_{i}" for i in range(n)]), np.array(['a','b']), hashes, np.arange(5))


@pytest.fixture(scope="module")
def primary():
    cfg = load_config(ROOT / "configs/repeated_cv.yaml")
    return cfg, synthetic_dataset(), build_plan(cfg, synthetic_dataset(), "synthetic_context")


def test_exact_ten_times_five_balanced_splits(primary):
    cfg, data, plan = primary
    assert plan['n_splits_total'] == 50
    assert len(plan['parts']) == 50
    for p in plan['parts']:
        assert len(p['train']) == 800 and len(p['test']) == 200
        assert np.all(np.bincount(data.y[p['train']]) == 160)
        assert np.all(np.bincount(data.y[p['test']]) == 40)
        assert not set(p['train']) & set(p['test'])
    assert plan['counts'] == {'pretrain': 250, 'fit': 1400, 'baseline': 1000}


def test_nested_labels_and_paired_seeds(primary):
    cfg, data, plan = primary
    tasks = [t for t in plan['tasks'].values() if t['kind'] == 'fit' and t['repeat'] == 0 and t['fold'] == 0]
    by_k = {}
    for t in tasks:
        by_k.setdefault(t['k'], []).append(t)
    prev = set()
    for k in sorted(by_k):
        first = by_k[k][0]
        selected = set(first['label_indices'])
        assert prev <= selected
        assert len(selected) == k * 5
        for t in by_k[k]:
            assert t['label_indices'] == first['label_indices']
            assert t['head_seed'] == first['head_seed']
            assert t['batch_seed'] == first['batch_seed']
            assert t['initialization_seed'] == first['initialization_seed']
        prev = selected
    enc = [t for t in plan['tasks'].values() if t['kind'] == 'pretrain' and t['repeat'] == 0 and t['fold'] == 0]
    assert len({t['initialization_seed'] for t in enc}) == 1
    assert len({t['crop_seed'] for t in enc}) == 1
    assert len({t['mask_seed'] for t in enc}) == 1
    assert len({t['corruption_seed'] for t in enc}) == 1


def test_fresh_label_and_training_seeds_between_splits(primary):
    _, _, plan = primary
    t = [t for t in plan['tasks'].values() if t['kind'] == 'fit' and t['method'] == 'scratch' and t['k'] == 1]
    assert len({x['initialization_seed'] for x in t}) == 50
    assert len({tuple(x['label_indices']) for x in t}) > 40


def test_pools_are_nested_and_not_label_anchored(primary):
    cfg, _, plan = primary
    p = plan['parts'][0]
    previous = set()
    for n in [1, 2, 5, 10, 20, 50, 100, 200, 400, 'full']:
        pool = set(pool_indices(p, n, 0, cfg['cv']))
        assert previous <= pool
        assert len(pool) == (800 if n == 'full' else n)
        assert pool <= set(p['train'])
        previous = pool
    assert pool_indices(p, 10, 0, cfg['cv']) != pool_indices(p, 10, 1, cfg['cv'])


def test_full_endpoint_reuses_only_new_protocol(primary):
    _, data, plan = primary
    cfg = load_config(ROOT / 'configs/pool_sizes.yaml')
    pool_plan = build_plan(cfg, data, 'synthetic_context')
    full = {i for i,t in pool_plan['tasks'].items() if t['pool'] == 'full'}
    assert len(full) == 200
    assert full <= set(plan['tasks'])
    assert pool_plan['counts'] == {'pretrain': 2800, 'fit': 2800, 'baseline': 0}


def test_crossed_initialization_and_labels_are_independent(primary):
    cfg = load_config(ROOT / 'configs/diagnostics/variability_k1.yaml')
    cfg['cv']['n_repeats'] = 1
    plan = build_plan(cfg, primary[1], 'synthetic_context')
    cells = [t for t in plan['tasks'].values() if t['kind'] == 'fit' and t['method'] == 'cora_maple_excerpt' and t['fold'] == 0]
    assert len(cells) == 9
    for label_draw in range(3):
        same_labels = [t for t in cells if t['label_draw'] == label_draw]
        assert len({tuple(t['label_indices']) for t in same_labels}) == 1
        assert len({t['initialization_seed'] for t in same_labels}) == 3
    for model_rep in range(3):
        same_init = [t for t in cells if t['model_rep'] == model_rep]
        assert len({t['dependency'] for t in same_init}) == 1
        assert len({t['head_seed'] for t in same_init}) == 1
        assert len({t['dropout_seed'] for t in same_init}) == 1
        assert len({t['batch_seed'] for t in same_init}) == 1
        assert len({tuple(t['label_indices']) for t in same_init}) == 3


def test_duplicate_inputs_never_cross_roles(primary):
    cfg, original, _ = primary
    n = 100
    data = synthetic_dataset(n)
    data.X[1::2] = data.X[::2]
    data.hashes[1::2] = data.hashes[::2]
    for p in partitions(data, cfg['cv']):
        assert not set(data.hashes[p['train']]) & set(data.hashes[p['test']])


@pytest.mark.parametrize('change', ['patch', 'probabilities', 'pool', 'batch'])
def test_invalid_settings_fail_before_execution(primary, change):
    cfg = deepcopy(primary[0])
    if change == 'patch': cfg['model']['patch_size'] = 7
    if change == 'probabilities': cfg['corruption']['probabilities'] = [.2,.2,.2]
    if change == 'pool': cfg['study']['pool_sizes'] = [0]
    if change == 'batch': cfg['training']['pretrain_batch'] = 1
    with pytest.raises(ValueError): validate_config(cfg)
