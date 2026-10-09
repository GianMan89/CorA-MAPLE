"""Paper-to-configuration consistency checks."""
from cora_maple.config import ROOT, METHODS, BASELINES, load_config


def test_primary_contains_exact_paper_methods():
    cfg = load_config(ROOT / 'configs/repeated_cv.yaml')
    assert cfg['study']['methods'] == [
        'scratch', 'scratch_aug', 'recon_excerpt', 'maple_excerpt', 'maple_full',
        'cora_maple_excerpt', 'cora_maple_full', *BASELINES,
    ]
    assert cfg['cv']['n_splits'] == 5
    assert cfg['cv']['n_repeats'] == 10
    assert cfg['study']['label_budgets'] == [1, 2, 5, 20]


def test_supervised_augmentation_matches_manuscript():
    cfg = load_config(ROOT / 'configs/repeated_cv.yaml')
    assert METHODS['scratch_aug']['augmentation'] == 'bank'
    assert cfg['training']['augmentation_bank_size'] == 8
    assert 'scratch_aug_online' not in cfg['study']['methods']
    assert METHODS['scratch_aug_online']['augmentation'] == 'online'


def test_paper_loss_and_training_settings():
    cfg = load_config(ROOT / 'configs/repeated_cv.yaml')
    assert cfg['objective'] == {'ema': .99, 'raw_weight': .25, 'var_weight': .1, 'gamma': .5, 'epsilon': 1e-4}
    assert cfg['training']['pretrain_steps'] == 800
    assert cfg['training']['finetune_steps'] == 400
    assert cfg['corruption']['operators'] == ['missing', 'spurious', 'timing']
    assert cfg['test']['operators'] == ['missing', 'spurious', 'timing', 'burst', 'tag_dropout', 'mixed']
