"""Aggregation tests use arbitrary synthetic numbers, not benchmark results."""
import numpy as np
import pandas as pd
import pytest
from cora_maple.report import aggregate_frames, paired_differences


def artificial_rows():
    rows=[]
    for rep in range(10):
        for fold in range(5):
            for method,offset in [('cora_maple_excerpt',.1),('maple_excerpt',0.)]:
                for draw in range(3):
                    rows.append(dict(task_id=f'{rep}-{fold}-{method}',repeat=rep,fold=fold,method=method,k=1,pool='full',pool_size=800,
                                     pool_draw=0,label_draw=0,model_rep=0,operator='spurious',severity=.3,draw=draw,
                                     macro_f1=.4+rep*.01+fold*.001+draw*.0001+offset,accuracy=.5))
    return pd.DataFrame(rows)


def test_rep_mean_sd_not_fifty_fold_ci():
    per_fit,per_split,per_repeat,summary=aggregate_frames(artificial_rows())
    assert len(per_fit)==100 and len(per_split)==100 and len(per_repeat)==20
    assert (summary.n_repeats==10).all()
    expected=np.std(np.arange(10)*.01,ddof=1)
    assert np.allclose(summary.macro_f1_sd,expected)


def test_paired_not_unpaired_differences():
    per_fit,*_=aggregate_frames(artificial_rows())
    pair,summary=paired_differences(per_fit)
    assert len(pair)==50
    assert summary.mean_difference.iloc[0]==pytest.approx(.1)
    assert summary.positive_repetitions.iloc[0]==10


def test_mc_draws_are_not_independent_replicates():
    rows=artificial_rows()
    # Copy MC rows ten times: mean should be invariant; repetition count still ten.
    *_,a=aggregate_frames(rows)
    *_,b=aggregate_frames(pd.concat([rows]*10,ignore_index=True))
    assert np.allclose(a.macro_f1_mean,b.macro_f1_mean)
    assert np.allclose(a.macro_f1_sd,b.macro_f1_sd)


def test_load_study_rejects_scope_filtering(tmp_path):
    import json
    from cora_maple.report import load_study
    from cora_maple.config import load_config, ROOT, PROTOCOL
    folder=tmp_path/'studies'/'primary';folder.mkdir(parents=True)
    cfg=load_config(ROOT/'configs/repeated_cv.yaml')
    original={'protocol':PROTOCOL,'tasks':{'a':{'id':'a','kind':'fit','method':'outside_scope'}},'evaluations':['a']}
    (folder/'plan.json').write_text(json.dumps(original))
    (folder/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='unsupported'):
        load_study(tmp_path,'primary')
    assert json.loads((folder/'plan.json').read_text())==original
