"""Additional unlabelled-pool sensitivity with fixed one-label/class targets.

This module adds an experiment; it does not modify the primary training algorithm.
Pool membership is selected without unseen class labels. Known labelled inputs are
always included (their labels are NOT passed to pretraining). Percentages refer to
the entire unique outer-training pool, INCLUDING these fixed anchor inputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import load_processed
from .models import Classifier, PatchEncoder
from .perturb import paired_test_suite, seed_from
from .run import score
from .train import pretrain, predict, train_classifier

ROOT = Path(__file__).resolve().parents[2]


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_pretraining_pools(
    full_ids: np.ndarray, labelled_ids: np.ndarray, fractions: list[float], seed: int,
) -> dict[float, np.ndarray]:
    """Select nested inputs using identities only, never unlabelled class labels.

    The five already-labelled examples are present at each size. Relative ordering
    remains the same as in full_ids, preserving the original 100% training input.
    A shared random permutation selects only the *additional* pool members.
    """
    full_ids = np.asarray(full_ids, dtype=np.int64)
    labelled_ids = np.asarray(labelled_ids, dtype=np.int64)
    if len(np.unique(full_ids)) != len(full_ids):
        raise ValueError('full_ids must be unique canonical recording indices.')
    if len(np.unique(labelled_ids)) != len(labelled_ids):
        raise ValueError('labelled_ids must not contain duplicates.')
    if not set(labelled_ids).issubset(set(full_ids)):
        raise ValueError('Labelled observations must belong to the outer training pool.')
    if any(not (0 < float(f) <= 1) for f in fractions):
        raise ValueError('Pretraining fractions must lie in (0,1].')
    rng = np.random.default_rng(seed)
    candidates = full_ids[~np.isin(full_ids, labelled_ids)]
    permutation = rng.permutation(candidates)
    pools: dict[float, np.ndarray] = {}
    for fraction in sorted(set(float(f) for f in fractions)):
        n = len(full_ids) if fraction == 1 else math.floor(fraction * len(full_ids))
        if n < len(labelled_ids):
            raise ValueError('Fraction has insufficient capacity for the fixed labelled inputs.')
        selected = np.concatenate([labelled_ids, permutation[:n - len(labelled_ids)]])
        pools[fraction] = full_ids[np.isin(full_ids, selected)]
        assert len(pools[fraction]) == n
    return pools


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_source_roles(cfg: dict[str, Any], fold: int):
    """Reuse saved identities rather than regenerate folds with a new sklearn version."""
    d = load_processed(_resolve(cfg['data_path']), cfg['observation_samples'])
    role_path = _resolve(cfg['source_run']) / 'tep' / f'fold_{fold}' / 'seed_0/provenance.json'
    provenance = json.loads(role_path.read_text())
    id_to_index = {str(identity): i for i, identity in enumerate(d.ids)}
    def indices(key: str) -> np.ndarray:
        return np.asarray([id_to_index[str(x)] for x in provenance[key]], dtype=np.int64)
    train = indices('train_ids'); test = indices('test_ids'); full = indices('ssl_ids')
    labelled = np.asarray([id_to_index[str(x)] for x in provenance['label_ids']['1']], dtype=np.int64)
    if set(train) & set(test) or not set(full).issubset(set(train)):
        raise ValueError('Invalid source role separation.')
    if set(d.hashes[full]) & set(d.hashes[test]):
        raise ValueError('An exact input overlaps training and test roles.')
    if not np.array_equal(np.bincount(d.y[labelled], minlength=len(d.classes)), np.ones(len(d.classes))):
        raise ValueError('The fixed supervised set must contain exactly one input per class.')
    return d, provenance, train, test, full, labelled


def make_test_cache(cfg: dict[str, Any], out: Path, fold: int) -> dict[str, Any]:
    """Create the exact original paired test suite once per fold, then memory-map it."""
    d, prov, train, test, full, labelled = read_source_roles(cfg, fold)
    cache = out / '_test_cache' / f'fold_{fold}'
    cache.mkdir(parents=True, exist_ok=True)
    conditions = [('clean', 0., 0)] + [(kind, float(s), draw)
        for kind in cfg['perturbations'] for s in cfg['severities'] if s != 0
        for draw in range(cfg['perturbation_draws'])]
    map_path = cache / 'test_inputs.npy'
    tensor = np.lib.format.open_memmap(map_path, mode='w+', dtype=np.float32,
        shape=(len(conditions),len(test),d.X.shape[1],d.X.shape[2]))
    diagnostics = []
    for j,(kind,s,draw,x,diag) in enumerate(paired_test_suite(
            d.X[test],d.ids[test],cfg['perturbations'],cfg['severities'],
            cfg['perturbation_draws'],seed_from(cfg['partition_seed'],'tep',fold))):
        if (kind,s,draw) != conditions[j]:
            raise RuntimeError('Unexpected test-condition order.')
        tensor[j] = x
        diagnostics.extend({'fold':fold,'perturbation':kind,'severity':s,'draw':draw,**r} for r in diag)
    tensor.flush(); del tensor
    pd.DataFrame(diagnostics).to_csv(cache/'diagnostics.csv.gz',index=False,compression='gzip')
    keys = [f'{kind}|{s}|{draw}' for kind,s,draw in conditions]
    np.savez_compressed(cache/'metadata.npz',conditions=np.asarray(keys),y=d.y[test],ids=d.ids[test],classes=d.classes)
    manifest = {'fold':fold,'n_train':len(train),'n_test':len(test),'n_pool':len(full),
        'n_labelled':len(labelled),'conditions':keys,'source_roles_sha256':digest_file(
        _resolve(cfg['source_run'])/'tep'/f'fold_{fold}'/'seed_0/provenance.json'),
        'test_inputs_sha256':digest_file(map_path),'metadata_sha256':digest_file(cache/'metadata.npz')}
    (cache/'cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def _fit_one(job: dict[str, Any]) -> dict[str, Any]:
    cfg=job['cfg'];out=Path(job['out']);fold=int(job['fold']);method=job['method']
    fraction=float(job['fraction']);repeat=int(job['pool_repeat'])
    directory=out/'tep'/f'fold_{fold}'/f'pool_{repeat}'/f'pct_{int(round(100*fraction)):03}'/method
    if (directory/'complete.json').exists():
        return json.loads((directory/'complete.json').read_text())
    d,source,train,test,full,labelled=read_source_roles(cfg,fold)
    pool_seed=seed_from(cfg['pool_seed_base'],'tep',fold,repeat,'unlabelled_pool')
    members=nested_pretraining_pools(full,labelled,cfg['pool_fractions'],pool_seed)[fraction]
    directory.mkdir(parents=True,exist_ok=True)
    original_seed=int(source['seed'])
    pre_seed=seed_from(original_seed,'pretrain'); fine_seed=seed_from(original_seed,'finetune',1)
    provenance={'fold':fold,'pool_repeat':repeat,'fraction':fraction,'pool_seed':pool_seed,
        'pretrain_seed':pre_seed,'finetune_seed':fine_seed,'method':method,
        'policy':cfg['pool_policy'],'full_pool_n':len(full),'selected_pool_n':len(members),
        'additional_unlabelled_n':len(members)-len(labelled),'k':1,
        'selected_ids':d.ids[members].tolist(),'labelled_ids':d.ids[labelled].tolist(),
        'test_ids':d.ids[test].tolist(),
        'classes_used_to_select_pool':False,'training_labels_used_in_pretraining':False,
        'is_primary_endpoint_reuse':False}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    start=time.perf_counter()
    pretrained=pretrain(d.X[members],d.X.shape[1],cfg,pre_seed,method,directory/'pretrain')
    model,fit=train_classifier(d.X[labelled],d.y[labelled],len(d.classes),cfg,fine_seed,
        pretrained['state'],False,directory/'k_1')
    cache=out/'_test_cache'/f'fold_{fold}'
    xs=np.load(cache/'test_inputs.npy',mmap_mode='r')
    meta=np.load(cache/'metadata.npz',allow_pickle=False)
    preds=[];rows=[]
    for j,key in enumerate(meta['conditions']):
        kind,s,draw=str(key).split('|')
        # Copy each condition, not the full memory map, to obtain a writable torch input.
        pred=predict(model,np.array(xs[j],copy=True)).argmax(1).astype(np.int16)
        preds.append(pred)
        rows.append({'dataset':'tep','fold':fold,'pool_repeat':repeat,'fraction':fraction,
            'pool_n':len(members),'additional_unlabelled_n':len(members)-len(labelled),
            'method':method,'k':1,'perturbation':kind,'severity':float(s),'draw':int(draw),
            'n_test':len(test),'source':'new_fit',**score(meta['y'],pred,len(d.classes))})
    np.savez_compressed(directory/'predictions.npz',predictions=np.stack(preds),
        conditions=meta['conditions'],ids=meta['ids'],y=meta['y'],classes=meta['classes'])
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    summary={'fold':fold,'pool_repeat':repeat,'fraction':fraction,'pool_n':len(members),
        'method':method,'pretrain_seconds':pretrained['seconds'],'finetune_seconds':fit['seconds'],
        'train_accuracy':fit['train_accuracy'],'train_loss':fit['train_loss'],
        'pretrain_stats':pretrained['stats'],'elapsed_seconds':time.perf_counter()-start,
        'classifier_sha256':digest_file(directory/'k_1/classifier.pt'),
        'predictions_sha256':digest_file(directory/'predictions.npz'),
        'clean_f1':rows[0]['macro_f1'],
        'spurious_f1_s03':float(np.mean([r['macro_f1'] for r in rows if r['perturbation']=='spurious' and r['severity']==.3]))}
    (directory/'complete.json').write_text(json.dumps(summary,indent=2))
    return summary


def reuse_full_endpoints(cfg: dict[str,Any],out:Path) -> list[dict[str,Any]]:
    """Replay the original full-pool model on regenerated inputs; do not count duplicates."""
    rows=[];checks=[]
    torch.set_num_threads(cfg['threads'])
    for fold in cfg['folds']:
        d,source,tr,te,full,labelled=read_source_roles(cfg,fold)
        cache=out/'_test_cache'/f'fold_{fold}'
        xs=np.load(cache/'test_inputs.npy',mmap_mode='r');meta=np.load(cache/'metadata.npz')
        for method in cfg['methods']:
            original=_resolve(cfg['source_run'])/'tep'/f'fold_{fold}'/'seed_0'/method
            checkpoint=torch.load(original/'k_1/classifier.pt',map_location='cpu',weights_only=False)
            model=Classifier(PatchEncoder(checkpoint['n_tags'],**checkpoint['model']),checkpoint['n_classes'])
            model.load_state_dict(checkpoint['state_dict'])
            predictions=np.load(original/'k_1/predictions.npz')
            if not np.array_equal(predictions['ids'],meta['ids']) or not np.array_equal(predictions['conditions'],meta['conditions']):
                raise ValueError('Original test identity or condition mismatch.')
            for j,key in enumerate(meta['conditions']):
                yhat=predict(model,np.array(xs[j],copy=True)).argmax(1).astype(np.int16)
                if not np.array_equal(yhat,predictions['predictions'][j]):
                    raise AssertionError(f'Full endpoint replay mismatch: fold={fold},method={method},condition={key}')
                kind,s,draw=str(key).split('|')
                rows.append({'dataset':'tep','fold':fold,'pool_repeat':-1,'fraction':1.,'pool_n':len(full),
                    'additional_unlabelled_n':len(full)-len(labelled),'method':method,'k':1,
                    'perturbation':kind,'severity':float(s),'draw':int(draw),'n_test':len(te),
                    'source':'reused_primary',**score(meta['y'],yhat,len(d.classes))})
            checks.append({'fold':fold,'method':method,'n_conditions':len(meta['conditions']),
                'n_predictions':int(predictions['predictions'].size),'all_exact':True,
                'classifier_path':str((original/'k_1/classifier.pt').relative_to(ROOT)),
                'classifier_sha256':digest_file(original/'k_1/classifier.pt'),
                'predictions_path':str((original/'k_1/predictions.npz').relative_to(ROOT)),
                'predictions_sha256':digest_file(original/'k_1/predictions.npz')})
    pd.DataFrame(rows).to_csv(out/'full_pool_reused_metrics.csv',index=False)
    (out/'full_pool_replay.json').write_text(json.dumps(checks,indent=2))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/tep_unlabelled_scaling.yaml')
    parser.add_argument('--output',default='results/tep_unlabelled_scaling')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(_resolve(args.config).read_text());out=_resolve(args.output)
    if (out/'run_manifest.json').exists() and not args.resume:
        raise SystemExit('Existing execution; select another output path or use --resume.')
    out.mkdir(parents=True,exist_ok=True)
    lock=yaml.safe_dump(cfg,sort_keys=True)
    if (out/'config.lock.yaml').exists() and (out/'config.lock.yaml').read_text()!=lock:
        raise SystemExit('Locked configuration differs.')
    (out/'config.lock.yaml').write_text(lock)
    if cfg['seeds'] != [0]:
        raise SystemExit('This controlled add-on reuses the primary seed_0 roles; independent training-seed replication requires a separate experiment.')
    if len(set(cfg['pool_repeats'])) != len(cfg['pool_repeats']) or len(set(cfg['pool_fractions'])) != len(cfg['pool_fractions']):
        raise SystemExit('Pool repeats and fractions must be unique.')
    if cfg['datasets']!=['tep'] or cfg['label_budgets']!=[1]:
        raise SystemExit('This experiment is TEP-only at fixed k=1.')
    if 1. not in cfg['pool_fractions']:
        raise SystemExit('The comparison requires the original full-pool endpoint.')
    manifest={'experiment':'additional unlabelled-data scaling at k=1',
        'start_utc':pd.Timestamp.now(tz='UTC').isoformat(),'config_sha256':hashlib.sha256(lock.encode()).hexdigest(),
        'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'platform':platform.platform(),'workers':cfg['workers'],'threads_per_worker':cfg['threads'],
        'fitting_policy':'fixed schedules, same seeds within fold, only pool selection varies',
        'source_data_sha256':digest_file(_resolve(cfg['data_path'])),
        'source_run_manifest_sha256':digest_file(_resolve(cfg['source_run'])/'run_manifest.json'),
        'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest(),
        'selection_policy':cfg['pool_policy'],'status':'running'}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2));start=time.perf_counter()
    for fold in cfg['folds']:
        cache=out/'_test_cache'/f'fold_{fold}'
        if not args.resume or not (cache/'cache_manifest.json').exists() or not (cache/'test_inputs.npy').exists():
            c=make_test_cache(cfg,out,fold);print('CACHE',c['fold'],c['n_pool'],c['n_test'],flush=True)
    if not args.resume or not (out/'full_pool_replay.json').exists():
        print('Verifying full-pool reference predictions...',flush=True)
        checks=reuse_full_endpoints(cfg,out);print('Full-pool replay exact:',sum(c['n_predictions'] for c in checks),flush=True)
    jobs=[dict(cfg=cfg,out=str(out),fold=fold,pool_repeat=rep,fraction=fraction,method=method)
        for fraction in cfg['pool_fractions'] if fraction < 1
        for rep in cfg['pool_repeats'] for fold in cfg['folds'] for method in cfg['methods']]
    if args.resume:
        jobs=[j for j in jobs if not (out/'tep'/f'fold_{j["fold"]}'/f'pool_{j["pool_repeat"]}'/f'pct_{int(round(j["fraction"]*100)):03}'/j['method']/'complete.json').exists()]
    print(f'Executing {len(jobs)} new fits with {cfg["workers"]} workers',flush=True)
    ctx=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=ctx) as pool:
        futures=[pool.submit(_fit_one,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            v=f.result();print(f'{i}/{len(jobs)} fold={v["fold"]} pool={v["pool_repeat"]} p={v["fraction"]:.2f} {v["method"]} clean={v["clean_f1"]:.4f} spur={v["spurious_f1_s03"]:.4f} secs={v["elapsed_seconds"]:.1f}',flush=True)
    metrics=[pd.read_csv(out/'full_pool_reused_metrics.csv')]+[
        pd.read_csv(p) for p in sorted((out/'tep').glob('fold_*/pool_*/pct_*/*/metrics.csv'))]
    all_metrics=pd.concat(metrics,ignore_index=True);all_metrics.to_csv(out/'metrics.csv',index=False)
    summaries=[json.loads(p.read_text()) for p in sorted((out/'tep').glob('fold_*/pool_*/pct_*/*/complete.json'))]
    pd.json_normalize(summaries,sep='_').to_csv(out/'training_summary.csv',index=False)
    manifest.update(status='complete',end_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.perf_counter()-start,
        n_new_pretraining_runs=len(summaries),n_new_classifiers=len(summaries),n_reused_full_pool_models=2*len(cfg['folds']),
        n_metric_rows=len(all_metrics),n_new_metric_rows=int((all_metrics.source=='new_fit').sum()))
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',json.dumps(manifest),flush=True)

if __name__=='__main__':
    main()
