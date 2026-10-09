"""Absolute unlabelled-source scaling, conditional on the primary k=1 split.

N_U counts distinct pretraining recordings, not examples per class. Pools are
uniform nested subsets of the outer training fold. The five labelled fine-tuning
inputs are not forced into the pool; N_U=1 and 2 are thus possible without hidden
class access. Source sampling within SSL minibatches remains with replacement.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, multiprocessing, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .scaling import ROOT, _resolve, digest_file, read_source_roles, make_test_cache, reuse_full_endpoints
from .train import pretrain, train_classifier, predict
from .perturb import seed_from
from .run import score


def nested_absolute_pools(full_ids: np.ndarray, sizes: list[int], seed: int) -> dict[int,np.ndarray]:
    """Uniform membership without class access; preserve canonical source order."""
    full=np.asarray(full_ids,dtype=np.int64)
    if full.ndim!=1 or len(np.unique(full))!=len(full):
        raise ValueError('full_ids must be a vector of unique source indices.')
    if any(isinstance(n,bool) or int(n)!=n or n<1 or n>len(full) for n in sizes):
        raise ValueError('Each pool size must be an integer in [1,len(full_ids)].')
    order=np.random.default_rng(seed).permutation(full)
    return {int(n):full[np.isin(full,order[:int(n)])] for n in sorted(set(sizes))}


def fit_one(job:dict) -> dict:
    cfg=job['cfg'];out=Path(job['out']);fold=int(job['fold']);method=job['method']
    size=int(job['pool_n']);repeat=int(job['pool_repeat'])
    directory=out/'tep'/f'fold_{fold}'/f'pool_{repeat}'/f'n_{size:04d}'/method
    if (directory/'complete.json').exists(): return json.loads((directory/'complete.json').read_text())
    d,source,train,test,full,labelled=read_source_roles(cfg,fold)
    pool_seed=seed_from(cfg['pool_seed_base'],'tep',fold,repeat,'absolute_unlabelled_pool')
    members=nested_absolute_pools(full,cfg['pool_sizes'],pool_seed)[size]
    directory.mkdir(parents=True,exist_ok=True)
    seed=int(source['seed']);pre_seed=seed_from(seed,'pretrain');fine_seed=seed_from(seed,'finetune',1)
    overlap=int(np.isin(members,labelled).sum())
    # Class coverage is an after-selection diagnostic and never changes membership.
    provenance={'fold':fold,'pool_repeat':repeat,'pool_n':size,'pool_seed':pool_seed,
        'pretrain_seed':pre_seed,'finetune_seed':fine_seed,'method':method,
        'policy':cfg['pool_policy'],'full_pool_n':len(full),'selected_pool_n':len(members),
        'labelled_overlap_n':overlap,'k':1,'selected_ids':d.ids[members].tolist(),
        'labelled_ids':d.ids[labelled].tolist(),'test_ids':d.ids[test].tolist(),
        'classes_used_to_select_pool':False,'training_labels_used_in_pretraining':False,
        'posthoc_class_counts':np.bincount(d.y[members],minlength=len(d.classes)).tolist(),
        'is_primary_endpoint_reuse':False}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    start=time.perf_counter()
    pretrained=pretrain(d.X[members],d.X.shape[1],cfg,pre_seed,method,directory/'pretrain')
    model,fit=train_classifier(d.X[labelled],d.y[labelled],len(d.classes),cfg,fine_seed,
                              pretrained['state'],False,directory/'k_1')
    cache=out/'_test_cache'/f'fold_{fold}';xs=np.load(cache/'test_inputs.npy',mmap_mode='r')
    meta=np.load(cache/'metadata.npz',allow_pickle=False);preds=[];rows=[]
    for j,key in enumerate(meta['conditions']):
        kind,s,draw=str(key).split('|')
        pred=predict(model,np.array(xs[j],copy=True)).argmax(1).astype(np.int16)
        preds.append(pred)
        rows.append({'dataset':'tep','fold':fold,'pool_repeat':repeat,'pool_n':size,
            'budget':str(size),'labelled_overlap_n':overlap,'method':method,'k':1,
            'perturbation':kind,'severity':float(s),'draw':int(draw),'n_test':len(test),
            'source':'new_fit',**score(meta['y'],pred,len(d.classes))})
    np.savez_compressed(directory/'predictions.npz',predictions=np.stack(preds),
        conditions=meta['conditions'],ids=meta['ids'],y=meta['y'],classes=meta['classes'])
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    summary={'fold':fold,'pool_repeat':repeat,'pool_n':size,'method':method,
        'pretrain_seconds':pretrained['seconds'],'finetune_seconds':fit['seconds'],
        'train_accuracy':fit['train_accuracy'],'train_loss':fit['train_loss'],
        'pretrain_stats':pretrained['stats'],'elapsed_seconds':time.perf_counter()-start,
        'classifier_sha256':digest_file(directory/'k_1/classifier.pt'),
        'predictions_sha256':digest_file(directory/'predictions.npz'),'clean_f1':rows[0]['macro_f1'],
        'spurious_f1_s03':float(np.mean([r['macro_f1'] for r in rows if r['perturbation']=='spurious' and r['severity']==.3]))}
    (directory/'complete.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/tep_absolute_scaling.yaml')
    p.add_argument('--output',default='results/tep_absolute_scaling')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(_resolve(a.config).read_text());out=_resolve(a.output)
    if (out/'run_manifest.json').exists() and not a.resume:
        raise SystemExit('Use a new output path or --resume; existing runs are immutable.')
    out.mkdir(parents=True,exist_ok=True);lock=yaml.safe_dump(cfg,sort_keys=True)
    if (out/'config.lock.yaml').exists() and (out/'config.lock.yaml').read_text()!=lock:
        raise SystemExit('Locked configuration differs.')
    (out/'config.lock.yaml').write_text(lock)
    if cfg['datasets']!=['tep'] or cfg['label_budgets']!=[1] or cfg['seeds']!=[0]:
        raise SystemExit('This add-on conditions on the saved primary TEP k=1 seed_0 roles.')
    manifest={'experiment':'additional absolute unlabelled pool scaling at fixed k=1',
        'start_utc':pd.Timestamp.now(tz='UTC').isoformat(),'status':'running',
        'config_sha256':hashlib.sha256(lock.encode()).hexdigest(),'python':sys.version,
        'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'platform':platform.platform(),'workers':cfg['workers'],'threads_per_worker':cfg['threads'],
        'source_data_sha256':digest_file(_resolve(cfg['data_path'])),
        'selection_policy':cfg['pool_policy'],
        'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest()}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2));start=time.perf_counter()
    for fold in cfg['folds']:
        cache=out/'_test_cache'/f'fold_{fold}'
        if not a.resume or not (cache/'cache_manifest.json').exists() or not (cache/'test_inputs.npy').exists():
            c=make_test_cache(cfg,out,fold);print('CACHE',c['fold'],c['n_pool'],c['n_test'],flush=True)
    if not a.resume or not (out/'full_pool_replay.json').exists():
        checks=reuse_full_endpoints(cfg,out)
        print('Full-pool replay exact:',sum(c['n_predictions'] for c in checks),flush=True)
    jobs=[dict(cfg=cfg,out=str(out),fold=f,pool_repeat=r,pool_n=n,method=m)
          for n in cfg['pool_sizes'] for r in cfg['pool_repeats'] for f in cfg['folds'] for m in cfg['methods']]
    if a.resume:
        jobs=[j for j in jobs if not (out/'tep'/f'fold_{j["fold"]}'/f'pool_{j["pool_repeat"]}'/f'n_{j["pool_n"]:04d}'/j['method']/'complete.json').exists()]
    print('Executing',len(jobs),'fits',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
        for i,fut in enumerate(concurrent.futures.as_completed([executor.submit(fit_one,j) for j in jobs]),1):
            v=fut.result();print(f'{i}/{len(jobs)} fold={v["fold"]} pool={v["pool_repeat"]} n={v["pool_n"]} {v["method"]} clean={v["clean_f1"]:.4f} spur={v["spurious_f1_s03"]:.4f} secs={v["elapsed_seconds"]:.1f}',flush=True)
    reused=pd.read_csv(out/'full_pool_reused_metrics.csv').drop(columns=['fraction','additional_unlabelled_n'])
    reused['budget']='full';reused['labelled_overlap_n']=5
    metrics=[reused]+[pd.read_csv(p,dtype={'budget':str}) for p in sorted((out/'tep').glob('fold_*/pool_*/n_*/*/metrics.csv'))]
    data=pd.concat(metrics,ignore_index=True);data.to_csv(out/'metrics.csv',index=False)
    summaries=[json.loads(p.read_text()) for p in sorted((out/'tep').glob('fold_*/pool_*/n_*/*/complete.json'))]
    pd.json_normalize(summaries,sep='_').to_csv(out/'training_summary.csv',index=False)
    manifest.update(status='complete',end_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.perf_counter()-start,
        n_new_pretraining_runs=len(summaries),n_new_classifiers=len(summaries),n_reused_full_pool_models=2*len(cfg['folds']),
        n_metric_rows=len(data),n_new_metric_rows=int((data.source=='new_fit').sum()))
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
