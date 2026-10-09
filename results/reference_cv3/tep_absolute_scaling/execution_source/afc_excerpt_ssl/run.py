"""Executable source-separated TEP evaluation. Every reported score has saved instance predictions.
Fixed training budgets; neither test data nor test labels select checkpoints.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
from .data import load_processed,make_partition,nested_labels
from .train import pretrain,train_classifier,predict
from .perturb import paired_test_suite,seed_from
from afc_robustness.models.eac import EAC1NN
from afc_robustness.models.mbw_lr import MBWLogisticRegression
from afc_robustness.models.jaccard import Jaccard1NN
from afc_robustness.models.wdi import WDI1NN
from afc_robustness.models.acm_svm import ACMSVM
ROOT=Path(__file__).resolve().parents[2]

def baseline(name,seed):
    if name=='EAC-1NN':return EAC1NN(attenuation=.01)
    if name=='MBW-LR':return MBWLogisticRegression(C=1.)
    if name=='JAC-1NN':return Jaccard1NN()
    if name=='WDI-1NN':return WDI1NN()
    if name=='ACM-SVM':return ACMSVM(probability=False,random_state=seed)
    if name=='CASIM':
        from afc_robustness.models.casim import CASIM
        return CASIM(num_features=672,n_estimators=1,backend='sktime',random_state=seed)
    raise ValueError(name)

def base_predict(model,X):
    if isinstance(model,ACMSVM):
        # Native SVC classification; avoid small-sample Platt probability inversion.
        return model.clf_.predict(model.coactivation_features(X))
    return model.predict(X)

def score(y,pred,C):
    return {'macro_f1':f1_score(y,pred,labels=np.arange(C),average='macro',zero_division=0),
            'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred)}

def run(cfg:dict,out:Path,datasets:list[str]|None=None):
    selected = datasets or cfg['datasets']
    if selected != ['tep']:
        raise ValueError('This release evaluates TEP only.')
    if cfg.get('sample_period_minutes', 1.) != 1.:
        raise ValueError('Packaged data and perturbation parameters use one-minute sampling.')
    horizon = int(cfg.get('observation_samples', 60))
    if horizon % cfg['model']['patch_size']:
        raise ValueError('observation_samples must be divisible by patch_size.')
    if (out/'run_manifest.json').exists():
        raise FileExistsError('An execution already exists here. Use a new output directory to preserve provenance.')
    out.mkdir(parents=True,exist_ok=True); cfgpath=out/'config.lock.yaml'
    encoded=yaml.safe_dump(cfg,sort_keys=True)
    if cfgpath.exists() and cfgpath.read_text()!=encoded:raise RuntimeError('Output config differs; choose a new directory.')
    cfgpath.write_text(encoded)
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest()
    meta={'phase':cfg.get('phase','three_fold_reference'),'config_sha256':hashlib.sha256(encoded.encode()).hexdigest(),
          'code_sha256':code_hash,'python':sys.version,'torch':torch.__version__,'device':'cpu',
          'cpu':platform.processor(),'start_utc':pd.Timestamp.now(tz='UTC').isoformat(),
          'source_archive_sha256':json.loads((ROOT/'vendor/PROVENANCE.json').read_text())['archive_sha256'],
          'selection':'fixed schedule; final checkpoint; no performance-driven selection',
          'pretraining':'outer training recordings only; no held-out test inputs',
          'warning':'three_fold_reference cross-validation SD is descriptive; folds and draws are not independent'}
    (out/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    rows=[];stats=[];diags=[];start=time.perf_counter()
    for ds in datasets or cfg['datasets']:
        data_path = Path(cfg.get('data_path', 'data/processed/tep_60min.npz'))
        if not data_path.is_absolute(): data_path = ROOT / data_path
        D=load_processed(data_path, observation_samples=horizon);C=len(D.classes)
        parts=make_partition(D,seed=cfg['partition_seed'],n_splits=cfg['n_folds'])
        (out/f'{ds}_splits.json').write_text(json.dumps(parts,indent=2))
        for part in parts:
            fold=part['fold']
            if fold not in cfg['folds']:continue
            tr=np.array(part['train']);te=np.array(part['test']);ytest=D.y[te]
            _,u=np.unique(D.hashes[tr],return_index=True);ssl_ids=tr[u]
            print(f'BEGIN {ds} fold={fold} train={len(tr)} ssl_unique={len(u)} test={len(te)}',flush=True)
            suite=list(paired_test_suite(D.X[te],D.ids[te],cfg['perturbations'],cfg['severities'],cfg['perturbation_draws'],seed_from(cfg['partition_seed'],ds,fold)))
            for kind,s,draw,X,diag in suite:
                for r in diag:diags.append({'dataset':ds,'fold':fold,'perturbation':kind,'severity':s,'draw':draw,**r})
            for rep in cfg['seeds']:
                seed=seed_from(cfg['partition_seed'],ds,fold,rep)
                labels=nested_labels(D,tr,cfg['label_budgets'],seed_from(seed,'labels'))
                dest=out/ds/f'fold_{fold}'/f'seed_{rep}';dest.mkdir(parents=True,exist_ok=True)
                (dest/'provenance.json').write_text(json.dumps({'train_ids':D.ids[tr].tolist(),'test_ids':D.ids[te].tolist(),
                    'ssl_ids':D.ids[ssl_ids].tolist(),'label_ids':{k:D.ids[v].tolist() for k,v in labels.items()},'seed':seed},indent=2))
                states={}
                for method in cfg['methods']:
                    if method.startswith('scratch'):continue
                    pp=pretrain(D.X[ssl_ids],D.X.shape[1],cfg,seed_from(seed,'pretrain'),method,dest/method/'pretrain')
                    states[method]=pp['state'];stats.append({'dataset':ds,'fold':fold,'seed':rep,'method':method,'stage':'pretrain','seconds':pp['seconds'],**pp['stats']})
                    print(f'PRETRAIN {ds} f{fold} {method} {pp["seconds"]:.1f}s loss={pp["history"][-1]["loss"]:.4f}',flush=True)
                for k,li in labels.items():
                    for method in cfg['methods']+cfg['baselines']:
                        directory=dest/method/f'k_{k}';t=time.perf_counter()
                        if method in cfg['methods']:
                            model,st=train_classifier(D.X[li],D.y[li],C,cfg,seed_from(seed,'finetune',k),states.get(method),method=='scratch_aug',directory)
                            stats.append({'dataset':ds,'fold':fold,'seed':rep,'method':method,'k':k,'stage':'finetune',
                                'seconds':st['seconds'],'train_accuracy':st['train_accuracy'],'train_loss':st['train_loss']})
                            predict_fn=lambda x:predict(model,x).argmax(1)
                        else:
                            model=baseline(method,seed);model.fit(D.X[li],D.y[li]);directory.mkdir(parents=True,exist_ok=True)
                            predict_fn=lambda x:base_predict(model,x)
                            stats.append({'dataset':ds,'fold':fold,'seed':rep,'method':method,'k':k,'stage':'fit',
                                'seconds':time.perf_counter()-t,'train_accuracy':float((predict_fn(D.X[li])==D.y[li]).mean())})
                        preds=[];keys=[]
                        for kind,s,draw,X,diag in suite:
                            pred=predict_fn(X).astype(np.int16)
                            rows.append({'dataset':ds,'fold':fold,'seed':rep,'method':method,'k':k,
                                'perturbation':kind,'severity':s,'draw':draw,'n_test':len(te),**score(ytest,pred,C)})
                            preds.append(pred);keys.append(f'{kind}|{s}|{draw}')
                        np.savez_compressed(directory/'predictions.npz',predictions=np.stack(preds),conditions=np.array(keys),ids=D.ids[te],y=ytest,classes=D.classes)
                        print(f'FIT {ds} f{fold} k{k} {method} clean={rows[-len(suite)]["macro_f1"]:.3f} total={time.perf_counter()-t:.1f}s',flush=True)
                    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(stats).to_csv(out/'training_summary.csv',index=False)
            del suite
        pd.DataFrame(diags).to_csv(out/'perturbation_diagnostics.csv.gz',index=False,compression='gzip')
    meta.update(end_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.perf_counter()-start,n_metric_rows=len(rows))
    (out/'run_manifest.json').write_text(json.dumps(meta,indent=2));print('COMPLETED',meta['elapsed_seconds'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/tep_primary.yaml');p.add_argument('--output',default='results/new_tep_primary');p.add_argument('--datasets',nargs='+')
    a=p.parse_args();run(yaml.safe_load(Path(a.config).read_text()),Path(a.output),a.datasets)
if __name__=='__main__':main()
