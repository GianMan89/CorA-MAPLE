#!/usr/bin/env python3
"""Regenerate paper assets from a complete report snapshot for the declared protocol.

For normal runs use `python -m cora_maple export-paper --workspace ...` instead.
Reports must match the configured methods, conditions, and scientific settings.
"""
from __future__ import annotations
import argparse,json,shutil,sys,tempfile,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
import numpy as np
import pandas as pd
from cora_maple.common import atomic_json,file_hash
from cora_maple.config import PROTOCOL,validate_config,BASELINES
from cora_maple.report import aggregate_frames,write_group_summaries,make_assets
from cora_maple.transforms import test_conditions


def validate_report(folder:Path):
    status_path=folder/'REPORT_STATUS.json';config_path=folder/'config.json'
    if not status_path.exists() or not config_path.exists():
        raise ValueError(f'A complete report and its resolved configuration are required: {folder}')
    status=json.loads(status_path.read_text());cfg=json.loads(config_path.read_text())
    validate_config(cfg)
    if not status.get('complete') or status.get('protocol')!=PROTOCOL:
        raise ValueError('Only completed reports matching the declared protocol are accepted.')
    for filename,digest in status.get('files',{}).items():
        path=(folder/filename).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_hash(path)!=digest:
            raise ValueError(f'Recorded report content was altered: {filename}')
    rows=pd.read_csv(folder/'metrics_by_draw.csv',dtype={'pool':str})
    if set(rows.method)!=set(cfg['study']['methods']):
        raise ValueError('Recorded method set differs; no automatic subset filtering is allowed')
    expected={f'{op}|{s:g}|{draw}' for op,s,draw in test_conditions(cfg)}
    if rows.duplicated(['task_id','condition']).any() or any(set(x)!=expected for _,x in rows.groupby('task_id').condition):
        raise ValueError('Incomplete or duplicated perturbation conditions')
    # Exact classifier grid, including nested pools and crossed diagnostics.
    config_study=cfg['study'];expected_grid=set()
    for method,pool in itertools.product(config_study['methods'],config_study['pool_sizes']):
        models=[0] if method in BASELINES else config_study['model_seeds']
        draws=[0] if pool=='full' else config_study['pool_draws']
        for rep,fold,k,draw,label,model in itertools.product(range(cfg['cv']['n_repeats']),range(cfg['cv']['n_splits']),config_study['label_budgets'],draws,config_study['label_draws'],models):
            expected_grid.add((rep,fold,method,k,str(pool),draw,label,model))
    columns=['repeat','fold','method','k','pool','pool_draw','label_draw','model_rep']
    unique=rows[['task_id']+columns].drop_duplicates()
    if unique.duplicated(columns).any() or set(unique[columns].itertuples(index=False,name=None))!=expected_grid:
        raise ValueError('Recorded classifier grid differs from the frozen study')
    per_fit,per_split,per_repeat,summary=aggregate_frames(rows)
    if not (summary.n_repeats==cfg['cv']['n_repeats']).all():
        raise ValueError('Incomplete repetition coverage')
    counts=per_split.groupby(['repeat','method','k','pool','operator','severity']).fold.nunique()
    if not (counts==cfg['cv']['n_splits']).all():raise ValueError('Incomplete fold coverage')
    errors={}
    for name,new in [('per_fit',per_fit),('per_split',per_split),('per_repeat',per_repeat),('summary',summary)]:
        old=pd.read_csv(folder/f'{name}.csv',dtype={'pool':str})
        cols=['macro_f1_mean','macro_f1_sd','accuracy_mean','accuracy_sd'] if name=='summary' else ['macro_f1','accuracy']
        keys=[c for c in new if c not in cols+['n_repeats']]
        merged=new.merge(old,on=keys,suffixes=('_new','_old'),how='outer',indicator=True,validate='one_to_one')
        if len(merged)!=len(new) or not merged['_merge'].eq('both').all():raise ValueError(f'{name} identities differ')
        err=0.
        for c in cols:
            a,b=merged[c+'_new'],merged[c+'_old']
            if not np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True):raise ValueError(f'{name} numeric values differ')
            finite=(a-b).abs().dropna()
            if len(finite):err=max(err,float(finite.max()))
        errors[name]=err
    return cfg,summary,per_split,{'protocol':PROTOCOL,'complete':True,'metric_rows':len(rows),
        'max_aggregation_errors':errors,'scope':'Recorded metric aggregation, not model training or prediction replay.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True,help='Directory with complete primary/ and pool_sizes/ reports')
    p.add_argument('--destination',type=Path,default=ROOT/'paper/generated')
    p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()) and not args.overwrite:
        p.error('Use an empty destination or --overwrite')
    # Validate both studies before writing either one.
    loaded={name:validate_report(args.reports/name) for name in ['primary','pool_sizes']}
    for name,(cfg,summary,splits,_) in loaded.items():
        if cfg['study']['name']!=name:raise ValueError('Report folder and study name differ')
    def common(v):return {k:x for k,x in v.items() if k!='study'}
    if common(loaded['primary'][0])!=common(loaded['pool_sizes'][0]):
        raise ValueError('Primary and pool reports do not share the same scientific settings')
    with tempfile.TemporaryDirectory(prefix='cora-v2_1-assets-') as temporary:
        for name,(cfg,summary,splits,checks) in loaded.items():
            out=Path(temporary)/name;out.mkdir()
            write_group_summaries(splits,cfg,out);make_assets(summary,cfg,out)
            for category in ['tables','figures']:
                if (out/category).exists():shutil.copytree(out/category,args.destination/category,dirs_exist_ok=True)
            for artifact in out.glob('*'):
                if artifact.is_file() and artifact.name in ['ASSET_PROVENANCE.json','numeric_facts.json','paper_numeric_summary.csv']:
                    filename=f'{name}_provenance.json' if artifact.name=='ASSET_PROVENANCE.json' else artifact.name
                    args.destination.mkdir(parents=True,exist_ok=True);shutil.copy2(artifact,args.destination/filename)
    atomic_json(args.destination/'RESULTS_STATUS.json',{'protocol':PROTOCOL,'complete':True,
      'n_splits':loaded['primary'][0]['cv']['n_splits'],'n_repeats':loaded['primary'][0]['cv']['n_repeats'],
      'aggregation_checks':{k:v[3] for k,v in loaded.items()},'model_fitting_performed':False,'prediction_replay_performed':False})
    print(json.dumps({'destination':str(args.destination),'complete':True,'protocol':PROTOCOL},indent=2))

if __name__=='__main__':main()
