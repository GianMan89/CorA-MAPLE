"""Conference tables plus the six neural conditions and all per-operator results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import METHODS, BASELINES, PROPOSED, operator_groups, OPERATOR_LABELS
from .common import atomic_json


def label(name):
    return METHODS[name]["label"] if name in METHODS else name


def texlabel(name):
    return r"\emph{"+label(name).replace('_', r'\_')+"}"


def get(summary, method, k, op, s, pool="full"):
    x = summary[(summary.method==method)&(summary.k==k)&(summary.operator==op)
                & np.isclose(summary.severity,s)&(summary.pool.astype(str)==str(pool))]
    if len(x)!=1:
        raise ValueError(f"Expected one result for {method}, k={k}, {op}, s={s}, pool={pool}: {len(x)}")
    return x.iloc[0]


def cell(row, best=False):
    mean = f"{row.macro_f1_mean:.3f}"
    if best:
        mean = r"\mathbf{"+mean+"}"
    sd = f"{row.macro_f1_sd:.3f}" if np.isfinite(row.macro_f1_sd) else r"\text{--}"
    return "$"+mean+r"\pm"+sd+"$"


def _main_table(summary, cfg, methods, path):
    k,s=cfg["reporting"]["headline_k"],cfg["reporting"]["headline_severity"]
    ops=cfg["test"]["operators"]
    pre=set(cfg['corruption']['pretraining']['operators'])
    short={'false_full':'False/full','missing_full':'Missing/full','timing':'Timing','late_start':'Late start','missing':'Missing/event','spurious':'Spurious/event'}
    headers=['Clean']+[short[x]+('$^{P}$' if x in pre else '$^{U}$') for x in ops]
    rows=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l'+ 'c'*(len(ops)+1) + r'@{}}',r'\toprule',
          'Method & '+' & '.join(headers)+r' \\',r'\midrule']
    conditions=[('clean',0.)]+[(o,s) for o in ops]
    maxima={o:max(get(summary,m,k,o,v).macro_f1_mean for m in methods) for o,v in conditions}
    for m in methods:
        cells=[cell(get(summary,m,k,o,v),np.isclose(get(summary,m,k,o,v).macro_f1_mean,maxima[o],atol=1e-12,rtol=0)) for o,v in conditions]
        rows.append(texlabel(m)+' & '+' & '.join(cells)+r' \\')
    rows += [r'\bottomrule',r'\end{tabular*}']
    path.write_text('\n'.join(rows)+'\n')


def _factorial(summary,cfg,folder):
    group_path=folder/'group_summary.csv'
    if not group_path.exists():
        return
    group=pd.read_csv(group_path)
    k,s=cfg['reporting']['headline_k'],cfg['reporting']['headline_severity']
    names=[m for m in cfg['study']['methods'] if m in METHODS]
    rows=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllccc@{}}',r'\toprule',
          rf'Method & Pretraining & Classifier & Clean & Trained, $s={s:g}$ & Test-only, $s={s:g}$ \\',r'\midrule']
    for m in names:
        condition=METHODS[m]
        clean=get(summary,m,k,'clean',0.)
        tail=[]
        for g in ['trained','test_only']:
            x=group[(group.method==m)&(group.k==k)&(group.pool.astype(str)=='full')&np.isclose(group.severity,s)&(group.group==g)]
            if len(x)!=1:
                raise ValueError('Incomplete pretraining-comparison group summary')
            tail.append(cell(x.iloc[0]))
        p={'none':'None','clean':'Masked SSL','corrupted':'Corrupted SSL'}[condition['pretraining_state']]
        sup='Clean CE'
        rows.append(texlabel(m)+' & '+p+' & '+sup+' & '+cell(clean)+' & '+' & '.join(tail)+r' \\')
    rows += [r'\bottomrule',r'\end{tabular*}']
    (folder/'tables/ablation_factorial_table.tex').write_text('\n'.join(rows)+'\n')


def make_assets(summary: pd.DataFrame,cfg:dict,folder:Path):
    tables,figures=folder/'tables',folder/'figures'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    kind=cfg['study']['kind'];s=cfg['reporting']['headline_severity'];k=cfg['reporting']['headline_k']
    if kind=='primary':
        names=cfg['study']['methods']
        # Complete scientific record, plus a conference-length seven-method option.
        _main_table(summary,cfg,names,tables/'robustness_all_table.tex')
        selected=[m for m in cfg['reporting']['primary_table_methods'] if m in names]
        if selected:_main_table(summary,cfg,selected,tables/'robustness_main_table.tex')
        _factorial(summary,cfg,folder)
        from .plotting import render_profiles
        render_profiles(summary,cfg,figures)
        facts={}
        for m in names:
            facts[label(m)]={}
            for op,v in [('clean',0.)]+[(o,s) for o in cfg['test']['operators']]:
                row=get(summary,m,k,op,v)
                facts[label(m)][op]={'k':k,'severity':v,'mean':float(row.macro_f1_mean),
                    'sd':float(row.macro_f1_sd) if np.isfinite(row.macro_f1_sd) else None,
                    'n_repetitions':int(row.n_repeats)}
        atomic_json(folder/'numeric_facts.json',facts)
        summary.to_csv(folder/'paper_numeric_summary.csv',index=False)
    elif kind=='pool':
        op=cfg['reporting']['figure_operator']
        rows=[r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',
              r'& \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{'+{'false_full':'False/full','missing_full':'Missing/full','timing':'Timing','late_start':'Late start','missing':'Missing/events','spurious':'Spurious/events'}[op]+f', $s={s:g}$'+r'} \\',
              r'$H_U$ [h] & E & F & E & F \\',r'\midrule']
        for pool in cfg['study']['pool_sizes']:
            if pool=='full':rows.append(r'\midrule')
            text='Full' if pool=='full' else f"{int(pool)*cfg['data']['observation_samples']*cfg['data']['sample_period_minutes']/60:g}"
            cells=[cell(get(summary,m,1,o,v,pool)) for o,v in [('clean',0.),(op,s)] for m in PROPOSED]
            rows.append(text+' & '+' & '.join(cells)+r' \\')
        rows += [r'\bottomrule',r'\end{tabular*}']
        (tables/'unlabelled_pool_table_compact.tex').write_text('\n'.join(rows)+'\n')
        if (folder/'group_summary.csv').exists():
            group=pd.read_csv(folder/'group_summary.csv')
            lines=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',
                r'& \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Trained, $s='+f'{s:g}'+r'$} & \multicolumn{2}{c}{Test-only, $s='+f'{s:g}'+r'$} \\',
                r'$H_U$ [h] & E & F & E & F & E & F \\',r'\midrule']
            for pool in cfg['study']['pool_sizes']:
                text='Full' if pool=='full' else f"{int(pool)*cfg['data']['observation_samples']*cfg['data']['sample_period_minutes']/60:g}"
                vals=[cell(get(summary,m,1,'clean',0.,pool)) for m in PROPOSED]
                for g in ['trained','test_only']:
                    for m in PROPOSED:
                        rr=group[(group.method==m)&(group.k==1)&(group.pool.astype(str)==str(pool))&np.isclose(group.severity,s)&(group.group==g)]
                        if len(rr)!=1:raise ValueError('Incomplete history group')
                        vals.append(cell(rr.iloc[0]))
                lines.append(text+' & '+' & '.join(vals)+r' \\')
            lines += [r'\bottomrule',r'\end{tabular*}']
            (tables/'unlabelled_pool_by_group.tex').write_text('\n'.join(lines)+'\n')
    atomic_json(folder/'ASSET_PROVENANCE.json',{'protocol':cfg['protocol'],'study':cfg['study']['name'],
        'n_splits':cfg['cv']['n_splits'],'n_repeats':cfg['cv']['n_repeats'],'headline_severity':s,
        'aggregation':'Perturbation draws and pool draws, then folds within repetition; mean/sample SD of repetition means.',
        'group_definition':operator_groups(cfg),'event_initial_state':cfg['corruption']['event_initial_state'],
        'delay_policy':cfg['corruption']['delay_policy'],'hyperparameter_selection':'Fixed for the configured study; no score-dependent checkpoint selection. Repeated partitions of one dataset do not constitute independent external validation.'})
