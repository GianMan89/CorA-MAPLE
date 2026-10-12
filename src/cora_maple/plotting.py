"""Vector manuscript figures with explicit training support and corruption scope."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from .config import PROPOSED, BASELINES, OPERATOR_LABELS, operator_groups

STYLES = (
    ('EAC-1NN','EAC-1NN','#1f77b4','o',(0,(5,2)),False),
    ('MBW-LR','MBW-LR','#ff7f0e','s',(0,(3.5,1.8)),False),
    ('JAC-1NN','JAC-1NN','#2ca02c','^',(0,(1,1.6)),False),
    ('WDI-1NN','WDI-1NN','#d62728','v',(0,(4.5,1.7,1.1,1.7)),False),
    ('ACM-SVM','ACM-SVM','#9467bd','D',(0,(2.3,1.5)),False),
    ('cora_maple_excerpt','CorA-MAPLE-E','#8c564b','p','-',True),
    ('cora_maple_full','CorA-MAPLE-F','#e377c2','P','-',True),
)


def _style():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white','axes.unicode_minus':False})


def _get(s,m,k,op,v):
    x=s[(s.method==m)&(s.k==k)&(s.pool.astype(str)=='full')&(s.operator==op)&np.isclose(s.severity,v)]
    if len(x)!=1:raise ValueError(f'Incomplete plotted result: {m}, {op}, {v}, k={k}')
    return float(x.iloc[0].macro_f1_mean),float(x.iloc[0].macro_f1_sd)


def _curve(ax,summary,cfg,op,kind='corrupt',narrow=False):
    styles=[v for v in STYLES if v[0] in set(summary.method)]
    k=cfg['reporting']['headline_k']
    xs=cfg['study']['label_budgets'] if kind=='clean' else cfg['test']['severities']
    for m,lab,color,marker,dash,proposed in styles:
        points=[_get(summary,m,x,'clean',0) for x in xs] if kind=='clean' else [_get(summary,m,k,'clean' if x==0 else op,x) for x in xs]
        ys,err=np.asarray(points).T
        err=np.where(np.isfinite(err),err,0.)
        ax.errorbar(xs,ys,yerr=err,fmt='none',ecolor=color,elinewidth=.8,capsize=1.7,alpha=.75,zorder=1)
        ax.plot(xs,ys,color=color,linestyle=dash,linewidth=(2.3 if proposed else 1.55),marker=marker,
                markersize=(4.8 if proposed else 3.8),markeredgecolor='white',markeredgewidth=.28,zorder=3 if proposed else 2)
    ax.set_ylim(0,1.04)
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.tick_params(labelsize=7.3 if narrow else 8.7,pad=2,width=.8)
    ax.grid(axis='y',alpha=.23,linewidth=.55);ax.grid(axis='x',alpha=.12,linewidth=.5)
    if kind=='clean':
        ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator(xs));ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xs]));ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(xs)*.85,max(xs)*1.18)
        ax.set_xlabel('Labelled episodes per class, $k$',fontsize=7.6 if narrow else 9)
    else:
        ax.set_xlim(-.018,max(xs)+.018);ax.set_xticks(xs)
        ax.set_xlabel('Corruption severity, $s$',fontsize=7.6 if narrow else 9)
        groups=operator_groups(cfg)
        if op in groups['trained']:
            ax.axvspan(0,.30,color='#2f9d62',alpha=.055,zorder=0)
            ax.axvline(.30,color='#53666f',linestyle=':',linewidth=1.1,zorder=1)
    for spine in ax.spines.values():spine.set_linewidth(.8)
    return styles


def _handles(styles):
    return [Line2D([],[],label=n,color=c,marker=m,linestyle=l,linewidth=2.3 if p else 1.55,
                   markersize=4.8 if p else 3.8) for _,n,c,m,l,p in styles]


def _save(fig,out,stem):
    for extension in ['pdf','svg','png']:
        fig.savefig(out/f'{stem}.{extension}',dpi=220 if extension=='png' else None)
    plt.close(fig)


def render_profiles(summary:pd.DataFrame,cfg:dict,out:Path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);_style()
    op=cfg['reporting']['figure_operator']
    names=set(summary.method)
    if not set(PROPOSED)<=names:
        # A diagnostic may request only a subset; no false claim of full main figure.
        return
    fig=plt.figure(figsize=(180/25.4,2.03))
    a=fig.add_axes([.078,.28,.305,.655]);b=fig.add_axes([.420,.28,.305,.655],sharey=a)
    styles=_curve(a,summary,cfg,op,kind='clean');_curve(b,summary,cfg,op)
    a.set_ylabel('Macro-F1',fontsize=10,labelpad=4);b.tick_params(labelleft=False)
    b.set_xlabel('False/full severity, $s$',fontsize=9)
    fig.legend(handles=_handles(styles),loc='center left',bbox_to_anchor=(.754,.61),fontsize=8.3,
               frameon=False,borderaxespad=0,handlelength=2.3,labelspacing=.5)
    _save(fig,out,'figure1_two_columns')
    fig=plt.figure(figsize=(87.5/25.4,2.7))
    a=fig.add_axes([.17,.62,.465,.29]);b=fig.add_axes([.17,.185,.465,.29],sharey=a)
    _curve(a,summary,cfg,op,'clean',True);_curve(b,summary,cfg,op,narrow=True)
    fig.text(.027,.55,'Macro-F1',rotation=90,va='center',fontsize=8)
    fig.legend(handles=_handles(styles),loc='center left',bbox_to_anchor=(.674,.56),fontsize=6.9,
               frameon=False,borderaxespad=0,handlelength=1.5,handletextpad=.35)
    _save(fig,out,'figure1_one_column')
    # Optional six-panel appendix figure: only six published mechanisms.
    groups=operator_groups(cfg)
    test=cfg['test']['operators'];cols=3;rows=(len(test)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(12,3.1*rows),squeeze=False,sharey=True)
    for ax,operator in zip(axes.flat,test):
        _curve(ax,summary,cfg,operator)
        status='P: pretraining corruption' if operator in groups['pretraining_only'] else 'U: test-only structure'
        ax.set_title(OPERATOR_LABELS[operator]+'\n'+status,fontsize=10)
    for ax in list(axes.flat)[len(test):]:ax.set_visible(False)
    for ax in axes[:,0]:ax.set_ylabel('Macro-F1',fontsize=10)
    fig.subplots_adjust(left=.06,right=.81,bottom=.095,top=.9,hspace=.45,wspace=.19)
    fig.legend(handles=_handles(styles),loc='upper left',bbox_to_anchor=(.82,.84),frameon=False,fontsize=9)
    fig.suptitle(f"k={cfg['reporting']['headline_k']}: pretraining exposure and transfer to test-only structures",fontsize=12)
    fig.text(.825,.40,'P: self-supervised pretraining\nAll classifier training: clean only\n\nDotted line: pretraining ceiling 0.30\nGreen area: within training support\nU: absent from all training',fontsize=8,va='top')
    _save(fig,out,'all_six_robustness_profiles')
    # Machine-readable source rows, with explicit unrounded means and SDs.
    summary.to_csv(out/'plot_source.csv',index=False)
