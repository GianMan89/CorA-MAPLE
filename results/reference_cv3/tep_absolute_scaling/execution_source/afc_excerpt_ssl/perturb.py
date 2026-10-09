"""Offline adapter of AFC-RobustBench; reference semantics are unchanged.

The full first-hour input is perturbed once. No prefix scoring, progress average,
re-detection or selection of a different episode boundary takes place. Leading RTN
repair infers activity from the window start, as in the supplied benchmark. This
can amplify the effect of insertions and is explicitly diagnosed, not hidden.
"""
from __future__ import annotations
import hashlib
import numpy as np
from afc_robustness.perturbations import (MissingEvents,SpuriousEvents,TimingUncertainty,
    BurstSpuriousEvents,TagDropout,MixedPerturbation)
from afc_robustness.repair import TraceRepair
from afc_robustness.representations import series_to_episode,episode_to_series

OPS={'missing':MissingEvents(),'spurious':SpuriousEvents(),
     'timing':TimingUncertainty(max_shift=10.),'burst':BurstSpuriousEvents(half_width=2.),
     'tag_dropout':TagDropout(),
     'mixed':MixedPerturbation([MissingEvents(),SpuriousEvents()],name='missing_then_spurious')}
REPAIR=TraceRepair()

def seed_from(*parts) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')

def perturb_one(x:np.ndarray, kind:str,severity:float,rng:np.random.Generator,diagnostic:bool=False):
    if kind not in OPS:raise ValueError(f'Unknown perturbation: {kind}')
    if not 0<=severity<1:raise ValueError('Severity must be in [0,1).')
    ep=series_to_episode(x,dt=1.,initial_active_policy='act')
    raw=OPS[kind].apply(ep,severity,rng)
    repaired=REPAIR.apply(raw)
    y=episode_to_series(repaired,dt=1.,max_length=x.shape[-1]).astype(np.float32)
    if not diagnostic:return y
    intersection=np.logical_and(x,y).sum(); union=np.logical_or(x,y).sum()
    return y,{'raw_events':ep.n_events,'perturbed_events':raw.n_events,'repaired_events':repaired.n_events,
              'changed_cells':int(np.count_nonzero(x!=y)), 'jaccard':float(intersection/union) if union else 1.,
              'active_tags_clean':int((x.max(1)>0).sum()),'active_tags_corrupt':int((y.max(1)>0).sum())}

def training_view(x:np.ndarray,rng:np.random.Generator,max_severity:float=.2) -> np.ndarray:
    """Sample only individual missing/spurious/timing, never burst/tag dropout/mixed."""
    if rng.random()<.2:return x.copy()
    kind=str(rng.choice(['missing','spurious','timing']))
    return perturb_one(x,kind,float(rng.uniform(.05,max_severity)),rng)

def paired_test_suite(X:np.ndarray,ids:np.ndarray,kinds:list[str],severities:list[float],draws:int,seed:int):
    yield 'clean',0.,0,X.copy(),[]
    for kind in kinds:
        for s in severities:
            if s==0:continue
            for draw in range(draws):
                xs=[];diag=[]
                for x,identity in zip(X,ids):
                    rng=np.random.default_rng(seed_from(seed,str(identity),kind,s,draw))
                    y,d=perturb_one(x,kind,float(s),rng,True);xs.append(y);d['id']=str(identity);diag.append(d)
                yield kind,float(s),draw,np.stack(xs),diag
