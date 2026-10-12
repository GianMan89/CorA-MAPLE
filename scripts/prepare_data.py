#!/usr/bin/env python3
"""Prepare TEP observations at a configurable horizon, without fitting data statistics.
The packaged CSVs contain minutes 1--60; longer horizons require separately supplied
source recordings. Never substitute or extrapolate missing samples.
"""
from __future__ import annotations
import argparse,hashlib,json,io,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def prepare(source:Path,dest:Path,observation_samples:int=60)->dict:
    if observation_samples < 3: raise ValueError('At least three samples required.')
    dest.mkdir(parents=True,exist_ok=True)
    folder=source/'data/tep'
    if not folder.exists(): folder=source/'tep'
    if not folder.exists(): raise FileNotFoundError(folder)
    xs=[];rows=[];tags0=None
    for file in sorted(folder.glob('*/*.csv')):
        f=pd.read_csv(file); times=f.iloc[:,0].to_numpy()
        if len(times)<observation_samples or not np.allclose(times[:observation_samples],np.arange(1,observation_samples+1)):
            raise ValueError(f'{file}: required original minute samples not present.')
        a=f.iloc[:observation_samples,1:].to_numpy()
        if not np.isin(a,[0,1]).all():raise ValueError(f'{file}: non-binary data.')
        tags=list(f.columns[1:])
        if tags0 is None:tags0=tags
        if tags!=tags0:raise ValueError('Inconsistent channel order.')
        x=a.astype(np.uint8).T;diff=np.diff(x.astype(np.int16),axis=1)
        import re
        match=re.findall(r'(?:run|timeseries_)(\d+)',file.stem)
        if len(match)!=1:raise ValueError(f'Unknown recording ID: {file}')
        rows.append(dict(id='tep/'+file.relative_to(folder).as_posix(),label=file.parent.name,run=int(match[0]),hash=hashlib.sha256(x.tobytes()).hexdigest(),events=int(np.count_nonzero(diff)+x[:,0].sum()),onsets=int((diff>0).sum()+x[:,0].sum()),active_tags=int(x.max(1).sum())))
        xs.append(x)
    if not xs:raise ValueError('No TEP source CSVs.')
    t=pd.DataFrame(rows)
    if (t.groupby('hash').label.nunique()>1).any():raise ValueError('Duplicate observations with different labels.')
    np.savez_compressed(dest/f'tep_{observation_samples}min.npz',X=np.stack(xs),y=t.label.to_numpy(dtype=str),run_ids=t.run.to_numpy(),ids=t.id.to_numpy(dtype=str),tags=np.array(tags0),hashes=t.hash.to_numpy(dtype=str))
    t.to_csv(dest/'tep_manifest.csv',index=False)
    summary={'tep':dict(shape=list(np.stack(xs).shape),class_counts=t.label.value_counts().to_dict(),unique_recordings=int(t.hash.nunique()),onsets_mean=float(t.onsets.mean()),events_mean=float(t.events.mean()),active_tags_mean=float(t.active_tags.mean()),minutes_retained=[1,observation_samples])}
    (dest/'audit.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data/raw'));p.add_argument('--output',type=Path,default=Path('data/processed'));p.add_argument('--observation-samples',type=int,default=60)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.observation_samples),indent=2))
