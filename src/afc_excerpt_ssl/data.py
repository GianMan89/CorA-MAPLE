"""Strict first-hour loading, duplicate-aware partitions, and label-blind crops.

A source recording is indivisible. Never split overlapping excerpts between roles.
CSV rows correspond to original minutes 1..60; local upstream timestamps are 0..59.
The source filenames/class labels are never passed to a representation learner.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    tags: np.ndarray
    hashes: np.ndarray
    classes: np.ndarray

def load_processed(path: str|Path, observation_samples: int | None = None) -> Dataset:
    z=np.load(path,allow_pickle=False)
    X=z['X'].astype(np.float32)
    if X.ndim != 3 or not np.isin(X, [0, 1]).all():
        raise ValueError('Expected binary N x channels x time recordings.')
    horizon = X.shape[-1] if observation_samples is None else int(observation_samples)
    if horizon < 3 or horizon > X.shape[-1]:
        raise ValueError(f'Observation length {horizon} exceeds available samples {X.shape[-1]} or is too short.')
    X = X[..., :horizon]
    hashes = np.asarray([hashlib.sha256(v.astype(np.uint8).tobytes()).hexdigest() for v in X])
    le=LabelEncoder(); y=le.fit_transform(z['y'])
    # The same selected observation cannot carry conflicting ground truth labels.
    byhash={}
    for h,label in zip(hashes,y):byhash.setdefault(str(h),set()).add(int(label))
    if any(len(v)>1 for v in byhash.values()):raise ValueError('Conflicting labels for duplicate data.')
    return Dataset(X,y,z['ids'],z['tags'],hashes,le.classes_)

def make_partition(data: Dataset, seed: int=2027, n_splits: int=5) -> list[dict]:
    """Group exact first-hour duplicates. Each test recording is unseen in SSL.

    Fixed settings are used: no test or validation labels select checkpoints.
    The specified folds are rotated as held-out folds; crops only use outer training data.
    """
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    partitions=[]
    for fold,(tr,te) in enumerate(cv.split(data.X,data.y,groups=data.hashes)):
        assert not set(data.hashes[tr]) & set(data.hashes[te])
        assert set(data.y[tr])==set(data.y[te])==set(range(len(data.classes)))
        partitions.append({'fold':fold,'train':tr.tolist(),'test':te.tolist()})
    return partitions

def nested_labels(data:Dataset, train:np.ndarray, budgets:list[int], seed:int) -> dict[int,np.ndarray]:
    """k UNIQUE observed recordings per class; duplicated files do not buy labels."""
    rng=np.random.default_rng(seed); order=[]
    for c in range(len(data.classes)):
        idx=np.asarray(train)[data.y[train]==c]
        _,unique=np.unique(data.hashes[idx],return_index=True)
        idx=idx[unique];idx=rng.permutation(idx)
        if len(idx)<max(budgets):raise ValueError(f'Class {c} has only {len(idx)} unique training records.')
        order.append(idx)
    return {k:np.concatenate([idx[:k] for idx in order]) for k in budgets}

def sample_crops(X:np.ndarray,batch:int,length:int,rng:np.random.Generator) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Uniform source and start sampling; no flood detector, class or activity filter."""
    if not 1<=length<=X.shape[2]:raise ValueError('Invalid excerpt length.')
    idx=rng.integers(0,len(X),size=batch); starts=rng.integers(0,X.shape[2]-length+1,size=batch)
    result=np.stack([X[i,:,s:s+length] for i,s in zip(idx,starts)])
    return result.astype(np.float32),idx,starts
