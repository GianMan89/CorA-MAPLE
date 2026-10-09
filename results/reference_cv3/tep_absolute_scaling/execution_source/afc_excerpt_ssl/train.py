"""Seeded, fixed-budget training; no held-out labels choose model checkpoints."""
from __future__ import annotations
import copy,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .models import PatchEncoder,LatentPretrainer,Classifier
from .data import sample_crops
from .perturb import training_view,seed_from

def set_seed(seed:int,threads:int=1):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

def pretrain(X:np.ndarray,n_tags:int,cfg:dict,seed:int,mode:str,out:Path) -> dict:
    set_seed(seed,cfg.get('threads',1)); rng=np.random.default_rng(seed)
    encoder=PatchEncoder(n_tags,**cfg['model'])
    model=LatentPretrainer(encoder,**cfg['objective']).train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['pretrain_lr'],weight_decay=cfg['weight_decay'])
    lengths=[X.shape[-1]] if mode.endswith('_full') else cfg['crop_lengths']
    if not lengths or any(l < cfg['model']['patch_size'] or l > X.shape[-1] or l % cfg['model']['patch_size'] for l in lengths):
        raise ValueError('Crop lengths must fit the selected observation and be divisible by patch_size.')
    steps = cfg.get('pretrain_steps_by_method', {}).get(mode, cfg['pretrain_steps'])
    target_minutes = cfg.get('pretrain_target_sample_minutes_by_method', {}).get(mode)
    if target_minutes is not None and target_minutes % (cfg['pretrain_batch'] * cfg['model']['patch_size']):
        raise ValueError('Exposure target must be divisible by batch size times patch size.')
    robust='robust' in mode;objective='reconstruction' if mode.startswith('mae') else 'latent'
    hist=[];start=time.perf_counter();stats={'crops':0,'empty_crops':0,'minutes':0,'events':0}
    step = 0
    while ((stats['minutes'] < target_minutes) if target_minutes is not None else (step < steps)):
        length=int(rng.choice(lengths))
        if target_minutes is not None:
            length=min(length, int((target_minutes - stats['minutes']) / cfg['pretrain_batch']))
        clean,_,_=sample_crops(X,cfg['pretrain_batch'],length,rng)
        view=np.stack([training_view(x,rng,cfg['augmentation_max_severity']) for x in clean]) if robust else clean.copy()
        target=torch.from_numpy(clean);inp=torch.from_numpy(view)
        P=length//cfg['model']['patch_size']
        mask=np.zeros((len(clean),P),dtype=bool)
        # Equal token budget across pretext objectives, at least one masked patch.
        for row in mask: row[rng.choice(P,max(1,int(round(P*cfg['mask_ratio']))),replace=False)]=True
        mask=torch.from_numpy(mask)
        opt.zero_grad(set_to_none=True); loss,d=model.loss(target,inp,mask,objective)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite pretraining loss.')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.update_teacher()
        stats['crops']+=len(clean);stats['empty_crops']+=int((clean.sum((1,2))==0).sum());stats['minutes']+=len(clean)*length
        stats['events']+=int(np.count_nonzero(np.diff(clean,axis=2))+clean[:,:,0].sum())
        if step%50==0 or (target_minutes is not None and stats['minutes'] >= target_minutes) or (target_minutes is None and step == steps-1):hist.append({'step':step+1,'loss':float(loss.detach()),**d})
        step += 1
    stats['updates'] = step
    stats['patch_tokens'] = stats['minutes'] // cfg['model']['patch_size']
    state=copy.deepcopy(model.student.state_dict());elapsed=time.perf_counter()-start
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(hist).to_csv(out/'pretraining.csv',index=False)
    torch.save({'state_dict':state,'seed':seed,'mode':mode,'config':cfg,'n_tags':n_tags,'stats':stats,'seconds':elapsed},out/'encoder.pt')
    return {'state':state,'seconds':elapsed,'stats':stats,'history':hist}

def augment_bank(X:np.ndarray,seed:int,maxseverity:float,copies:int=8):
    rng=np.random.default_rng(seed)
    return np.stack([np.stack([training_view(x,rng,maxseverity) for x in X]) for _ in range(copies)])

def train_classifier(X:np.ndarray,y:np.ndarray,n_classes:int,cfg:dict,seed:int,state:dict|None=None,augment:bool=False,out:Path|None=None):
    set_seed(seed,cfg.get('threads',1));rng=np.random.default_rng(seed)
    encoder=PatchEncoder(X.shape[1],**cfg['model'])
    if state is not None:encoder.load_state_dict(state)
    # Identical freshly initialized heads in paired SSL / no-SSL comparisons.
    torch.manual_seed(seed_from(seed,'head'));model=Classifier(encoder,n_classes).train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['finetune_lr'],weight_decay=cfg['weight_decay'])
    inputs=torch.from_numpy(X);labels=torch.from_numpy(y.astype(np.int64))
    bank=augment_bank(X,seed_from(seed,'supervised_augment'),cfg['augmentation_max_severity']) if augment else None
    hist=[];start=time.perf_counter()
    for step in range(cfg['finetune_steps']):
        idx=rng.choice(len(X),size=min(cfg['finetune_batch'],len(X)),replace=False)
        if bank is None:batch=inputs[idx]
        else:batch=torch.from_numpy(bank[rng.integers(0,len(bank)),idx])
        logits=model(batch);loss=F.cross_entropy(logits,labels[idx]);opt.zero_grad(set_to_none=True)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0 or step==cfg['finetune_steps']-1:
            model.eval()
            with torch.no_grad():train_logits=model(inputs);train_loss=F.cross_entropy(train_logits,labels).item();acc=(train_logits.argmax(1)==labels).float().mean().item()
            model.train();hist.append({'step':step+1,'minibatch_loss':loss.item(),'clean_train_loss':train_loss,'clean_train_accuracy':acc})
    model.eval();elapsed=time.perf_counter()-start
    if out is not None:
        out.mkdir(parents=True,exist_ok=True);pd.DataFrame(hist).to_csv(out/'finetuning.csv',index=False)
        torch.save({'state_dict':model.state_dict(),'n_tags':X.shape[1],'n_classes':n_classes,'model':cfg['model'],'seed':seed},out/'classifier.pt')
    return model,{'seconds':elapsed,'train_loss':train_loss,'train_accuracy':acc,'history':hist}

@torch.no_grad()
def predict(model:nn.Module,X:np.ndarray,batch:int=128):
    model.eval();return torch.cat([model(torch.from_numpy(X[i:i+batch])).softmax(1) for i in range(0,len(X),batch)]).cpu().numpy()

@torch.no_grad()
def embeddings(encoder:PatchEncoder,X:np.ndarray,batch:int=128):
    encoder.eval();return torch.cat([encoder.pooled(torch.from_numpy(X[i:i+batch])) for i in range(0,len(X),batch)]).cpu().numpy()
