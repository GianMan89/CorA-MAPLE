"""Small patch-based temporal encoders and masked latent denoising.

Inspired by TS-JEPA (Ennadir et al., 2025), not a reproduction: multivariate binary
patches, all-position latent denoising, explicit variance control, and a low-weight
binary reconstruction auxiliary. No text-model weights or labels enter pretraining.
"""
from __future__ import annotations
import copy, math
import torch
from torch import nn
import torch.nn.functional as F

class PatchEncoder(nn.Module):
    def __init__(self,n_tags:int,d_model:int=48,patch_size:int=3,n_layers:int=2,dropout:float=.1):
        super().__init__()
        if patch_size < 1 or d_model < 4 or d_model % 4:
            raise ValueError('patch_size must be positive and d_model divisible by four attention heads.')
        self.n_tags=n_tags;self.patch_size=patch_size;self.d_model=d_model
        self.embed=nn.Linear(n_tags*patch_size,d_model)
        self.mask_token=nn.Parameter(torch.zeros(1,1,d_model))
        layer=nn.TransformerEncoderLayer(d_model,4,2*d_model,dropout=dropout,
                                       activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,n_layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d_model)
        self.register_buffer('pos',self._positions(128,d_model),persistent=False)
    @staticmethod
    def _positions(n:int,d:int):
        p=torch.arange(n)[:,None];div=torch.exp(torch.arange(0,d,2)*(-math.log(10000.)/d))
        out=torch.zeros(1,n,d);out[0,:,0::2]=torch.sin(p*div);out[0,:,1::2]=torch.cos(p*div)
        return out
    def patchify(self,x):
        if x.ndim!=3 or x.shape[1]!=self.n_tags or x.shape[2]%self.patch_size:
            raise ValueError('Input shape N x tags x length; length must be divisible by patch size.')
        return x.unfold(2,self.patch_size,self.patch_size).permute(0,2,1,3).flatten(2)
    def forward(self,x,mask=None):
        h=self.embed(self.patchify(x))
        if mask is not None:h=torch.where(mask[:,:,None],self.mask_token,h)
        positions = self.pos if h.shape[1] <= self.pos.shape[1] else self._positions(h.shape[1], self.d_model).to(h.device)
        h=h+positions[:,:h.shape[1]].to(h.dtype)
        return self.norm(self.encoder(h))
    def pooled(self,x):
        h=self(x);return torch.cat([h.mean(1),h.amax(1)],dim=1)

class Classifier(nn.Module):
    def __init__(self,encoder:PatchEncoder,n_classes:int):
        super().__init__();self.encoder=encoder;self.head=nn.Linear(2*encoder.d_model,n_classes)
    def forward(self,x):return self.head(self.encoder.pooled(x))

class LatentPretrainer(nn.Module):
    def __init__(self,encoder:PatchEncoder,ema:float=.99,raw_weight:float=.25,var_weight:float=.1):
        super().__init__();self.student=encoder;self.teacher=copy.deepcopy(encoder).eval()
        for p in self.teacher.parameters():p.requires_grad=False
        d=encoder.d_model
        self.predictor=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.decoder=nn.Linear(d,encoder.n_tags*encoder.patch_size)
        self.ema=ema;self.raw_weight=raw_weight;self.var_weight=var_weight
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def loss(self,clean,view,mask,objective='latent'):
        h=self.student(view,mask)
        target_raw=self.student.patchify(clean)
        # Equal weight for active and inactive cells within each minibatch.
        raw=F.binary_cross_entropy_with_logits(self.decoder(h),target_raw,reduction='none')
        positive=target_raw.sum().clamp_min(1);negative=(1-target_raw).sum().clamp_min(1)
        rec=.5*((raw*target_raw).sum()/positive+(raw*(1-target_raw)).sum()/negative)
        if objective=='reconstruction':return rec,{'latent':0.,'reconstruction':float(rec.detach()),'std':0.}
        with torch.no_grad():target=self.teacher(clean)
        pred=F.layer_norm(self.predictor(h),(h.shape[-1],))
        latent=F.mse_loss(pred,target)
        # Batch-wise variance across recording embeddings discourages a constant encoder.
        pooled=h.mean(1);std=torch.sqrt(pooled.var(0,unbiased=False)+1e-4)
        var=F.relu(.5-std).mean()
        loss=latent+self.raw_weight*rec+self.var_weight*var
        return loss,{'latent':float(latent.detach()),'reconstruction':float(rec.detach()),'std':float(std.mean().detach())}
    @torch.no_grad()
    def update_teacher(self):
        for p,q in zip(self.student.parameters(),self.teacher.parameters()):q.mul_(self.ema).add_(p,alpha=1-self.ema)
