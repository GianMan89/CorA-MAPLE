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
    def _prepare(self, x, lengths=None):
        if x.ndim != 3 or x.shape[1] != self.n_tags or x.shape[2] < 1:
            raise ValueError('Input shape must be N x tags x samples, with at least one real sample.')
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[2], device=x.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=x.device, dtype=torch.long)
        if lengths.shape != (x.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > x.shape[2]):
            raise ValueError('Every observation needs a valid, positive length within its supplied matrix.')
        # Supplied padding is never interpreted as observed inactivity, even if its
        # stored values are nonzero. Partial final patches are zero-filled explicitly.
        real = torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None]
        x = x.masked_fill(~real[:, None, :], 0.)
        extra = (-x.shape[2]) % self.patch_size
        if extra:
            x = F.pad(x, (0, extra))
            real = F.pad(real, (0, extra), value=False)
        fractions = real.unfold(1, self.patch_size, self.patch_size).to(x.dtype).mean(-1)
        return x, lengths, fractions

    def patchify(self, x, lengths=None):
        x, _, _ = self._prepare(x, lengths)
        return x.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)

    def patch_weights(self, x, lengths=None):
        return self._prepare(x, lengths)[2]

    def cell_mask(self, x, lengths=None):
        x, lengths, _ = self._prepare(x, lengths)
        valid = torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None]
        valid = valid[:, None, :].expand(-1, self.n_tags, -1)
        return valid.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)

    def forward(self, x, mask=None, lengths=None):
        prepared, lengths, weights = self._prepare(x, lengths)
        patches = prepared.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)
        h = self.embed(patches)
        valid = weights > 0
        if mask is not None:
            if mask.shape != valid.shape or torch.any(mask & ~valid):
                raise ValueError('Patch masking may select real patches only.')
            h = torch.where(mask[:, :, None], self.mask_token, h)
        positions = self.pos if h.shape[1] <= self.pos.shape[1] else self._positions(h.shape[1], self.d_model).to(h.device)
        h = h + positions[:, :h.shape[1]].to(h.dtype)
        # Avoid a different attention path when every supplied patch is real.
        padding_mask = None if bool(valid.all()) else ~valid
        h = self.norm(self.encoder(h, src_key_padding_mask=padding_mask))
        return h.masked_fill(~valid[:, :, None], 0.)

    def mean_tokens(self, h, x, lengths=None):
        weights = self.patch_weights(x, lengths)
        return (h * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-12)

    def pooled(self, x, lengths=None):
        h = self(x, lengths=lengths)
        weights = self.patch_weights(x, lengths)
        mean = self.mean_tokens(h, x, lengths)
        maximum = h.masked_fill((weights == 0)[:, :, None], -torch.inf).amax(1)
        return torch.cat([mean, maximum], dim=1)

class Classifier(nn.Module):
    def __init__(self,encoder:PatchEncoder,n_classes:int):
        super().__init__();self.encoder=encoder;self.head=nn.Linear(2*encoder.d_model,n_classes)
    def forward(self, x, lengths=None):
        return self.head(self.encoder.pooled(x, lengths=lengths))

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
