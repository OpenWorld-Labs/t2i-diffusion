"""
Simplified rectified flow transformer
"""

import torch
from torch import nn

from ..nn.attn import (
    StackedTransformer,
    PatchProjIn,
    PatchProjOut
)
from .nn.embeddings import TimestepEmbedding

import einops as eo

class RFTCore(nn.Module):
    def __init__(config : 'TransformerConfig'):
        super().__init__()

        self.proj_in = PatchProjIn(config.d_model, config.channels, config.patch_size)
        self.blocks = StackedTransformer(config)
        self.proj_out = PatchProjOut(config.sample_size, config.d_model, config.channels, config.patch_size)

        self.t_embed = TimestepEmbedding(config.d_model)

    def forward(self, x, t):
        cond = self.t_embed(t)

        x = self.proj_in(x)
        x = self.blocks(x, cond)
        x = self.proj_out(x, cond)

        return x

class RFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = RFTCore(config)
    
    def forward(self, x):
        b,c,h,w = x.shape
        with torch.no_grad():
            ts = torch.rand(b,device=x.device,dtype=x.dtype).sigmoid()
            
            ts_exp = eo.repeat(ts, 'b -> b c h w', c=c,h=h,w=w)
            z = torch.randn_like(x)

            lerpd = x * (1. - ts_exp) + z * ts_exp
            target = z - x
        
        pred = self.core(lerpd, ts)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss