"""
Simplified rectified flow transformer
"""

import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..nn.attn import DiT, PatchProjIn, PatchProjOut, ProjOut, UViT
from ..nn.embeddings import LearnedPosEnc, TimestepEmbedding


class RFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.proj_in = (
            PatchProjIn(
                config.d_model, config.channels, config.patch_size, config.patch
            )
            if config.patch
            else nn.Linear(config.channels, config.d_model, bias=False)
        )
        self.pos_enc = (
            LearnedPosEnc(config.sample_size, config.d_model)
            if not config.patch
            else nn.Sequential()
        )
        self.blocks = UViT(config) if config.uvit else DiT(config)
        self.proj_out = (
            PatchProjOut(
                config.sample_size, config.d_model, config.channels, config.patch_size
            )
            if config.patch
            else ProjOut(config.d_model, config.channels)
        )

        self.t_embed = TimestepEmbedding(config.d_model)

    def forward(self, x, t):
        cond = self.t_embed(t)
        x = self.proj_in(x)
        x = self.pos_enc(x)
        x = self.blocks(x, cond)
        x = self.proj_out(x, cond)

        return x


class RFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = RFTCore(config)
        self.patch = config.patch

    def forward_patch(self, x):
        b, c, h, w = x.shape
        with torch.no_grad():
            ts = torch.randn(b, device=x.device, dtype=x.dtype).sigmoid()

            ts_exp = eo.repeat(ts, "b -> b c h w", c=c, h=h, w=w)
            z = torch.randn_like(x)

            lerpd = x * (1.0 - ts_exp) + z * ts_exp
            target = z - x

        pred = self.core(lerpd, ts)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss

    def forward_nopatch(self, x):
        b, n, d = x.shape
        with torch.no_grad():
            ts = torch.randn(b, device=x.device, dtype=x.dtype).sigmoid()

            ts_exp = eo.repeat(ts, "b -> b n d", n=n, d=d)
            z = torch.randn_like(x)

            lerpd = x * (1.0 - ts_exp) + z * ts_exp
            target = z - x

        pred = self.core(lerpd, ts)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss

    def forward(self, x):
        if self.patch:
            return self.forward_patch(x)
        else:
            return self.forward_nopatch(x)
