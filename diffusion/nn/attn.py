import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .embeddings import ImageRoPE
from .mlp import MLP

import einops as eo
from .mimetic import mimetic_init

from .modulation import AdaLN, Gate

torch.backends.cuda.enable_flash_sdp(enabled = True)

class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.rope = ImageRoPE(config)

        mimetic_init(self.qkv, self.out, config)

    def forward(self, x):

        q,k,v = eo.rearrange(self.qkv(x), 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)
        q,k = self.qk_norm(q,k)
        q,k = self.rope(q,k)
        x = F.scaled_dot_product_attention(q,k,v)
        x = eo.rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond):
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x)
        x = self.gate1(x, cond)
        x = res1 + x
        
        res2 = x.clone()
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = res2 + x

        return x

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(DiTBlock(config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)

        return x

# === VIT Specific Layers ===

class PatchProjIn(nn.Module):
    def __init__(self, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.proj_in = nn.Conv2d(channels, d_model, patch_size, patch_size, 0, bias=False)
    
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.proj_in(x)
        x = eo.rearrange(x, 'b c h w -> b (h w) c')
        return x

class PatchProjOut(nn.Module):
    def __init__(self, sample_size, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels*patch_size*patch_size)
        self.sample_size = sample_size
        self.patch_size = patch_size

        self.n_patches = self.sample_size//self.patch_size

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)
        x = eo.rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h = self.n_patches, ph = self.patch_size, pw = self.patch_size)

        return x

if __name__ == "__main__":
    layer = PatchProjOut(64, 384, 3, 4).cuda().bfloat16()
    x = torch.randn(1,256,384).cuda().bfloat16()

    with torch.no_grad():
        z = layer(x)
        print(z.shape)