import torch
from torch import nn
import torch.nn.functional as F

import einops as eo
from .mlp import MLPCustom

class ImageRoPE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        pass

    def forward(self, q, k):
        # q k both [b,h,n,d]
        pass

class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()

        self.p = nn.Parameter(torch.randn(n_seq,dim)*0.02)

    def forward(self, x):
        b,n,d = x.shape
        p = eo.repeat(self.p, 'n d -> b n d', b = b)
        return x + p

class SinCosEmbed(nn.Module):
    def __init__(self, dim, theta=300, mult=1000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.mult = mult

    def forward(self, x):
        # Handle different input types
        if isinstance(x, float):
            x = torch.tensor([x])
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
            
        x = x * self.mult
        
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.theta)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        
        # Match device and dtype of input
        emb = emb.to(device=x.device, dtype=x.dtype)
        
        # Compute sin/cos embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sincos = SinCosEmbed(512, theta=300, mult = 1000)
        self.mlp = MLP(512, dim * 4, dim)
    
    def forward(self, x):
        x = self.sincos(x)
        x = self.mlp(x)
        return x