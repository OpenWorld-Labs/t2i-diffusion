import einops as eo
import torch
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

from .mlp import MLPCustom


class ImageRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim=dim_head // 4,  # or // 2 depending on your needs
            freqs_for="pixel",
            max_freq=256,
        )

        n_patches = config.sample_size // config.patch_size
        self.rearrange_in = lambda x: eo.rearrange(
            x, "b h (n_y n_x) d -> b h n_y n_x d", n_y=n_patches
        )
        self.rearrange_out = lambda x: eo.rearrange(
            x, "b h n_y n_x d -> b h (n_y n_x) d"
        )
        self.get_freqs = lambda: self.pos_emb.get_axial_freqs(n_patches, n_patches)

    def forward(self, q, k):
        # q k both [b,h,n,d]
        q = self.rearrange_in(q)
        k = self.rearrange_in(k)
        freqs = self.get_freqs()
        q = apply_rotary_emb(freqs.float(), q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs.float(), k.float()).to(k.dtype)
        q = self.rearrange_out(q)
        k = self.rearrange_out(k)
        return q, k


class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()

        self.p = nn.Parameter(torch.randn(n_seq, dim) * 0.02)

    def forward(self, x):
        b, n, d = x.shape
        p = eo.repeat(self.p, "n d -> b n d", b=b)
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

        self.sincos = SinCosEmbed(512, theta=300, mult=1000)
        self.mlp = MLPCustom(512, dim * 4, dim)

    def forward(self, x):
        x = self.sincos(x)
        x = self.mlp(x)
        return x


class ConditionEmbedding(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()

        self.embedding = nn.Embedding(n_classes, dim)
        self.mlp = MLPCustom(dim, dim * 4, dim)

    def forward(self, x):
        # x is long tensor of [b,]
        x = self.embedding(x)
        x = self.mlp(x)
        return x
