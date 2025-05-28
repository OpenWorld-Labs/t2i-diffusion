import torch
import torch.nn.functional as F
from torch import nn

from .normalization import LayerNorm


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc_a = nn.Linear(dim, dim)
        self.fc_b = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim)

    def forward(self, x, cond):
        y = F.silu(cond)

        alpha = self.fc_a(y).unsqueeze(1)  # [b,1,d]
        beta = self.fc_b(y).unsqueeze(1)  # [b,1,d]

        x = self.norm(x) * (1.0 + alpha) + beta
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc_c = nn.Linear(dim, dim)

    def forward(self, x, cond):
        y = F.silu(cond)
        c = self.fc_c(y).unsqueeze(1)  # [b,1,d]

        return c * x
