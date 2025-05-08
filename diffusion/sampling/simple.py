import torch
from torch import nn
import torch.nn.functional as F

class SimpleSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, sampling_steps = 64):
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        for _ in range(sampling_steps):
            pred = model(x, ts)
            x = x - pred*dt
            ts = ts - dt

        return x

if __name__ == "__main__":
    model = lambda x,t: x

    sampler = SimpleSampler()
    x = sampler(model, torch.randn(4, 3, 64, 64), 4)
    print(x.shape)