"""Experimental finite-context detail branch; frozen clean model is external."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, d):
        super().__init__()
        self.a = nn.Conv2d(width, width, 3, padding=d, dilation=d)
        self.b = nn.Conv2d(width, width, 3, padding=d, dilation=d)

    def forward(self, x):
        return x + 0.1 * self.b(F.silu(self.a(x)))


class DetailBranch(nn.Module):
    # 1 + 2*(1+2+4+4+4+2+1) + 1 = 38 native pixels of context.
    radius = 38

    def __init__(self, width=32):
        super().__init__()
        self.entry = nn.Conv2d(4, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width, d) for d in (1, 2, 4, 4, 4, 2, 1)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, f):
        # Linear residual avoids an initially saturated output transform.
        return self.head(self.body(F.silu(self.entry(f))))


def compose_detail(f, residual, support):
    raw = f[:, 3:4] + residual * support
    return raw, raw.clamp(0, 1)
