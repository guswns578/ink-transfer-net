"""Native-resolution local residual model; no pooling, resize or PixelShuffle.

Architectural comparison, not an assertion that downsampling was the only cause.
All convolutions have stride=1. Finite receptive field permits haloed inference.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LocalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        return x + 0.1 * self.c2(F.silu(self.c1(x)))


class FullResReinker(nn.Module):
    # Intro + 12*2 convolutions + head => 53x53 receptive field (26px radius).
    radius = 26

    def __init__(self, width=48):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(*[LocalBlock(width) for _ in range(12)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, f):
        return torch.tanh(self.head(self.blocks(F.silu(self.intro(f)))))
