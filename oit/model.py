"""Small NAF-style residual baseline, trained from scratch on provisional pairs.

Not the old OIT checkpoint. No skeleton input, text encoder, diffusion, or remote
inference. Reuses the NAF block pattern already present in the local OIT project.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        var, mu = torch.var_mean(x.float(), dim=1, keepdim=True, unbiased=False)
        return ((x-mu)*torch.rsqrt(var+1e-6)).to(x.dtype)*self.weight + self.bias


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = Norm(c), Norm(c)
        self.expand = nn.Conv2d(c, 2*c, 1)
        self.depth = nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.project = nn.Conv2d(c, c, 1)
        self.ff1, self.ff2 = nn.Conv2d(c, 2*c, 1), nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        a, b = self.depth(self.expand(self.n1(x))).chunk(2, 1)
        z = a*b
        x = x+self.beta*self.project(z*self.attn(z))
        a, b = self.ff1(self.n2(x)).chunk(2, 1)
        return x+self.gamma*self.ff2(a*b)


class Reinker(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.enc = nn.ModuleList([nn.Sequential(Block(width)),
                                  nn.Sequential(Block(width*2), Block(width*2))])
        self.down = nn.ModuleList([nn.Conv2d(width, width*2, 2, 2), nn.Conv2d(width*2, width*4, 2, 2)])
        self.mid = nn.Sequential(*[Block(width*4) for _ in range(6)])
        self.up = nn.ModuleList([nn.Sequential(nn.Conv2d(width*4, width*8, 1), nn.PixelShuffle(2)),
                                 nn.Sequential(nn.Conv2d(width*2, width*4, 1), nn.PixelShuffle(2))])
        self.dec = nn.ModuleList([nn.Sequential(Block(width*2), Block(width*2)), nn.Sequential(Block(width), Block(width))])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        # Direct signed residual + raw-output training loss has no relu(0) trap.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, f):
        h, w = f.shape[-2:]
        f = F.pad(f, (0, (-w)%4, 0, (-h)%4), mode='replicate')
        z, skips = self.intro(f), []
        for enc, down in zip(self.enc, self.down):
            z = enc(z)
            skips.append(z)
            z = down(z)
        z = self.mid(z)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            z = dec(up(z)+skip)
        return torch.tanh(self.head(z))[..., :h, :w]


def compose(f, r, support):
    raw = f[:, :1]+r*support
    return raw, raw.clamp(0, 1)
