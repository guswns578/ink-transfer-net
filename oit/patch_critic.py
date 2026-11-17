"""Small conditional patch critic, used in training only."""

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class PatchCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(2, 32, 4, 2, 1)),
                spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
                spectral_norm(nn.Conv2d(64, 96, 4, 2, 1)),
                spectral_norm(nn.Conv2d(96, 1, 3, 1, 1)),
            ]
        )

    def forward(self, x, y):
        z = torch.cat([x, y], 1)
        fs = []
        for layer in self.layers[:-1]:
            z = torch.nn.functional.leaky_relu(layer(z), 0.2)
            fs.append(z)
        return self.layers[-1](z), fs
