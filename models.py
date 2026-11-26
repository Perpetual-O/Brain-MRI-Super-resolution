"""CNN super-resolution models (PyTorch).

Both models take a bicubic-upsampled low-resolution slice (same size as the target)
and predict the high-resolution slice. That keeps the baseline and the CNNs
directly comparable: the CNN only has to learn the *missing detail*.

  SRCNN : 3 conv layers (Dong et al., 2016).
  VDSR  : deep stack of 3x3 convs with a global residual connection (Kim et al., 2016).
          `depth` defaults to 10 here (the paper uses 20) so it trains in reasonable
          time on a laptop/Colab; raise it if you have the compute.
"""
import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.body(x)


class VDSR(nn.Module):
    def __init__(self, channels: int = 1, depth: int = 10, width: int = 64):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.body(x)  # global residual: learn only the missing detail


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "srcnn":
        return SRCNN()
    if name == "vdsr":
        return VDSR()
    raise ValueError(f"unknown model: {name}")
