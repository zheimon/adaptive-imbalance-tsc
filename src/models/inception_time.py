"""InceptionTime 1D-CNN for time-series classification.

Reference: Fawaz et al. 2020, "InceptionTime: Finding AlexNet for Time Series Classification"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseModel


class InceptionModule(nn.Module):
    """Single Inception module with 3 conv branches + MaxPool branch."""

    def __init__(self, in_channels: int, nb_filters: int = 32):
        super().__init__()
        # bottleneck before the 3 conv branches
        bottleneck_size = nb_filters
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)

        self.conv9 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=9, padding=4, bias=False)
        self.conv19 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=19, padding=9, bias=False)
        self.conv39 = nn.Conv1d(bottleneck_size, nb_filters, kernel_size=39, padding=19, bias=False)

        # MaxPool branch operates on original input
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        out_channels = nb_filters * 4
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        b = self.bottleneck(x)
        c1 = self.conv9(b)
        c2 = self.conv19(b)
        c3 = self.conv39(b)
        mp = self.mp_conv(self.maxpool(x))
        out = torch.cat([c1, c2, c3, mp], dim=1)
        return self.relu(self.bn(out))


class ResidualBlock(nn.Module):
    """Residual block wrapping two Inception modules."""

    def __init__(self, in_channels: int, nb_filters: int = 32):
        super().__init__()
        inception_out = nb_filters * 4
        self.inc1 = InceptionModule(in_channels, nb_filters)
        self.inc2 = InceptionModule(inception_out, nb_filters)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, inception_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(inception_out),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = self.shortcut(x)
        out = self.inc1(x)
        out = self.inc2(out)
        return self.relu(out + res)


class InceptionTime(BaseModel):
    """InceptionTime 1D-CNN classifier.

    3 Inception modules in a residual stack, GlobalAveragePooling, Linear head.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        nb_filters: int = 32,
        depth: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        layers = []
        current_channels = in_channels
        inception_out = nb_filters * 4
        n_blocks = depth // 2  # each ResidualBlock has 2 Inception modules

        for _ in range(max(n_blocks, 1)):
            layers.append(ResidualBlock(current_channels, nb_filters))
            current_channels = inception_out

        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, T)
        out = self.blocks(x)
        out = self.gap(out).squeeze(-1)  # (N, channels)
        return self.head(out)

    @property
    def name(self) -> str:
        return "InceptionTime"
