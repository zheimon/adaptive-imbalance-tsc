"""PatchTST classification model.

Patches the time series, embeds each patch linearly, and passes through a
Transformer encoder. Classification via mean-pooling over patch tokens.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseModel


class PatchTST(BaseModel):
    """PatchTST for time-series classification.

    Args:
        in_channels: number of input channels (variates).
        n_classes: number of output classes.
        seq_len: length of the input time series.
        patch_len: length of each patch.
        stride: stride between patches.
        d_model: transformer hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        seq_len: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Linear patch embedding: (patch_len * in_channels) -> d_model
        self.patch_embed = nn.Linear(patch_len * in_channels, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding for patches + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _patchify(self, x: Tensor) -> Tensor:
        """Extract patches from input.

        Args:
            x: (N, C, T)

        Returns:
            patches: (N, n_patches, patch_len * C)
        """
        N, C, T = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, :, start:end]  # (N, C, patch_len)
            patches.append(patch.reshape(N, -1))  # (N, C * patch_len)
        return torch.stack(patches, dim=1)  # (N, n_patches, C * patch_len)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]

        # Patchify and embed
        patches = self._patchify(x)  # (N, n_patches, patch_len * C)
        tokens = self.patch_embed(patches)  # (N, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (N, n_patches+1, d_model)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Use CLS token for classification
        cls_out = tokens[:, 0, :]  # (N, d_model)
        return self.head(cls_out)

    @property
    def name(self) -> str:
        return "PatchTST"
