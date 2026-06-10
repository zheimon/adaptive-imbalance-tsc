"""Abstract base class for all models."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Abstract base model. All models return logits of shape (N, C)."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Returns logits (N, C)."""
        raise NotImplementedError

    def get_logits(self, x: Tensor) -> Tensor:
        """Alias for forward."""
        return self.forward(x)

    def count_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        ...
