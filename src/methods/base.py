"""Abstract base class for imbalance methods."""
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class ImbalanceMethod(nn.Module, ABC):
    """Abstract base for all imbalance-handling loss methods."""

    @abstractmethod
    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute the (scalar) loss given logits and targets."""
        ...

    def forward(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return self.compute_loss(logits, targets, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name."""
        ...
