"""Vanilla cross-entropy loss."""
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class VanillaCE(ImbalanceMethod):
    """Standard cross-entropy loss — no imbalance correction."""

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.cross_entropy(logits, targets)

    @property
    def name(self) -> str:
        return "VanillaCE"
