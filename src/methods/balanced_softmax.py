"""Balanced Softmax (Ren et al. 2020)."""
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class BalancedSoftmax(ImbalanceMethod):
    """Augment logits with log(n_k) before softmax.

    Args:
        class_counts: number of samples per class.
    """

    def __init__(self, class_counts: list):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        log_counts = torch.log(counts)
        self.register_buffer("log_counts", log_counts)

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        adjusted = logits + self.log_counts.to(logits.device)
        return F.cross_entropy(adjusted, targets)

    @property
    def name(self) -> str:
        return "BalancedSoftmax"
