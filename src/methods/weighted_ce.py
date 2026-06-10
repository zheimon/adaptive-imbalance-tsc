"""Inverse-frequency weighted cross-entropy."""
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class WeightedCE(ImbalanceMethod):
    """Inverse-frequency class weights: w_c = N / (K * n_c)."""

    def __init__(self, class_counts: list):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        N = counts.sum()
        K = len(counts)
        weights = N / (K * counts)
        self.register_buffer("weights", weights)

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))

    @property
    def name(self) -> str:
        return "WeightedCE"
