"""Logit Adjustment (Menon et al. 2021)."""
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class LogitAdjustment(ImbalanceMethod):
    """Adjust logits by log(pi_y) at train and/or test time.

    Args:
        class_counts: number of samples per class.
        tau: temperature scaling for the adjustment (default 1.0).
        apply_at_train: whether to apply adjustment during training.
    """

    def __init__(self, class_counts: list, tau: float = 1.0, apply_at_train: bool = True):
        super().__init__()
        self.tau = tau
        self.apply_at_train = apply_at_train

        counts = torch.tensor(class_counts, dtype=torch.float)
        log_prior = torch.log(counts / counts.sum())
        self.register_buffer("log_prior", log_prior)

    def adjust(self, logits: Tensor) -> Tensor:
        """Post-hoc test-time logit adjustment."""
        return logits + self.tau * self.log_prior.to(logits.device)

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        if self.apply_at_train:
            logits = self.adjust(logits)
        return F.cross_entropy(logits, targets)

    @property
    def name(self) -> str:
        return "LogitAdjustment"
