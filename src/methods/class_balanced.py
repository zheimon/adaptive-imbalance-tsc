"""Class-Balanced Loss (Cui et al. 2019)."""
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class ClassBalancedLoss(ImbalanceMethod):
    """Effective number weighting: E_n = (1 - beta^n) / (1 - beta).

    Args:
        class_counts: number of samples per class.
        beta: hyperparameter in (0, 1), default 0.9999.
        loss_type: 'focal' or 'ce'.
        gamma: focal loss gamma (used when loss_type='focal').
    """

    def __init__(
        self,
        class_counts: list,
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 0.5,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma

        counts = torch.tensor(class_counts, dtype=torch.float)
        effective_num = (1.0 - beta ** counts) / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights)

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        weights = self.weights.to(logits.device)
        if self.loss_type == "focal":
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** self.gamma
            at = weights[targets]
            return (at * focal_weight * ce).mean()
        else:
            return F.cross_entropy(logits, targets, weight=weights)

    @property
    def name(self) -> str:
        return "ClassBalancedLoss"
