"""Focal Loss (Lin et al. 2017)."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class FocalLoss(ImbalanceMethod):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default 2.0).
        alpha: per-class weight list or None for uniform weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[list] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            alpha_t = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer("alpha", alpha_t)
        else:
            self.alpha = None

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal_weight = at * focal_weight

        return (focal_weight * ce).mean()

    @property
    def name(self) -> str:
        return "FocalLoss"
