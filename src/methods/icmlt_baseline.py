"""ICMLT baseline method from zheimon/Stock-Predictor submission."""
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class ICMLTBaseline(ImbalanceMethod):
    """Baseline from zheimon/Stock-Predictor ICMLT submission.

    SMOTE oversampling + Focal Loss (gamma=2) + inverse-frequency class weights.
    This is the method being improved upon, NOT our contribution.

    Note: SMOTE is applied at data-prep time. This class handles only the loss
    component: Focal Loss with alpha=inverse_freq_weights, gamma=2.0.
    """

    def __init__(self, class_counts: list):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        inv_freq = 1.0 / counts
        inv_freq = inv_freq / inv_freq.sum()  # normalize to sum to 1
        self.register_buffer("alpha", inv_freq)
        self.gamma = 2.0

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        alpha = self.alpha.to(logits.device)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma
        at = alpha[targets]
        return (at * focal_weight * ce).mean()

    @property
    def name(self) -> str:
        return "ICMLTBaseline"
