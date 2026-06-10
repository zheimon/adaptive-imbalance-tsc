"""LDAM Loss with optional Deferred Re-Weighting (Cao et al. 2019)."""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ImbalanceMethod


class LDAMLoss(ImbalanceMethod):
    """Label-Distribution-Aware Margin Loss.

    Margins proportional to n_j^{-1/4}. Supports deferred re-weighting (DRW):
    uniform class weights for the first `drw_epoch` epochs, then inverse-freq
    weights thereafter.

    Args:
        class_counts: number of samples per class.
        max_m: maximum margin value (default 0.5).
        s: logit scale factor (default 30).
        drw_epoch: epoch at which DRW kicks in; None means always uniform weights.
    """

    def __init__(
        self,
        class_counts: list,
        max_m: float = 0.5,
        s: float = 30,
        drw_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.s = s
        self.drw_epoch = drw_epoch
        self._current_epoch = 0

        counts = torch.tensor(class_counts, dtype=torch.float)
        # Margins: Delta_j = C / n_j^{1/4}
        m_list = 1.0 / (counts ** 0.25)
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)

        # Inverse-frequency weights for DRW phase
        inv_freq = 1.0 / counts
        inv_freq = inv_freq / inv_freq.sum() * len(counts)
        self.register_buffer("inv_freq_weights", inv_freq)

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for DRW scheduling."""
        self._current_epoch = epoch

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        # Apply per-class margin to the target class logit
        m_list = self.m_list.to(logits.device)
        batch_m = m_list[targets]  # (N,)
        # Subtract margin from the correct-class logit
        logits_m = logits.clone()
        logits_m[torch.arange(len(targets)), targets] -= batch_m

        scaled = self.s * logits_m

        # DRW: use inverse-freq weights after drw_epoch
        if self.drw_epoch is not None and self._current_epoch >= self.drw_epoch:
            weights = self.inv_freq_weights.to(logits.device)
            return F.cross_entropy(scaled, targets, weight=weights)
        else:
            return F.cross_entropy(scaled, targets)

    @property
    def name(self) -> str:
        return "LDAMLoss"
