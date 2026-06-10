"""AdaCAL: Adaptive Convergence-Aware Loss — hybrid scheduler combining all signals."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .adaptive_base import AdaptiveScheduler
from .adaptive_loss_trajectory import LossTrajectoryScheduler
from .adaptive_gradient_norm import GradientNormScheduler
from .adaptive_f1_gap import F1GapScheduler
from .base import ImbalanceMethod


class AdaCAL(ImbalanceMethod, AdaptiveScheduler):
    """Adaptive Convergence-Aware Loss (AdaCAL).

    Combines three convergence signals:
      1. Loss trajectory plateau detection (LossTrajectoryScheduler)
      2. Gradient norm stagnation detection (GradientNormScheduler)
      3. Validation F1 gap (F1GapScheduler)

    The composite score is a weighted sum of the individual scores:
        s_c(t) = alpha_1 * plateau_score_c + alpha_2 * grad_score_c + alpha_3 * f1_gap_score_c

    Scores are zero-sum normalized, then used in exponential weight updates.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    eta : float
        Exponential update learning rate (default 0.1).
    alpha : tuple of 3 floats
        Weights for (plateau, gradient, f1_gap) signals. Must sum to 1.
        Default (0.4, 0.3, 0.3).
    k : int
        Lookback window for loss trajectory (default 5).
    eps : float
        Numerical stability constant (default 1e-8).
    """

    def __init__(
        self,
        n_classes: int,
        eta: float = 0.1,
        alpha: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        k: int = 5,
        eps: float = 1e-8,
    ):
        # Initialize AdaptiveScheduler (sets _weights, weight_history)
        AdaptiveScheduler.__init__(self, n_classes)
        # Initialize nn.Module
        ImbalanceMethod.__init__(self)

        self.n_classes = n_classes
        self.eta = eta
        self.alpha = alpha
        self.k = k
        self.eps = eps

        self._loss_sched = LossTrajectoryScheduler(n_classes, eta=eta, k=k, eps=eps)
        self._grad_sched = GradientNormScheduler(n_classes, eta=eta, eps=eps)
        self._f1_sched = F1GapScheduler(n_classes, eta=eta, eps=eps)

    # ------------------------------------------------------------------
    # AdaptiveScheduler interface
    # ------------------------------------------------------------------

    def update(self, epoch: int, signals: dict) -> np.ndarray:
        """Combine sub-scheduler signals and update weights.

        Parameters
        ----------
        epoch : int
        signals : dict
            Keys: 'per_class_loss', 'per_class_grad_norm', 'per_class_val_f1'

        Returns
        -------
        np.ndarray (K,) — updated weights (mean == 1).
        """
        alpha_1, alpha_2, alpha_3 = self.alpha

        # ---- plateau score ----
        per_class_loss = np.asarray(signals["per_class_loss"], dtype=np.float64)
        # Sync sub-scheduler loss history by feeding current loss
        self._loss_sched.update(epoch, {"per_class_loss": per_class_loss})

        # Re-compute plateau_score directly (binary: in plateau or not)
        plateau_score = np.zeros(self.n_classes, dtype=np.float64)
        for c in range(self.n_classes):
            hist = self._loss_sched._loss_history[c]
            if len(hist) >= self.k + 1:
                l_now = hist[-1]
                l_prev = hist[-(self.k + 1)]
                delta = abs(l_now - l_prev) / (abs(l_prev) + self.eps)
                if delta < self._loss_sched.tau_plateau:
                    plateau_score[c] = 1.0

        # ---- gradient score ----
        grad_norms = np.asarray(signals["per_class_grad_norm"], dtype=np.float64)
        self._grad_sched.update(epoch, {"per_class_grad_norm": grad_norms})

        mean_norm = grad_norms.mean() + self.eps
        normalized_grad = grad_norms / mean_norm
        grad_score = np.maximum(0.0, self._grad_sched.tau_grad - normalized_grad)

        # ---- f1 gap score ----
        val_f1 = np.asarray(signals["per_class_val_f1"], dtype=np.float64)
        self._f1_sched.update(epoch, {"per_class_val_f1": val_f1})

        f1_max = val_f1.max()
        f1_gap = f1_max - val_f1
        mean_gap = f1_gap.mean()
        std_gap = f1_gap.std() + self.eps
        f1_gap_score = (f1_gap - mean_gap) / std_gap

        # ---- composite signal ----
        s_c = alpha_1 * plateau_score + alpha_2 * grad_score + alpha_3 * f1_gap_score

        # Zero-sum normalize
        s_c = s_c - s_c.mean()

        # Exponential update
        self._weights *= np.exp(self.eta * s_c)

        # Renormalize
        self._renormalize()
        self._record_history()
        return self.weights

    def reset(self, n_classes: int) -> None:
        AdaptiveScheduler.reset(self, n_classes)
        self.n_classes = n_classes
        self._loss_sched.reset(n_classes)
        self._grad_sched.reset(n_classes)
        self._f1_sched.reset(n_classes)

    # ------------------------------------------------------------------
    # ImbalanceMethod interface
    # ------------------------------------------------------------------

    def compute_loss(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute per-class weighted cross-entropy.

        Parameters
        ----------
        logits : Tensor (N, K)
        targets : Tensor (N,) — class indices
        **kwargs :
            update_weights : bool (default False)
                If True, call self.update() using signals from kwargs.
                Required keys when True: 'epoch', 'per_class_loss',
                'per_class_grad_norm', 'per_class_val_f1'.

        Returns
        -------
        Scalar loss Tensor.
        """
        if kwargs.get("update_weights", False):
            epoch = kwargs.get("epoch", 0)
            signals = {
                "per_class_loss": kwargs["per_class_loss"],
                "per_class_grad_norm": kwargs["per_class_grad_norm"],
                "per_class_val_f1": kwargs["per_class_val_f1"],
            }
            self.update(epoch, signals)

        device = logits.device
        weight_tensor = self.get_weight_tensor(device)
        return F.cross_entropy(logits, targets, weight=weight_tensor)

    @property
    def name(self) -> str:
        return "AdaCAL"
