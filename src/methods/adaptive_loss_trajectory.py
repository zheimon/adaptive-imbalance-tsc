"""Loss-trajectory-based adaptive scheduler (plateau detection)."""
from collections import deque

import numpy as np

from .adaptive_base import AdaptiveScheduler


class LossTrajectoryScheduler(AdaptiveScheduler):
    """Upweights classes whose per-class loss has plateaued.

    A class is in plateau at epoch t if:
        delta_c(t) = |L_c(t) - L_c(t-k)| / (L_c(t-k) + eps) < tau_plateau

    For classes that have been in plateau for > consecutive_thresh consecutive
    epochs a 2x boost multiplier is applied on top of the exponential update.

    Parameters
    ----------
    n_classes : int
    eta : float
        Exponential update learning rate (default 0.1).
    tau_plateau : float
        Relative change threshold to declare plateau (default 0.01).
    k : int
        Lookback window in epochs (default 5).
    eps : float
        Numerical stability constant (default 1e-8).
    consecutive_thresh : int
        Number of consecutive plateau epochs before 2x boost (default 3).
    """

    def __init__(
        self,
        n_classes: int,
        eta: float = 0.1,
        tau_plateau: float = 0.01,
        k: int = 5,
        eps: float = 1e-8,
        consecutive_thresh: int = 3,
    ):
        super().__init__(n_classes)
        self.eta = eta
        self.tau_plateau = tau_plateau
        self.k = k
        self.eps = eps
        self.consecutive_thresh = consecutive_thresh

        # History buffer: one deque per class, max length k+1
        self._loss_history: list[deque] = [
            deque(maxlen=k + 1) for _ in range(n_classes)
        ]
        # Count consecutive plateau epochs per class
        self._plateau_streak: np.ndarray = np.zeros(n_classes, dtype=int)

    def update(self, epoch: int, signals: dict) -> np.ndarray:
        """Update weights based on per-class loss plateau signals.

        Parameters
        ----------
        epoch : int
        signals : dict
            Must contain 'per_class_loss': np.ndarray (K,).

        Returns
        -------
        np.ndarray (K,) — updated weights.
        """
        per_class_loss: np.ndarray = np.asarray(
            signals["per_class_loss"], dtype=np.float64
        )

        # Record loss in history
        for c in range(self._n_classes):
            self._loss_history[c].append(per_class_loss[c])

        # Compute plateau scores
        plateau_score = np.zeros(self._n_classes, dtype=np.float64)
        for c in range(self._n_classes):
            hist = self._loss_history[c]
            if len(hist) >= self.k + 1:
                l_now = hist[-1]
                l_prev = hist[-(self.k + 1)]
                delta = abs(l_now - l_prev) / (abs(l_prev) + self.eps)
                in_plateau = delta < self.tau_plateau
                if in_plateau:
                    self._plateau_streak[c] += 1
                    plateau_score[c] = 1.0
                else:
                    self._plateau_streak[c] = 0

        # Zero-sum normalize scores
        mean_score = plateau_score.mean()
        centered = plateau_score - mean_score

        # Exponential update
        self._weights *= np.exp(self.eta * centered)

        # Apply 2x boost for classes in long plateau
        for c in range(self._n_classes):
            if self._plateau_streak[c] > self.consecutive_thresh:
                self._weights[c] *= 2.0

        # Renormalize so mean weight == 1
        self._renormalize()
        self._record_history()
        return self.weights
