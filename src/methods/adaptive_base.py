"""Abstract base class for AdaCAL adaptive schedulers."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class AdaptiveScheduler(ABC):
    """Abstract interface for AdaCAL adaptive schedulers.

    Subclasses maintain per-class weights that are updated each epoch
    based on convergence signals (loss trajectory, gradient norms, F1 gaps).
    """

    def __init__(self, n_classes: int):
        self._n_classes = n_classes
        self._weights: np.ndarray = np.ones(n_classes, dtype=np.float64)
        self.weight_history: list[np.ndarray] = []

    @abstractmethod
    def update(self, epoch: int, signals: dict) -> np.ndarray:
        """Compute updated weights given convergence signals.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        signals : dict
            Dictionary with keys:
            - 'per_class_loss': np.ndarray (K,) — mean training loss per class
            - 'per_class_grad_norm': np.ndarray (K,) — mean gradient norm per class
            - 'per_class_val_f1': np.ndarray (K,) — validation F1 per class

        Returns
        -------
        np.ndarray of shape (K,) — updated per-class weights.
        """
        ...

    @property
    def weights(self) -> np.ndarray:
        """Current per-class weights (copy)."""
        return self._weights.copy()

    def reset(self, n_classes: int) -> None:
        """Reset to uniform weights and clear history."""
        self._n_classes = n_classes
        self._weights = np.ones(n_classes, dtype=np.float64)
        self.weight_history = []

    def _renormalize(self) -> None:
        """Renormalize weights so mean == 1 (sum == n_classes)."""
        mean_w = self._weights.mean()
        if mean_w > 0:
            self._weights /= mean_w

    def _record_history(self) -> None:
        """Append a copy of current weights to history."""
        self.weight_history.append(self._weights.copy())

    def get_weight_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return current weights as a float32 tensor on the given device.

        Parameters
        ----------
        device : torch.device or None
            Target device (CPU if None).

        Returns
        -------
        torch.Tensor of shape (K,), dtype float32.
        """
        t = torch.tensor(self._weights, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t
