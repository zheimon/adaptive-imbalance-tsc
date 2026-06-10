"""Gradient-norm-based adaptive scheduler (stagnation detection)."""
import numpy as np

from .adaptive_base import AdaptiveScheduler


class GradientNormScheduler(AdaptiveScheduler):
    """Upweights classes whose gradient norm is below a relative threshold.

    Gradient stagnation score:
        grad_score_c = max(0, tau_grad - normalized_grad_c)

    where normalized_grad_c = ||nabla L_c||_2 / mean_c(||nabla L_c||_2).

    Scores are zero-sum normalized before the exponential weight update.

    Parameters
    ----------
    n_classes : int
    eta : float
        Exponential update learning rate (default 0.1).
    tau_grad : float
        Normalized gradient threshold below which stagnation is declared
        (default 0.5).
    eps : float
        Numerical stability constant (default 1e-8).
    """

    def __init__(
        self,
        n_classes: int,
        eta: float = 0.1,
        tau_grad: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__(n_classes)
        self.eta = eta
        self.tau_grad = tau_grad
        self.eps = eps

    def update(self, epoch: int, signals: dict) -> np.ndarray:
        """Update weights based on per-class gradient norm signals.

        Parameters
        ----------
        epoch : int
        signals : dict
            Must contain 'per_class_grad_norm': np.ndarray (K,).

        Returns
        -------
        np.ndarray (K,) — updated weights.
        """
        grad_norms: np.ndarray = np.asarray(
            signals["per_class_grad_norm"], dtype=np.float64
        )

        # Normalize gradient norms across classes
        mean_norm = grad_norms.mean() + self.eps
        normalized = grad_norms / mean_norm

        # Stagnation score: positive only for classes below tau_grad
        grad_score = np.maximum(0.0, self.tau_grad - normalized)

        # Zero-sum normalize so total weight mass is preserved
        mean_score = grad_score.mean()
        centered = grad_score - mean_score

        # Exponential weight update
        self._weights *= np.exp(self.eta * centered)

        # Renormalize
        self._renormalize()
        self._record_history()
        return self.weights
