"""F1-gap-based adaptive scheduler."""
import numpy as np

from .adaptive_base import AdaptiveScheduler


class F1GapScheduler(AdaptiveScheduler):
    """Upweights classes whose validation F1 lags behind the best class.

    F1 gap for class c:
        f1_gap_c = F1_max - F1_c   (>= 0)

    Standardized score:
        score_c = (f1_gap_c - mean(f1_gap)) / (std(f1_gap) + eps)

    The standardized scores are already zero-mean across classes, so they
    serve directly as zero-sum signals for the exponential weight update.

    Parameters
    ----------
    n_classes : int
    eta : float
        Exponential update learning rate (default 0.1).
    eps : float
        Numerical stability constant (default 1e-8).
    """

    def __init__(
        self,
        n_classes: int,
        eta: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__(n_classes)
        self.eta = eta
        self.eps = eps

    def update(self, epoch: int, signals: dict) -> np.ndarray:
        """Update weights based on per-class validation F1 signals.

        Parameters
        ----------
        epoch : int
        signals : dict
            Must contain 'per_class_val_f1': np.ndarray (K,).

        Returns
        -------
        np.ndarray (K,) — updated weights.
        """
        val_f1: np.ndarray = np.asarray(
            signals["per_class_val_f1"], dtype=np.float64
        )

        # Compute F1 gap from best-performing class
        f1_max = val_f1.max()
        f1_gap = f1_max - val_f1  # >= 0 for all classes

        # Standardize gaps (zero-mean by construction after centering)
        mean_gap = f1_gap.mean()
        std_gap = f1_gap.std() + self.eps
        score = (f1_gap - mean_gap) / std_gap  # zero-sum (mean == 0)

        # Exponential weight update
        self._weights *= np.exp(self.eta * score)

        # Renormalize
        self._renormalize()
        self._record_history()
        return self.weights
