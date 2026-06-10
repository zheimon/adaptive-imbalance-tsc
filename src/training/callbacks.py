"""Training callbacks for AdaCAL signal collection."""
from typing import List, Optional, TYPE_CHECKING
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.methods.adaptive_base import AdaptiveScheduler


class PerClassMonitorCallback:
    """Collects per-class training signals and publishes them to AdaptiveSchedulers.

    At the end of each epoch this callback:
      1. Receives per-class mean training loss (passed in via on_epoch_end).
      2. Receives per-class validation F1 (passed in via on_epoch_end).
      3. Computes per-class gradient norms using a small probe backward pass.
      4. Publishes the combined signals dict to all registered schedulers.
      5. Stores the full signal history for later analysis.

    Parameters
    ----------
    n_classes : int
    schedulers : list of AdaptiveScheduler, optional
        Schedulers to notify on each epoch end.
    """

    def __init__(
        self,
        n_classes: int,
        schedulers: Optional[List["AdaptiveScheduler"]] = None,
    ):
        self.n_classes = n_classes
        self.schedulers: List["AdaptiveScheduler"] = list(schedulers or [])
        self.signal_history: List[dict] = []

    def register_scheduler(self, scheduler: "AdaptiveScheduler") -> None:
        """Register a scheduler to receive signal updates."""
        self.schedulers.append(scheduler)

    def on_epoch_end(
        self,
        epoch: int,
        signals: dict,
    ) -> None:
        """Called at end of each training epoch.

        Parameters
        ----------
        epoch : int
        signals : dict
            Expected keys:
            - 'per_class_loss': np.ndarray (K,) — mean training loss per class
            - 'per_class_val_f1': np.ndarray (K,) — validation F1 per class
            - 'per_class_grad_norm': np.ndarray (K,), optional — if not provided,
              will default to uniform norms.
        """
        per_class_loss = np.asarray(
            signals.get("per_class_loss", np.zeros(self.n_classes)),
            dtype=np.float64,
        )
        per_class_val_f1 = np.asarray(
            signals.get("per_class_val_f1", np.zeros(self.n_classes)),
            dtype=np.float64,
        )
        per_class_grad_norm = np.asarray(
            signals.get("per_class_grad_norm", np.ones(self.n_classes)),
            dtype=np.float64,
        )

        full_signals = {
            "per_class_loss": per_class_loss,
            "per_class_val_f1": per_class_val_f1,
            "per_class_grad_norm": per_class_grad_norm,
        }

        # Store history
        self.signal_history.append({"epoch": epoch, **full_signals})

        # Publish to all registered schedulers
        for sched in self.schedulers:
            sched.update(epoch, full_signals)


def collect_per_class_grad_norms(
    model: torch.nn.Module,
    probe_batches: List[Tensor],  # list of (x, y) tensors, one per class
    criterion,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute per-class gradient norms via probe backward passes.

    For each class c, run a forward+backward on a small probe batch and
    record the L2 norm of the gradient w.r.t. model parameters.

    Parameters
    ----------
    model : nn.Module
    probe_batches : list of (x_c, y_c) tensors
        One entry per class with a small probe batch.
    criterion : callable
        Loss function: criterion(logits, targets) -> scalar.
    device : torch.device, optional

    Returns
    -------
    np.ndarray (K,) — per-class gradient norms.
    """
    n_classes = len(probe_batches)
    grad_norms = np.zeros(n_classes, dtype=np.float64)

    original_grads = {
        name: (p.grad.clone() if p.grad is not None else None)
        for name, p in model.named_parameters()
    }

    for c, (x_c, y_c) in enumerate(probe_batches):
        if device is not None:
            x_c = x_c.to(device)
            y_c = y_c.to(device)

        model.zero_grad()
        logits = model(x_c)
        loss = criterion(logits, y_c)
        loss.backward()

        norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                norm_sq += p.grad.detach().pow(2).sum().item()
        grad_norms[c] = float(norm_sq ** 0.5)

    # Restore original gradients
    model.zero_grad()
    for name, p in model.named_parameters():
        if original_grads[name] is not None:
            p.grad = original_grads[name]

    return grad_norms
