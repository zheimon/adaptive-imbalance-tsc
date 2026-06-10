"""Tests for AdaCAL adaptive schedulers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.methods.adaptive_loss_trajectory import LossTrajectoryScheduler
from src.methods.adaptive_f1_gap import F1GapScheduler
from src.methods.adaptive_gradient_norm import GradientNormScheduler
from src.methods.adaptive_hybrid import AdaCAL


def make_signals(loss=None, grad=None, f1=None, n=2):
    return {
        "per_class_loss": np.array(loss or [1.0] * n),
        "per_class_grad_norm": np.array(grad or [1.0] * n),
        "per_class_val_f1": np.array(f1 or [0.5] * n),
    }


# ---------------------------------------------------------------------------
# Test 1: stagnant class gets higher weight from LossTrajectoryScheduler
# ---------------------------------------------------------------------------
def test_stagnant_class_gets_higher_weight():
    """Class 0 has flat loss (plateau), class 1 has decreasing loss.
    After 6 epochs, w[0] should be > w[1].
    """
    sched = LossTrajectoryScheduler(n_classes=2, eta=0.1, tau_plateau=0.01, k=5)

    for epoch in range(6):
        # Class 0: constant loss = 1.0 (plateau)
        # Class 1: decreasing loss
        loss_0 = 1.0
        loss_1 = 1.0 - 0.1 * (epoch + 1)
        sched.update(epoch, {"per_class_loss": np.array([loss_0, loss_1])})

    w = sched.weights
    assert w[0] > w[1], f"Expected w[0]={w[0]:.4f} > w[1]={w[1]:.4f}"


# ---------------------------------------------------------------------------
# Test 2: F1 gap upweights weak class
# ---------------------------------------------------------------------------
def test_f1_gap_upweights_weak_class():
    """Class 0 has high F1=0.9, class 1 has low F1=0.3.
    After one update, w[1] should be > w[0].
    """
    sched = F1GapScheduler(n_classes=2, eta=0.5)
    sched.update(0, {"per_class_val_f1": np.array([0.9, 0.3]),
                     "per_class_loss": np.array([0.5, 0.5]),
                     "per_class_grad_norm": np.array([1.0, 1.0])})
    w = sched.weights
    assert w[1] > w[0], f"Expected w[1]={w[1]:.4f} > w[0]={w[0]:.4f}"


# ---------------------------------------------------------------------------
# Test 3: grad stagnation upweights class with small gradient
# ---------------------------------------------------------------------------
def test_grad_stagnation_upweights_class():
    """Class 0 has tiny gradient norm (0.01), class 1 has large (1.0).
    After one update, w[0] should be > w[1].
    """
    sched = GradientNormScheduler(n_classes=2, eta=0.5, tau_grad=0.5)
    sched.update(0, {
        "per_class_grad_norm": np.array([0.01, 1.0]),
        "per_class_loss": np.array([1.0, 1.0]),
        "per_class_val_f1": np.array([0.5, 0.5]),
    })
    w = sched.weights
    assert w[0] > w[1], f"Expected w[0]={w[0]:.4f} > w[1]={w[1]:.4f}"


# ---------------------------------------------------------------------------
# Test 4: AdaCAL weights sum to n_classes after updates
# ---------------------------------------------------------------------------
def test_adacal_weights_sum_to_n_classes():
    """After 5 updates, sum of AdaCAL weights should equal n_classes."""
    n_classes = 4
    adacal = AdaCAL(n_classes=n_classes, eta=0.1)

    rng = np.random.default_rng(42)
    for epoch in range(5):
        signals = {
            "per_class_loss": rng.uniform(0.5, 2.0, size=n_classes),
            "per_class_grad_norm": rng.uniform(0.1, 2.0, size=n_classes),
            "per_class_val_f1": rng.uniform(0.2, 0.9, size=n_classes),
        }
        adacal.update(epoch, signals)

    w = adacal.weights
    assert abs(w.sum() - n_classes) < 1e-6, (
        f"Expected weights sum = {n_classes}, got {w.sum():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5: AdaCAL compute_loss returns finite scalar
# ---------------------------------------------------------------------------
def test_adacal_compute_loss_finite():
    """compute_loss should return a finite scalar tensor."""
    n_classes = 3
    adacal = AdaCAL(n_classes=n_classes, eta=0.1)

    logits = torch.randn(16, n_classes)
    targets = torch.randint(0, n_classes, (16,))

    loss = adacal.compute_loss(logits, targets)
    assert loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 6: weight history length
# ---------------------------------------------------------------------------
def test_weight_history_length():
    """After k updates, len(scheduler.weight_history) == k."""
    k = 7
    sched = LossTrajectoryScheduler(n_classes=3, eta=0.1, k=5)

    for epoch in range(k):
        sched.update(
            epoch,
            {"per_class_loss": np.array([1.0, 0.9 - 0.05 * epoch, 0.8])},
        )

    assert len(sched.weight_history) == k, (
        f"Expected {k} history entries, got {len(sched.weight_history)}"
    )
