"""Tests for the Trainer class."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm import LSTMClassifier
from src.methods.cross_entropy import VanillaCE
from src.training.trainer import Trainer


N_TRAIN = 64
N_VAL = 32
IN_CHANNELS = 3
SEQ_LEN = 20
N_CLASSES = 3


def make_synthetic_loader(n_samples: int, batch_size: int = 16) -> DataLoader:
    """Create a simple synthetic DataLoader."""
    torch.manual_seed(0)
    x = torch.randn(n_samples, IN_CHANNELS, SEQ_LEN)
    y = torch.randint(0, N_CLASSES, (n_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def make_trainer(epochs: int = 2, patience: int = None) -> Trainer:
    model = LSTMClassifier(
        in_channels=IN_CHANNELS, n_classes=N_CLASSES, hidden_dim=16, num_layers=1
    )
    method = VanillaCE()
    config = {
        "training": {"epochs": epochs, "patience": patience},
        "run_name": "test_run",
    }
    return Trainer(model=model, method=method, config=config)


# ---------------------------------------------------------------------------
# Test 1: trainer runs 2 epochs and produces finite loss
# ---------------------------------------------------------------------------
def test_trainer_runs_2_epochs():
    """Train for 2 epochs and verify loss values are finite."""
    trainer = make_trainer(epochs=2)
    train_loader = make_synthetic_loader(N_TRAIN)
    val_loader = make_synthetic_loader(N_VAL)

    history = trainer.fit(train_loader, val_loader)

    assert len(history["train_loss"]) == 2
    assert len(history["val_macro_f1"]) == 2
    for loss_val in history["train_loss"]:
        assert np.isfinite(loss_val), f"Train loss not finite: {loss_val}"
    for loss_val in history["val_loss"]:
        assert np.isfinite(loss_val), f"Val loss not finite: {loss_val}"


# ---------------------------------------------------------------------------
# Test 2: evaluate returns all expected metric keys
# ---------------------------------------------------------------------------
def test_evaluate_returns_all_metrics():
    """evaluate() should return all expected metric keys."""
    trainer = make_trainer(epochs=1)
    train_loader = make_synthetic_loader(N_TRAIN)
    val_loader = make_synthetic_loader(N_VAL)

    trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(val_loader)

    expected_keys = {"val_loss", "val_macro_f1", "val_balanced_acc", "per_class_prf", "per_class_f1"}
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"

    assert np.isfinite(metrics["val_loss"])
    assert np.isfinite(metrics["val_macro_f1"])
    assert np.isfinite(metrics["val_balanced_acc"])


# ---------------------------------------------------------------------------
# Test 3: early stopping triggers when val_f1 never improves
# ---------------------------------------------------------------------------
def test_early_stopping():
    """With patience=1 and a model/data that never improves val F1,
    training should stop before the max epoch count.
    """
    # Use a very short training run with patience=1
    # To reliably trigger early stopping: use many epochs but low patience
    max_epochs = 10
    trainer = make_trainer(epochs=max_epochs, patience=1)

    # Use fixed val data with identical labels — macro F1 will be constant
    torch.manual_seed(99)
    x_train = torch.randn(N_TRAIN, IN_CHANNELS, SEQ_LEN)
    y_train = torch.zeros(N_TRAIN, dtype=torch.long)  # all class 0
    x_val = torch.randn(N_VAL, IN_CHANNELS, SEQ_LEN)
    y_val = torch.zeros(N_VAL, dtype=torch.long)  # all class 0

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

    history = trainer.fit(train_loader, val_loader)

    # With patience=1 and stable val_f1, should stop well before max_epochs
    assert len(history["train_loss"]) < max_epochs, (
        f"Expected early stopping before {max_epochs} epochs, "
        f"but trained for {len(history['train_loss'])} epochs"
    )
