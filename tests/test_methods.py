"""Tests for imbalance methods."""
import torch
import pytest

from src.methods.cross_entropy import VanillaCE
from src.methods.weighted_ce import WeightedCE
from src.methods.focal_loss import FocalLoss
from src.methods.class_balanced import ClassBalancedLoss
from src.methods.ldam import LDAMLoss
from src.methods.logit_adjustment import LogitAdjustment
from src.methods.balanced_softmax import BalancedSoftmax
from src.methods.icmlt_baseline import ICMLTBaseline

CLASS_COUNTS = [100, 50, 25, 10]
N_CLASSES = len(CLASS_COUNTS)
BATCH_SIZE = 8


@pytest.fixture
def dummy_logits():
    t = torch.randn(BATCH_SIZE, N_CLASSES, requires_grad=True)
    return t


@pytest.fixture
def dummy_targets():
    return torch.randint(0, N_CLASSES, (BATCH_SIZE,))


def _check_loss(loss):
    assert loss.dim() == 0, "loss must be scalar"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.requires_grad, "loss must require grad for backprop"


class TestVanillaCE:
    def test_loss(self, dummy_logits, dummy_targets):
        method = VanillaCE()
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestWeightedCE:
    def test_loss(self, dummy_logits, dummy_targets):
        method = WeightedCE(CLASS_COUNTS)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestFocalLoss:
    def test_loss_no_alpha(self, dummy_logits, dummy_targets):
        method = FocalLoss(gamma=2.0)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)

    def test_loss_with_alpha(self, dummy_logits, dummy_targets):
        alpha = [1.0 / c for c in CLASS_COUNTS]
        method = FocalLoss(gamma=2.0, alpha=alpha)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestClassBalancedLoss:
    def test_loss_focal(self, dummy_logits, dummy_targets):
        method = ClassBalancedLoss(CLASS_COUNTS, loss_type="focal")
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)

    def test_loss_ce(self, dummy_logits, dummy_targets):
        method = ClassBalancedLoss(CLASS_COUNTS, loss_type="ce")
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestLDAMLoss:
    def test_loss(self, dummy_logits, dummy_targets):
        method = LDAMLoss(CLASS_COUNTS)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)

    def test_drw(self, dummy_logits, dummy_targets):
        method = LDAMLoss(CLASS_COUNTS, drw_epoch=5)
        method.set_epoch(10)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestLogitAdjustment:
    def test_loss(self, dummy_logits, dummy_targets):
        method = LogitAdjustment(CLASS_COUNTS)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)

    def test_adjust(self, dummy_logits):
        method = LogitAdjustment(CLASS_COUNTS)
        adjusted = method.adjust(dummy_logits)
        assert adjusted.shape == dummy_logits.shape


class TestBalancedSoftmax:
    def test_loss(self, dummy_logits, dummy_targets):
        method = BalancedSoftmax(CLASS_COUNTS)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)


class TestICMLTBaseline:
    def test_loss(self, dummy_logits, dummy_targets):
        method = ICMLTBaseline(CLASS_COUNTS)
        loss = method.compute_loss(dummy_logits, dummy_targets)
        _check_loss(loss)
