"""Tests for model implementations."""
import torch
import pytest

from src.models.lstm import LSTMClassifier
from src.models.inception_time import InceptionTime
from src.models.patchtst import PatchTST


N_CLASSES = 4
IN_CHANNELS = 1
SEQ_LEN = 128
BATCH_SIZE = 4


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)


class TestLSTMClassifier:
    def setup_method(self):
        self.model = LSTMClassifier(
            in_channels=IN_CHANNELS, n_classes=N_CLASSES
        )

    def test_output_shape(self, dummy_input):
        out = self.model(dummy_input)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_count_params(self):
        assert self.model.count_params() > 0

    def test_no_nan(self, dummy_input):
        out = self.model(dummy_input)
        assert not torch.isnan(out).any()

    def test_get_logits_alias(self, dummy_input):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(dummy_input)
            out2 = self.model.get_logits(dummy_input)
        assert torch.allclose(out1, out2)


class TestInceptionTime:
    def setup_method(self):
        self.model = InceptionTime(
            in_channels=IN_CHANNELS, n_classes=N_CLASSES
        )

    def test_output_shape(self, dummy_input):
        out = self.model(dummy_input)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_count_params(self):
        assert self.model.count_params() > 0

    def test_no_nan(self, dummy_input):
        out = self.model(dummy_input)
        assert not torch.isnan(out).any()

    def test_get_logits_alias(self, dummy_input):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(dummy_input)
            out2 = self.model.get_logits(dummy_input)
        assert torch.allclose(out1, out2)


class TestPatchTST:
    def setup_method(self):
        self.model = PatchTST(
            in_channels=IN_CHANNELS,
            n_classes=N_CLASSES,
            seq_len=SEQ_LEN,
        )

    def test_output_shape(self, dummy_input):
        out = self.model(dummy_input)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_count_params(self):
        assert self.model.count_params() > 0

    def test_no_nan(self, dummy_input):
        out = self.model(dummy_input)
        assert not torch.isnan(out).any()

    def test_get_logits_alias(self, dummy_input):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(dummy_input)
            out2 = self.model.get_logits(dummy_input)
        assert torch.allclose(out1, out2)
