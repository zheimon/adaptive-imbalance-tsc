"""2-layer LSTM classifier for time-series."""
import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseModel


class LSTMClassifier(BaseModel):
    """2-layer LSTM, hidden_dim=64, dropout=0.3, then linear head.

    Input: (N, C, T) — permuted to (N, T, C) before LSTM.
    Output: logits (N, n_classes).
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, T) -> (N, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        # Take last time step
        out = out[:, -1, :]
        return self.head(out)

    @property
    def name(self) -> str:
        return "LSTMClassifier"
