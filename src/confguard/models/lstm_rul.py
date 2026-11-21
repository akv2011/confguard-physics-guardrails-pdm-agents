"""Bidirectional LSTM RUL regressor.

Two-layer bidirectional LSTM (hidden 64), last time step, MLP head, and a sigmoid
scaled by the RUL cap so predictions stay in [0, rul_cap].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    n_features: int = 17
    hidden_size: int = 64
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.0
    head_hidden: int = 64
    head_dropout: float = 0.3
    rul_cap: float = 125.0


class LSTMRUL(nn.Module):
    def __init__(self, cfg: LSTMConfig | None = None):
        super().__init__()
        cfg = cfg or LSTMConfig()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, window, n_features] -> RUL in cycles, shape [batch]."""
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1]).squeeze(-1)) * self.cfg.rul_cap
