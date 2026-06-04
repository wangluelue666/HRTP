#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPredictorV3(nn.Module):
    """
    LSTM-based multi-step classification predictor.

    Input
    -----
    x: (B, T, D)

    Output
    ------
    logits: (B, 6, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_steps: int = 6,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_steps = int(num_steps)
        self.num_classes = int(num_classes)

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, 6, C)
        """
        out, _ = self.lstm(x)              # (B, T, H)
        last = out[:, -1, :]               # (B, H)

        # expand to 6 steps
        last = last.unsqueeze(1).repeat(1, self.num_steps, 1)  # (B, 6, H)

        logits = self.head(last)           # (B, 6, C)
        return logits