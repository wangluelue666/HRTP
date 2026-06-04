#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerPredictorV3(nn.Module):
    """
    Standard Transformer encoder-based predictor.

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
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_steps: int = 6,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.num_steps = int(num_steps)

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, 6, C)
        """
        h = self.input_proj(x)         # (B, T, D)
        h = self.encoder(h)            # (B, T, D)

        last = h[:, -1, :]             # (B, D)
        last = last.unsqueeze(1).repeat(1, self.num_steps, 1)

        logits = self.head(last)       # (B, 6, C)
        return logits