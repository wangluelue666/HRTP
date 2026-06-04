#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    """
    Mean pooling with binary mask support.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x   : [B, T, D]
            mask: [B, T], 1 valid, 0 invalid

        Returns:
            pooled: [B, D]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,D], got shape={tuple(x.shape)}")

        if mask is None:
            return x.mean(dim=1)

        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"mask shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
            )

        mask_f = mask.to(x.dtype).unsqueeze(-1)
        x_masked = x * mask_f
        denom = mask_f.sum(dim=1).clamp_min(self.eps)
        pooled = x_masked.sum(dim=1) / denom
        return pooled


class AttentiveMaskedPooling(nn.Module):
    """
    Learnable attentive pooling over temporal sequences.

    This module produces:
        - pooled sequence summary
        - attention weights over valid positions
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.temperature = float(temperature)

        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x            : [B, T, D]
            mask         : [B, T], 1 valid, 0 invalid
            return_weights: whether to return attention weights

        Returns:
            pooled       : [B, D]
            weights      : [B, T], optional
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,D], got shape={tuple(x.shape)}")

        scores = self.score_mlp(x).squeeze(-1) / self.temperature  # [B, T]

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
                )
            mask_bool = mask > 0
            scores = scores.masked_fill(~mask_bool, float("-inf"))
        else:
            mask_bool = torch.ones_like(scores, dtype=torch.bool)

        # Prevent fully masked rows from producing NaN.
        fully_masked = (~mask_bool).all(dim=1)
        if fully_masked.any():
            scores = scores.clone()
            scores[fully_masked] = 0.0

        weights = torch.softmax(scores, dim=1)  # [B, T]

        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        pooled = self.out_norm(pooled)

        if return_weights:
            return pooled, weights
        return pooled


if __name__ == "__main__":
    B, T, D = 4, 15, 256
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[0, 12:] = 0
    mask[1, 10:] = 0

    mean_pool = MaskedMeanPooling()
    attn_pool = AttentiveMaskedPooling(d_model=D, hidden_dim=D, dropout=0.1)

    y_mean = mean_pool(x, mask=mask)
    y_attn, w = attn_pool(x, mask=mask, return_weights=True)

    print("[INFO] mean pooled shape:", tuple(y_mean.shape))
    print("[INFO] attn pooled shape:", tuple(y_attn.shape))
    print("[INFO] attn weights shape:", tuple(w.shape))