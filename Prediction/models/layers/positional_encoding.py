#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_future_year_ids(
    batch_size: int,
    future_steps: int,
    start_year: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build future year ids for autoregressive decoding.

    Returns:
        year_ids: [B, F]
    """
    years = torch.arange(start_year, start_year + future_steps, device=device, dtype=torch.long)
    return years.unsqueeze(0).expand(batch_size, -1).contiguous()


def build_future_recency_ids(
    batch_size: int,
    future_steps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build future recency ids for autoregressive decoding.

    Returns:
        recency_ids: [B, F], values in [0, F-1]
    """
    recency = torch.arange(future_steps, device=device, dtype=torch.long)
    return recency.unsqueeze(0).expand(batch_size, -1).contiguous()


class LearnedTemporalPositionalEncoding(nn.Module):
    """
    Learned temporal positional encoding for aligned yearly sequences.

    This module supports:
        - year embeddings
        - recency embeddings
        - optional learned scaling
        - mask-aware output cleanup

    Expected inputs:
        x          : [B, T, D]
        year_ids   : [B, T], e.g. 2000...2014
        recency_ids: [B, T], e.g. 0...14
        mask       : [B, T], 1 for valid, 0 for invalid
    """

    def __init__(
        self,
        d_model: int,
        max_year_tokens: int = 64,
        max_recency_tokens: int = 64,
        base_year: int = 2000,
        use_year_embedding: bool = True,
        use_recency_embedding: bool = True,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        learned_scale: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.base_year = int(base_year)
        self.max_year_tokens = int(max_year_tokens)
        self.max_recency_tokens = int(max_recency_tokens)
        self.use_year_embedding = bool(use_year_embedding)
        self.use_recency_embedding = bool(use_recency_embedding)

        if self.use_year_embedding:
            self.year_embedding = nn.Embedding(self.max_year_tokens, d_model)
        else:
            self.year_embedding = None

        if self.use_recency_embedding:
            self.recency_embedding = nn.Embedding(self.max_recency_tokens, d_model)
        else:
            self.recency_embedding = None

        if learned_scale:
            self.year_scale = nn.Parameter(torch.tensor(1.0))
            self.recency_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("year_scale", torch.tensor(1.0), persistent=False)
            self.register_buffer("recency_scale", torch.tensor(1.0), persistent=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.year_embedding is not None:
            nn.init.normal_(self.year_embedding.weight, mean=0.0, std=0.02)
        if self.recency_embedding is not None:
            nn.init.normal_(self.recency_embedding.weight, mean=0.0, std=0.02)

    def _normalize_year_ids(self, year_ids: torch.Tensor) -> torch.Tensor:
        rel_year = year_ids - self.base_year
        rel_year = torch.clamp(rel_year, min=0, max=self.max_year_tokens - 1)
        return rel_year

    def _normalize_recency_ids(self, recency_ids: torch.Tensor) -> torch.Tensor:
        recency_ids = torch.clamp(recency_ids, min=0, max=self.max_recency_tokens - 1)
        return recency_ids

    def forward(
        self,
        x: torch.Tensor,
        year_ids: Optional[torch.Tensor] = None,
        recency_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : [B, T, D]
            year_ids  : [B, T]
            recency_ids: [B, T]
            mask      : [B, T], 1 valid, 0 invalid

        Returns:
            out       : [B, T, D]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,D], got shape={tuple(x.shape)}")

        out = x

        if self.use_year_embedding:
            if year_ids is None:
                raise ValueError("year_ids is required when use_year_embedding=True")
            if year_ids.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"year_ids shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(year_ids.shape)}"
                )
            year_index = self._normalize_year_ids(year_ids)
            year_embed = self.year_embedding(year_index)
            out = out + self.year_scale * year_embed

        if self.use_recency_embedding:
            if recency_ids is None:
                raise ValueError("recency_ids is required when use_recency_embedding=True")
            if recency_ids.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"recency_ids shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(recency_ids.shape)}"
                )
            recency_index = self._normalize_recency_ids(recency_ids)
            recency_embed = self.recency_embedding(recency_index)
            out = out + self.recency_scale * recency_embed

        out = self.norm(out)
        out = self.dropout(out)

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
                )
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out


if __name__ == "__main__":
    B, T, D = 4, 15, 256
    x = torch.randn(B, T, D)
    year_ids = torch.arange(2000, 2000 + T).unsqueeze(0).expand(B, -1)
    recency_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, T)

    pe = LearnedTemporalPositionalEncoding(
        d_model=D,
        max_year_tokens=32,
        max_recency_tokens=32,
        base_year=2000,
        use_year_embedding=True,
        use_recency_embedding=True,
        dropout=0.1,
        use_layernorm=True,
        learned_scale=True,
    )

    y = pe(x, year_ids=year_ids, recency_ids=recency_ids, mask=mask)
    print("[INFO] positional output shape:", tuple(y.shape))

    fy = build_future_year_ids(batch_size=B, future_steps=6, start_year=2015, device=x.device)
    fr = build_future_recency_ids(batch_size=B, future_steps=6, device=x.device)
    print("[INFO] future year ids shape:", tuple(fy.shape))
    print("[INFO] future recency ids shape:", tuple(fr.shape))