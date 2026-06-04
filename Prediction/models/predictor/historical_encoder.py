#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn

# Support both:
#   1) python -m models.predictor.historical_encoder
#   2) python models/predictor/historical_encoder.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layers import (
    LearnedTemporalPositionalEncoding,
    TransformerEncoderBlock,
)


class StreamInputProjection(nn.Module):
    """
    Input projection block for a single historical stream.

    It projects raw input features to the shared model space and applies:
        - dropout
        - optional layer normalization
        - residual refinement in model space
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.refine_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        for module in self.refine:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x   : [B, T, D_in]
            mask: [B, T], 1 valid, 0 invalid
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,T,D_in], got shape={tuple(x.shape)}")

        out = self.proj(x)
        out = self.norm(out)
        out = self.dropout(out)

        refine = self.refine(out)
        out = self.refine_norm(out + refine)

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
                )
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out


class SingleStreamEncoder(nn.Module):
    """
    A full encoder stack for one historical stream.

    Pipeline:
        input -> projection -> temporal encoding -> encoder layers
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_year_embedding: bool = True,
        use_recency_embedding: bool = True,
        max_year_tokens: int = 64,
        max_recency_tokens: int = 64,
        base_year: int = 2000,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_proj = StreamInputProjection(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
            use_layernorm=True,
        )

        self.temporal_encoding = LearnedTemporalPositionalEncoding(
            d_model=d_model,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            use_year_embedding=use_year_embedding,
            use_recency_embedding=use_recency_embedding,
            dropout=dropout,
            use_layernorm=True,
            learned_scale=True,
        )

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        year_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Returns:
            {
                "input_proj": [B,T,D],
                "encoded": [B,T,D],
                "layer_outputs": list[[B,T,D]]
            }
        """
        h = self.input_proj(x, mask=mask)
        h = self.temporal_encoding(
            h,
            year_ids=year_ids,
            recency_ids=recency_ids,
            mask=mask,
        )

        layer_outputs: List[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h, mask=mask, return_attn=False)
            if return_intermediates:
                layer_outputs.append(h)

        h = self.out_norm(h)
        h = h * mask.unsqueeze(-1).to(h.dtype)

        out = {
            "input_proj": h if not return_intermediates else None,
            "encoded": h,
        }
        if return_intermediates:
            out["layer_outputs"] = layer_outputs
        return out


class HistoricalTriStreamEncoder(nn.Module):
    """
    Historical encoder for three aligned streams:
        - stock
        - flow
        - joint

    Outputs encoded representations for each stream.

    This module intentionally uses separate parameters for the three streams,
    because they represent distinct semantics and should not be forced into
    a single shared encoder.
    """

    def __init__(
        self,
        input_dim_stock: int,
        input_dim_flow: int,
        input_dim_joint: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_year_embedding: bool = True,
        use_recency_embedding: bool = True,
        max_year_tokens: int = 32,
        max_recency_tokens: int = 32,
        base_year: int = 2000,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)

        self.stock_encoder = SingleStreamEncoder(
            input_dim=input_dim_stock,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_year_embedding=use_year_embedding,
            use_recency_embedding=use_recency_embedding,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            use_pre_norm=use_pre_norm,
        )
        self.flow_encoder = SingleStreamEncoder(
            input_dim=input_dim_flow,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_year_embedding=use_year_embedding,
            use_recency_embedding=use_recency_embedding,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            use_pre_norm=use_pre_norm,
        )
        self.joint_encoder = SingleStreamEncoder(
            input_dim=input_dim_joint,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_year_embedding=use_year_embedding,
            use_recency_embedding=use_recency_embedding,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            use_pre_norm=use_pre_norm,
        )

    def forward(
        self,
        x_stock: torch.Tensor,
        x_flow: torch.Tensor,
        x_joint: torch.Tensor,
        mask: torch.Tensor,
        year_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor | Dict]:
        """
        Args:
            x_stock   : [B, T, D_s]
            x_flow    : [B, T, D_f]
            x_joint   : [B, T, D_j]
            mask      : [B, T]
            year_ids  : [B, T]
            recency_ids: [B, T]

        Returns:
            {
                "h_stock": [B,T,D],
                "h_flow":  [B,T,D],
                "h_joint": [B,T,D],
                "stream_details": {...} optional
            }
        """
        if x_stock.ndim != 3 or x_flow.ndim != 3 or x_joint.ndim != 3:
            raise ValueError(
                "All stream inputs must be [B,T,D], got "
                f"x_stock={tuple(x_stock.shape)}, "
                f"x_flow={tuple(x_flow.shape)}, "
                f"x_joint={tuple(x_joint.shape)}"
            )

        B, T = x_stock.shape[:2]
        if x_flow.shape[:2] != (B, T) or x_joint.shape[:2] != (B, T):
            raise ValueError(
                "Temporal axes mismatch across streams: "
                f"x_stock={tuple(x_stock.shape)}, "
                f"x_flow={tuple(x_flow.shape)}, "
                f"x_joint={tuple(x_joint.shape)}"
            )
        if mask.shape != (B, T):
            raise ValueError(f"mask must be [B,T], got shape={tuple(mask.shape)}")
        if year_ids.shape != (B, T):
            raise ValueError(f"year_ids must be [B,T], got shape={tuple(year_ids.shape)}")
        if recency_ids.shape != (B, T):
            raise ValueError(f"recency_ids must be [B,T], got shape={tuple(recency_ids.shape)}")

        stock_out = self.stock_encoder(
            x=x_stock,
            mask=mask,
            year_ids=year_ids,
            recency_ids=recency_ids,
            return_intermediates=return_intermediates,
        )
        flow_out = self.flow_encoder(
            x=x_flow,
            mask=mask,
            year_ids=year_ids,
            recency_ids=recency_ids,
            return_intermediates=return_intermediates,
        )
        joint_out = self.joint_encoder(
            x=x_joint,
            mask=mask,
            year_ids=year_ids,
            recency_ids=recency_ids,
            return_intermediates=return_intermediates,
        )

        outputs: Dict[str, torch.Tensor | Dict] = {
            "h_stock": stock_out["encoded"],
            "h_flow": flow_out["encoded"],
            "h_joint": joint_out["encoded"],
        }

        if return_intermediates:
            outputs["stream_details"] = {
                "stock": stock_out,
                "flow": flow_out,
                "joint": joint_out,
            }

        return outputs


if __name__ == "__main__":
    B, T = 4, 15
    Ds, Df, Dj = 256, 256, 256
    x_stock = torch.randn(B, T, Ds)
    x_flow = torch.randn(B, T, Df)
    x_joint = torch.randn(B, T, Dj)
    mask = torch.ones(B, T)
    mask[0, 12:] = 0
    year_ids = torch.arange(2000, 2000 + T).unsqueeze(0).expand(B, -1)
    recency_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

    model = HistoricalTriStreamEncoder(
        input_dim_stock=Ds,
        input_dim_flow=Df,
        input_dim_joint=Dj,
        d_model=256,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        attn_dropout=0.1,
        use_year_embedding=True,
        use_recency_embedding=True,
        max_year_tokens=32,
        max_recency_tokens=32,
        base_year=2000,
        use_pre_norm=True,
    )

    out = model(
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        mask=mask,
        year_ids=year_ids,
        recency_ids=recency_ids,
        return_intermediates=True,
    )

    print("[INFO] h_stock shape:", tuple(out["h_stock"].shape))
    print("[INFO] h_flow shape:", tuple(out["h_flow"].shape))
    print("[INFO] h_joint shape:", tuple(out["h_joint"].shape))