#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadCrossStreamAttention(nn.Module):
    """
    Cross-stream attention fusion at each time step.

    For each time step t:
        query  = h_joint[t]
        key    = {h_stock[t], h_flow[t]}
        value  = {h_stock[t], h_flow[t]}

    Then the fused representation is combined with the original joint stream
    through residual connection and position-wise feed-forward refinement.

    Inputs:
        h_stock: [B, T, D]
        h_flow : [B, T, D]
        h_joint: [B, T, D]
        mask   : [B, T], 1 valid, 0 invalid

    Outputs:
        fused  : [B, T, D]
        attn_weights(optional): [B, T, 2]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_hidden_dim: Optional[int] = None,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        ff_hidden_dim = ff_hidden_dim or (2 * d_model)
        self.use_pre_norm = bool(use_pre_norm)

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_out_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_stock: torch.Tensor,
        h_flow: torch.Tensor,
        h_joint: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if h_stock.ndim != 3 or h_flow.ndim != 3 or h_joint.ndim != 3:
            raise ValueError(
                "All inputs must be [B,T,D], got "
                f"h_stock={tuple(h_stock.shape)}, "
                f"h_flow={tuple(h_flow.shape)}, "
                f"h_joint={tuple(h_joint.shape)}"
            )

        if not (h_stock.shape == h_flow.shape == h_joint.shape):
            raise ValueError(
                "Input stream shapes must match exactly, got "
                f"h_stock={tuple(h_stock.shape)}, "
                f"h_flow={tuple(h_flow.shape)}, "
                f"h_joint={tuple(h_joint.shape)}"
            )

        B, T, D = h_joint.shape

        if mask is not None and mask.shape != (B, T):
            raise ValueError(f"mask must be [B,T], got shape={tuple(mask.shape)}")

        q = self.q_norm(h_joint) if self.use_pre_norm else h_joint
        stock = self.kv_norm(h_stock) if self.use_pre_norm else h_stock
        flow = self.kv_norm(h_flow) if self.use_pre_norm else h_flow

        # Flatten the temporal axis so each time step independently attends over {stock_t, flow_t}.
        q_bt = q.reshape(B * T, 1, D)  # [BT, 1, D]
        kv_bt = torch.stack([stock, flow], dim=2).reshape(B * T, 2, D)  # [BT, 2, D]

        fused_bt, attn_weights_bt = self.attn(
            query=q_bt,
            key=kv_bt,
            value=kv_bt,
            need_weights=True,
            average_attn_weights=True,
        )  # fused: [BT,1,D], weights: [BT,1,2]

        fused = fused_bt.reshape(B, T, D)
        attn_weights = attn_weights_bt.reshape(B, T, 2)

        out = h_joint + self.attn_dropout(fused)
        out = self.attn_out_norm(out)

        ff_out = self.ffn(out)
        out = self.ffn_norm(out + ff_out)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(out.dtype)
            out = out * mask_f
            attn_weights = attn_weights * mask.unsqueeze(-1).to(attn_weights.dtype)

        if return_attn:
            return out, attn_weights
        return out


if __name__ == "__main__":
    B, T, D = 4, 15, 256
    h_stock = torch.randn(B, T, D)
    h_flow = torch.randn(B, T, D)
    h_joint = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[0, 12:] = 0
    mask[1, 10:] = 0

    layer = MultiHeadCrossStreamAttention(
        d_model=D,
        num_heads=4,
        dropout=0.1,
        ff_hidden_dim=512,
        use_pre_norm=True,
    )
    y, a = layer(h_stock, h_flow, h_joint, mask=mask, return_attn=True)

    print("[INFO] fused shape:", tuple(y.shape))
    print("[INFO] attn weights shape:", tuple(a.shape))