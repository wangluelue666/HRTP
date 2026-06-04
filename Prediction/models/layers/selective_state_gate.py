#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SelectiveStateGate(nn.Module):
    """
    Context-aware selective state gating inspired by selective state propagation.

    Given current state x and optional context c:
        gate      = sigmoid(G([x;c]))
        candidate = tanh(U(x))
        out       = gate * candidate + (1 - gate) * x

    This formulation is stronger than a simple element-wise gate because:
        - it uses context-conditioned selection
        - it preserves residual state paths
        - it allows controlled state rewriting
    """

    def __init__(
        self,
        d_model: int,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        context_dim = context_dim or d_model
        in_dim = d_model + context_dim

        self.use_layernorm = bool(use_layernorm)
        self.x_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if use_layernorm else nn.Identity()

        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.candidate_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.out_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x       : [B, L, D] or [B, D]
            context : same leading shape as x, last dim can differ
            mask    : [B, L] or [B], 1 valid, 0 invalid
        """
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,D]
            if context is not None and context.ndim == 2:
                context = context.unsqueeze(1)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(1)
            squeeze_back = True

        if x.ndim != 3:
            raise ValueError(f"x must be [B,L,D] or [B,D], got shape={tuple(x.shape)}")

        x_in = self.x_norm(x)

        if context is None:
            context = x_in
        else:
            if context.ndim == 2:
                context = context.unsqueeze(1)
            if context.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"context leading shape mismatch: x={tuple(x.shape)}, context={tuple(context.shape)}"
                )
            context = self.context_norm(context)

        gate_input = torch.cat([x_in, context], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        candidate = torch.tanh(self.candidate_proj(x_in))

        out = gate * candidate + (1.0 - gate) * x
        out = self.out_dropout(out)
        out = self.out_norm(out)

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch: expected {tuple(x.shape[:2])}, got {tuple(mask.shape)}"
                )
            out = out * mask.unsqueeze(-1).to(out.dtype)
            gate = gate * mask.unsqueeze(-1).to(gate.dtype)

        if squeeze_back:
            out = out.squeeze(1)
            gate = gate.squeeze(1)

        if return_gate:
            return out, gate
        return out


if __name__ == "__main__":
    B, L, D = 4, 6, 256
    x = torch.randn(B, L, D)
    c = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[0, 4:] = 0

    gate = SelectiveStateGate(
        d_model=D,
        context_dim=D,
        hidden_dim=512,
        dropout=0.1,
        use_layernorm=True,
    )

    y, g = gate(x, context=c, mask=mask, return_gate=True)
    print("[INFO] gated output shape:", tuple(y.shape))
    print("[INFO] gate shape:", tuple(g.shape))