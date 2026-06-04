#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Support both:
#   1) python -m models.predictor.rollout_cell
#   2) python models/predictor/rollout_cell.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layers import SelectiveStateGate


class StepTokenComposer(nn.Module):
    """
    Compose a single-step future token from:

        - previous latent state z_{k-1}
        - previous class probability p_{k-1}
        - global historical context c_hist
        - current future year embedding
        - current future recency embedding

    Output:
        step token u_k in the shared model space.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.prev_state_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.prev_class_proj = nn.Sequential(
            nn.Linear(num_classes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.hist_context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.year_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.recency_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.fuse_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_prev: torch.Tensor,
        p_prev: torch.Tensor,
        c_hist: torch.Tensor,
        year_embed: torch.Tensor,
        recency_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_prev      : [B, D]
            p_prev      : [B, C]
            c_hist      : [B, D]
            year_embed  : [B, D]
            recency_embed: [B, D]

        Returns:
            step_token  : [B, D]
        """
        if z_prev.ndim != 2:
            raise ValueError(f"z_prev must be [B,D], got {tuple(z_prev.shape)}")
        if p_prev.ndim != 2:
            raise ValueError(f"p_prev must be [B,C], got {tuple(p_prev.shape)}")
        if c_hist.ndim != 2:
            raise ValueError(f"c_hist must be [B,D], got {tuple(c_hist.shape)}")
        if year_embed.ndim != 2:
            raise ValueError(f"year_embed must be [B,D], got {tuple(year_embed.shape)}")
        if recency_embed.ndim != 2:
            raise ValueError(f"recency_embed must be [B,D], got {tuple(recency_embed.shape)}")

        B, D = z_prev.shape
        if c_hist.shape != (B, D):
            raise ValueError(
                f"c_hist shape mismatch: expected {(B, D)}, got {tuple(c_hist.shape)}"
            )
        if year_embed.shape != (B, D):
            raise ValueError(
                f"year_embed shape mismatch: expected {(B, D)}, got {tuple(year_embed.shape)}"
            )
        if recency_embed.shape != (B, D):
            raise ValueError(
                f"recency_embed shape mismatch: expected {(B, D)}, got {tuple(recency_embed.shape)}"
            )

        z_feat = self.prev_state_proj(z_prev)
        p_feat = self.prev_class_proj(p_prev)
        c_feat = self.hist_context_proj(c_hist)
        y_feat = self.year_proj(year_embed)
        r_feat = self.recency_proj(recency_embed)

        token = torch.cat([z_feat, p_feat, c_feat, y_feat, r_feat], dim=-1)
        token = self.fuse(token)
        token = self.fuse_norm(token)
        return token


class HistoryCrossAttention(nn.Module):
    """
    Single-step cross-attention from current decoder state to historical memory.

    Query:
        current step token / state

    Key / Value:
        historical relational memory H_rel
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_state : [B, D]
            memory      : [B, T, D]
            memory_mask : [B, T], 1 valid, 0 invalid

        Returns:
            context     : [B, D]
            attn_weights: [B, 1, T] if return_attn=True
        """
        if query_state.ndim != 2:
            raise ValueError(f"query_state must be [B,D], got {tuple(query_state.shape)}")
        if memory.ndim != 3:
            raise ValueError(f"memory must be [B,T,D], got {tuple(memory.shape)}")

        B, T, D = memory.shape
        if query_state.shape != (B, D):
            raise ValueError(
                f"query_state shape mismatch: expected {(B, D)}, got {tuple(query_state.shape)}"
            )

        q = self.query_norm(query_state).unsqueeze(1)   # [B,1,D]
        kv = self.memory_norm(memory)                   # [B,T,D]

        key_padding_mask = None
        if memory_mask is not None:
            if memory_mask.shape != (B, T):
                raise ValueError(
                    f"memory_mask shape mismatch: expected {(B, T)}, got {tuple(memory_mask.shape)}"
                )
            key_padding_mask = ~(memory_mask > 0)

        context, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )  # context [B,1,D]

        context = context.squeeze(1)
        context = self.out_norm(query_state + self.out_dropout(context))

        if return_attn:
            return context, attn_weights
        return context


class StepTransitionBlock(nn.Module):
    """
    Single-step latent transition block.

    It refines the current step state using:
        - self transformation
        - history cross-attention context
        - selective state gating conditioned on global history
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_selective_gate: bool = True,
    ) -> None:
        super().__init__()

        self.self_refine = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.self_refine_norm = nn.LayerNorm(d_model)

        self.history_attn = HistoryCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        self.use_selective_gate = bool(use_selective_gate)
        self.gate = SelectiveStateGate(
            d_model=d_model,
            context_dim=d_model,
            hidden_dim=ffn_dim,
            dropout=dropout,
            use_layernorm=True,
        ) if self.use_selective_gate else None

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        step_token: torch.Tensor,
        c_hist: torch.Tensor,
        h_rel: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        return_gate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            step_token : [B, D]
            c_hist     : [B, D]
            h_rel      : [B, T, D]
            history_mask: [B, T]

        Returns:
            {
                "z_next": [B, D],
                "cross_attn": [B,1,T], optional,
                "gate_values": [B,D], optional
            }
        """
        if step_token.ndim != 2:
            raise ValueError(f"step_token must be [B,D], got {tuple(step_token.shape)}")
        if c_hist.ndim != 2:
            raise ValueError(f"c_hist must be [B,D], got {tuple(c_hist.shape)}")
        if h_rel.ndim != 3:
            raise ValueError(f"h_rel must be [B,T,D], got {tuple(h_rel.shape)}")

        B, T, D = h_rel.shape
        if step_token.shape != (B, D):
            raise ValueError(
                f"step_token shape mismatch: expected {(B, D)}, got {tuple(step_token.shape)}"
            )
        if c_hist.shape != (B, D):
            raise ValueError(
                f"c_hist shape mismatch: expected {(B, D)}, got {tuple(c_hist.shape)}"
            )

        self_part = self.self_refine_norm(step_token + self.self_refine(step_token))

        if return_attn:
            hist_context, cross_attn = self.history_attn(
                query_state=self_part,
                memory=h_rel,
                memory_mask=history_mask,
                return_attn=True,
            )
        else:
            hist_context = self.history_attn(
                query_state=self_part,
                memory=h_rel,
                memory_mask=history_mask,
                return_attn=False,
            )
            cross_attn = None

        fused = torch.cat([self_part, hist_context], dim=-1)
        z = self.fusion(fused)
        z = self.fusion_norm(self_part + z)

        gate_values = None
        if self.use_selective_gate:
            if return_gate:
                z, gate_values = self.gate(
                    x=z,
                    context=c_hist,
                    mask=None,
                    return_gate=True,
                )
            else:
                z = self.gate(
                    x=z,
                    context=c_hist,
                    mask=None,
                    return_gate=False,
                )

        z = self.out_norm(z)

        out = {"z_next": z}
        if return_attn and cross_attn is not None:
            out["cross_attn"] = cross_attn
        if return_gate and gate_values is not None:
            out["gate_values"] = gate_values
        return out


class RolloutCell(nn.Module):
    """
    Single-step rollout cell for HARP.

    For future step k, this cell consumes:
        - previous latent state z_{k-1}
        - previous class probability p_{k-1}
        - global historical context c_hist
        - historical relational memory h_rel
        - current year embedding
        - current recency embedding

    and produces:
        - current latent state z_k

    This cell does not produce logits by itself.
    Logits should be produced by the shared decision head outside.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_selective_gate: bool = True,
    ) -> None:
        super().__init__()

        self.token_composer = StepTokenComposer(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.transition = StepTransitionBlock(
            d_model=d_model,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_selective_gate=use_selective_gate,
        )

    def forward(
        self,
        z_prev: torch.Tensor,
        p_prev: torch.Tensor,
        c_hist: torch.Tensor,
        h_rel: torch.Tensor,
        history_mask: Optional[torch.Tensor],
        year_embed: torch.Tensor,
        recency_embed: torch.Tensor,
        return_analysis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_prev       : [B, D]
            p_prev       : [B, C]
            c_hist       : [B, D]
            h_rel        : [B, T, D]
            history_mask : [B, T]
            year_embed   : [B, D]
            recency_embed: [B, D]

        Returns:
            {
                "z_next": [B, D],
                "cross_attn": [B,1,T], optional,
                "gate_values": [B,D], optional
            }
        """
        step_token = self.token_composer(
            z_prev=z_prev,
            p_prev=p_prev,
            c_hist=c_hist,
            year_embed=year_embed,
            recency_embed=recency_embed,
        )

        out = self.transition(
            step_token=step_token,
            c_hist=c_hist,
            h_rel=h_rel,
            history_mask=history_mask,
            return_attn=return_analysis,
            return_gate=return_analysis,
        )
        return out


if __name__ == "__main__":
    B, T_hist, D, C = 4, 15, 256, 4

    z_prev = torch.randn(B, D)
    p_prev = torch.softmax(torch.randn(B, C), dim=-1)
    c_hist = torch.randn(B, D)
    h_rel = torch.randn(B, T_hist, D)
    history_mask = torch.ones(B, T_hist)
    history_mask[0, 12:] = 0

    year_embed = torch.randn(B, D)
    recency_embed = torch.randn(B, D)

    cell = RolloutCell(
        d_model=D,
        num_classes=C,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        use_selective_gate=True,
    )

    out = cell(
        z_prev=z_prev,
        p_prev=p_prev,
        c_hist=c_hist,
        h_rel=h_rel,
        history_mask=history_mask,
        year_embed=year_embed,
        recency_embed=recency_embed,
        return_analysis=True,
    )

    print("[INFO] z_next shape:", tuple(out["z_next"].shape))
    print("[INFO] cross_attn shape:", tuple(out["cross_attn"].shape))
    print("[INFO] gate_values shape:", tuple(out["gate_values"].shape))