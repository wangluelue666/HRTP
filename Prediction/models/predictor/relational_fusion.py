#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn

# Support both:
#   1) python -m models.predictor.relational_fusion
#   2) python models/predictor/relational_fusion.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layers import (
    MultiHeadCrossStreamAttention,
    AttentiveMaskedPooling,
    TransformerEncoderBlock,
)


class RelationalFusionModule(nn.Module):
    """
    Relational fusion over the tri-stream historical representations.

    Steps:
        1) Per-time-step cross-stream attention fusion:
           joint queries stock/flow at each year.
        2) Temporal refinement over fused sequence.
        3) Attentive masked pooling to obtain global historical context.

    Outputs:
        - H_rel   : [B, T, D]
        - C_hist  : [B, D]
        - optional analysis tensors:
            * cross_stream_attn
            * pooling_weights
    """

    def __init__(
        self,
        d_model: int = 256,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1,
        temporal_refine_layers: int = 1,
        temporal_refine_heads: int = 4,
        temporal_refine_ffn_dim: int = 512,
        temporal_refine_attn_dropout: float = 0.1,
        pooling_hidden_dim: Optional[int] = None,
        pooling_dropout: float = 0.1,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.cross_stream_fusion = MultiHeadCrossStreamAttention(
            d_model=d_model,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
            ff_hidden_dim=temporal_refine_ffn_dim,
            use_pre_norm=use_pre_norm,
        )

        self.temporal_refine_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=temporal_refine_heads,
                ffn_dim=temporal_refine_ffn_dim,
                dropout=fusion_dropout,
                attn_dropout=temporal_refine_attn_dropout,
                use_pre_norm=use_pre_norm,
            )
            for _ in range(temporal_refine_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

        self.pooling = AttentiveMaskedPooling(
            d_model=d_model,
            hidden_dim=pooling_hidden_dim or d_model,
            dropout=pooling_dropout,
            temperature=1.0,
        )

    def forward(
        self,
        h_stock: torch.Tensor,
        h_flow: torch.Tensor,
        h_joint: torch.Tensor,
        mask: torch.Tensor,
        return_analysis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_stock: [B, T, D]
            h_flow : [B, T, D]
            h_joint: [B, T, D]
            mask   : [B, T]

        Returns:
            {
                "h_rel": [B,T,D],
                "c_hist": [B,D],
                "cross_stream_attn": [B,T,2], optional,
                "pooling_weights": [B,T], optional
            }
        """
        if h_stock.ndim != 3 or h_flow.ndim != 3 or h_joint.ndim != 3:
            raise ValueError(
                "All inputs must be [B,T,D], got "
                f"h_stock={tuple(h_stock.shape)}, "
                f"h_flow={tuple(h_flow.shape)}, "
                f"h_joint={tuple(h_joint.shape)}"
            )

        B, T, D = h_joint.shape
        if h_stock.shape != (B, T, D) or h_flow.shape != (B, T, D):
            raise ValueError(
                "Input shapes must match exactly, got "
                f"h_stock={tuple(h_stock.shape)}, "
                f"h_flow={tuple(h_flow.shape)}, "
                f"h_joint={tuple(h_joint.shape)}"
            )

        if mask.shape != (B, T):
            raise ValueError(f"mask must be [B,T], got shape={tuple(mask.shape)}")

        h_rel, cross_attn = self.cross_stream_fusion(
            h_stock=h_stock,
            h_flow=h_flow,
            h_joint=h_joint,
            mask=mask,
            return_attn=True,
        )

        for layer in self.temporal_refine_layers:
            h_rel = layer(h_rel, mask=mask, return_attn=False)

        h_rel = self.out_norm(h_rel)
        h_rel = h_rel * mask.unsqueeze(-1).to(h_rel.dtype)

        c_hist, pooling_weights = self.pooling(
            h_rel,
            mask=mask,
            return_weights=True,
        )

        outputs: Dict[str, torch.Tensor] = {
            "h_rel": h_rel,
            "c_hist": c_hist,
        }

        if return_analysis:
            outputs["cross_stream_attn"] = cross_attn
            outputs["pooling_weights"] = pooling_weights

        return outputs


if __name__ == "__main__":
    B, T, D = 4, 15, 256
    h_stock = torch.randn(B, T, D)
    h_flow = torch.randn(B, T, D)
    h_joint = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[0, 12:] = 0
    mask[1, 10:] = 0

    model = RelationalFusionModule(
        d_model=D,
        fusion_num_heads=4,
        fusion_dropout=0.1,
        temporal_refine_layers=1,
        temporal_refine_heads=4,
        temporal_refine_ffn_dim=512,
        temporal_refine_attn_dropout=0.1,
        pooling_hidden_dim=D,
        pooling_dropout=0.1,
        use_pre_norm=True,
    )

    out = model(
        h_stock=h_stock,
        h_flow=h_flow,
        h_joint=h_joint,
        mask=mask,
        return_analysis=True,
    )

    print("[INFO] h_rel shape:", tuple(out["h_rel"].shape))
    print("[INFO] c_hist shape:", tuple(out["c_hist"].shape))
    print("[INFO] cross_stream_attn shape:", tuple(out["cross_stream_attn"].shape))
    print("[INFO] pooling_weights shape:", tuple(out["pooling_weights"].shape))