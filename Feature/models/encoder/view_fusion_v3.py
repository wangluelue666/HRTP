#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RingCrossViewBlock(nn.Module):
    """
    Cross-view interaction at ring level.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.stock_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Sigmoid(),
        )
        self.flow_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Sigmoid(),
        )

        self.stock_upd = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.flow_upd = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.stock_norm = nn.LayerNorm(dim)
        self.flow_norm = nn.LayerNorm(dim)

    def forward(self, stock_x: torch.Tensor, flow_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pair_feat = torch.cat(
            [stock_x, flow_x, torch.abs(stock_x - flow_x), stock_x * flow_x],
            dim=-1,
        )

        s_gate = self.stock_gate(pair_feat)
        f_gate = self.flow_gate(pair_feat)

        stock_new = stock_x + s_gate * self.stock_upd(pair_feat)
        flow_new = flow_x + f_gate * self.flow_upd(pair_feat)

        stock_new = self.stock_norm(stock_new)
        flow_new = self.flow_norm(flow_new)

        return {
            "stock": stock_new,
            "flow": flow_new,
        }


class ViewFusionV3(nn.Module):
    """
    Cross-view fusion with:
      1) ring-level interaction
      2) view-level interaction
      3) final joint representation
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        self.ring1_cross = RingCrossViewBlock(dim, dropout)
        self.ring2_cross = RingCrossViewBlock(dim, dropout)
        self.view_cross = RingCrossViewBlock(dim, dropout)

        self.stock_proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.flow_proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.stock_norm = nn.LayerNorm(dim)
        self.flow_norm = nn.LayerNorm(dim)

        self.joint_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Sigmoid(),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.joint_norm = nn.LayerNorm(dim)

    def forward(
        self,
        stock_view: Dict[str, torch.Tensor],
        flow_view: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # ring-level cross interaction
        ring1_out = self.ring1_cross(stock_view["ring1_repr"], flow_view["ring1_repr"])
        ring2_out = self.ring2_cross(stock_view["ring2_repr"], flow_view["ring2_repr"])

        stock_ring1 = ring1_out["stock"]
        flow_ring1 = ring1_out["flow"]
        stock_ring2 = ring2_out["stock"]
        flow_ring2 = ring2_out["flow"]

        # enrich each view by its updated ring hierarchy
        stock_view_enh = self.stock_norm(
            stock_view["view_repr"] + self.stock_proj(
                torch.cat([stock_view["view_repr"], stock_ring1, stock_ring2], dim=-1)
            )
        )
        flow_view_enh = self.flow_norm(
            flow_view["view_repr"] + self.flow_proj(
                torch.cat([flow_view["view_repr"], flow_ring1, flow_ring2], dim=-1)
            )
        )

        # final view-level interaction
        view_out = self.view_cross(stock_view_enh, flow_view_enh)
        z_stock = view_out["stock"]
        z_flow = view_out["flow"]

        pair_feat = torch.cat(
            [z_stock, z_flow, torch.abs(z_stock - z_flow), z_stock * z_flow],
            dim=-1,
        )
        g = self.joint_gate(pair_feat)
        z_joint = self.joint_norm(z_stock + z_flow + g * self.joint_mlp(pair_feat))

        return {
            "stock_ring1_repr": stock_ring1,
            "flow_ring1_repr": flow_ring1,
            "stock_ring2_repr": stock_ring2,
            "flow_ring2_repr": flow_ring2,
            "z_stock": z_stock,
            "z_flow": z_flow,
            "z_joint": z_joint,
        }


if __name__ == "__main__":
    bsz, dim = 4, 64
    fusion = ViewFusionV3(dim=dim, dropout=0.1)

    stock_view = {
        "ring1_repr": torch.randn(bsz, dim),
        "ring2_repr": torch.randn(bsz, dim),
        "view_repr": torch.randn(bsz, dim),
    }
    flow_view = {
        "ring1_repr": torch.randn(bsz, dim),
        "ring2_repr": torch.randn(bsz, dim),
        "view_repr": torch.randn(bsz, dim),
    }

    out = fusion(stock_view, flow_view)
    for k, v in out.items():
        print(k, tuple(v.shape))