#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RingConditionBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, base: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([base, cond], dim=-1)
        g = self.gate(cat)
        upd = self.mix(cat)
        out = base + g * upd
        return self.norm(out)


class RingHierarchyAggregatorV3(nn.Module):
    """
    Hierarchical ring aggregation:
      - ring0 target anchors ring1
      - ring1 absorbs ring2 context
      - ring2 is lightly conditioned by ring1
      - final view repr fuses all three rings
    """

    def __init__(self, hidden_dim: int = 128, target_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_norm = nn.LayerNorm(hidden_dim)

        self.r1_by_t = RingConditionBlock(hidden_dim, dropout)
        self.r1_by_r2 = RingConditionBlock(hidden_dim, dropout)
        self.r2_by_r1 = RingConditionBlock(hidden_dim, dropout)

        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        target_x: torch.Tensor,
        ring1_repr: torch.Tensor,
        ring2_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_repr = self.target_norm(self.target_proj(target_x))

        ring1_cond = self.r1_by_t(ring1_repr, target_repr)
        ring1_cond = self.r1_by_r2(ring1_cond, ring2_repr)

        ring2_cond = self.r2_by_r1(ring2_repr, ring1_repr)

        view_repr = self.final_norm(
            self.final_fuse(torch.cat([target_repr, ring1_cond, ring2_cond], dim=-1))
        )

        return {
            "target_repr": target_repr,
            "ring1_repr": ring1_cond,
            "ring2_repr": ring2_cond,
            "view_repr": view_repr,
        }


if __name__ == "__main__":
    bsz = 4
    model = RingHierarchyAggregatorV3(hidden_dim=64, target_dim=13, dropout=0.1)

    target_x = torch.randn(bsz, 13)
    ring1_repr = torch.randn(bsz, 64)
    ring2_repr = torch.randn(bsz, 64)

    out = model(target_x, ring1_repr, ring2_repr)
    for k, v in out.items():
        print(k, tuple(v.shape))