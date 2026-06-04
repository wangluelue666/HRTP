#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.feature_schema_v3 import FEATURE_SCHEMA
from models.encoder.ring_aggregator_v3 import RingHierarchyAggregatorV3
from models.encoder.type_encoder_v3 import RingTypeEncoderV3
from models.encoder.view_fusion_v3 import ViewFusionV3


class SingleViewEncoderV3(nn.Module):
    """
    Encode one view (stock or flow) using:
      - ring1 typed relation encoder
      - ring2 typed relation encoder
      - ring hierarchy aggregator
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        self.ring1_encoder = RingTypeEncoderV3(
            paper_in_dim=FEATURE_SCHEMA.get_dim("paper"),
            author_in_dim=FEATURE_SCHEMA.get_dim("author"),
            topic_in_dim=FEATURE_SCHEMA.get_dim("topic"),
            venue_in_dim=FEATURE_SCHEMA.get_dim("venue"),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.ring2_encoder = RingTypeEncoderV3(
            paper_in_dim=FEATURE_SCHEMA.get_dim("paper"),
            author_in_dim=FEATURE_SCHEMA.get_dim("author"),
            topic_in_dim=FEATURE_SCHEMA.get_dim("topic"),
            venue_in_dim=FEATURE_SCHEMA.get_dim("venue"),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.ring_agg = RingHierarchyAggregatorV3(
            hidden_dim=hidden_dim,
            target_dim=FEATURE_SCHEMA.get_dim("target"),
            dropout=dropout,
        )

    def forward(self, view_batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ring0 = view_batch["ring0"]
        ring1 = view_batch["ring1"]
        ring2 = view_batch["ring2"]

        ring1_out = self.ring1_encoder(ring1)
        ring2_out = self.ring2_encoder(ring2)

        agg_out = self.ring_agg(
            target_x=ring0["target_x"],
            ring1_repr=ring1_out["ring_repr"],
            ring2_repr=ring2_out["ring_repr"],
        )

        return {
            "target_repr": agg_out["target_repr"],
            "ring1_repr": agg_out["ring1_repr"],
            "ring2_repr": agg_out["ring2_repr"],
            "view_repr": agg_out["view_repr"],
            "ring1_type": ring1_out,
            "ring2_type": ring2_out,
        }


class HypergraphEncoderV3(nn.Module):
    """
    Full paired encoder for stock / flow subgraphs.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.stock_encoder = SingleViewEncoderV3(hidden_dim=hidden_dim, dropout=dropout)
        self.flow_encoder = SingleViewEncoderV3(hidden_dim=hidden_dim, dropout=dropout)

        self.view_fusion = ViewFusionV3(dim=hidden_dim, dropout=dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.hs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.hf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        stock_out = self.stock_encoder(batch["stock"])
        flow_out = self.flow_encoder(batch["flow"])

        fusion_out = self.view_fusion(stock_out, flow_out)

        aux_cls_logits = self.cls_head(fusion_out["z_joint"])
        aux_hs_logits = self.hs_head(fusion_out["z_stock"])
        aux_hf_logits = self.hf_head(fusion_out["z_flow"])

        return {
            # final embeddings
            "z_stock": fusion_out["z_stock"],
            "z_flow": fusion_out["z_flow"],
            "z_joint": fusion_out["z_joint"],

            # ring-aligned embeddings after cross-view interaction
            "stock_ring1_repr": fusion_out["stock_ring1_repr"],
            "flow_ring1_repr": fusion_out["flow_ring1_repr"],
            "stock_ring2_repr": fusion_out["stock_ring2_repr"],
            "flow_ring2_repr": fusion_out["flow_ring2_repr"],

            # pre-fusion per-view representations
            "stock_view_repr": stock_out["view_repr"],
            "flow_view_repr": flow_out["view_repr"],
            "stock_target_repr": stock_out["target_repr"],
            "flow_target_repr": flow_out["target_repr"],

            # auxiliary heads
            "aux_cls_logits": aux_cls_logits,
            "aux_hs_logits": aux_hs_logits,
            "aux_hf_logits": aux_hf_logits,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.hypergraph_dataset_v3 import HypergraphDatasetV3
    from dataset.collate_fn_v3 import collate_paired_subgraphs_v3

    ds = HypergraphDatasetV3(dataset="aps", split="train")
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_paired_subgraphs_v3)

    batch = next(iter(dl))
    model = HypergraphEncoderV3(hidden_dim=64, dropout=0.1, num_classes=4)
    out = model(batch)

    for k, v in out.items():
        print(k, tuple(v.shape))