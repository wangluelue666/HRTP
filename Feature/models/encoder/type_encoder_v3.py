#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 0:
        return x.new_zeros(x.size(0), x.size(-1))
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 0:
        return x.new_zeros(x.size(0), x.size(-1))
    neg_inf = torch.full_like(x, -1e9)
    x_masked = torch.where(mask.unsqueeze(-1), x, neg_inf)
    out, _ = x_masked.max(dim=1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


class NodeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TypePool(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x.new_zeros(x.size(0), x.size(-1))

        logits = self.attn(x).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        attn = F.softmax(logits, dim=1)
        attn = torch.where(mask, attn, torch.zeros_like(attn))

        attn_sum = attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
        attn = attn / attn_sum

        weighted = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        mean_p = masked_mean(x, mask)
        max_p = masked_max(x, mask)

        out = self.proj(torch.cat([weighted, mean_p, max_p], dim=-1))
        return self.norm(out)


class RelationMessagePassing(nn.Module):
    """
    Lightweight typed relation propagation centered on paper nodes.

    This version removes the batch-level Python for-loop and uses
    flattened global indexing with index_add_ for much better throughput.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.author_to_paper = nn.Linear(dim, dim)
        self.topic_to_paper = nn.Linear(dim, dim)
        self.venue_to_paper = nn.Linear(dim, dim)

        self.paper_to_author = nn.Linear(dim, dim)
        self.paper_to_topic = nn.Linear(dim, dim)
        self.paper_to_venue = nn.Linear(dim, dim)

        self.paper_update = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.author_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.topic_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.venue_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.paper_norm = nn.LayerNorm(dim)
        self.author_norm = nn.LayerNorm(dim)
        self.topic_norm = nn.LayerNorm(dim)
        self.venue_norm = nn.LayerNorm(dim)

    def forward(
        self,
        paper_x: torch.Tensor,
        paper_mask: torch.Tensor,
        author_x: torch.Tensor,
        author_mask: torch.Tensor,
        topic_x: torch.Tensor,
        topic_mask: torch.Tensor,
        venue_x: torch.Tensor,
        venue_mask: torch.Tensor,
        pa_edges: torch.Tensor,
        pa_edge_mask: torch.Tensor,
        pt_edges: torch.Tensor,
        pt_edge_mask: torch.Tensor,
        pv_edges: torch.Tensor,
        pv_edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        paper_msg_from_author = self._aggregate_to_paper_vectorized(
            src_x=author_x,
            src_mask=author_mask,
            edges=pa_edges,
            edge_mask=pa_edge_mask,
            transform=self.author_to_paper,
            num_papers=paper_x.size(1),
        )
        paper_msg_from_topic = self._aggregate_to_paper_vectorized(
            src_x=topic_x,
            src_mask=topic_mask,
            edges=pt_edges,
            edge_mask=pt_edge_mask,
            transform=self.topic_to_paper,
            num_papers=paper_x.size(1),
        )
        paper_msg_from_venue = self._aggregate_to_paper_vectorized(
            src_x=venue_x,
            src_mask=venue_mask,
            edges=pv_edges,
            edge_mask=pv_edge_mask,
            transform=self.venue_to_paper,
            num_papers=paper_x.size(1),
        )

        paper_new = self.paper_update(
            torch.cat(
                [paper_x, paper_msg_from_author, paper_msg_from_topic, paper_msg_from_venue],
                dim=-1,
            )
        )
        paper_x = self.paper_norm(paper_x + paper_new)

        author_msg = self._aggregate_from_paper_vectorized(
            paper_x=paper_x,
            paper_mask=paper_mask,
            edges=pa_edges,
            edge_mask=pa_edge_mask,
            transform=self.paper_to_author,
            num_dst=author_x.size(1),
        )
        topic_msg = self._aggregate_from_paper_vectorized(
            paper_x=paper_x,
            paper_mask=paper_mask,
            edges=pt_edges,
            edge_mask=pt_edge_mask,
            transform=self.paper_to_topic,
            num_dst=topic_x.size(1),
        )
        venue_msg = self._aggregate_from_paper_vectorized(
            paper_x=paper_x,
            paper_mask=paper_mask,
            edges=pv_edges,
            edge_mask=pv_edge_mask,
            transform=self.paper_to_venue,
            num_dst=venue_x.size(1),
        )

        author_x = self.author_norm(author_x + self.author_update(torch.cat([author_x, author_msg], dim=-1)))
        topic_x = self.topic_norm(topic_x + self.topic_update(torch.cat([topic_x, topic_msg], dim=-1)))
        venue_x = self.venue_norm(venue_x + self.venue_update(torch.cat([venue_x, venue_msg], dim=-1)))

        author_x = author_x * author_mask.unsqueeze(-1).float()
        topic_x = topic_x * topic_mask.unsqueeze(-1).float()
        venue_x = venue_x * venue_mask.unsqueeze(-1).float()
        paper_x = paper_x * paper_mask.unsqueeze(-1).float()

        return paper_x, author_x, topic_x, venue_x

    def _aggregate_to_paper_vectorized(
        self,
        src_x: torch.Tensor,
        src_mask: torch.Tensor,
        edges: torch.Tensor,
        edge_mask: torch.Tensor,
        transform: nn.Module,
        num_papers: int,
    ) -> torch.Tensor:
        """
        Aggregate source node messages to paper nodes.

        edges: [B, E, 2] where each edge = [paper_idx, src_idx]
        """
        bsz, num_src, dim = src_x.shape
        if num_papers == 0 or num_src == 0 or edges.size(1) == 0:
            return src_x.new_zeros(bsz, num_papers, dim)

        src_h = transform(src_x)  # [B, S, D]

        valid_pos = edge_mask.nonzero(as_tuple=False)  # [M, 2] -> [batch_idx, edge_pos]
        if valid_pos.numel() == 0:
            return src_x.new_zeros(bsz, num_papers, dim)

        batch_idx = valid_pos[:, 0].long()
        edge_pos = valid_pos[:, 1].long()

        e = edges[batch_idx, edge_pos]  # [M, 2]
        paper_idx = e[:, 0].long()
        src_idx = e[:, 1].long()

        # filter invalid source nodes by mask
        valid_src = src_mask[batch_idx, src_idx]
        if valid_src.sum().item() == 0:
            return src_x.new_zeros(bsz, num_papers, dim)

        batch_idx = batch_idx[valid_src]
        paper_idx = paper_idx[valid_src]
        src_idx = src_idx[valid_src]

        global_paper_idx = batch_idx * num_papers + paper_idx
        global_src_idx = batch_idx * num_src + src_idx

        src_h_flat = src_h.reshape(bsz * num_src, dim)
        msg = src_h_flat[global_src_idx]  # [M_valid, D]

        out_flat = src_x.new_zeros(bsz * num_papers, dim)
        deg_flat = src_x.new_zeros(bsz * num_papers, 1)

        out_flat.index_add_(0, global_paper_idx, msg)
        deg_flat.index_add_(
            0,
            global_paper_idx,
            torch.ones((global_paper_idx.size(0), 1), device=src_x.device, dtype=src_x.dtype),
        )

        out_flat = out_flat / deg_flat.clamp_min(1.0)
        return out_flat.view(bsz, num_papers, dim)

    def _aggregate_from_paper_vectorized(
        self,
        paper_x: torch.Tensor,
        paper_mask: torch.Tensor,
        edges: torch.Tensor,
        edge_mask: torch.Tensor,
        transform: nn.Module,
        num_dst: int,
    ) -> torch.Tensor:
        """
        Aggregate paper messages to destination nodes.

        edges: [B, E, 2] where each edge = [paper_idx, dst_idx]
        """
        bsz, num_papers, dim = paper_x.shape
        if num_dst == 0 or num_papers == 0 or edges.size(1) == 0:
            return paper_x.new_zeros(bsz, num_dst, dim)

        paper_h = transform(paper_x)  # [B, P, D]

        valid_pos = edge_mask.nonzero(as_tuple=False)
        if valid_pos.numel() == 0:
            return paper_x.new_zeros(bsz, num_dst, dim)

        batch_idx = valid_pos[:, 0].long()
        edge_pos = valid_pos[:, 1].long()

        e = edges[batch_idx, edge_pos]
        paper_idx = e[:, 0].long()
        dst_idx = e[:, 1].long()

        valid_paper = paper_mask[batch_idx, paper_idx]
        if valid_paper.sum().item() == 0:
            return paper_x.new_zeros(bsz, num_dst, dim)

        batch_idx = batch_idx[valid_paper]
        paper_idx = paper_idx[valid_paper]
        dst_idx = dst_idx[valid_paper]

        global_dst_idx = batch_idx * num_dst + dst_idx
        global_paper_idx = batch_idx * num_papers + paper_idx

        paper_h_flat = paper_h.reshape(bsz * num_papers, dim)
        msg = paper_h_flat[global_paper_idx]

        out_flat = paper_x.new_zeros(bsz * num_dst, dim)
        deg_flat = paper_x.new_zeros(bsz * num_dst, 1)

        out_flat.index_add_(0, global_dst_idx, msg)
        deg_flat.index_add_(
            0,
            global_dst_idx,
            torch.ones((global_dst_idx.size(0), 1), device=paper_x.device, dtype=paper_x.dtype),
        )

        out_flat = out_flat / deg_flat.clamp_min(1.0)
        return out_flat.view(bsz, num_dst, dim)


class RingTypeEncoderV3(nn.Module):
    """
    Encode one ring using:
      1) type-specific MLP projection
      2) relation-aware message passing centered on paper
      3) type pooling
      4) fused ring representation
    """

    def __init__(
        self,
        paper_in_dim: int,
        author_in_dim: int,
        topic_in_dim: int,
        venue_in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.paper_proj = NodeMLP(paper_in_dim, hidden_dim, hidden_dim, dropout)
        self.author_proj = NodeMLP(author_in_dim, hidden_dim, hidden_dim, dropout)
        self.topic_proj = NodeMLP(topic_in_dim, hidden_dim, hidden_dim, dropout)
        self.venue_proj = NodeMLP(venue_in_dim, hidden_dim, hidden_dim, dropout)

        self.rel_block = RelationMessagePassing(hidden_dim, dropout)

        self.paper_pool = TypePool(hidden_dim, dropout)
        self.author_pool = TypePool(hidden_dim, dropout)
        self.topic_pool = TypePool(hidden_dim, dropout)
        self.venue_pool = TypePool(hidden_dim, dropout)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, ring: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        paper_x = self.paper_proj(ring["paper_x"])
        author_x = self.author_proj(ring["author_x"])
        topic_x = self.topic_proj(ring["topic_x"])
        venue_x = self.venue_proj(ring["venue_x"])

        paper_mask = ring["paper_mask"]
        author_mask = ring["author_mask"]
        topic_mask = ring["topic_mask"]
        venue_mask = ring["venue_mask"]

        paper_x, author_x, topic_x, venue_x = self.rel_block(
            paper_x=paper_x,
            paper_mask=paper_mask,
            author_x=author_x,
            author_mask=author_mask,
            topic_x=topic_x,
            topic_mask=topic_mask,
            venue_x=venue_x,
            venue_mask=venue_mask,
            pa_edges=ring["paper_author_edges"],
            pa_edge_mask=ring["paper_author_edge_mask"],
            pt_edges=ring["paper_topic_edges"],
            pt_edge_mask=ring["paper_topic_edge_mask"],
            pv_edges=ring["paper_venue_edges"],
            pv_edge_mask=ring["paper_venue_edge_mask"],
        )

        paper_repr = self.paper_pool(paper_x, paper_mask)
        author_repr = self.author_pool(author_x, author_mask)
        topic_repr = self.topic_pool(topic_x, topic_mask)
        venue_repr = self.venue_pool(venue_x, venue_mask)

        ring_repr = self.norm(
            self.fuse(torch.cat([paper_repr, author_repr, topic_repr, venue_repr], dim=-1))
        )

        return {
            "paper_x": paper_x,
            "author_x": author_x,
            "topic_x": topic_x,
            "venue_x": venue_x,
            "paper_repr": paper_repr,
            "author_repr": author_repr,
            "topic_repr": topic_repr,
            "venue_repr": venue_repr,
            "ring_repr": ring_repr,
        }


if __name__ == "__main__":
    import torch
    from config.feature_schema_v3 import FEATURE_SCHEMA

    bsz = 2
    ring = {
        "paper_x": torch.randn(bsz, 5, FEATURE_SCHEMA.get_dim("paper")),
        "paper_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool),
        "author_x": torch.randn(bsz, 4, FEATURE_SCHEMA.get_dim("author")),
        "author_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool),
        "topic_x": torch.randn(bsz, 3, FEATURE_SCHEMA.get_dim("topic")),
        "topic_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "venue_x": torch.randn(bsz, 2, FEATURE_SCHEMA.get_dim("venue")),
        "venue_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        "paper_author_edges": torch.tensor(
            [[[0, 0], [0, 1], [1, 2], [-1, -1]], [[0, 0], [1, 0], [-1, -1], [-1, -1]]],
            dtype=torch.long,
        ),
        "paper_author_edge_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        "paper_topic_edges": torch.tensor(
            [[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [-1, -1]]],
            dtype=torch.long,
        ),
        "paper_topic_edge_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "paper_venue_edges": torch.tensor(
            [[[0, 0], [1, 1]], [[0, 0], [-1, -1]]],
            dtype=torch.long,
        ),
        "paper_venue_edge_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    }

    model = RingTypeEncoderV3(
        paper_in_dim=FEATURE_SCHEMA.get_dim("paper"),
        author_in_dim=FEATURE_SCHEMA.get_dim("author"),
        topic_in_dim=FEATURE_SCHEMA.get_dim("topic"),
        venue_in_dim=FEATURE_SCHEMA.get_dim("venue"),
        hidden_dim=64,
        dropout=0.1,
    )
    out = model(ring)
    for k, v in out.items():
        print(k, tuple(v.shape))