#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from config.feature_schema_v3 import FEATURE_SCHEMA


def _pad_feature_list(
    nodes: List[List[Dict]],
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(nodes)
    max_n = max((len(x) for x in nodes), default=0)

    x = torch.zeros(batch_size, max_n, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

    for b, node_list in enumerate(nodes):
        for i, node in enumerate(node_list):
            feat = node.get("feature", None)
            if feat is None:
                raise ValueError("Node missing 'feature' field in collate_fn_v3.")
            if len(feat) != feat_dim:
                raise ValueError(
                    f"Feature dim mismatch: expected {feat_dim}, got {len(feat)}"
                )
            x[b, i] = torch.tensor(feat, dtype=torch.float32)
            mask[b, i] = True

    return x, mask


def _pad_edge_list(
    edge_lists: List[List[List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(edge_lists)
    max_e = max((len(x) for x in edge_lists), default=0)

    edges = torch.full((batch_size, max_e, 2), fill_value=-1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_e, dtype=torch.bool)

    for b, e_list in enumerate(edge_lists):
        for i, e in enumerate(e_list):
            if len(e) != 2:
                raise ValueError(f"Edge format error: expected len=2, got {e}")
            edges[b, i, 0] = int(e[0])
            edges[b, i, 1] = int(e[1])
            mask[b, i] = True

    return edges, mask


def _collate_single_ring(ring_batch: List[Dict]) -> Dict[str, torch.Tensor]:
    author_dim = FEATURE_SCHEMA.get_dim("author")
    paper_dim = FEATURE_SCHEMA.get_dim("paper")
    topic_dim = FEATURE_SCHEMA.get_dim("topic")
    venue_dim = FEATURE_SCHEMA.get_dim("venue")

    papers = [x.get("papers", []) for x in ring_batch]
    authors = [x.get("authors", []) for x in ring_batch]
    topics = [x.get("topics", []) for x in ring_batch]
    venues = [x.get("venues", []) for x in ring_batch]

    pa_edges = [x.get("paper_author_edges", []) for x in ring_batch]
    pt_edges = [x.get("paper_topic_edges", []) for x in ring_batch]
    pv_edges = [x.get("paper_venue_edges", []) for x in ring_batch]

    paper_x, paper_mask = _pad_feature_list(papers, paper_dim)
    author_x, author_mask = _pad_feature_list(authors, author_dim)
    topic_x, topic_mask = _pad_feature_list(topics, topic_dim)
    venue_x, venue_mask = _pad_feature_list(venues, venue_dim)

    pa_edge_index, pa_edge_mask = _pad_edge_list(pa_edges)
    pt_edge_index, pt_edge_mask = _pad_edge_list(pt_edges)
    pv_edge_index, pv_edge_mask = _pad_edge_list(pv_edges)

    return {
        "paper_x": paper_x,
        "paper_mask": paper_mask,
        "author_x": author_x,
        "author_mask": author_mask,
        "topic_x": topic_x,
        "topic_mask": topic_mask,
        "venue_x": venue_x,
        "venue_mask": venue_mask,
        "paper_author_edges": pa_edge_index,
        "paper_author_edge_mask": pa_edge_mask,
        "paper_topic_edges": pt_edge_index,
        "paper_topic_edge_mask": pt_edge_mask,
        "paper_venue_edges": pv_edge_index,
        "paper_venue_edge_mask": pv_edge_mask,
    }


def _collate_ring0(ring0_batch: List[Dict]) -> Dict[str, torch.Tensor]:
    target_dim = FEATURE_SCHEMA.get_dim("target")
    targets = [x["target"] for x in ring0_batch]

    x = torch.zeros(len(targets), target_dim, dtype=torch.float32)
    for b, node in enumerate(targets):
        feat = node.get("feature", None)
        if feat is None:
            raise ValueError("Ring0 target node missing 'feature'.")
        if len(feat) != target_dim:
            raise ValueError(
                f"Ring0 feature dim mismatch: expected {target_dim}, got {len(feat)}"
            )
        x[b] = torch.tensor(feat, dtype=torch.float32)

    return {"target_x": x}


def _collate_single_view(view_batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    ring0_batch = [x["ring0"] for x in view_batch]
    ring1_batch = [x["ring1"] for x in view_batch]
    ring2_batch = [x["ring2"] for x in view_batch]

    return {
        "ring0": _collate_ring0(ring0_batch),
        "ring1": _collate_single_ring(ring1_batch),
        "ring2": _collate_single_ring(ring2_batch),
    }


def collate_paired_subgraphs_v3(batch: List[Dict]) -> Dict:
    stock_batch = [x["subgraph"]["stock"] for x in batch]
    flow_batch = [x["subgraph"]["flow"] for x in batch]

    author_idx = torch.tensor([x["author_idx"] for x in batch], dtype=torch.long)
    year = torch.tensor([x["year"] for x in batch], dtype=torch.long)
    hist_hs = torch.tensor([x["hist_hs"] for x in batch], dtype=torch.long)
    hist_hf = torch.tensor([x["hist_hf"] for x in batch], dtype=torch.long)
    hist_cls = torch.tensor([x["hist_cls"] for x in batch], dtype=torch.long)

    author_ids = [x["author_id"] for x in batch]

    return {
        "author_idx": author_idx,
        "author_id": author_ids,
        "year": year,
        "hist_hs": hist_hs,
        "hist_hf": hist_hf,
        "hist_cls": hist_cls,
        "stock": _collate_single_view(stock_batch),
        "flow": _collate_single_view(flow_batch),
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.hypergraph_dataset_v3 import HypergraphDatasetV3

    ds = HypergraphDatasetV3(dataset="aps", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_paired_subgraphs_v3)

    batch = next(iter(dl))
    print(batch.keys())
    print(batch["stock"]["ring0"]["target_x"].shape)
    print(batch["stock"]["ring1"]["paper_x"].shape)
    print(batch["flow"]["ring1"]["paper_author_edges"].shape)