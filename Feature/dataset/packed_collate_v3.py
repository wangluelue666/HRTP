#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from config.feature_schema_v3 import FEATURE_SCHEMA


def _pad_2d_feat(batch_arrays: List[np.ndarray], feat_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    bsz = len(batch_arrays)
    max_n = max((x.shape[0] for x in batch_arrays), default=0)

    out = np.zeros((bsz, max_n, feat_dim), dtype=np.float32)
    mask = np.zeros((bsz, max_n), dtype=np.bool_)

    for b, arr in enumerate(batch_arrays):
        n = arr.shape[0]
        if n == 0:
            continue
        out[b, :n] = np.asarray(arr, dtype=np.float32)
        mask[b, :n] = True
    return out, mask


def _pad_edges(batch_edges: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    bsz = len(batch_edges)
    max_e = max((x.shape[0] for x in batch_edges), default=0)

    out = np.full((bsz, max_e, 2), fill_value=-1, dtype=np.int64)
    mask = np.zeros((bsz, max_e), dtype=np.bool_)

    for b, arr in enumerate(batch_edges):
        e = arr.shape[0]
        if e == 0:
            continue
        out[b, :e] = np.asarray(arr, dtype=np.int64)
        mask[b, :e] = True
    return out, mask


def _collate_ring(ring_batch: List[Dict]) -> Dict[str, np.ndarray]:
    paper_dim = FEATURE_SCHEMA.get_dim("paper")
    author_dim = FEATURE_SCHEMA.get_dim("author")
    topic_dim = FEATURE_SCHEMA.get_dim("topic")
    venue_dim = FEATURE_SCHEMA.get_dim("venue")

    paper_x, paper_mask = _pad_2d_feat([x["paper_x"] for x in ring_batch], paper_dim)
    author_x, author_mask = _pad_2d_feat([x["author_x"] for x in ring_batch], author_dim)
    topic_x, topic_mask = _pad_2d_feat([x["topic_x"] for x in ring_batch], topic_dim)
    venue_x, venue_mask = _pad_2d_feat([x["venue_x"] for x in ring_batch], venue_dim)

    pa_edges, pa_mask = _pad_edges([x["paper_author_edges"] for x in ring_batch])
    pt_edges, pt_mask = _pad_edges([x["paper_topic_edges"] for x in ring_batch])
    pv_edges, pv_mask = _pad_edges([x["paper_venue_edges"] for x in ring_batch])

    return {
        "paper_x": paper_x,
        "paper_mask": paper_mask,
        "author_x": author_x,
        "author_mask": author_mask,
        "topic_x": topic_x,
        "topic_mask": topic_mask,
        "venue_x": venue_x,
        "venue_mask": venue_mask,
        "paper_author_edges": pa_edges,
        "paper_author_edge_mask": pa_mask,
        "paper_topic_edges": pt_edges,
        "paper_topic_edge_mask": pt_mask,
        "paper_venue_edges": pv_edges,
        "paper_venue_edge_mask": pv_mask,
    }


def _collate_view(view_batch: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    target_x = np.stack(
        [np.asarray(x["ring0"]["target_x"], dtype=np.float32) for x in view_batch],
        axis=0,
    )

    return {
        "ring0": {"target_x": target_x},
        "ring1": _collate_ring([x["ring1"] for x in view_batch]),
        "ring2": _collate_ring([x["ring2"] for x in view_batch]),
    }


def collate_packed_subgraphs_v3(batch: List[Dict]) -> Dict:
    return {
        "author_idx": np.asarray([x["author_idx"] for x in batch], dtype=np.int64),
        "year": np.asarray([x["year"] for x in batch], dtype=np.int64),
        "hist_hs": np.asarray([x["hist_hs"] for x in batch], dtype=np.int64),
        "hist_hf": np.asarray([x["hist_hf"] for x in batch], dtype=np.int64),
        "hist_cls": np.asarray([x["hist_cls"] for x in batch], dtype=np.int64),
        "stock": _collate_view([x["stock"] for x in batch]),
        "flow": _collate_view([x["flow"] for x in batch]),
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset.packed_hypergraph_dataset_v3 import PackedHypergraphDatasetV3

    ds = PackedHypergraphDatasetV3(dataset="aps", split="train")
    print(ds.summary())
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_packed_subgraphs_v3)
        batch = next(iter(dl))
        print(batch.keys())
        print(batch["stock"]["ring0"]["target_x"].shape)