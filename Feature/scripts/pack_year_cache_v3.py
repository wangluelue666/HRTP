#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config.feature_schema_v3 import FEATURE_SCHEMA
from config.path_config_v3 import PATHS
from utils.packed_cache_v3 import PackedYearCacheV3
from utils.shard_cache_v3 import ShardedSubgraphCacheV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack object cache into numeric packed cache for one split/year.")
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--year", required=True, type=int)
    return parser.parse_args()


def _concat_or_empty(chunks: List[np.ndarray], shape_tail: tuple, dtype) -> np.ndarray:
    if len(chunks) == 0:
        return np.zeros((0,) + shape_tail, dtype=dtype)
    return np.concatenate(chunks, axis=0)


def _ptr_from_lengths(lengths: List[int]) -> np.ndarray:
    ptr = [0]
    cur = 0
    for x in lengths:
        cur += int(x)
        ptr.append(cur)
    return np.asarray(ptr, dtype=np.int64)


def _as_float2d(feature_list: List[List[float]], feat_dim: int) -> np.ndarray:
    if len(feature_list) == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)
    arr = np.asarray(feature_list, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != feat_dim:
        raise ValueError(f"Feature shape mismatch, expected (*, {feat_dim}), got {arr.shape}")
    return arr


def _as_int2d(edge_list: List[List[int]]) -> np.ndarray:
    if len(edge_list) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    arr = np.asarray(edge_list, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Edge shape mismatch, expected (*, 2), got {arr.shape}")
    return arr


def _pack_view(arrays: Dict[str, np.ndarray], prefix: str, subgraphs: List[Dict[str, Any]]) -> None:
    target_dim = FEATURE_SCHEMA.get_dim("target")
    paper_dim = FEATURE_SCHEMA.get_dim("paper")
    author_dim = FEATURE_SCHEMA.get_dim("author")
    topic_dim = FEATURE_SCHEMA.get_dim("topic")
    venue_dim = FEATURE_SCHEMA.get_dim("venue")

    target_x_list: List[List[float]] = []

    # ring1
    ring1_paper_feat_chunks: List[np.ndarray] = []
    ring1_author_feat_chunks: List[np.ndarray] = []
    ring1_topic_feat_chunks: List[np.ndarray] = []
    ring1_venue_feat_chunks: List[np.ndarray] = []
    ring1_pa_edge_chunks: List[np.ndarray] = []
    ring1_pt_edge_chunks: List[np.ndarray] = []
    ring1_pv_edge_chunks: List[np.ndarray] = []

    ring1_paper_lens: List[int] = []
    ring1_author_lens: List[int] = []
    ring1_topic_lens: List[int] = []
    ring1_venue_lens: List[int] = []
    ring1_pa_lens: List[int] = []
    ring1_pt_lens: List[int] = []
    ring1_pv_lens: List[int] = []

    # ring2
    ring2_paper_feat_chunks: List[np.ndarray] = []
    ring2_author_feat_chunks: List[np.ndarray] = []
    ring2_topic_feat_chunks: List[np.ndarray] = []
    ring2_venue_feat_chunks: List[np.ndarray] = []
    ring2_pa_edge_chunks: List[np.ndarray] = []
    ring2_pt_edge_chunks: List[np.ndarray] = []
    ring2_pv_edge_chunks: List[np.ndarray] = []

    ring2_paper_lens: List[int] = []
    ring2_author_lens: List[int] = []
    ring2_topic_lens: List[int] = []
    ring2_venue_lens: List[int] = []
    ring2_pa_lens: List[int] = []
    ring2_pt_lens: List[int] = []
    ring2_pv_lens: List[int] = []

    for sg in subgraphs:
        view = sg[prefix]

        target_feat = view["ring0"]["target"]["feature"]
        if len(target_feat) != target_dim:
            raise ValueError(f"{prefix} target dim mismatch: {len(target_feat)} != {target_dim}")
        target_x_list.append(target_feat)

        # ring1
        r1 = view["ring1"]
        r1_paper = _as_float2d([x["feature"] for x in r1["papers"]], feat_dim=paper_dim)
        r1_author = _as_float2d([x["feature"] for x in r1["authors"]], feat_dim=author_dim)
        r1_topic = _as_float2d([x["feature"] for x in r1["topics"]], feat_dim=topic_dim)
        r1_venue = _as_float2d([x["feature"] for x in r1["venues"]], feat_dim=venue_dim)
        r1_pa = _as_int2d(r1["paper_author_edges"])
        r1_pt = _as_int2d(r1["paper_topic_edges"])
        r1_pv = _as_int2d(r1["paper_venue_edges"])

        ring1_paper_feat_chunks.append(r1_paper)
        ring1_author_feat_chunks.append(r1_author)
        ring1_topic_feat_chunks.append(r1_topic)
        ring1_venue_feat_chunks.append(r1_venue)
        ring1_pa_edge_chunks.append(r1_pa)
        ring1_pt_edge_chunks.append(r1_pt)
        ring1_pv_edge_chunks.append(r1_pv)

        ring1_paper_lens.append(r1_paper.shape[0])
        ring1_author_lens.append(r1_author.shape[0])
        ring1_topic_lens.append(r1_topic.shape[0])
        ring1_venue_lens.append(r1_venue.shape[0])
        ring1_pa_lens.append(r1_pa.shape[0])
        ring1_pt_lens.append(r1_pt.shape[0])
        ring1_pv_lens.append(r1_pv.shape[0])

        # ring2
        r2 = view["ring2"]
        r2_paper = _as_float2d([x["feature"] for x in r2["papers"]], feat_dim=paper_dim)
        r2_author = _as_float2d([x["feature"] for x in r2["authors"]], feat_dim=author_dim)
        r2_topic = _as_float2d([x["feature"] for x in r2["topics"]], feat_dim=topic_dim)
        r2_venue = _as_float2d([x["feature"] for x in r2["venues"]], feat_dim=venue_dim)
        r2_pa = _as_int2d(r2["paper_author_edges"])
        r2_pt = _as_int2d(r2["paper_topic_edges"])
        r2_pv = _as_int2d(r2["paper_venue_edges"])

        ring2_paper_feat_chunks.append(r2_paper)
        ring2_author_feat_chunks.append(r2_author)
        ring2_topic_feat_chunks.append(r2_topic)
        ring2_venue_feat_chunks.append(r2_venue)
        ring2_pa_edge_chunks.append(r2_pa)
        ring2_pt_edge_chunks.append(r2_pt)
        ring2_pv_edge_chunks.append(r2_pv)

        ring2_paper_lens.append(r2_paper.shape[0])
        ring2_author_lens.append(r2_author.shape[0])
        ring2_topic_lens.append(r2_topic.shape[0])
        ring2_venue_lens.append(r2_venue.shape[0])
        ring2_pa_lens.append(r2_pa.shape[0])
        ring2_pt_lens.append(r2_pt.shape[0])
        ring2_pv_lens.append(r2_pv.shape[0])

    arrays[f"{prefix}_target_x"] = np.asarray(target_x_list, dtype=np.float32)

    arrays[f"{prefix}_ring1_paper_x"] = _concat_or_empty(ring1_paper_feat_chunks, (paper_dim,), np.float32)
    arrays[f"{prefix}_ring1_paper_ptr"] = _ptr_from_lengths(ring1_paper_lens)
    arrays[f"{prefix}_ring1_author_x"] = _concat_or_empty(ring1_author_feat_chunks, (author_dim,), np.float32)
    arrays[f"{prefix}_ring1_author_ptr"] = _ptr_from_lengths(ring1_author_lens)
    arrays[f"{prefix}_ring1_topic_x"] = _concat_or_empty(ring1_topic_feat_chunks, (topic_dim,), np.float32)
    arrays[f"{prefix}_ring1_topic_ptr"] = _ptr_from_lengths(ring1_topic_lens)
    arrays[f"{prefix}_ring1_venue_x"] = _concat_or_empty(ring1_venue_feat_chunks, (venue_dim,), np.float32)
    arrays[f"{prefix}_ring1_venue_ptr"] = _ptr_from_lengths(ring1_venue_lens)
    arrays[f"{prefix}_ring1_pa_edges"] = _concat_or_empty(ring1_pa_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring1_pa_ptr"] = _ptr_from_lengths(ring1_pa_lens)
    arrays[f"{prefix}_ring1_pt_edges"] = _concat_or_empty(ring1_pt_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring1_pt_ptr"] = _ptr_from_lengths(ring1_pt_lens)
    arrays[f"{prefix}_ring1_pv_edges"] = _concat_or_empty(ring1_pv_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring1_pv_ptr"] = _ptr_from_lengths(ring1_pv_lens)

    arrays[f"{prefix}_ring2_paper_x"] = _concat_or_empty(ring2_paper_feat_chunks, (paper_dim,), np.float32)
    arrays[f"{prefix}_ring2_paper_ptr"] = _ptr_from_lengths(ring2_paper_lens)
    arrays[f"{prefix}_ring2_author_x"] = _concat_or_empty(ring2_author_feat_chunks, (author_dim,), np.float32)
    arrays[f"{prefix}_ring2_author_ptr"] = _ptr_from_lengths(ring2_author_lens)
    arrays[f"{prefix}_ring2_topic_x"] = _concat_or_empty(ring2_topic_feat_chunks, (topic_dim,), np.float32)
    arrays[f"{prefix}_ring2_topic_ptr"] = _ptr_from_lengths(ring2_topic_lens)
    arrays[f"{prefix}_ring2_venue_x"] = _concat_or_empty(ring2_venue_feat_chunks, (venue_dim,), np.float32)
    arrays[f"{prefix}_ring2_venue_ptr"] = _ptr_from_lengths(ring2_venue_lens)
    arrays[f"{prefix}_ring2_pa_edges"] = _concat_or_empty(ring2_pa_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring2_pa_ptr"] = _ptr_from_lengths(ring2_pa_lens)
    arrays[f"{prefix}_ring2_pt_edges"] = _concat_or_empty(ring2_pt_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring2_pt_ptr"] = _ptr_from_lengths(ring2_pt_lens)
    arrays[f"{prefix}_ring2_pv_edges"] = _concat_or_empty(ring2_pv_edge_chunks, (2,), np.int32)
    arrays[f"{prefix}_ring2_pv_ptr"] = _ptr_from_lengths(ring2_pv_lens)


def main() -> None:
    args = parse_args()
    if args.year < 2000 or args.year > 2014:
        raise ValueError(f"--year must be in [2000, 2014], got {args.year}")

    PATHS.ensure_v3_dirs()

    object_root = PATHS.get_cache_year_root(args.dataset, args.split, args.year)
    if not (object_root / "index.npz").exists():
        raise FileNotFoundError(f"Missing object cache index: {object_root / 'index.npz'}")

    packed_root = (
        Path("/root/autodl-tmp/WH2/my_method_v3/artifacts/cache_packed")
        / args.dataset / args.split / str(args.year)
    )
    packed_root.mkdir(parents=True, exist_ok=True)

    object_cache = ShardedSubgraphCacheV3(object_root)
    object_cache.load_index_arrays()

    assert object_cache.author_idx_arr is not None
    assert object_cache.year_arr is not None
    assert object_cache.shard_id_arr is not None
    assert object_cache.offset_arr is not None

    author_idx_list: List[int] = []
    year_list: List[int] = []
    hist_hs_list: List[int] = []
    hist_hf_list: List[int] = []
    hist_cls_list: List[int] = []
    subgraphs: List[Dict[str, Any]] = []

    n = len(object_cache.author_idx_arr)
    for i in range(n):
        author_idx = int(object_cache.author_idx_arr[i])
        year = int(object_cache.year_arr[i])
        shard_id = int(object_cache.shard_id_arr[i])
        offset = int(object_cache.offset_arr[i])

        entry = object_cache.get_by_location(shard_id, offset)

        if "hist_hs" not in entry or "hist_hf" not in entry or "hist_cls" not in entry:
            raise KeyError(
                "Object cache entry missing hist labels. "
                "Rebuild object cache with updated build_paired_subgraphs_v3.py first."
            )

        author_idx_list.append(author_idx)
        year_list.append(year)
        hist_hs_list.append(int(entry["hist_hs"]))
        hist_hf_list.append(int(entry["hist_hf"]))
        hist_cls_list.append(int(entry["hist_cls"]))
        subgraphs.append(entry["subgraph"])

    arrays: Dict[str, np.ndarray] = {
        "author_idx": np.asarray(author_idx_list, dtype=np.int64),
        "year": np.asarray(year_list, dtype=np.int16),
        "hist_hs": np.asarray(hist_hs_list, dtype=np.int8),
        "hist_hf": np.asarray(hist_hf_list, dtype=np.int8),
        "hist_cls": np.asarray(hist_cls_list, dtype=np.int8),
    }

    _pack_view(arrays, "stock", subgraphs)
    _pack_view(arrays, "flow", subgraphs)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "year": int(args.year),
        "n_samples": int(len(author_idx_list)),
        "source_object_cache": str(object_root),
    }

    packed_cache = PackedYearCacheV3(packed_root)
    packed_cache.save(arrays, meta)

    print(json.dumps(packed_cache.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()