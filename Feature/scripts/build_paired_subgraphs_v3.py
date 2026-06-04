#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List

from config.path_config_v3 import PATHS
from subgraph.sample_author_subgraph_v3 import AuthorSubgraphSamplerV3
from utils.resource_manager_v3 import ResourceManagerV3
from utils.shard_cache_v3 import ShardedSubgraphCacheV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3 paired subgraph cache for one split and one year.")
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--limit_authors", type=int, default=-1)
    parser.add_argument("--max_entries", type=int, default=-1)
    return parser.parse_args()


def iter_entries(
    rm: ResourceManagerV3,
    sampler: AuthorSubgraphSamplerV3,
    split_indices: List[int],
    year: int,
    max_entries: int = -1,
) -> Iterable[Dict]:
    labels = rm.get_history_labels()
    history_mask = labels["mask"]
    history_hs = labels["hs"]
    history_hf = labels["hf"]
    history_cls = labels["cls"]

    t = year - 2000
    count = 0

    for author_idx in split_indices:
        author_idx = int(author_idx)
        if history_mask[author_idx, t] == 0:
            continue

        author_id = rm.author_ids_history[author_idx]
        subgraph = sampler.sample(author_id, year)

        yield {
            "author_idx": author_idx,
            "author_id": author_id,
            "year": year,
            "hist_hs": int(history_hs[author_idx, t]),
            "hist_hf": int(history_hf[author_idx, t]),
            "hist_cls": int(history_cls[author_idx, t]),
            "subgraph": subgraph,
        }

        count += 1
        if max_entries > 0 and count >= max_entries:
            return


def main() -> None:
    args = parse_args()

    if args.year < 2000 or args.year > 2014:
        raise ValueError(f"--year must be in [2000, 2014], got {args.year}")

    PATHS.ensure_v3_dirs()

    rm = ResourceManagerV3(args.dataset)
    sampler = AuthorSubgraphSamplerV3(rm)

    split_indices = rm.get_split_indices(args.split).tolist()
    if args.limit_authors > 0:
        split_indices = split_indices[: args.limit_authors]

    cache_root = PATHS.get_cache_year_root(args.dataset, args.split, args.year)
    cache_root.mkdir(parents=True, exist_ok=True)

    cache = ShardedSubgraphCacheV3(cache_root)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "year": int(args.year),
        "limit_authors": int(args.limit_authors),
        "max_entries": int(args.max_entries),
    }

    entries = iter_entries(
        rm=rm,
        sampler=sampler,
        split_indices=split_indices,
        year=args.year,
        max_entries=args.max_entries,
    )

    cache.build_from_entries(
        entries=entries,
        shard_size=args.shard_size,
        meta=meta,
    )

    print(json.dumps(cache.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()