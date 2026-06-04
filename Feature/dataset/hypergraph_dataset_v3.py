#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from config.path_config_v3 import PATHS
from utils.shard_cache_v3 import ShardedSubgraphCacheV3


class HypergraphDatasetV3(Dataset):
    """
    Lightweight dataset for encoder training with year-split cache layout.

    Cache layout
    ------------
    artifacts/cache/{dataset}/{split}/{year}/
      - index.npz
      - meta.json
      - shard_*.pkl

    Important
    ---------
    1. Do NOT load ResourceManagerV3 here.
    2. Do NOT load raw dataset json here.
    3. Do NOT build giant python dict indices here.
    4. Aggregate all valid year-level caches into compact sample arrays.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        year_start: int = 2000,
        year_end: int = 2014,
        max_open_shards: int = 8,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset.lower().strip()
        self.split = split.lower().strip()
        self.year_start = int(year_start)
        self.year_end = int(year_end)

        self.dp = PATHS.get_dataset_paths(self.dataset_name)

        # Lightweight label loading
        self.history_mask = np.load(self.dp.history_mask, mmap_mode="r")
        self.history_hs = np.load(self.dp.history_hs, mmap_mode="r")
        self.history_hf = np.load(self.dp.history_hf, mmap_mode="r")
        self.history_cls = np.load(self.dp.history_cls, mmap_mode="r")

        with open(self.dp.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        self.split_indices = np.asarray(split_data[self.split], dtype=np.int64)

        with open(self.dp.author_ids_history, "r", encoding="utf-8") as f:
            self.author_ids_history = json.load(f)

        # one cache object per year
        self.year_caches: Dict[int, ShardedSubgraphCacheV3] = {}
        self.cache_slot_to_year: List[int] = []

        # flattened sample arrays
        self.author_idx_arr: np.ndarray
        self.year_arr: np.ndarray
        self.cache_slot_arr: np.ndarray
        self.shard_id_arr: np.ndarray
        self.offset_arr: np.ndarray

        self._build_samples(max_open_shards=max_open_shards)

    # ------------------------------------------------------------------
    # Build compact sample arrays from year-level caches
    # ------------------------------------------------------------------
    def _build_samples(self, max_open_shards: int) -> None:
        author_idx_parts: List[np.ndarray] = []
        year_parts: List[np.ndarray] = []
        cache_slot_parts: List[np.ndarray] = []
        shard_id_parts: List[np.ndarray] = []
        offset_parts: List[np.ndarray] = []

        split_set = set(self.split_indices.tolist())
        cache_slot = 0

        for year in range(max(self.year_start, 2000), min(self.year_end, 2014) + 1):
            cache_root = PATHS.get_cache_year_root(self.dataset_name, self.split, year)
            index_file = cache_root / "index.npz"
            if not index_file.exists():
                continue

            cache = ShardedSubgraphCacheV3(cache_root, max_open_shards=max_open_shards)
            cache.load_index_arrays()

            assert cache.author_idx_arr is not None
            assert cache.year_arr is not None
            assert cache.shard_id_arr is not None
            assert cache.offset_arr is not None

            author_idx_arr = cache.author_idx_arr.astype(np.int64, copy=False)
            year_arr = cache.year_arr.astype(np.int64, copy=False)
            shard_id_arr = cache.shard_id_arr.astype(np.int32, copy=False)
            offset_arr = cache.offset_arr.astype(np.int32, copy=False)

            # split membership re-check
            keep_split = np.isin(author_idx_arr, self.split_indices)

            # year consistency re-check
            keep_year = (year_arr == year)

            # history mask re-check
            t_arr = year_arr - 2000
            keep_hist = self.history_mask[author_idx_arr, t_arr] == 1

            keep = keep_split & keep_year & keep_hist
            if not np.any(keep):
                continue

            author_idx_parts.append(author_idx_arr[keep])
            year_parts.append(year_arr[keep].astype(np.int16, copy=False))
            shard_id_parts.append(shard_id_arr[keep])
            offset_parts.append(offset_arr[keep])
            cache_slot_parts.append(
                np.full(int(np.sum(keep)), fill_value=cache_slot, dtype=np.int16)
            )

            self.year_caches[year] = cache
            self.cache_slot_to_year.append(year)
            cache_slot += 1

        if len(author_idx_parts) == 0:
            self.author_idx_arr = np.zeros((0,), dtype=np.int64)
            self.year_arr = np.zeros((0,), dtype=np.int16)
            self.cache_slot_arr = np.zeros((0,), dtype=np.int16)
            self.shard_id_arr = np.zeros((0,), dtype=np.int32)
            self.offset_arr = np.zeros((0,), dtype=np.int32)
            return

        self.author_idx_arr = np.concatenate(author_idx_parts, axis=0)
        self.year_arr = np.concatenate(year_parts, axis=0)
        self.cache_slot_arr = np.concatenate(cache_slot_parts, axis=0)
        self.shard_id_arr = np.concatenate(shard_id_parts, axis=0)
        self.offset_arr = np.concatenate(offset_parts, axis=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.author_idx_arr)

    def __getitem__(self, index: int) -> Dict:
        author_idx = int(self.author_idx_arr[index])
        year = int(self.year_arr[index])
        cache_slot = int(self.cache_slot_arr[index])
        shard_id = int(self.shard_id_arr[index])
        offset = int(self.offset_arr[index])

        cache_year = self.cache_slot_to_year[cache_slot]
        cache = self.year_caches[cache_year]
        entry = cache.get_by_location(shard_id, offset)

        t = year - 2000
        hs = int(self.history_hs[author_idx, t])
        hf = int(self.history_hf[author_idx, t])
        cls = int(self.history_cls[author_idx, t])

        return {
            "author_idx": author_idx,
            "author_id": entry["author_id"],
            "year": year,
            "hist_hs": hs,
            "hist_hf": hf,
            "hist_cls": cls,
            "subgraph": entry["subgraph"],
        }

    # ------------------------------------------------------------------
    # Debug summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        if len(self) == 0:
            return {
                "dataset": self.dataset_name,
                "split": self.split,
                "n_samples": 0,
                "year_min": None,
                "year_max": None,
                "n_year_caches": 0,
            }

        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "n_samples": int(len(self)),
            "year_min": int(self.year_arr.min()),
            "year_max": int(self.year_arr.max()),
            "n_year_caches": int(len(self.year_caches)),
        }


if __name__ == "__main__":
    ds = HypergraphDatasetV3(dataset="aps", split="train")
    print(ds.summary())
    if len(ds) > 0:
        item = ds[0]
        print(item.keys())
        print(item["author_id"], item["year"], item["hist_cls"])