#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from config.path_config_v3 import PATHS
from utils.packed_cache_v3 import PackedYearCacheV3


class PackedHypergraphDatasetV3(Dataset):
    """
    Aggregate packed year caches for one dataset/split over [year_start, year_end].
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        year_start: int = 2000,
        year_end: int = 2014,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset.lower().strip()
        self.split = split.lower().strip()
        self.year_start = int(year_start)
        self.year_end = int(year_end)

        self.caches: Dict[int, PackedYearCacheV3] = {}
        self.cache_slot_to_year: List[int] = []

        self.sample_cache_slot: np.ndarray
        self.sample_local_idx: np.ndarray

        self._build_index()

    def _build_index(self) -> None:
        cache_slot_parts: List[np.ndarray] = []
        local_idx_parts: List[np.ndarray] = []

        cache_slot = 0
        for year in range(max(self.year_start, 2000), min(self.year_end, 2014) + 1):
            root = (
                PATHS.root / "artifacts" / "cache_packed" / self.dataset_name / self.split / str(year)
            )
            if not (root / "sample_index.npz").exists():
                continue

            cache = PackedYearCacheV3(root)
            idx = cache.load_index()
            n = idx["author_idx"].shape[0]
            if n == 0:
                continue

            self.caches[year] = cache
            self.cache_slot_to_year.append(year)

            cache_slot_parts.append(np.full(n, cache_slot, dtype=np.int16))
            local_idx_parts.append(np.arange(n, dtype=np.int32))
            cache_slot += 1

        if len(cache_slot_parts) == 0:
            self.sample_cache_slot = np.zeros((0,), dtype=np.int16)
            self.sample_local_idx = np.zeros((0,), dtype=np.int32)
        else:
            self.sample_cache_slot = np.concatenate(cache_slot_parts, axis=0)
            self.sample_local_idx = np.concatenate(local_idx_parts, axis=0)

    def __len__(self) -> int:
        return len(self.sample_local_idx)

    @staticmethod
    def _slice_by_ptr(arr: np.ndarray, ptr: np.ndarray, i: int) -> np.ndarray:
        s = int(ptr[i])
        e = int(ptr[i + 1])
        return arr[s:e]

    def _load_view_sample(self, cache: PackedYearCacheV3, prefix: str, local_i: int) -> Dict:
        target_x = cache.load_array(f"{prefix}_target_x")[local_i]

        # ring1
        r1_paper_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_paper_x"),
            cache.load_array(f"{prefix}_ring1_paper_ptr"),
            local_i,
        )
        r1_author_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_author_x"),
            cache.load_array(f"{prefix}_ring1_author_ptr"),
            local_i,
        )
        r1_topic_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_topic_x"),
            cache.load_array(f"{prefix}_ring1_topic_ptr"),
            local_i,
        )
        r1_venue_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_venue_x"),
            cache.load_array(f"{prefix}_ring1_venue_ptr"),
            local_i,
        )
        r1_pa = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_pa_edges"),
            cache.load_array(f"{prefix}_ring1_pa_ptr"),
            local_i,
        )
        r1_pt = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_pt_edges"),
            cache.load_array(f"{prefix}_ring1_pt_ptr"),
            local_i,
        )
        r1_pv = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring1_pv_edges"),
            cache.load_array(f"{prefix}_ring1_pv_ptr"),
            local_i,
        )

        # ring2
        r2_paper_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_paper_x"),
            cache.load_array(f"{prefix}_ring2_paper_ptr"),
            local_i,
        )
        r2_author_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_author_x"),
            cache.load_array(f"{prefix}_ring2_author_ptr"),
            local_i,
        )
        r2_topic_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_topic_x"),
            cache.load_array(f"{prefix}_ring2_topic_ptr"),
            local_i,
        )
        r2_venue_x = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_venue_x"),
            cache.load_array(f"{prefix}_ring2_venue_ptr"),
            local_i,
        )
        r2_pa = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_pa_edges"),
            cache.load_array(f"{prefix}_ring2_pa_ptr"),
            local_i,
        )
        r2_pt = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_pt_edges"),
            cache.load_array(f"{prefix}_ring2_pt_ptr"),
            local_i,
        )
        r2_pv = self._slice_by_ptr(
            cache.load_array(f"{prefix}_ring2_pv_edges"),
            cache.load_array(f"{prefix}_ring2_pv_ptr"),
            local_i,
        )

        return {
            "ring0": {"target_x": target_x},
            "ring1": {
                "paper_x": r1_paper_x,
                "author_x": r1_author_x,
                "topic_x": r1_topic_x,
                "venue_x": r1_venue_x,
                "paper_author_edges": r1_pa,
                "paper_topic_edges": r1_pt,
                "paper_venue_edges": r1_pv,
            },
            "ring2": {
                "paper_x": r2_paper_x,
                "author_x": r2_author_x,
                "topic_x": r2_topic_x,
                "venue_x": r2_venue_x,
                "paper_author_edges": r2_pa,
                "paper_topic_edges": r2_pt,
                "paper_venue_edges": r2_pv,
            },
        }

    def __getitem__(self, index: int) -> Dict:
        cache_slot = int(self.sample_cache_slot[index])
        local_i = int(self.sample_local_idx[index])

        year = self.cache_slot_to_year[cache_slot]
        cache = self.caches[year]
        idx = cache.load_index()

        author_idx = int(idx["author_idx"][local_i])
        year = int(idx["year"][local_i])
        hist_hs = int(idx["hist_hs"][local_i])
        hist_hf = int(idx["hist_hf"][local_i])
        hist_cls = int(idx["hist_cls"][local_i])

        return {
            "author_idx": author_idx,
            "year": year,
            "hist_hs": hist_hs,
            "hist_hf": hist_hf,
            "hist_cls": hist_cls,
            "stock": self._load_view_sample(cache, "stock", local_i),
            "flow": self._load_view_sample(cache, "flow", local_i),
        }

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
            "year_min": int(min(self.cache_slot_to_year)),
            "year_max": int(max(self.cache_slot_to_year)),
            "n_year_caches": int(len(self.cache_slot_to_year)),
        }


if __name__ == "__main__":
    ds = PackedHypergraphDatasetV3(dataset="aps", split="train")
    print(ds.summary())
    if len(ds) > 0:
        item = ds[0]
        print(item.keys())
        print(item["author_idx"], item["year"], item["hist_cls"])