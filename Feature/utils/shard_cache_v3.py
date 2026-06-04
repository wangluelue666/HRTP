#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class ShardedSubgraphCacheV3:
    """
    Disk-backed sharded cache for paired subgraphs.

    Training stage should prefer array-based index access rather than building
    a giant python dict for all (author_idx, year) pairs.
    """

    def __init__(self, cache_root: str | Path, max_open_shards: int = 8) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.index_file = self.cache_root / "index.npz"
        self.meta_file = self.cache_root / "meta.json"

        self.max_open_shards = int(max_open_shards)
        self._open_shards: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()

        # optional heavy dict mode
        self._index_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._loaded_index_dict = False

        # light array mode
        self.author_idx_arr: Optional[np.ndarray] = None
        self.year_arr: Optional[np.ndarray] = None
        self.shard_id_arr: Optional[np.ndarray] = None
        self.offset_arr: Optional[np.ndarray] = None
        self._loaded_index_arrays = False

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def build_from_entries(
        self,
        entries: Iterable[Dict[str, Any]],
        shard_size: int = 1024,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        shard_size = max(int(shard_size), 1)

        author_idx_list: List[int] = []
        year_list: List[int] = []
        shard_id_list: List[int] = []
        offset_list: List[int] = []

        shard_entries: List[Dict[str, Any]] = []
        shard_id = 0
        n_total = 0
        n_written_shards = 0

        for entry in entries:
            author_idx = int(entry["author_idx"])
            year = int(entry["year"])

            offset = len(shard_entries)
            shard_entries.append(entry)

            author_idx_list.append(author_idx)
            year_list.append(year)
            shard_id_list.append(shard_id)
            offset_list.append(offset)

            n_total += 1

            if len(shard_entries) >= shard_size:
                self._dump_shard(shard_id, shard_entries)
                shard_entries = []
                shard_id += 1
                n_written_shards += 1

        if shard_entries:
            self._dump_shard(shard_id, shard_entries)
            n_written_shards += 1

        np.savez_compressed(
            self.index_file,
            author_idx=np.asarray(author_idx_list, dtype=np.int64),
            year=np.asarray(year_list, dtype=np.int16),
            shard_id=np.asarray(shard_id_list, dtype=np.int32),
            offset=np.asarray(offset_list, dtype=np.int32),
        )

        meta_out = {
            "n_entries": int(n_total),
            "n_shards": int(n_written_shards),
            "shard_size": int(shard_size),
        }
        if meta:
            meta_out.update(meta)

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

        self._loaded_index_dict = False
        self._index_map.clear()

        self._loaded_index_arrays = False
        self.author_idx_arr = None
        self.year_arr = None
        self.shard_id_arr = None
        self.offset_arr = None

        self._open_shards.clear()

    def _dump_shard(self, shard_id: int, entries: List[Dict[str, Any]]) -> None:
        shard_path = self.cache_root / f"shard_{shard_id:06d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Light array index mode
    # ------------------------------------------------------------------
    def load_index_arrays(self) -> None:
        if self._loaded_index_arrays:
            return
        if not self.index_file.exists():
            raise FileNotFoundError(f"Missing cache index: {self.index_file}")

        data = np.load(self.index_file)
        self.author_idx_arr = data["author_idx"]
        self.year_arr = data["year"]
        self.shard_id_arr = data["shard_id"]
        self.offset_arr = data["offset"]
        self._loaded_index_arrays = True

    # ------------------------------------------------------------------
    # Optional heavy dict mode
    # ------------------------------------------------------------------
    def load_index(self) -> None:
        if self._loaded_index_dict:
            return
        self.load_index_arrays()

        assert self.author_idx_arr is not None
        assert self.year_arr is not None
        assert self.shard_id_arr is not None
        assert self.offset_arr is not None

        self._index_map = {}
        for a, y, s, o in zip(
            self.author_idx_arr,
            self.year_arr,
            self.shard_id_arr,
            self.offset_arr,
        ):
            self._index_map[(int(a), int(y))] = (int(s), int(o))

        self._loaded_index_dict = True

    def has(self, author_idx: int, year: int) -> bool:
        self.load_index()
        return (int(author_idx), int(year)) in self._index_map

    def get(self, author_idx: int, year: int) -> Dict[str, Any]:
        self.load_index()
        key = (int(author_idx), int(year))
        if key not in self._index_map:
            raise KeyError(f"Cache miss for (author_idx={author_idx}, year={year})")
        shard_id, offset = self._index_map[key]
        return self.get_by_location(shard_id, offset)

    def get_by_location(self, shard_id: int, offset: int) -> Dict[str, Any]:
        shard_entries = self._load_shard(int(shard_id))
        return shard_entries[int(offset)]

    def _load_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        if shard_id in self._open_shards:
            entries = self._open_shards.pop(shard_id)
            self._open_shards[shard_id] = entries
            return entries

        shard_path = self.cache_root / f"shard_{shard_id:06d}.pkl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        with open(shard_path, "rb") as f:
            entries = pickle.load(f)

        self._open_shards[shard_id] = entries
        while len(self._open_shards) > self.max_open_shards:
            self._open_shards.popitem(last=False)

        return entries

    # ------------------------------------------------------------------
    # Meta / debug
    # ------------------------------------------------------------------
    def load_meta(self) -> Dict[str, Any]:
        if not self.meta_file.exists():
            return {}
        with open(self.meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def summary(self) -> Dict[str, Any]:
        meta = self.load_meta()
        return {
            "cache_root": str(self.cache_root),
            "index_exists": self.index_file.exists(),
            "meta_exists": self.meta_file.exists(),
            "meta": meta,
        }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ShardedSubgraphCacheV3(tmpdir)
        demo_entries = [
            {"author_idx": 0, "author_id": "a0", "year": 2014, "subgraph": {"demo": 1}},
            {"author_idx": 1, "author_id": "a1", "year": 2013, "subgraph": {"demo": 2}},
        ]
        cache.build_from_entries(demo_entries, shard_size=1, meta={"dataset": "demo"})
        cache.load_index_arrays()
        print(cache.summary())
        print(cache.get_by_location(int(cache.shard_id_arr[0]), int(cache.offset_arr[0])))