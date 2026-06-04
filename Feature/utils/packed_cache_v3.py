#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class PackedYearCacheV3:
    """
    Packed numeric cache for one dataset/split/year.

    Layout
    ------
    cache_packed/{dataset}/{split}/{year}/
      - meta.json
      - sample_index.npz

      - target_x.npy
      - hist_labels.npy

      - ring1_paper_x.npy
      - ring1_paper_ptr.npy
      - ring1_author_x.npy
      - ring1_author_ptr.npy
      - ring1_topic_x.npy
      - ring1_topic_ptr.npy
      - ring1_venue_x.npy
      - ring1_venue_ptr.npy
      - ring1_pa_edges.npy
      - ring1_pa_ptr.npy
      - ring1_pt_edges.npy
      - ring1_pt_ptr.npy
      - ring1_pv_edges.npy
      - ring1_pv_ptr.npy

      - ring2_paper_x.npy
      - ring2_paper_ptr.npy
      - ring2_author_x.npy
      - ring2_author_ptr.npy
      - ring2_topic_x.npy
      - ring2_topic_ptr.npy
      - ring2_venue_x.npy
      - ring2_venue_ptr.npy
      - ring2_pa_edges.npy
      - ring2_pa_ptr.npy
      - ring2_pt_edges.npy
      - ring2_pt_ptr.npy
      - ring2_pv_edges.npy
      - ring2_pv_ptr.npy
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta_file = self.root / "meta.json"
        self.index_file = self.root / "sample_index.npz"

        self._arrays: Dict[str, np.ndarray] = {}
        self._meta: Optional[Dict] = None
        self._index: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        required_index_keys = ["author_idx", "year", "hist_hs", "hist_hf", "hist_cls"]
        for k in required_index_keys:
            if k not in arrays:
                raise KeyError(f"Missing required packed index array: {k}")

        index_payload = {
            "author_idx": arrays["author_idx"].astype(np.int64, copy=False),
            "year": arrays["year"].astype(np.int16, copy=False),
            "hist_hs": arrays["hist_hs"].astype(np.int8, copy=False),
            "hist_hf": arrays["hist_hf"].astype(np.int8, copy=False),
            "hist_cls": arrays["hist_cls"].astype(np.int8, copy=False),
        }
        np.savez_compressed(self.index_file, **index_payload)

        # save everything else as standalone npy files
        for key, value in arrays.items():
            if key in index_payload:
                continue
            np.save(self.root / f"{key}.npy", value)

        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_meta(self) -> Dict:
        if self._meta is None:
            if not self.meta_file.exists():
                raise FileNotFoundError(f"Missing meta file: {self.meta_file}")
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    def load_index(self) -> Dict[str, np.ndarray]:
        if self._index is None:
            if not self.index_file.exists():
                raise FileNotFoundError(f"Missing sample index: {self.index_file}")
            data = np.load(self.index_file, mmap_mode="r")
            self._index = {k: data[k] for k in data.files}
        return self._index

    def load_array(self, name: str, mmap_mode: str = "r") -> np.ndarray:
        if name not in self._arrays:
            path = self.root / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing packed array: {path}")
            self._arrays[name] = np.load(path, mmap_mode=mmap_mode)
        return self._arrays[name]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        meta = self.load_meta()
        index = self.load_index()
        return {
            "root": str(self.root),
            "n_samples": int(index["author_idx"].shape[0]),
            "dataset": meta.get("dataset"),
            "split": meta.get("split"),
            "year": meta.get("year"),
        }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PackedYearCacheV3(tmpdir)
        arrays = {
            "author_idx": np.array([1, 2], dtype=np.int64),
            "year": np.array([2000, 2000], dtype=np.int16),
            "hist_hs": np.array([0, 1], dtype=np.int8),
            "hist_hf": np.array([1, 0], dtype=np.int8),
            "hist_cls": np.array([1, 2], dtype=np.int8),
            "target_x": np.random.randn(2, 13).astype(np.float32),
            "ring1_paper_x": np.random.randn(3, 16).astype(np.float32),
            "ring1_paper_ptr": np.array([0, 2, 3], dtype=np.int64),
        }
        meta = {"dataset": "demo", "split": "train", "year": 2000}
        cache.save(arrays, meta)
        print(cache.summary())
        print(cache.load_array("target_x").shape)