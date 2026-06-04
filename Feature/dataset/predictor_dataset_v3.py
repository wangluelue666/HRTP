#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

from config.path_config_v3 import PATHS


class PredictorDatasetV3(Dataset):
    """
    Predictor dataset for v3 merged embeddings.

    Inputs
    ------
    X_concat: (N, T, D)
    M_year  : (N, T)

    Targets
    -------
    eval_cls: (N, 6) or equivalent classification target matrix
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        input_name: str = "X_concat",
        use_mask: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset.lower().strip()
        self.split = split.lower().strip()
        self.input_name = input_name
        self.use_mask = bool(use_mask)

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.dp = PATHS.get_dataset_paths(self.dataset_name)

        merged_root = self.dp.merged_embed_root
        self.X = np.load(merged_root / f"{self.input_name}.npy", mmap_mode="r").astype(np.float32, copy=False)
        self.M = np.load(merged_root / "M_year.npy", mmap_mode="r").astype(np.float32, copy=False)

        self.Y = np.load(self.dp.eval_cls, mmap_mode="r").astype(np.int64, copy=False)

        with open(self.dp.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        self.indices = np.asarray(split_data[self.split], dtype=np.int64)

        with open(self.dp.author_ids_history, "r", encoding="utf-8") as f:
            self.author_ids = json.load(f)

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"Input/target size mismatch: X has N={self.X.shape[0]}, Y has N={self.Y.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        aidx = int(self.indices[index])

        x = np.asarray(self.X[aidx], dtype=np.float32)
        m = np.asarray(self.M[aidx], dtype=np.float32)
        y = np.asarray(self.Y[aidx], dtype=np.int64)

        if self.use_mask:
            x = x * m[:, None]

        return {
            "author_idx": aidx,
            "author_id": self.author_ids[aidx],
            "x": x,
            "mask": m,
            "y": y,
        }

    def summary(self) -> Dict:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "input_name": self.input_name,
            "use_mask": self.use_mask,
            "n_samples": len(self),
            "seq_len": int(self.X.shape[1]),
            "input_dim": int(self.X.shape[2]),
            "target_shape_tail": tuple(self.Y.shape[1:]),
        }


if __name__ == "__main__":
    ds = PredictorDatasetV3(dataset="aps", split="train", input_name="X_concat", use_mask=True)
    print(ds.summary())
    item = ds[0]
    print(item.keys())
    print(item["author_id"], item["x"].shape, item["y"].shape)