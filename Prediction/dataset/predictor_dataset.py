#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Support both:
#   1) python -m dataset.predictor_dataset
#   2) python dataset/predictor_dataset.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.path_config import (
    HARPPathConfig,
    build_harp_path_config,
    ensure_required_dirs,
    dump_path_manifest,
    assert_required_input_files,
)
SplitName = Literal["train", "val", "test"]


@dataclass
class HARPDataBundle:
    x_stock: np.ndarray
    x_flow: np.ndarray
    x_joint: np.ndarray
    labels: np.ndarray
    m_year: np.ndarray
    valid_len: np.ndarray
    year_ids: np.ndarray
    recency_ids: np.ndarray
    author_ids: List[str]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_author_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        # Prefer common key names if present.
        for key in ["author_ids", "ids", "active_author_ids"]:
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
        # Fall back to dictionary keys.
        return [str(k) for k in data.keys()]

    raise TypeError(f"Unsupported author id file format: {path}")


def _normalize_split_indices(obj) -> np.ndarray:
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.int64)
        return arr

    if isinstance(obj, dict):
        for key in ["indices", "idx", "index"]:
            if key in obj:
                return np.asarray(obj[key], dtype=np.int64)

    raise TypeError("Unsupported split index format.")


def _load_split_indices(split_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_data = _load_json(split_path)

    train_keys = ["train", "train_idx", "train_indices"]
    val_keys = ["val", "valid", "validation", "val_idx", "val_indices"]
    test_keys = ["test", "test_idx", "test_indices"]

    def _find_key(candidates: List[str]) -> str:
        for k in candidates:
            if k in split_data:
                return k
        raise KeyError(f"Cannot find any of keys: {candidates}")

    train_key = _find_key(train_keys)
    val_key = _find_key(val_keys)
    test_key = _find_key(test_keys)

    train_indices = _normalize_split_indices(split_data[train_key])
    val_indices = _normalize_split_indices(split_data[val_key])
    test_indices = _normalize_split_indices(split_data[test_key])

    return train_indices, val_indices, test_indices


def _build_year_ids(num_samples: int, num_steps: int, start_year: int = 2000) -> np.ndarray:
    year_row = np.arange(start_year, start_year + num_steps, dtype=np.int64)
    return np.tile(year_row[None, :], (num_samples, 1))


def _build_recency_ids(num_samples: int, num_steps: int) -> np.ndarray:
    recency_row = np.arange(num_steps, dtype=np.int64)
    return np.tile(recency_row[None, :], (num_samples, 1))


def _build_valid_len_from_m_year(m_year: np.ndarray) -> np.ndarray:
    """
    Build valid lengths from yearly mask.

    Expected m_year:
        shape [N, T]
        non-zero means valid year
    """
    if m_year.ndim != 2:
        raise ValueError(f"m_year must be 2D [N,T], got shape={m_year.shape}")

    valid_len = (m_year > 0).sum(axis=1).astype(np.int64)
    valid_len = np.clip(valid_len, 1, m_year.shape[1])
    return valid_len


def _assert_same_num_samples(arrays: Dict[str, np.ndarray]) -> int:
    sizes = {name: arr.shape[0] for name, arr in arrays.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(f"Inconsistent sample counts: {sizes}")
    return next(iter(unique_sizes))


def _assert_input_shapes(
    x_stock: np.ndarray,
    x_flow: np.ndarray,
    x_joint: np.ndarray,
    labels: np.ndarray,
    m_year: np.ndarray,
) -> None:
    if x_stock.ndim != 3:
        raise ValueError(f"x_stock must be [N,T,D], got {x_stock.shape}")
    if x_flow.ndim != 3:
        raise ValueError(f"x_flow must be [N,T,D], got {x_flow.shape}")
    if x_joint.ndim != 3:
        raise ValueError(f"x_joint must be [N,T,D], got {x_joint.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be [N,F], got {labels.shape}")
    if m_year.ndim != 2:
        raise ValueError(f"m_year must be [N,T], got {m_year.shape}")

    n = _assert_same_num_samples({
        "x_stock": x_stock,
        "x_flow": x_flow,
        "x_joint": x_joint,
        "labels": labels,
        "m_year": m_year,
    })

    t_stock = x_stock.shape[1]
    t_flow = x_flow.shape[1]
    t_joint = x_joint.shape[1]
    t_mask = m_year.shape[1]

    if not (t_stock == t_flow == t_joint == t_mask):
        raise ValueError(
            "Inconsistent history lengths: "
            f"x_stock={x_stock.shape}, x_flow={x_flow.shape}, "
            f"x_joint={x_joint.shape}, m_year={m_year.shape}"
        )

    if labels.shape[1] != 6:
        raise ValueError(f"labels second dim must be 6, got {labels.shape}")

    if np.any(labels < 0) or np.any(labels > 3):
        bad_min = int(labels.min())
        bad_max = int(labels.max())
        raise ValueError(
            f"labels must be in [0,3], got min={bad_min}, max={bad_max}"
        )

    if n <= 0:
        raise ValueError("Empty dataset is not allowed.")


def _assert_split_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    num_samples: int,
) -> None:
    for name, arr in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        if arr.ndim != 1:
            raise ValueError(f"{name}_indices must be 1D, got shape={arr.shape}")
        if arr.dtype.kind not in {"i", "u"}:
            raise TypeError(f"{name}_indices must be integer array, got dtype={arr.dtype}")
        if len(arr) == 0:
            raise ValueError(f"{name}_indices is empty.")
        if arr.min() < 0 or arr.max() >= num_samples:
            raise IndexError(
                f"{name}_indices out of range: min={arr.min()}, max={arr.max()}, N={num_samples}"
            )

    inter_train_val = np.intersect1d(train_indices, val_indices)
    inter_train_test = np.intersect1d(train_indices, test_indices)
    inter_val_test = np.intersect1d(val_indices, test_indices)

    if len(inter_train_val) > 0 or len(inter_train_test) > 0 or len(inter_val_test) > 0:
        raise ValueError(
            "Split indices overlap detected: "
            f"train∩val={len(inter_train_val)}, "
            f"train∩test={len(inter_train_test)}, "
            f"val∩test={len(inter_val_test)}"
        )


def _save_processed_arrays(
    cfg: HARPPathConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    valid_len: np.ndarray,
    year_ids: np.ndarray,
    recency_ids: np.ndarray,
) -> None:
    Path(cfg.processed_predictor_inputs_root).mkdir(parents=True, exist_ok=True)
    np.save(cfg.train_indices_path, train_indices)
    np.save(cfg.val_indices_path, val_indices)
    np.save(cfg.test_indices_path, test_indices)
    np.save(cfg.valid_len_path, valid_len)
    np.save(cfg.year_ids_path, year_ids)
    np.save(cfg.recency_ids_path, recency_ids)


def _save_split_summary(
    cfg: HARPPathConfig,
    num_samples: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    valid_len: np.ndarray,
    x_stock: np.ndarray,
    x_flow: np.ndarray,
    x_joint: np.ndarray,
    labels: np.ndarray,
) -> None:
    summary = {
        "dataset": cfg.dataset,
        "model_name": cfg.model_name,
        "num_samples": int(num_samples),
        "history_steps": int(x_stock.shape[1]),
        "future_steps": int(labels.shape[1]),
        "stock_dim": int(x_stock.shape[2]),
        "flow_dim": int(x_flow.shape[2]),
        "joint_dim": int(x_joint.shape[2]),
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "test_size": int(len(test_indices)),
        "valid_len_min": int(valid_len.min()),
        "valid_len_max": int(valid_len.max()),
        "valid_len_mean": float(valid_len.mean()),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
    }

    Path(cfg.cache_stats_root).mkdir(parents=True, exist_ok=True)
    with open(cfg.split_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_harp_data_bundle(cfg: HARPPathConfig) -> HARPDataBundle:
    ensure_required_dirs(cfg)
    dump_path_manifest(cfg)
    assert_required_input_files(cfg)

    x_stock = np.load(cfg.x_stock_path)
    x_flow = np.load(cfg.x_flow_path)
    x_joint = np.load(cfg.x_joint_path)
    labels = np.load(cfg.cls_label_path)
    m_year = np.load(cfg.m_year_path)

    _assert_input_shapes(
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        labels=labels,
        m_year=m_year,
    )

    num_samples = x_stock.shape[0]
    num_steps = x_stock.shape[1]

    author_ids = _load_author_ids(cfg.author_ids_path)
    if len(author_ids) != num_samples:
        raise ValueError(
            f"author_ids length mismatch: len(author_ids)={len(author_ids)}, N={num_samples}"
        )

    valid_len = _build_valid_len_from_m_year(m_year)
    year_ids = _build_year_ids(num_samples=num_samples, num_steps=num_steps, start_year=2000)
    recency_ids = _build_recency_ids(num_samples=num_samples, num_steps=num_steps)

    train_indices, val_indices, test_indices = _load_split_indices(cfg.split_path)
    _assert_split_indices(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        num_samples=num_samples,
    )

    _save_processed_arrays(
        cfg=cfg,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        valid_len=valid_len,
        year_ids=year_ids,
        recency_ids=recency_ids,
    )

    _save_split_summary(
        cfg=cfg,
        num_samples=num_samples,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        valid_len=valid_len,
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        labels=labels,
    )

    return HARPDataBundle(
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        labels=labels,
        m_year=m_year,
        valid_len=valid_len,
        year_ids=year_ids,
        recency_ids=recency_ids,
        author_ids=author_ids,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


class HARPPredictorDataset(Dataset):
    """
    Dataset for HARP scholar impact classification.

    Each sample contains:
        - three aligned historical feature streams
        - yearly validity mask and valid sequence length
        - fixed year ids and recency ids
        - 6-year future four-class labels
        - author id
        - global index
    """

    def __init__(
        self,
        cfg: HARPPathConfig,
        split: SplitName,
        preload_bundle: Optional[HARPDataBundle] = None,
        return_numpy: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.return_numpy = return_numpy

        self.bundle = preload_bundle if preload_bundle is not None else load_harp_data_bundle(cfg)

        if split == "train":
            self.indices = self.bundle.train_indices
        elif split == "val":
            self.indices = self.bundle.val_indices
        else:
            self.indices = self.bundle.test_indices

        self.num_history_steps = int(self.bundle.x_stock.shape[1])
        self.num_future_steps = int(self.bundle.labels.shape[1])
        self.stock_dim = int(self.bundle.x_stock.shape[2])
        self.flow_dim = int(self.bundle.x_flow.shape[2])
        self.joint_dim = int(self.bundle.x_joint.shape[2])

    def __len__(self) -> int:
        return int(len(self.indices))

    def _to_tensor_or_numpy(self, x, dtype=None):
        if self.return_numpy:
            if dtype is not None:
                return x.astype(dtype, copy=False)
            return x
        return torch.as_tensor(x, dtype=dtype)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        global_idx = int(self.indices[idx])

        x_stock = self.bundle.x_stock[global_idx]
        x_flow = self.bundle.x_flow[global_idx]
        x_joint = self.bundle.x_joint[global_idx]
        labels = self.bundle.labels[global_idx]
        m_year = self.bundle.m_year[global_idx]
        valid_len = self.bundle.valid_len[global_idx]
        year_ids = self.bundle.year_ids[global_idx]
        recency_ids = self.bundle.recency_ids[global_idx]
        author_id = self.bundle.author_ids[global_idx]

        sample = {
            "x_stock": self._to_tensor_or_numpy(x_stock, dtype=torch.float32 if not self.return_numpy else np.float32),
            "x_flow": self._to_tensor_or_numpy(x_flow, dtype=torch.float32 if not self.return_numpy else np.float32),
            "x_joint": self._to_tensor_or_numpy(x_joint, dtype=torch.float32 if not self.return_numpy else np.float32),
            "labels": self._to_tensor_or_numpy(labels, dtype=torch.long if not self.return_numpy else np.int64),
            "m_year": self._to_tensor_or_numpy(m_year, dtype=torch.float32 if not self.return_numpy else np.float32),
            "valid_len": int(valid_len),
            "year_ids": self._to_tensor_or_numpy(year_ids, dtype=torch.long if not self.return_numpy else np.int64),
            "recency_ids": self._to_tensor_or_numpy(recency_ids, dtype=torch.long if not self.return_numpy else np.int64),
            "author_id": str(author_id),
            "global_idx": global_idx,
        }
        return sample

    def get_split_indices(self) -> np.ndarray:
        return self.indices.copy()

    def get_meta(self) -> Dict[str, object]:
        return {
            "dataset": self.cfg.dataset,
            "split": self.split,
            "num_samples": len(self),
            "num_history_steps": self.num_history_steps,
            "num_future_steps": self.num_future_steps,
            "stock_dim": self.stock_dim,
            "flow_dim": self.flow_dim,
            "joint_dim": self.joint_dim,
        }


def build_harp_datasets(
    dataset: str,
    model_name: str = "HARP",
    project_root: str = "/root/autodl-tmp/WH2/HARP",
    source_feature_root: str = "/root/autodl-tmp/WH2/my_method_v3",
    data_root: str = "/root/autodl-tmp/WH2/data",
    result_root: str = "/root/autodl-tmp/WH2/results",
    return_numpy: bool = False,
) -> Tuple[HARPPathConfig, HARPPredictorDataset, HARPPredictorDataset, HARPPredictorDataset]:
    cfg = build_harp_path_config(
        dataset=dataset,
        model_name=model_name,
        project_root=project_root,
        source_feature_root=source_feature_root,
        data_root=data_root,
        result_root=result_root,
    )

    bundle = load_harp_data_bundle(cfg)

    train_dataset = HARPPredictorDataset(
        cfg=cfg,
        split="train",
        preload_bundle=bundle,
        return_numpy=return_numpy,
    )
    val_dataset = HARPPredictorDataset(
        cfg=cfg,
        split="val",
        preload_bundle=bundle,
        return_numpy=return_numpy,
    )
    test_dataset = HARPPredictorDataset(
        cfg=cfg,
        split="test",
        preload_bundle=bundle,
        return_numpy=return_numpy,
    )

    return cfg, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    cfg, train_set, val_set, test_set = build_harp_datasets(
        dataset="aps",
        model_name="HARP",
    )

    print("[INFO] HARP datasets built successfully.")
    print("[INFO] train meta:", train_set.get_meta())
    print("[INFO] val meta:", val_set.get_meta())
    print("[INFO] test meta:", test_set.get_meta())

    sample = train_set[0]
    print("[INFO] sample keys:", list(sample.keys()))
    print("[INFO] x_stock shape:", tuple(sample["x_stock"].shape))
    print("[INFO] x_flow shape:", tuple(sample["x_flow"].shape))
    print("[INFO] x_joint shape:", tuple(sample["x_joint"].shape))
    print("[INFO] labels shape:", tuple(sample["labels"].shape))