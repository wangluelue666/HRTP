#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Support both:
#   1) python -m scripts.01_build_class_stats
#   2) python scripts/01_build_class_stats.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.path_config import (
    build_harp_path_config,
    ensure_required_dirs,
    dump_path_manifest,
    assert_required_input_files,
)
from dataset.predictor_dataset import load_harp_data_bundle


VALID_DATASETS = {"acm", "aps", "dblp"}
CLASS_NAMES = ["LL", "LH", "HL", "HH"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build class statistics for HARP training."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=sorted(VALID_DATASETS),
        help="Dataset name.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HARP",
        help="Model name used in path config.",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/root/autodl-tmp/WH2/HARP",
        help="HARP project root.",
    )
    parser.add_argument(
        "--source_feature_root",
        type=str,
        default="/root/autodl-tmp/WH2/my_method_v3",
        help="Read-only source feature root.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/WH2/data",
        help="Shared data root.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="/root/autodl-tmp/WH2/results",
        help="Shared result root.",
    )
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="inverse_sqrt",
        choices=["inverse", "inverse_sqrt", "effective_num"],
        help="How to build class weights from class counts.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9999,
        help="Beta used in effective number weighting.",
    )
    parser.add_argument(
        "--year_weight_mode",
        type=str,
        default="linear",
        choices=["uniform", "linear", "custom"],
        help="How to build year weights.",
    )
    parser.add_argument(
        "--year_weight_custom",
        type=float,
        nargs=6,
        default=None,
        help="Custom 6-step year weights when year_weight_mode=custom.",
    )
    parser.add_argument(
        "--normalize_class_weight",
        action="store_true",
        help="Normalize class weights to mean=1.",
    )
    parser.add_argument(
        "--normalize_year_weight",
        action="store_true",
        help="Normalize year weights to mean=1.",
    )
    return parser.parse_args()


def _validate_labels(labels: np.ndarray) -> None:
    if labels.ndim != 2:
        raise ValueError(f"labels must be [N,F], got shape={labels.shape}")
    if labels.shape[1] != 6:
        raise ValueError(f"labels second dim must be 6, got shape={labels.shape}")
    if np.any(labels < 0) or np.any(labels > 3):
        raise ValueError(
            f"labels must be in [0,3], got min={labels.min()}, max={labels.max()}"
        )


def _count_global_classes(labels: np.ndarray) -> np.ndarray:
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=4).astype(np.int64)
    return counts


def _count_yearwise_classes(labels: np.ndarray) -> np.ndarray:
    """
    Args:
        labels: [N, 6]

    Returns:
        year_counts: [6, 4]
    """
    future_steps = labels.shape[1]
    year_counts = []
    for t in range(future_steps):
        c = np.bincount(labels[:, t], minlength=4).astype(np.int64)
        year_counts.append(c)
    return np.stack(year_counts, axis=0)


def _safe_ratio(numer: np.ndarray, denom: np.ndarray | float) -> np.ndarray:
    denom_arr = np.asarray(denom, dtype=np.float64)
    return numer.astype(np.float64) / np.maximum(denom_arr, 1e-12)


def build_class_weight(
    class_counts: np.ndarray,
    mode: str = "inverse_sqrt",
    beta: float = 0.9999,
    normalize_to_mean_one: bool = False,
) -> np.ndarray:
    """
    Build class weights from total class counts.

    Supported modes:
        - inverse:      1 / count
        - inverse_sqrt: 1 / sqrt(count)
        - effective_num: (1 - beta) / (1 - beta^count)
    """
    if class_counts.shape != (4,):
        raise ValueError(f"class_counts must be shape (4,), got {class_counts.shape}")

    counts = class_counts.astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if mode == "inverse":
        weight = 1.0 / counts
    elif mode == "inverse_sqrt":
        weight = 1.0 / np.sqrt(counts)
    elif mode == "effective_num":
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0,1), got {beta}")
        effective_num = 1.0 - np.power(beta, counts)
        weight = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    if normalize_to_mean_one:
        weight = weight / np.maximum(weight.mean(), 1e-12)

    return weight.astype(np.float32)


def build_year_weight(
    future_steps: int = 6,
    mode: str = "linear",
    custom_weights: List[float] | None = None,
    normalize_to_mean_one: bool = False,
) -> np.ndarray:
    """
    Build year weights for future steps.

    Modes:
        - uniform: all 1.0
        - linear : progressively upweight later years
        - custom : use provided values
    """
    if future_steps != 6:
        raise ValueError(f"Expected future_steps=6, got {future_steps}")

    if mode == "uniform":
        weight = np.ones((future_steps,), dtype=np.float32)
    elif mode == "linear":
        weight = np.asarray([0.85, 0.90, 1.00, 1.10, 1.20, 1.35], dtype=np.float32)
    elif mode == "custom":
        if custom_weights is None or len(custom_weights) != future_steps:
            raise ValueError(
                f"custom year weights must have length {future_steps}, got {custom_weights}"
            )
        weight = np.asarray(custom_weights, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported year weight mode: {mode}")

    if normalize_to_mean_one:
        weight = weight / max(float(weight.mean()), 1e-12)

    return weight.astype(np.float32)


def summarize_class_stats(
    labels_train: np.ndarray,
    class_weight: np.ndarray,
    year_weight: np.ndarray,
    class_weight_mode: str,
    year_weight_mode: str,
    beta: float,
) -> Dict:
    _validate_labels(labels_train)

    total_counts = _count_global_classes(labels_train)          # [4]
    total_ratio = _safe_ratio(total_counts, total_counts.sum()) # [4]

    year_counts = _count_yearwise_classes(labels_train)         # [6,4]
    year_ratio = _safe_ratio(year_counts, year_counts.sum(axis=1, keepdims=True))

    stats = {
        "num_train_samples": int(labels_train.shape[0]),
        "future_steps": int(labels_train.shape[1]),
        "class_names": CLASS_NAMES,

        "global_class_counts": total_counts.tolist(),
        "global_class_ratio": total_ratio.tolist(),

        "yearwise_class_counts": year_counts.tolist(),
        "yearwise_class_ratio": year_ratio.tolist(),

        "class_weight_mode": class_weight_mode,
        "year_weight_mode": year_weight_mode,
        "beta": float(beta),

        "class_weight": class_weight.tolist(),
        "year_weight": year_weight.tolist(),
    }
    return stats


def save_json(obj: Dict, path: str) -> None:
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    with path_p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    cfg = build_harp_path_config(
        dataset=args.dataset,
        model_name=args.model_name,
        project_root=args.project_root,
        source_feature_root=args.source_feature_root,
        data_root=args.data_root,
        result_root=args.result_root,
    )

    ensure_required_dirs(cfg)
    dump_path_manifest(cfg)
    assert_required_input_files(cfg)

    bundle = load_harp_data_bundle(cfg)

    train_indices = bundle.train_indices
    labels_train = bundle.labels[train_indices]
    _validate_labels(labels_train)

    class_counts = _count_global_classes(labels_train)
    class_weight = build_class_weight(
        class_counts=class_counts,
        mode=args.class_weight_mode,
        beta=args.beta,
        normalize_to_mean_one=args.normalize_class_weight,
    )

    year_weight = build_year_weight(
        future_steps=labels_train.shape[1],
        mode=args.year_weight_mode,
        custom_weights=args.year_weight_custom,
        normalize_to_mean_one=args.normalize_year_weight,
    )

    stats = summarize_class_stats(
        labels_train=labels_train,
        class_weight=class_weight,
        year_weight=year_weight,
        class_weight_mode=args.class_weight_mode,
        year_weight_mode=args.year_weight_mode,
        beta=args.beta,
    )

    save_json(stats, cfg.class_stats_train_path)

    print(f"[INFO] dataset={cfg.dataset}")
    print(f"[INFO] train_size={len(train_indices)}")
    print(f"[INFO] saved class stats to: {cfg.class_stats_train_path}")
    print(f"[INFO] global_class_counts={stats['global_class_counts']}")
    print(f"[INFO] class_weight={stats['class_weight']}")
    print(f"[INFO] year_weight={stats['year_weight']}")


if __name__ == "__main__":
    main()