#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Support both:
#   1) python -m scripts.06_feature_ablation_harp_train
#   2) python scripts/06_feature_ablation_harp_train.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.path_config import build_harp_path_config
from losses import HARPLoss
from models.predictor import HARPPredictor


VALID_DATASETS = {"acm", "aps", "dblp"}
VALID_FEATURE_MODELS = {"gat", "han", "hgnn", "hgnnp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HARP for feature ablation baselines."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=sorted(VALID_DATASETS))
    parser.add_argument("--feature_model", type=str, required=True, choices=sorted(VALID_FEATURE_MODELS))

    # Reuse original HARP path system
    parser.add_argument("--model_name", type=str, default="HARP")
    parser.add_argument("--project_root", type=str, default="/root/autodl-tmp/WH2/HARP")
    parser.add_argument("--source_feature_root", type=str, default="/root/autodl-tmp/WH2/my_method_v3")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/WH2/data")

    # New roots for feature ablation
    parser.add_argument("--feature_root", type=str, default="/root/autodl-tmp/WH2/Feature_ablation/features")
    parser.add_argument("--result_root", type=str, default="/root/autodl-tmp/WH2/results/Feature_ablation")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--fusion_num_heads", type=int, default=4)
    parser.add_argument("--temporal_refine_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)

    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--aux_stock_weight", type=float, default=0.15)
    parser.add_argument("--aux_flow_weight", type=float, default=0.15)
    parser.add_argument("--smoothness_weight", type=float, default=0.02)

    parser.add_argument("--scheduled_sampling_start", type=float, default=1.0)
    parser.add_argument("--scheduled_sampling_end", type=float, default=0.2)
    parser.add_argument(
        "--tf_start_mode",
        type=str,
        default="uniform",
        choices=["uniform", "zeros", "first_label"],
    )
    parser.add_argument("--tf_smoothing", type=float, default=0.0)

    parser.add_argument(
        "--eval_rollout_mode",
        type=str,
        default="pure",
        choices=["pure", "gt"],
        help="pure: pure autoregressive rollout at evaluation; gt: use gt-shifted prev class sequence.",
    )

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_train_val_test_predictions", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(device_str)
    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _cfg_items(cfg) -> Dict[str, Any]:
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return dict(cfg)


def find_existing_cfg_path(
    cfg,
    exact_names: List[str] | None = None,
    contains_all: List[str] | None = None,
    suffixes: Tuple[str, ...] | None = None,
) -> Path | None:
    items = _cfg_items(cfg)
    exact_names = exact_names or []
    contains_all = contains_all or []

    for name in exact_names:
        if name in items:
            p = Path(str(items[name]))
            if p.exists():
                return p

    for k, v in items.items():
        if not isinstance(v, (str, Path)):
            continue
        lk = k.lower()
        if contains_all and not all(token in lk for token in contains_all):
            continue
        p = Path(str(v))
        if suffixes is not None and p.suffix not in suffixes:
            continue
        if p.exists():
            return p
    return None


def resolve_harp_reference_paths(args: argparse.Namespace):
    """
    Reuse original HARP path system for:
    - labels
    - split
    - canonical author_ids
    - class stats
    """
    cfg = build_harp_path_config(
        dataset=args.dataset,
        model_name=args.model_name,
        project_root=args.project_root,
        source_feature_root=args.source_feature_root,
        data_root=args.data_root,
        result_root=args.result_root,
    )

    label_path = find_existing_cfg_path(
        cfg,
        exact_names=[
            "cls_label_path",
            "label_path",
            "labels_path",
            "target_cls_path",
            "targets_path",
        ],
        contains_all=["label"],
        suffixes=(".npy", ".npz"),
    )
    if label_path is None:
        label_path = find_existing_cfg_path(
            cfg,
            contains_all=["cls"],
            suffixes=(".npy", ".npz"),
        )

    split_path = find_existing_cfg_path(
        cfg,
        exact_names=["split_path", "split_json_path"],
        contains_all=["split"],
        suffixes=(".json",),
    )

    author_ids_path = find_existing_cfg_path(
        cfg,
        exact_names=[
            "author_ids_path",
            "history_author_ids_path",
            "author_id_path",
        ],
        contains_all=["author", "ids"],
        suffixes=(".json",),
    )

    class_stats_path = find_existing_cfg_path(
        cfg,
        exact_names=["class_stats_train_path"],
        contains_all=["class", "stats"],
        suffixes=(".json",),
    )

    if label_path is None:
        raise FileNotFoundError(
            f"Cannot resolve label path from original HARP path config for dataset={args.dataset}. "
            f"cfg keys={list(_cfg_items(cfg).keys())}"
        )
    if split_path is None:
        raise FileNotFoundError(
            f"Cannot resolve split path from original HARP path config for dataset={args.dataset}. "
            f"cfg keys={list(_cfg_items(cfg).keys())}"
        )

    return cfg, label_path, split_path, author_ids_path, class_stats_path


def load_split_indices(path: Path) -> Dict[str, np.ndarray]:
    obj = load_json(path)

    key_map = {
        "train": ["train", "train_idx", "train_indices"],
        "val": ["val", "valid", "val_idx", "valid_idx", "val_indices", "valid_indices"],
        "test": ["test", "test_idx", "test_indices"],
    }

    out = {}
    for target_key, candidate_keys in key_map.items():
        found = None
        for k in candidate_keys:
            if k in obj:
                found = np.asarray(obj[k], dtype=np.int64)
                break
        if found is None:
            raise KeyError(f"Split file missing key for '{target_key}'. Available keys: {list(obj.keys())}")
        out[target_key] = found
    return out


def maybe_align_features_to_canonical_author_order(
    x_stock: np.ndarray,
    x_flow: np.ndarray,
    x_joint: np.ndarray,
    feature_author_ids: List[str],
    canonical_author_ids: List[str] | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if canonical_author_ids is None:
        return x_stock, x_flow, x_joint, feature_author_ids

    if list(feature_author_ids) == list(canonical_author_ids):
        return x_stock, x_flow, x_joint, canonical_author_ids

    feat_id2idx = {aid: i for i, aid in enumerate(feature_author_ids)}
    reordered_stock = np.zeros((len(canonical_author_ids),) + x_stock.shape[1:], dtype=x_stock.dtype)
    reordered_flow = np.zeros((len(canonical_author_ids),) + x_flow.shape[1:], dtype=x_flow.dtype)
    reordered_joint = np.zeros((len(canonical_author_ids),) + x_joint.shape[1:], dtype=x_joint.dtype)

    matched = 0
    for i, aid in enumerate(canonical_author_ids):
        j = feat_id2idx.get(aid)
        if j is None:
            continue
        reordered_stock[i] = x_stock[j]
        reordered_flow[i] = x_flow[j]
        reordered_joint[i] = x_joint[j]
        matched += 1

    print(f"[INFO] Reordered features to canonical author order. matched={matched}/{len(canonical_author_ids)}")
    return reordered_stock, reordered_flow, reordered_joint, canonical_author_ids


def load_feature_author_ids(feature_dir: Path, dataset: str) -> List[str]:
    direct = feature_dir / f"author_ids_{dataset}.json"
    if direct.exists():
        return load_json(direct)

    cache_fallback = Path("/root/autodl-tmp/WH2/Feature_ablation/cache") / dataset / "basic_cache.pkl"
    if cache_fallback.exists():
        cache = load_pickle(cache_fallback)
        print(f"[WARN] {direct} not found. Fallback to {cache_fallback}::author_ids_history")
        return cache["author_ids_history"]

    raise FileNotFoundError(
        f"Missing feature author ids file: {direct}; "
        f"fallback cache not found: {cache_fallback}"
    )


def load_feature_bundle(
    args: argparse.Namespace,
    label_path: Path,
    split_path: Path,
    canonical_author_ids_path: Path | None,
    class_stats_path: Path | None,
) -> Dict[str, Any]:
    feature_dir = Path(args.feature_root) / args.feature_model / args.dataset
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    x_stock_path = feature_dir / "X_stock.npy"
    x_flow_path = feature_dir / "X_flow.npy"
    x_joint_path = feature_dir / "X_joint.npy"

    for p in [x_stock_path, x_flow_path, x_joint_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing feature file: {p}")

    x_stock = np.load(x_stock_path).astype(np.float32)
    x_flow = np.load(x_flow_path).astype(np.float32)
    x_joint = np.load(x_joint_path).astype(np.float32)

    feature_author_ids = load_feature_author_ids(feature_dir, args.dataset)
    canonical_author_ids = load_json(canonical_author_ids_path) if canonical_author_ids_path is not None else None

    x_stock, x_flow, x_joint, author_ids = maybe_align_features_to_canonical_author_order(
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        feature_author_ids=feature_author_ids,
        canonical_author_ids=canonical_author_ids,
    )

    labels = np.load(label_path).astype(np.int64)
    split_idx = load_split_indices(split_path)
    class_stats = load_json(class_stats_path) if class_stats_path is not None and class_stats_path.exists() else None

    if labels.shape[0] != x_stock.shape[0]:
        raise ValueError(
            f"Feature/label size mismatch: x_stock={x_stock.shape}, labels={labels.shape}, "
            f"dataset={args.dataset}, feature_model={args.feature_model}"
        )
    if labels.shape[1] != 6:
        raise ValueError(f"Expected labels shape (N, 6), got {labels.shape}")

    return {
        "feature_dir": str(feature_dir),
        "x_stock": x_stock,
        "x_flow": x_flow,
        "x_joint": x_joint,
        "labels": labels,
        "author_ids": author_ids,
        "split_idx": split_idx,
        "class_stats": class_stats,
        "label_path": str(label_path),
        "split_path": str(split_path),
        "canonical_author_ids_path": str(canonical_author_ids_path) if canonical_author_ids_path is not None else None,
        "class_stats_path": str(class_stats_path) if class_stats_path is not None else None,
    }


class FeatureAblationHARPDataset(Dataset):
    def __init__(
        self,
        x_stock: np.ndarray,
        x_flow: np.ndarray,
        x_joint: np.ndarray,
        labels: np.ndarray,
        author_ids: List[str],
        indices: np.ndarray,
        start_year: int = 2000,
    ) -> None:
        super().__init__()
        self.x_stock = x_stock
        self.x_flow = x_flow
        self.x_joint = x_joint
        self.labels = labels
        self.author_ids = author_ids
        self.indices = np.asarray(indices, dtype=np.int64)

        self.t_hist = int(self.x_stock.shape[1])
        self.start_year = int(start_year)

        self.year_ids = np.arange(self.start_year, self.start_year + self.t_hist, dtype=np.int64)
        self.recency_ids = np.arange(self.t_hist - 1, -1, -1, dtype=np.int64)
        self.mask = np.ones((self.t_hist,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        gidx = int(self.indices[i])
        y = self.labels[gidx].astype(np.int64)

        stock_targets = np.isin(y, [2, 3]).astype(np.float32)
        flow_targets = np.isin(y, [1, 3]).astype(np.float32)

        return {
            "global_idx": np.asarray(gidx, dtype=np.int64),
            "author_id": self.author_ids[gidx],
            "x_stock": self.x_stock[gidx].astype(np.float32),
            "x_flow": self.x_flow[gidx].astype(np.float32),
            "x_joint": self.x_joint[gidx].astype(np.float32),
            "mask": self.mask.copy(),
            "year_ids": self.year_ids.copy(),
            "recency_ids": self.recency_ids.copy(),
            "labels": y,
            "stock_targets": stock_targets,
            "flow_targets": flow_targets,
        }


def harp_feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tensor_keys_float = ["x_stock", "x_flow", "x_joint", "mask", "stock_targets", "flow_targets"]
    tensor_keys_long = ["year_ids", "recency_ids", "labels", "global_idx"]

    for k in tensor_keys_float:
        out[k] = torch.tensor(np.stack([b[k] for b in batch], axis=0), dtype=torch.float32)

    for k in tensor_keys_long:
        out[k] = torch.tensor(np.stack([b[k] for b in batch], axis=0), dtype=torch.long)

    out["author_id"] = [b["author_id"] for b in batch]
    return out


def build_model_from_sample(batch: Dict[str, Any], args: argparse.Namespace) -> HARPPredictor:
    d_stock = int(batch["x_stock"].shape[-1])
    d_flow = int(batch["x_flow"].shape[-1])
    d_joint = int(batch["x_joint"].shape[-1])
    t_hist = int(batch["x_stock"].shape[1])
    t_future = int(batch["labels"].shape[1])

    model = HARPPredictor(
        input_dim_stock=d_stock,
        input_dim_flow=d_flow,
        input_dim_joint=d_joint,
        d_model=args.d_model,
        num_classes=4,
        num_history_steps=t_hist,
        num_future_steps=t_future,
        encoder_layers=args.encoder_layers,
        num_heads=args.num_heads,
        fusion_num_heads=args.fusion_num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        fusion_dropout=args.fusion_dropout,
        use_year_embedding=True,
        use_recency_embedding=True,
        max_year_tokens=64,
        max_recency_tokens=64,
        base_year=2000,
        future_year_start=2015,
        temporal_refine_layers=args.temporal_refine_layers,
        pooling_hidden_dim=args.d_model,
        use_selective_gate=True,
        use_pre_norm=True,
    )
    return model


def build_loss_from_stats(
    args: argparse.Namespace,
    stats: Dict[str, Any] | None,
    device: torch.device,
) -> HARPLoss:
    class_weight = None
    year_weight = None

    if stats is not None:
        if "class_weight" in stats:
            class_weight = torch.tensor(stats["class_weight"], dtype=torch.float32, device=device)
        if "year_weight" in stats:
            year_weight = torch.tensor(stats["year_weight"], dtype=torch.float32, device=device)

    return HARPLoss(
        focal_gamma=args.focal_gamma,
        class_weight=class_weight,
        year_weight=year_weight,
        label_smoothing=args.label_smoothing,
        aux_stock_weight=args.aux_stock_weight,
        aux_flow_weight=args.aux_flow_weight,
        smoothness_weight=args.smoothness_weight,
        smoothness_p=1,
    )


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )


def compute_teacher_forcing_ratio(epoch_idx: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return float(end)
    progress = epoch_idx / float(total_epochs - 1)
    ratio = start + (end - start) * progress
    return float(max(0.0, min(1.0, ratio)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(yt, yp)),
    }


def compute_yearwise_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    future_steps = y_true.shape[1]
    out = {}
    for t in range(future_steps):
        out[f"year_{2015 + t}_macro_f1"] = float(
            f1_score(y_true[:, t], y_pred[:, t], average="macro", zero_division=0)
        )
    return out


def run_forward_with_prev(
    model: HARPPredictor,
    batch: Dict[str, Any],
    prev_class_probs_override: torch.Tensor | None,
) -> Dict[str, Any]:
    return model(
        x_stock=batch["x_stock"],
        x_flow=batch["x_flow"],
        x_joint=batch["x_joint"],
        mask=batch["mask"],
        year_ids=batch["year_ids"],
        recency_ids=batch["recency_ids"],
        prev_class_probs_override=prev_class_probs_override,
        return_analysis=False,
        return_intermediates=False,
    )


@torch.no_grad()
def build_mixed_prev_probs(
    model: HARPPredictor,
    batch: Dict[str, Any],
    teacher_forcing_ratio: float,
    tf_start_mode: str,
    tf_smoothing: float,
) -> torch.Tensor:
    gt_prev = model.build_prev_class_probs_from_labels(
        labels=batch["labels"],
        start_mode=tf_start_mode,
        smoothing=tf_smoothing,
    ).to(batch["labels"].device)

    warm_outputs = run_forward_with_prev(
        model=model,
        batch=batch,
        prev_class_probs_override=None,
    )
    pred_prev = model.build_prev_class_probs_from_prediction(
        class_probs=warm_outputs["class_probs"],
        start_mode=tf_start_mode,
    ).to(batch["labels"].device)

    mixed_prev = model.mix_prev_class_probs(
        gt_prev_probs=gt_prev,
        pred_prev_probs=pred_prev,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )
    return mixed_prev


def train_one_epoch(
    model: HARPPredictor,
    loader: DataLoader,
    criterion: HARPLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()

    tf_ratio = compute_teacher_forcing_ratio(
        epoch_idx=epoch_idx,
        total_epochs=total_epochs,
        start=args.scheduled_sampling_start,
        end=args.scheduled_sampling_end,
    )

    total_loss = 0.0
    total_cls = 0.0
    total_stock = 0.0
    total_flow = 0.0
    total_smooth = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        mixed_prev = build_mixed_prev_probs(
            model=model,
            batch=batch,
            teacher_forcing_ratio=tf_ratio,
            tf_start_mode=args.tf_start_mode,
            tf_smoothing=args.tf_smoothing,
        )

        outputs = run_forward_with_prev(
            model=model,
            batch=batch,
            prev_class_probs_override=mixed_prev,
        )

        loss_dict = criterion(
            class_logits=outputs["class_logits"],
            class_probs=outputs["class_probs"],
            labels=batch["labels"],
            stock_probs=outputs["stock_probs"],
            flow_probs=outputs["flow_probs"],
            stock_targets=batch["stock_targets"],
            flow_targets=batch["flow_targets"],
        )

        if not torch.isfinite(loss_dict["loss"]):
            print("[WARN] Non-finite loss detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        total_loss += float(loss_dict["loss"].detach().cpu().item())
        total_cls += float(loss_dict["loss_cls"].detach().cpu().item())
        total_stock += float(loss_dict["loss_stock"].detach().cpu().item())
        total_flow += float(loss_dict["loss_flow"].detach().cpu().item())
        total_smooth += float(loss_dict["loss_smooth"].detach().cpu().item())
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "loss_cls": total_cls / max(total_batches, 1),
        "loss_stock": total_stock / max(total_batches, 1),
        "loss_flow": total_flow / max(total_batches, 1),
        "loss_smooth": total_smooth / max(total_batches, 1),
        "teacher_forcing_ratio": tf_ratio,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: HARPPredictor,
    loader: DataLoader,
    criterion: HARPLoss,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_stock = 0.0
    total_flow = 0.0
    total_smooth = 0.0
    total_batches = 0

    all_true = []
    all_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if args.eval_rollout_mode == "gt":
            prev_override = model.build_prev_class_probs_from_labels(
                labels=batch["labels"],
                start_mode=args.tf_start_mode,
                smoothing=args.tf_smoothing,
            ).to(device)
        else:
            prev_override = None

        outputs = run_forward_with_prev(
            model=model,
            batch=batch,
            prev_class_probs_override=prev_override,
        )

        loss_dict = criterion(
            class_logits=outputs["class_logits"],
            class_probs=outputs["class_probs"],
            labels=batch["labels"],
            stock_probs=outputs["stock_probs"],
            flow_probs=outputs["flow_probs"],
            stock_targets=batch["stock_targets"],
            flow_targets=batch["flow_targets"],
        )

        total_loss += float(loss_dict["loss"].detach().cpu().item())
        total_cls += float(loss_dict["loss_cls"].detach().cpu().item())
        total_stock += float(loss_dict["loss_stock"].detach().cpu().item())
        total_flow += float(loss_dict["loss_flow"].detach().cpu().item())
        total_smooth += float(loss_dict["loss_smooth"].detach().cpu().item())
        total_batches += 1

        all_true.append(batch["labels"].detach().cpu().numpy())
        all_pred.append(outputs["class_pred"].detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics.update(compute_yearwise_macro_f1(y_true, y_pred))

    return {
        "loss": total_loss / max(total_batches, 1),
        "loss_cls": total_cls / max(total_batches, 1),
        "loss_stock": total_stock / max(total_batches, 1),
        "loss_flow": total_flow / max(total_batches, 1),
        "loss_smooth": total_smooth / max(total_batches, 1),
        **metrics,
    }


@torch.no_grad()
def collect_full_predictions(
    model: HARPPredictor,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    model.eval()

    preds = []
    labels = []
    global_idx = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if args.eval_rollout_mode == "gt":
            prev_override = model.build_prev_class_probs_from_labels(
                labels=batch["labels"],
                start_mode=args.tf_start_mode,
                smoothing=args.tf_smoothing,
            ).to(device)
        else:
            prev_override = None

        outputs = run_forward_with_prev(
            model=model,
            batch=batch,
            prev_class_probs_override=prev_override,
        )

        preds.append(outputs["class_pred"].detach().cpu().numpy().astype(np.uint8))
        labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
        global_idx.append(batch["global_idx"].detach().cpu().numpy().astype(np.int64))

    return {
        "pred": np.concatenate(preds, axis=0),
        "label": np.concatenate(labels, axis=0),
        "global_idx": np.concatenate(global_idx, axis=0),
    }


def save_predictions_and_reports(
    result_dir: Path,
    dataset: str,
    train_pack: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    test_pack: Dict[str, np.ndarray],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    np.save(result_dir / f"pred_train_{dataset}.npy", train_pack["pred"])
    np.save(result_dir / f"pred_val_{dataset}.npy", val_pack["pred"])
    np.save(result_dir / f"pred_test_{dataset}.npy", test_pack["pred"])

    np.save(result_dir / f"true_train_{dataset}.npy", train_pack["label"])
    np.save(result_dir / f"true_val_{dataset}.npy", val_pack["label"])
    np.save(result_dir / f"true_test_{dataset}.npy", test_pack["label"])

    np.save(result_dir / f"idx_train_{dataset}.npy", train_pack["global_idx"])
    np.save(result_dir / f"idx_val_{dataset}.npy", val_pack["global_idx"])
    np.save(result_dir / f"idx_test_{dataset}.npy", test_pack["global_idx"])

    save_json(
        {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        str(result_dir / f"train_summary_{dataset}.json"),
    )


def build_full_author_prediction(
    num_authors: int,
    train_pack: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    test_pack: Dict[str, np.ndarray],
) -> np.ndarray:
    full_pred = np.zeros((num_authors, 6), dtype=np.uint8)
    for pack in [train_pack, val_pack, test_pack]:
        full_pred[pack["global_idx"]] = pack["pred"]
    return full_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    cfg, label_path, split_path, canonical_author_ids_path, class_stats_path = resolve_harp_reference_paths(args)

    bundle = load_feature_bundle(
        args=args,
        label_path=label_path,
        split_path=split_path,
        canonical_author_ids_path=canonical_author_ids_path,
        class_stats_path=class_stats_path,
    )

    result_dir = Path(args.result_root) / args.dataset / args.feature_model
    result_dir.mkdir(parents=True, exist_ok=True)

    train_set = FeatureAblationHARPDataset(
        x_stock=bundle["x_stock"],
        x_flow=bundle["x_flow"],
        x_joint=bundle["x_joint"],
        labels=bundle["labels"],
        author_ids=bundle["author_ids"],
        indices=bundle["split_idx"]["train"],
    )
    val_set = FeatureAblationHARPDataset(
        x_stock=bundle["x_stock"],
        x_flow=bundle["x_flow"],
        x_joint=bundle["x_joint"],
        labels=bundle["labels"],
        author_ids=bundle["author_ids"],
        indices=bundle["split_idx"]["val"],
    )
    test_set = FeatureAblationHARPDataset(
        x_stock=bundle["x_stock"],
        x_flow=bundle["x_flow"],
        x_joint=bundle["x_joint"],
        labels=bundle["labels"],
        author_ids=bundle["author_ids"],
        indices=bundle["split_idx"]["test"],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_feature_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_feature_collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_feature_collate_fn,
        drop_last=False,
    )

    first_batch = next(iter(train_loader))
    first_batch = move_batch_to_device(first_batch, device)

    class_stats = bundle["class_stats"]
    model = build_model_from_sample(first_batch, args).to(device)
    criterion = build_loss_from_stats(args, class_stats, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer)

    config_snapshot = vars(args).copy()
    config_snapshot["feature_dir"] = bundle["feature_dir"]
    config_snapshot["label_path"] = bundle["label_path"]
    config_snapshot["split_path"] = bundle["split_path"]
    config_snapshot["canonical_author_ids_path"] = bundle["canonical_author_ids_path"]
    config_snapshot["class_stats_path"] = bundle["class_stats_path"]
    if class_stats is not None:
        config_snapshot["class_weight"] = class_stats.get("class_weight")
        config_snapshot["year_weight"] = class_stats.get("year_weight")
    save_json(config_snapshot, str(result_dir / "config_snapshot.json"))

    best_metric = -1.0
    best_epoch = -1
    best_state_dict = None
    history: List[Dict[str, Any]] = []
    patience_counter = 0

    print(f"[INFO] dataset={args.dataset} device={device}")
    print(f"[INFO] feature_model={args.feature_model}")
    print(f"[INFO] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[INFO] batch_size={args.batch_size} epochs={args.epochs}")
    print(
        f"[INFO] feature_dims: stock={bundle['x_stock'].shape[-1]} "
        f"flow={bundle['x_flow'].shape[-1]} joint={bundle['x_joint'].shape[-1]}"
    )
    print(f"[INFO] label_path={bundle['label_path']}")
    print(f"[INFO] split_path={bundle['split_path']}")
    if class_stats is not None:
        print(f"[INFO] class_weight={class_stats.get('class_weight')}")
        print(f"[INFO] year_weight={class_stats.get('year_weight')}")

    best_ckpt_path = result_dir / "best.pt"
    last_ckpt_path = result_dir / "last.pt"
    train_state_path = result_dir / "train_state.json"

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            args=args,
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        scheduler.step(val_metrics["macro_f1"])

        epoch_log = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_log)

        print(
            f"[EPOCH {epoch + 1:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_cls={train_metrics['loss_cls']:.6f} "
            f"train_tf={train_metrics['teacher_forcing_ratio']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"val_bal_acc={val_metrics['balanced_acc']:.6f}"
        )

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_metric,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"[SAVE] best checkpoint: {best_ckpt_path}")
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_metric,
                    "args": vars(args),
                },
                last_ckpt_path,
            )

        save_json(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_metric,
                "history": history,
            },
            str(train_state_path),
        )

        if patience_counter >= args.patience:
            print(f"[EARLY STOP] no improvement for {args.patience} epochs.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without saving a best checkpoint.")

    model.load_state_dict(best_state_dict)

    train_eval = evaluate_one_epoch(model, train_loader, criterion, device, args)
    val_eval = evaluate_one_epoch(model, val_loader, criterion, device, args)
    test_eval = evaluate_one_epoch(model, test_loader, criterion, device, args)

    print(f"[FINAL] best_epoch={best_epoch}")
    print(f"[FINAL] train_macro_f1={train_eval['macro_f1']:.6f}")
    print(f"[FINAL] val_macro_f1={val_eval['macro_f1']:.6f}")
    print(f"[FINAL] test_macro_f1={test_eval['macro_f1']:.6f}")

    train_pack = collect_full_predictions(model, train_loader, device, args)
    val_pack = collect_full_predictions(model, val_loader, device, args)
    test_pack = collect_full_predictions(model, test_loader, device, args)

    if args.save_train_val_test_predictions:
        save_predictions_and_reports(
            result_dir=result_dir,
            dataset=args.dataset,
            train_pack=train_pack,
            val_pack=val_pack,
            test_pack=test_pack,
            train_metrics=train_eval,
            val_metrics=val_eval,
            test_metrics=test_eval,
        )

    num_authors = len(bundle["author_ids"])
    full_pred = build_full_author_prediction(
        num_authors=num_authors,
        train_pack=train_pack,
        val_pack=val_pack,
        test_pack=test_pack,
    )

    final_pred_path = result_dir / f"pred_feature_ablation_harp_{args.feature_model}_{args.dataset}.npy"
    final_author_ids_out_path = result_dir / f"author_ids_{args.dataset}.json"
    np.save(final_pred_path, full_pred)
    final_author_ids_out_path.write_text(
        json.dumps(bundle["author_ids"], ensure_ascii=False),
        encoding="utf-8",
    )

    print("[DONE] Saved outputs:")
    print(f"       pred: {final_pred_path} shape={full_pred.shape} dtype={full_pred.dtype}")
    print(f"       author_ids: {final_author_ids_out_path} N={num_authors}")


if __name__ == "__main__":
    main()