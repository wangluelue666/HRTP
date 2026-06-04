#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

# Support both:
#   1) python -m scripts.03_train_harp
#   2) python scripts/03_train_harp.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.path_config import build_harp_path_config, ensure_required_dirs, dump_path_manifest
from dataset import build_harp_datasets, harp_predictor_collate_fn
from losses import HARPLoss
from models.predictor import HARPPredictor


VALID_DATASETS = {"acm", "aps", "dblp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train step-wise rollout HARP for scholar impact classification."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=sorted(VALID_DATASETS))
    parser.add_argument("--model_name", type=str, default="HARP")
    parser.add_argument("--project_root", type=str, default="/root/autodl-tmp/WH2/HARP")
    parser.add_argument("--source_feature_root", type=str, default="/root/autodl-tmp/WH2/my_method_v3")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/WH2/data")
    parser.add_argument("--result_root", type=str, default="/root/autodl-tmp/WH2/results")

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


def load_class_stats_if_exists(cfg) -> Dict[str, Any] | None:
    path = Path(cfg.class_stats_train_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    """
    Build mixed previous-class sequence for scheduled sampling.

    gt_prev  : shifted one-hot ground truth
    pred_prev: shifted model prediction from a pure rollout warmup pass

    mixed = ratio * gt_prev + (1-ratio) * pred_prev
    """
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

        if not torch.isfinite(outputs["class_logits"]).all():
            print("[WARN] Non-finite class_logits detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(outputs["class_probs"]).all():
            print("[WARN] Non-finite class_probs detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(outputs["stock_probs"]).all():
            print("[WARN] Non-finite stock_probs detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(outputs["flow_probs"]).all():
            print("[WARN] Non-finite flow_probs detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

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


def save_json(obj: Dict[str, Any], path: str) -> None:
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    with path_p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_predictions_and_reports(
    cfg,
    train_pack: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    test_pack: Dict[str, np.ndarray],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    output_predict_root = Path(cfg.output_predict_root)
    output_predict_root.mkdir(parents=True, exist_ok=True)

    np.save(output_predict_root / f"pred_train_{cfg.dataset}.npy", train_pack["pred"])
    np.save(output_predict_root / f"pred_val_{cfg.dataset}.npy", val_pack["pred"])
    np.save(output_predict_root / f"pred_test_{cfg.dataset}.npy", test_pack["pred"])

    np.save(output_predict_root / f"true_train_{cfg.dataset}.npy", train_pack["label"])
    np.save(output_predict_root / f"true_val_{cfg.dataset}.npy", val_pack["label"])
    np.save(output_predict_root / f"true_test_{cfg.dataset}.npy", test_pack["label"])

    np.save(output_predict_root / f"idx_train_{cfg.dataset}.npy", train_pack["global_idx"])
    np.save(output_predict_root / f"idx_val_{cfg.dataset}.npy", val_pack["global_idx"])
    np.save(output_predict_root / f"idx_test_{cfg.dataset}.npy", test_pack["global_idx"])

    reports = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(reports, str(Path(cfg.output_reports_root) / f"train_summary_{cfg.dataset}.json"))


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


def copy_author_ids(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

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

    cfg, train_set, val_set, test_set = build_harp_datasets(
        dataset=args.dataset,
        model_name=args.model_name,
        project_root=args.project_root,
        source_feature_root=args.source_feature_root,
        data_root=args.data_root,
        result_root=args.result_root,
        return_numpy=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )

    first_batch = next(iter(train_loader))
    first_batch = move_batch_to_device(first_batch, device)

    class_stats = load_class_stats_if_exists(cfg)
    model = build_model_from_sample(first_batch, args).to(device)
    criterion = build_loss_from_stats(args, class_stats, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer)

    config_snapshot = vars(args).copy()
    if class_stats is not None:
        config_snapshot["class_stats_path"] = cfg.class_stats_train_path
        config_snapshot["class_weight"] = class_stats.get("class_weight")
        config_snapshot["year_weight"] = class_stats.get("year_weight")
    save_json(config_snapshot, cfg.config_snapshot_path)

    best_metric = -1.0
    best_epoch = -1
    best_state_dict = None
    history: List[Dict[str, Any]] = []
    patience_counter = 0

    print(f"[INFO] dataset={args.dataset} device={device}")
    print(f"[INFO] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[INFO] batch_size={args.batch_size} epochs={args.epochs}")
    if class_stats is not None:
        print(f"[INFO] class_weight={class_stats.get('class_weight')}")
        print(f"[INFO] year_weight={class_stats.get('year_weight')}")

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
                cfg.predictor_best_ckpt_path,
            )
            print(f"[SAVE] best checkpoint: {cfg.predictor_best_ckpt_path}")
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
                cfg.predictor_last_ckpt_path,
            )

        save_json(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_metric,
                "history": history,
            },
            cfg.predictor_train_state_path,
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

    if args.save_train_val_test_predictions:
        train_pack = collect_full_predictions(model, train_loader, device, args)
        val_pack = collect_full_predictions(model, val_loader, device, args)
        test_pack = collect_full_predictions(model, test_loader, device, args)

        save_predictions_and_reports(
            cfg=cfg,
            train_pack=train_pack,
            val_pack=val_pack,
            test_pack=test_pack,
            train_metrics=train_eval,
            val_metrics=val_eval,
            test_metrics=test_eval,
        )

        num_authors = len(train_set.bundle.author_ids)
        full_pred = build_full_author_prediction(
            num_authors=num_authors,
            train_pack=train_pack,
            val_pack=val_pack,
            test_pack=test_pack,
        )
    else:
        test_pack = collect_full_predictions(model, test_loader, device, args)
        num_authors = len(train_set.bundle.author_ids)
        full_pred = np.zeros((num_authors, 6), dtype=np.uint8)
        full_pred[test_pack["global_idx"]] = test_pack["pred"]

    final_result_root = Path(cfg.final_result_model_root)
    final_result_root.mkdir(parents=True, exist_ok=True)
    np.save(cfg.final_pred_path, full_pred)
    copy_author_ids(cfg.author_ids_path, cfg.final_author_ids_out_path)

    print("[DONE] Saved outputs:")
    print(f"       pred: {cfg.final_pred_path} shape={full_pred.shape} dtype={full_pred.dtype}")
    print(f"       author_ids: {cfg.final_author_ids_out_path} N={num_authors}")


if __name__ == "__main__":
    main()
