#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.path_config import build_harp_path_config, ensure_required_dirs
from dataset import build_harp_datasets, harp_predictor_collate_fn
from losses import HARPLoss
from models.predictor import HARPPredictor


VALID_DATASETS = {"acm", "aps", "dblp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-trial hyperparameter training for HARP."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=sorted(VALID_DATASETS))
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--trial_id", type=str, required=True)
    parser.add_argument("--trial_root", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="HARP")
    parser.add_argument("--project_root", type=str, default="/root/autodl-tmp/WH2/HARP")
    parser.add_argument("--source_feature_root", type=str, default="/root/autodl-tmp/WH2/my_method_v3")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/WH2/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # The 6 searched hyperparameters
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--ffn_dim", type=int, required=True)
    parser.add_argument("--encoder_layers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--scheduled_sampling_end", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)

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


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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
    return load_json(path)


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
        num_heads=4,
        fusion_num_heads=4,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        attn_dropout=args.dropout,
        fusion_dropout=args.dropout,
        use_year_embedding=True,
        use_recency_embedding=True,
        max_year_tokens=64,
        max_recency_tokens=64,
        base_year=2000,
        future_year_start=2015,
        temporal_refine_layers=1,
        pooling_hidden_dim=args.d_model,
        use_selective_gate=True,
        use_pre_norm=True,
    )
    return model


def build_loss_from_stats(stats: Dict[str, Any] | None, device: torch.device) -> HARPLoss:
    class_weight = None
    year_weight = None
    if stats is not None:
        if "class_weight" in stats:
            class_weight = torch.tensor(stats["class_weight"], dtype=torch.float32, device=device)
        if "year_weight" in stats:
            year_weight = torch.tensor(stats["year_weight"], dtype=torch.float32, device=device)

    return HARPLoss(
        focal_gamma=1.5,
        class_weight=class_weight,
        year_weight=year_weight,
        label_smoothing=0.0,
        aux_stock_weight=0.15,
        aux_flow_weight=0.15,
        smoothness_weight=0.02,
        smoothness_p=1,
    )


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
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
) -> torch.Tensor:
    gt_prev = model.build_prev_class_probs_from_labels(
        labels=batch["labels"],
        start_mode="uniform",
        smoothing=0.0,
    ).to(batch["labels"].device)

    warm_outputs = run_forward_with_prev(
        model=model,
        batch=batch,
        prev_class_probs_override=None,
    )
    pred_prev = model.build_prev_class_probs_from_prediction(
        class_probs=warm_outputs["class_probs"],
        start_mode="uniform",
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
        start=1.0,
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
        if not torch.isfinite(loss_dict["loss"]).all():
            print("[WARN] Non-finite loss detected. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
) -> Dict[str, float]:
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

        outputs = run_forward_with_prev(
            model=model,
            batch=batch,
            prev_class_probs_override=None,
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
def collect_predictions(
    model: HARPPredictor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()

    pred_list = []
    label_list = []
    idx_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = run_forward_with_prev(
            model=model,
            batch=batch,
            prev_class_probs_override=None,
        )

        pred_list.append(outputs["class_pred"].detach().cpu().numpy().astype(np.uint8))
        label_list.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
        idx_list.append(batch["global_idx"].detach().cpu().numpy().astype(np.int64))

    return {
        "pred": np.concatenate(pred_list, axis=0),
        "label": np.concatenate(label_list, axis=0),
        "global_idx": np.concatenate(idx_list, axis=0),
    }


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


def build_trial_config(args: argparse.Namespace, batch_size: int, epochs: int, patience: int) -> Dict[str, Any]:
    return {
        "trial_id": args.trial_id,
        "dataset": args.dataset,
        "device": args.device,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
        "project_root": args.project_root,
        "source_feature_root": args.source_feature_root,
        "data_root": args.data_root,
        "d_model": args.d_model,
        "ffn_dim": args.ffn_dim,
        "encoder_layers": args.encoder_layers,
        "num_heads": 4,
        "fusion_num_heads": 4,
        "temporal_refine_layers": 1,
        "dropout": args.dropout,
        "attn_dropout": args.dropout,
        "fusion_dropout": args.dropout,
        "scheduled_sampling_start": 1.0,
        "scheduled_sampling_end": args.scheduled_sampling_end,
        "tf_start_mode": "uniform",
        "tf_smoothing": 0.0,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "focal_gamma": 1.5,
        "label_smoothing": 0.0,
        "aux_stock_weight": 0.15,
        "aux_flow_weight": 0.15,
        "smoothness_weight": 0.02,
        "eval_rollout_mode": "pure",
        "use_selective_gate": True,
        "use_pre_norm": True,
        "max_year_tokens": 64,
        "max_recency_tokens": 64,
        "future_year_start": 2015,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    batch_size = 512 if args.dataset == "dblp" else 1024
    epochs = 50
    patience = 8

    trial_root = Path(args.trial_root).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)

    cfg = build_harp_path_config(
        dataset=args.dataset,
        model_name=args.model_name,
        project_root=args.project_root,
        source_feature_root=args.source_feature_root,
        data_root=args.data_root,
        result_root=str(trial_root),
    )
    ensure_required_dirs(cfg)

    cfg, train_set, val_set, test_set = build_harp_datasets(
        dataset=args.dataset,
        model_name=args.model_name,
        project_root=args.project_root,
        source_feature_root=args.source_feature_root,
        data_root=args.data_root,
        result_root=str(trial_root),
        return_numpy=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
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
    criterion = build_loss_from_stats(class_stats, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer)

    config_snapshot = build_trial_config(args, batch_size=batch_size, epochs=epochs, patience=patience)
    if class_stats is not None:
        config_snapshot["class_stats_path"] = cfg.class_stats_train_path
        config_snapshot["class_weight"] = class_stats.get("class_weight")
        config_snapshot["year_weight"] = class_stats.get("year_weight")
    save_json(config_snapshot, trial_root / "config.json")

    best_metric = -1.0
    best_epoch = -1
    best_state_dict = None
    history: List[Dict[str, Any]] = []
    patience_counter = 0

    print(f"[INFO] trial={args.trial_id} dataset={args.dataset} device={device}")
    print(f"[INFO] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[INFO] batch_size={batch_size} epochs={epochs}")
    if class_stats is not None:
        print(f"[INFO] class_weight={class_stats.get('class_weight')}")
        print(f"[INFO] year_weight={class_stats.get('year_weight')}")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=epochs,
            args=args,
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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
                    "config": config_snapshot,
                },
                trial_root / "best.pt",
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_metric,
                "config": config_snapshot,
            },
            trial_root / "last.pt",
        )

        save_json(
            {
                "trial_id": args.trial_id,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_metric,
                "history": history,
            },
            trial_root / "train_state.json",
        )

        if patience_counter >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state_dict)

    train_eval = evaluate_one_epoch(model, train_loader, criterion, device)
    val_eval = evaluate_one_epoch(model, val_loader, criterion, device)
    test_eval = evaluate_one_epoch(model, test_loader, criterion, device)

    train_pack = collect_predictions(model, train_loader, device)
    val_pack = collect_predictions(model, val_loader, device)
    test_pack = collect_predictions(model, test_loader, device)

    author_ids = load_json(Path(cfg.author_ids_path))
    num_authors = len(author_ids)
    full_pred = build_full_author_prediction(num_authors, train_pack, val_pack, test_pack)

    np.save(trial_root / f"pred_HARP_cls_2015_2020_{args.dataset}.npy", full_pred)
    save_json(author_ids, trial_root / f"author_ids_{args.dataset}.json")

    np.save(trial_root / f"pred_train_{args.dataset}.npy", train_pack["pred"])
    np.save(trial_root / f"pred_val_{args.dataset}.npy", val_pack["pred"])
    np.save(trial_root / f"pred_test_{args.dataset}.npy", test_pack["pred"])

    np.save(trial_root / f"true_train_{args.dataset}.npy", train_pack["label"])
    np.save(trial_root / f"true_val_{args.dataset}.npy", val_pack["label"])
    np.save(trial_root / f"true_test_{args.dataset}.npy", test_pack["label"])

    np.save(trial_root / f"idx_train_{args.dataset}.npy", train_pack["global_idx"])
    np.save(trial_root / f"idx_val_{args.dataset}.npy", val_pack["global_idx"])
    np.save(trial_root / f"idx_test_{args.dataset}.npy", test_pack["global_idx"])

    trial_result = {
        "trial_id": args.trial_id,
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "train": train_eval,
        "val": val_eval,
        "test": test_eval,
        "config": config_snapshot,
    }
    save_json(trial_result, trial_root / "trial_result.json")

    print("[DONE] Trial finished.")
    print(f"       trial_root: {trial_root}")
    print(f"       best_epoch: {best_epoch}")
    print(f"       best_val_macro_f1: {best_metric:.6f}")
    print(f"       test_macro_f1: {test_eval['macro_f1']:.6f}")


if __name__ == "__main__":
    main()