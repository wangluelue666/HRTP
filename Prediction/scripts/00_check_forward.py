#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

# Support both:
#   1) python -m scripts.00_check_forward
#   2) python scripts/00_check_forward.py
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
        description="Run full forward/loss sanity check for the step-wise rollout HARP."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=sorted(VALID_DATASETS))
    parser.add_argument("--model_name", type=str, default="HARP")
    parser.add_argument("--project_root", type=str, default="/root/autodl-tmp/WH2/HARP")
    parser.add_argument("--source_feature_root", type=str, default="/root/autodl-tmp/WH2/my_method_v3")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/WH2/data")
    parser.add_argument("--result_root", type=str, default="/root/autodl-tmp/WH2/results")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--save_report", action="store_true")

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

    parser.add_argument(
        "--prev_mode",
        type=str,
        default="gt",
        choices=["gt", "pure"],
        help="gt: use gt-shifted previous-class sequence; pure: pure rollout with no override.",
    )
    parser.add_argument(
        "--tf_start_mode",
        type=str,
        default="uniform",
        choices=["uniform", "zeros", "first_label"],
    )
    parser.add_argument("--tf_smoothing", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
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


def build_model_from_batch(batch: Dict[str, Any], args: argparse.Namespace) -> HARPPredictor:
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
    class_stats: Dict[str, Any] | None,
    device: torch.device,
) -> HARPLoss:
    class_weight = None
    year_weight = None

    if class_stats is not None:
        if "class_weight" in class_stats:
            class_weight = torch.tensor(class_stats["class_weight"], dtype=torch.float32, device=device)
        if "year_weight" in class_stats:
            year_weight = torch.tensor(class_stats["year_weight"], dtype=torch.float32, device=device)

    criterion = HARPLoss(
        focal_gamma=args.focal_gamma,
        class_weight=class_weight,
        year_weight=year_weight,
        label_smoothing=args.label_smoothing,
        aux_stock_weight=args.aux_stock_weight,
        aux_flow_weight=args.aux_flow_weight,
        smoothness_weight=args.smoothness_weight,
        smoothness_p=1,
    )
    return criterion


def tensor_shape(x: torch.Tensor) -> list[int]:
    return list(x.shape)


def collect_forward_summary(
    batch: Dict[str, Any],
    outputs: Dict[str, Any],
    loss_dict: Dict[str, torch.Tensor],
    prev_mode: str,
) -> Dict[str, Any]:
    summary = {
        "prev_mode": prev_mode,
        "batch": {
            "x_stock": tensor_shape(batch["x_stock"]),
            "x_flow": tensor_shape(batch["x_flow"]),
            "x_joint": tensor_shape(batch["x_joint"]),
            "labels": tensor_shape(batch["labels"]),
            "stock_targets": tensor_shape(batch["stock_targets"]),
            "flow_targets": tensor_shape(batch["flow_targets"]),
            "mask": tensor_shape(batch["mask"]),
            "year_ids": tensor_shape(batch["year_ids"]),
            "recency_ids": tensor_shape(batch["recency_ids"]),
        },
        "outputs": {
            "class_logits": tensor_shape(outputs["class_logits"]),
            "class_probs": tensor_shape(outputs["class_probs"]),
            "class_pred": tensor_shape(outputs["class_pred"]),
            "stock_probs": tensor_shape(outputs["stock_probs"]),
            "flow_probs": tensor_shape(outputs["flow_probs"]),
            "future_states": tensor_shape(outputs["future_states"]),
            "rollout_feedback_logits": tensor_shape(outputs["rollout_feedback_logits"]),
            "rollout_feedback_probs": tensor_shape(outputs["rollout_feedback_probs"]),
            "h_rel": tensor_shape(outputs["h_rel"]),
            "c_hist": tensor_shape(outputs["c_hist"]),
        },
        "loss": {
            "loss": float(loss_dict["loss"].detach().cpu().item()),
            "loss_cls": float(loss_dict["loss_cls"].detach().cpu().item()),
            "loss_stock": float(loss_dict["loss_stock"].detach().cpu().item()),
            "loss_flow": float(loss_dict["loss_flow"].detach().cpu().item()),
            "loss_smooth": float(loss_dict["loss_smooth"].detach().cpu().item()),
        },
    }

    if "cross_stream_attn" in outputs:
        summary["outputs"]["cross_stream_attn"] = tensor_shape(outputs["cross_stream_attn"])
    if "pooling_weights" in outputs:
        summary["outputs"]["pooling_weights"] = tensor_shape(outputs["pooling_weights"])
    if "step_cross_attn" in outputs:
        summary["outputs"]["num_step_cross_attn"] = len(outputs["step_cross_attn"])
    if "step_gate_values" in outputs:
        summary["outputs"]["num_step_gate_values"] = len(outputs["step_gate_values"])

    return summary


def main() -> None:
    args = parse_args()
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

    if args.split == "train":
        dataset = train_set
    elif args.split == "val":
        dataset = val_set
    else:
        dataset = test_set

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=harp_predictor_collate_fn,
        drop_last=False,
    )

    class_stats = load_class_stats_if_exists(cfg)
    report_all: Dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "used_class_stats": class_stats is not None,
        "prev_mode": args.prev_mode,
        "batches": [],
    }

    model = None
    criterion = None

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        batch = move_batch_to_device(batch, device)

        if model is None:
            model = build_model_from_batch(batch, args).to(device)
            criterion = build_loss_from_stats(args, class_stats, device)

        model.eval()
        with torch.no_grad():
            if args.prev_mode == "gt":
                prev_class_probs = model.build_prev_class_probs_from_labels(
                    labels=batch["labels"],
                    start_mode=args.tf_start_mode,
                    smoothing=args.tf_smoothing,
                ).to(device)
            else:
                prev_class_probs = None

            outputs = model(
                x_stock=batch["x_stock"],
                x_flow=batch["x_flow"],
                x_joint=batch["x_joint"],
                mask=batch["mask"],
                year_ids=batch["year_ids"],
                recency_ids=batch["recency_ids"],
                prev_class_probs_override=prev_class_probs,
                return_analysis=True,
                return_intermediates=False,
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

        batch_summary = collect_forward_summary(
            batch=batch,
            outputs=outputs,
            loss_dict=loss_dict,
            prev_mode=args.prev_mode,
        )
        report_all["batches"].append(batch_summary)

        print(f"[INFO] batch={batch_idx}")
        print(f"[INFO] prev_mode={args.prev_mode}")
        print(f"[INFO] class_logits={batch_summary['outputs']['class_logits']}")
        print(f"[INFO] rollout_feedback_logits={batch_summary['outputs']['rollout_feedback_logits']}")
        print(f"[INFO] future_states={batch_summary['outputs']['future_states']}")
        print(f"[INFO] loss={batch_summary['loss']['loss']:.6f}")
        print(f"[INFO] loss_cls={batch_summary['loss']['loss_cls']:.6f}")
        print(f"[INFO] loss_stock={batch_summary['loss']['loss_stock']:.6f}")
        print(f"[INFO] loss_flow={batch_summary['loss']['loss_flow']:.6f}")
        print(f"[INFO] loss_smooth={batch_summary['loss']['loss_smooth']:.6f}")

    if args.save_report:
        report_path = Path(cfg.output_reports_root) / f"forward_check_{args.dataset}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_all, f, ensure_ascii=False, indent=2)
        print(f"[INFO] saved report to: {report_path}")


if __name__ == "__main__":
    main()
