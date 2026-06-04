#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Subset

from config.path_config_v3 import PATHS
from dataset.packed_collate_v3 import collate_packed_subgraphs_v3
from dataset.packed_hypergraph_dataset_v3 import PackedHypergraphDatasetV3
from models.encoder.hypergraph_encoder_v3 import HypergraphEncoderV3
from models.encoder.losses_v3 import EncoderLossBundleV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 hypergraph encoder with packed cache.")
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_sample_budget", type=int, default=200000)
    parser.add_argument("--val_sample_budget", type=int, default=50000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = move_to_device(v, device)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.bool, non_blocking=True)
            elif np.issubdtype(v.dtype, np.integer):
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.long, non_blocking=True)
            else:
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.float32, non_blocking=True)
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def make_epoch_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    sample_budget: int,
    shuffle: bool,
):
    n = len(dataset)
    if sample_budget > 0 and sample_budget < n:
        idx = np.random.choice(n, size=sample_budget, replace=False)
        ds = Subset(dataset, idx.tolist())
    else:
        ds = dataset

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_packed_subgraphs_v3,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=1 if num_workers > 0 else None,
    )


def run_one_epoch(
    model: HypergraphEncoderV3,
    loss_bundle: EncoderLossBundleV3,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_view = 0.0
    total_ring = 0.0
    total_cons = 0.0
    total_cls = 0.0
    total_hs = 0.0
    total_hf = 0.0
    total_cls_acc = 0.0
    total_hs_acc = 0.0
    total_hf_acc = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            metrics = loss_bundle(
                outputs=outputs,
                hist_cls=batch["hist_cls"],
                hist_hs=batch["hist_hs"],
                hist_hf=batch["hist_hf"],
            )
            loss = metrics["loss"]

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_view += float(metrics["loss_view"].cpu())
        total_ring += float(metrics["loss_ring"].cpu())
        total_cons += float(metrics["loss_consistency"].cpu())
        total_cls += float(metrics["loss_cls"].cpu())
        total_hs += float(metrics["loss_hs"].cpu())
        total_hf += float(metrics["loss_hf"].cpu())
        total_cls_acc += float(metrics["cls_acc"].cpu())
        total_hs_acc += float(metrics["hs_acc"].cpu())
        total_hf_acc += float(metrics["hf_acc"].cpu())
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": 0.0,
            "loss_view": 0.0,
            "loss_ring": 0.0,
            "loss_consistency": 0.0,
            "loss_cls": 0.0,
            "loss_hs": 0.0,
            "loss_hf": 0.0,
            "cls_acc": 0.0,
            "hs_acc": 0.0,
            "hf_acc": 0.0,
        }

    return {
        "loss": total_loss / n_batches,
        "loss_view": total_view / n_batches,
        "loss_ring": total_ring / n_batches,
        "loss_consistency": total_cons / n_batches,
        "loss_cls": total_cls / n_batches,
        "loss_hs": total_hs / n_batches,
        "loss_hf": total_hf / n_batches,
        "cls_acc": total_cls_acc / n_batches,
        "hs_acc": total_hs_acc / n_batches,
        "hf_acc": total_hf_acc / n_batches,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    mp.set_sharing_strategy("file_system")
    PATHS.ensure_v3_dirs()

    dp = PATHS.get_dataset_paths(args.dataset)
    ckpt_root = Path(dp.encoder_ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    train_ds = PackedHypergraphDatasetV3(dataset=args.dataset, split="train")
    val_ds = PackedHypergraphDatasetV3(dataset=args.dataset, split="val")

    print(f"[INFO] dataset={args.dataset} device={device}")
    print(f"[INFO] train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(
        f"[INFO] hidden_dim={args.hidden_dim} dropout={args.dropout} "
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} "
        f"train_budget={args.train_sample_budget} val_budget={args.val_sample_budget}"
    )

    model = HypergraphEncoderV3(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=4,
    ).to(device)

    loss_bundle = EncoderLossBundleV3().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loader = make_epoch_loader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_budget=args.train_sample_budget,
            shuffle=True,
        )
        val_loader = make_epoch_loader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_budget=args.val_sample_budget,
            shuffle=False,
        )

        train_metrics = run_one_epoch(model, loss_bundle, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, loss_bundle, val_loader, None, device)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_cls_acc={train_metrics['cls_acc']:.4f} "
            f"val_cls_acc={val_metrics['cls_acc']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_root / "last.pt",
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val,
                },
                ckpt_root / "best.pt",
            )
            print(f"[SAVE] best checkpoint -> {ckpt_root / 'best.pt'}")

    with open(ckpt_root / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "history": history,
                "best_val_loss": best_val,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] packed encoder training finished")
    print(f"[OUT] best_ckpt = {ckpt_root / 'best.pt'}")
    print(f"[OUT] last_ckpt = {ckpt_root / 'last.pt'}")
    print(f"[OUT] train_state = {ckpt_root / 'train_state.json'}")


if __name__ == "__main__":
    main()