#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.path_config_v3 import PATHS
from dataset.predictor_dataset_v3 import PredictorDatasetV3
from models.predictor.lstm_predictor_v3 import LSTMPredictorV3
from models.predictor.transformer_predictor_v3 import TransformerPredictorV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export predictions for my_method_v3 predictors."
    )
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--input_name",
        type=str,
        default="X_concat",
        choices=["X_stock", "X_flow", "X_joint", "X_concat"],
    )
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_predictor_v3(batch):
    x = torch.tensor(np.stack([b["x"] for b in batch], axis=0), dtype=torch.float32)
    m = torch.tensor(np.stack([b["mask"] for b in batch], axis=0), dtype=torch.float32)
    y = torch.tensor(np.stack([b["y"] for b in batch], axis=0), dtype=torch.long)
    author_idx = torch.tensor([b["author_idx"] for b in batch], dtype=torch.long)
    author_id = [b["author_id"] for b in batch]
    return {
        "author_idx": author_idx,
        "author_id": author_id,
        "x": x,
        "mask": m,
        "y": y,
    }


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compute_loss_and_acc(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits: (B, 6, C)
    y     : (B, 6)
    """
    bsz, steps, num_classes = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(bsz * steps, num_classes),
        y.reshape(bsz * steps),
    )

    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean()
    return loss, acc, pred


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(batch["x"])
            loss, acc, _ = compute_loss_and_acc(logits, batch["y"])

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_acc += float(acc.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "acc": 0.0}

    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
    }


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return
    ------
    author_idx: (M,)
    pred      : (M, 6)
    """
    model.eval()

    author_idx_all = []
    pred_all = []

    for batch in loader:
        author_idx = batch["author_idx"].cpu().numpy()
        batch = move_to_device(batch, device)
        logits = model(batch["x"])
        pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.uint8)

        author_idx_all.append(author_idx)
        pred_all.append(pred)

    if len(author_idx_all) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 6), dtype=np.uint8)

    author_idx_all = np.concatenate(author_idx_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    return author_idx_all, pred_all


def build_loader(
    dataset: PredictorDatasetV3,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_predictor_v3,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    return DataLoader(**loader_kwargs)


def build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "lstm":
        model = LSTMPredictorV3(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_steps=6,
            num_classes=4,
        )
    elif model_name == "transformer":
        model = TransformerPredictorV3(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_steps=6,
            num_classes=4,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    PATHS.ensure_v3_dirs()

    model_tag = args.model.upper()
    model_name = f"my_M_v3_{model_tag}"

    dp = PATHS.get_dataset_paths(args.dataset)

    ckpt_root = Path(dp.predictor_ckpt_root) / model_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    result_root = PATHS.wh_root / "results" / model_name
    result_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    train_ds = PredictorDatasetV3(
        dataset=args.dataset,
        split="train",
        input_name=args.input_name,
        use_mask=args.use_mask,
    )
    val_ds = PredictorDatasetV3(
        dataset=args.dataset,
        split="val",
        input_name=args.input_name,
        use_mask=args.use_mask,
    )
    test_ds = PredictorDatasetV3(
        dataset=args.dataset,
        split="test",
        input_name=args.input_name,
        use_mask=args.use_mask,
    )

    input_dim = int(train_ds.X.shape[2])

    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)

    model = build_model(
        model_name=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    history = []

    print(f"[INFO] dataset={args.dataset} device={device}")
    print(f"[INFO] model={model_name}")
    print(f"[INFO] input_name={args.input_name} use_mask={args.use_mask}")
    print(
        f"[INFO] train_samples={len(train_ds)} "
        f"val_samples={len(val_ds)} test_samples={len(test_ds)}"
    )
    print(
        f"[INFO] input_dim={input_dim} hidden_dim={args.hidden_dim} "
        f"num_layers={args.num_layers} dropout={args.dropout}"
    )
    if args.model == "transformer":
        print(f"[INFO] num_heads={args.num_heads}")
    print(f"[INFO] batch_size={args.batch_size} epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, None, device)

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
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
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

    best_ckpt = torch.load(ckpt_root / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    train_loader_pred = build_loader(train_ds, args.batch_size, args.num_workers, shuffle=False)
    val_loader_pred = build_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_loader_pred = build_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    with open(dp.author_ids_history, "r", encoding="utf-8") as f:
        author_ids = json.load(f)

    n_authors = len(author_ids)
    full_pred = np.zeros((n_authors, 6), dtype=np.uint8)

    for split_name, loader in [
        ("train", train_loader_pred),
        ("val", val_loader_pred),
        ("test", test_loader_pred),
    ]:
        author_idx_arr, pred_arr = predict_split(model, loader, device)
        full_pred[author_idx_arr] = pred_arr
        print(f"[PRED] split={split_name} filled={len(author_idx_arr)}")

    pred_path = result_root / f"pred_{model_name}_cls_2015_2020_{args.dataset}.npy"
    author_ids_path = result_root / f"author_ids_{args.dataset}.json"

    np.save(pred_path, full_pred)
    with open(author_ids_path, "w", encoding="utf-8") as f:
        json.dump(author_ids, f, ensure_ascii=False)

    print("[DONE] predictor training and export finished")
    print(f"[OUT] best_ckpt     = {ckpt_root / 'best.pt'}")
    print(f"[OUT] train_state   = {ckpt_root / 'train_state.json'}")
    print(f"[OUT] pred          = {pred_path}")
    print(f"[OUT] author_ids    = {author_ids_path}")
    print(f"[OUT] pred_shape    = {full_pred.shape}")
    print(f"[OUT] pred_dtype    = {full_pred.dtype}")


if __name__ == "__main__":
    main()