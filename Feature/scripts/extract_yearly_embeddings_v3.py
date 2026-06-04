#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset

from config.path_config_v3 import PATHS
from dataset.packed_collate_v3 import collate_packed_subgraphs_v3
from models.encoder.hypergraph_encoder_v3 import HypergraphEncoderV3
from utils.packed_cache_v3 import PackedYearCacheV3


class PackedSingleYearAllSplitsDatasetV3(Dataset):
    def __init__(self, dataset: str, year: int) -> None:
        super().__init__()
        self.dataset_name = dataset.lower().strip()
        self.year = int(year)

        self.caches: Dict[str, PackedYearCacheV3] = {}
        self.slot_to_split: List[str] = []
        self.slot_arr: np.ndarray
        self.local_idx_arr: np.ndarray

        self._build()

    def _build(self) -> None:
        slot_parts: List[np.ndarray] = []
        local_parts: List[np.ndarray] = []
        slot = 0

        for split in ("train", "val", "test"):
            root = PATHS.root / "artifacts" / "cache_packed" / self.dataset_name / split / str(self.year)
            if not (root / "sample_index.npz").exists():
                continue

            cache = PackedYearCacheV3(root)
            idx = cache.load_index()
            n = idx["author_idx"].shape[0]
            if n == 0:
                continue

            self.caches[split] = cache
            self.slot_to_split.append(split)

            slot_parts.append(np.full(n, slot, dtype=np.int16))
            local_parts.append(np.arange(n, dtype=np.int32))
            slot += 1

        if len(slot_parts) == 0:
            self.slot_arr = np.zeros((0,), dtype=np.int16)
            self.local_idx_arr = np.zeros((0,), dtype=np.int32)
        else:
            self.slot_arr = np.concatenate(slot_parts, axis=0)
            self.local_idx_arr = np.concatenate(local_parts, axis=0)

    def __len__(self) -> int:
        return len(self.local_idx_arr)

    @staticmethod
    def _slice_by_ptr(arr: np.ndarray, ptr: np.ndarray, i: int) -> np.ndarray:
        s = int(ptr[i])
        e = int(ptr[i + 1])
        return arr[s:e]

    def _load_view_sample(self, cache: PackedYearCacheV3, prefix: str, local_i: int) -> Dict:
        return {
            "ring0": {"target_x": cache.load_array(f"{prefix}_target_x")[local_i]},
            "ring1": {
                "paper_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_paper_x"), cache.load_array(f"{prefix}_ring1_paper_ptr"), local_i),
                "author_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_author_x"), cache.load_array(f"{prefix}_ring1_author_ptr"), local_i),
                "topic_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_topic_x"), cache.load_array(f"{prefix}_ring1_topic_ptr"), local_i),
                "venue_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_venue_x"), cache.load_array(f"{prefix}_ring1_venue_ptr"), local_i),
                "paper_author_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_pa_edges"), cache.load_array(f"{prefix}_ring1_pa_ptr"), local_i),
                "paper_topic_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_pt_edges"), cache.load_array(f"{prefix}_ring1_pt_ptr"), local_i),
                "paper_venue_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring1_pv_edges"), cache.load_array(f"{prefix}_ring1_pv_ptr"), local_i),
            },
            "ring2": {
                "paper_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_paper_x"), cache.load_array(f"{prefix}_ring2_paper_ptr"), local_i),
                "author_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_author_x"), cache.load_array(f"{prefix}_ring2_author_ptr"), local_i),
                "topic_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_topic_x"), cache.load_array(f"{prefix}_ring2_topic_ptr"), local_i),
                "venue_x": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_venue_x"), cache.load_array(f"{prefix}_ring2_venue_ptr"), local_i),
                "paper_author_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_pa_edges"), cache.load_array(f"{prefix}_ring2_pa_ptr"), local_i),
                "paper_topic_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_pt_edges"), cache.load_array(f"{prefix}_ring2_pt_ptr"), local_i),
                "paper_venue_edges": self._slice_by_ptr(cache.load_array(f"{prefix}_ring2_pv_edges"), cache.load_array(f"{prefix}_ring2_pv_ptr"), local_i),
            },
        }

    def __getitem__(self, index: int) -> Dict:
        slot = int(self.slot_arr[index])
        local_i = int(self.local_idx_arr[index])

        split = self.slot_to_split[slot]
        cache = self.caches[split]
        idx = cache.load_index()

        return {
            "author_idx": int(idx["author_idx"][local_i]),
            "year": int(idx["year"][local_i]),
            "hist_hs": int(idx["hist_hs"][local_i]),
            "hist_hf": int(idx["hist_hf"][local_i]),
            "hist_cls": int(idx["hist_cls"][local_i]),
            "stock": self._load_view_sample(cache, "stock", local_i),
            "flow": self._load_view_sample(cache, "flow", local_i),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract yearly embeddings with packed cache.")
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt", type=str, default="")
    return parser.parse_args()

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

def main() -> None:
    args = parse_args()
    PATHS.ensure_v3_dirs()
    mp.set_sharing_strategy("file_system")
    dp = PATHS.get_dataset_paths(args.dataset)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(dp.encoder_ckpt_root) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ds = PackedSingleYearAllSplitsDatasetV3(args.dataset, args.year)
    print(f"[INFO] dataset={args.dataset} year={args.year} samples={len(ds)}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_packed_subgraphs_v3,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=1 if args.num_workers > 0 else None,
    )

    model = HypergraphEncoderV3(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=4,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with open(dp.author_ids_history, "r", encoding="utf-8") as f:
        author_ids_history = json.load(f)

    n_authors = len(author_ids_history)
    hidden_dim = args.hidden_dim

    X_stock = np.zeros((n_authors, hidden_dim), dtype=np.float32)
    X_flow = np.zeros((n_authors, hidden_dim), dtype=np.float32)
    X_joint = np.zeros((n_authors, hidden_dim), dtype=np.float32)
    mask = np.zeros((n_authors,), dtype=np.uint8)

    with torch.no_grad():
        for batch in dl:
            author_idx = np.asarray(batch["author_idx"], dtype=np.int64)
            batch = move_to_device(batch, device)
            outputs = model(batch)

            z_stock = outputs["z_stock"].detach().cpu().numpy()
            z_flow = outputs["z_flow"].detach().cpu().numpy()
            z_joint = outputs["z_joint"].detach().cpu().numpy()

            for i, aidx in enumerate(author_idx):
                X_stock[aidx] = z_stock[i]
                X_flow[aidx] = z_flow[i]
                X_joint[aidx] = z_joint[i]
                mask[aidx] = 1

    out_root = Path(dp.yearly_embed_root)
    out_root.mkdir(parents=True, exist_ok=True)

    np.save(out_root / f"stock_{args.year}.npy", X_stock)
    np.save(out_root / f"flow_{args.year}.npy", X_flow)
    np.save(out_root / f"joint_{args.year}.npy", X_joint)
    np.save(out_root / f"mask_{args.year}.npy", mask)

    with open(out_root / f"meta_{args.year}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "year": int(args.year),
                "checkpoint": str(ckpt_path),
                "hidden_dim": int(hidden_dim),
                "n_authors": int(n_authors),
                "n_filled": int(mask.sum()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] year={args.year}")
    print(f"  stock: {out_root / f'stock_{args.year}.npy'}")
    print(f"  flow : {out_root / f'flow_{args.year}.npy'}")
    print(f"  joint: {out_root / f'joint_{args.year}.npy'}")
    print(f"  mask : {out_root / f'mask_{args.year}.npy'}")


if __name__ == "__main__":
    main()