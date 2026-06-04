#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import torch


def _build_mask_from_m_year(m_year: torch.Tensor) -> torch.Tensor:
    """
    Build the historical validity mask directly from m_year.

    Args:
        m_year: [B, T], non-zero means valid.

    Returns:
        mask: [B, T], float32
    """
    if m_year.ndim != 2:
        raise ValueError(f"m_year must be [B,T], got shape={tuple(m_year.shape)}")
    return (m_year > 0).to(torch.float32)


def _validate_valid_len_against_mask(
    valid_len: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """
    Validate that valid_len matches the total number of valid historical steps,
    without assuming prefix-style continuity.
    """
    if valid_len.ndim != 1:
        raise ValueError(f"valid_len must be [B], got shape={tuple(valid_len.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be [B,T], got shape={tuple(mask.shape)}")
    if valid_len.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Batch size mismatch: valid_len={tuple(valid_len.shape)}, mask={tuple(mask.shape)}"
        )

    expected_valid_len = mask.sum(dim=1).to(torch.long)
    if not torch.equal(valid_len, expected_valid_len):
        diff = (valid_len != expected_valid_len).sum().item()
        raise ValueError(
            f"valid_len does not match mask valid counts, diff_count={diff}"
        )


def _derive_stock_flow_targets(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert four-class labels into stock / flow binary targets.

    Class mapping:
        0 -> LL
        1 -> LH
        2 -> HL
        3 -> HH

    Therefore:
        stock = y // 2
        flow  = y % 2
    """
    stock_targets = torch.div(labels, 2, rounding_mode="floor").to(torch.float32)
    flow_targets = torch.remainder(labels, 2).to(torch.float32)

    return {
        "stock_targets": stock_targets,
        "flow_targets": flow_targets,
    }


def harp_predictor_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    if len(batch) == 0:
        raise ValueError("Empty batch is not allowed.")

    x_stock = torch.stack([item["x_stock"] for item in batch], dim=0).to(torch.float32)
    x_flow = torch.stack([item["x_flow"] for item in batch], dim=0).to(torch.float32)
    x_joint = torch.stack([item["x_joint"] for item in batch], dim=0).to(torch.float32)

    labels = torch.stack([item["labels"] for item in batch], dim=0).to(torch.long)
    m_year = torch.stack([item["m_year"] for item in batch], dim=0).to(torch.float32)

    year_ids = torch.stack([item["year_ids"] for item in batch], dim=0).to(torch.long)
    recency_ids = torch.stack([item["recency_ids"] for item in batch], dim=0).to(torch.long)

    valid_len = torch.as_tensor(
        [item["valid_len"] for item in batch],
        dtype=torch.long,
    )

    author_ids = [str(item["author_id"]) for item in batch]
    global_idx = torch.as_tensor(
        [int(item["global_idx"]) for item in batch],
        dtype=torch.long,
    )

    batch_size, num_steps, _ = x_stock.shape

    if x_flow.shape[:2] != (batch_size, num_steps):
        raise ValueError(f"x_flow shape mismatch: {tuple(x_flow.shape)}")
    if x_joint.shape[:2] != (batch_size, num_steps):
        raise ValueError(f"x_joint shape mismatch: {tuple(x_joint.shape)}")
    if m_year.shape != (batch_size, num_steps):
        raise ValueError(f"m_year shape mismatch: {tuple(m_year.shape)}")
    if year_ids.shape != (batch_size, num_steps):
        raise ValueError(f"year_ids shape mismatch: {tuple(year_ids.shape)}")
    if recency_ids.shape != (batch_size, num_steps):
        raise ValueError(f"recency_ids shape mismatch: {tuple(recency_ids.shape)}")
    if labels.shape[0] != batch_size or labels.shape[1] != 6:
        raise ValueError(f"labels shape mismatch: {tuple(labels.shape)}")

    if torch.any(labels < 0) or torch.any(labels > 3):
        raise ValueError(
            f"labels must be within [0,3], got min={labels.min().item()}, max={labels.max().item()}"
        )

    # Use the true yearly availability structure directly.
    mask = _build_mask_from_m_year(m_year=m_year)
    _validate_valid_len_against_mask(valid_len=valid_len, mask=mask)

    aux_targets = _derive_stock_flow_targets(labels=labels)

    collated = {
        "x_stock": x_stock,                 # [B, T, D_s]
        "x_flow": x_flow,                   # [B, T, D_f]
        "x_joint": x_joint,                 # [B, T, D_j]
        "labels": labels,                   # [B, 6]
        "stock_targets": aux_targets["stock_targets"],  # [B, 6]
        "flow_targets": aux_targets["flow_targets"],    # [B, 6]

        "m_year": m_year,                   # [B, T]
        "mask": mask,                       # [B, T]
        "valid_len": valid_len,             # [B]
        "year_ids": year_ids,               # [B, T]
        "recency_ids": recency_ids,         # [B, T]

        "author_ids": author_ids,           # list[str]
        "global_idx": global_idx,           # [B]
    }
    return collated


if __name__ == "__main__":
    import numpy as np

    dummy = []
    for i in range(4):
        T = 15
        Ds, Df, Dj = 256, 256, 256

        # Simulate non-prefix yearly availability.
        m_year = np.zeros((T,), dtype=np.float32)
        active_positions = [0, 1, 2, 4, 5, 7, 8, 10]
        m_year[active_positions] = 1.0
        valid_len = int(m_year.sum())

        dummy.append({
            "x_stock": torch.randn(T, Ds),
            "x_flow": torch.randn(T, Df),
            "x_joint": torch.randn(T, Dj),
            "labels": torch.randint(0, 4, (6,), dtype=torch.long),
            "m_year": torch.from_numpy(m_year),
            "valid_len": valid_len,
            "year_ids": torch.arange(2000, 2000 + T, dtype=torch.long),
            "recency_ids": torch.arange(T, dtype=torch.long),
            "author_id": f"a{i}",
            "global_idx": i,
        })

    out = harp_predictor_collate_fn(dummy)
    print("[INFO] batch keys:", list(out.keys()))
    print("[INFO] x_stock:", tuple(out["x_stock"].shape))
    print("[INFO] mask:", tuple(out["mask"].shape))
    print("[INFO] labels:", tuple(out["labels"].shape))
    print("[INFO] stock_targets:", tuple(out["stock_targets"].shape))
    print("[INFO] flow_targets:", tuple(out["flow_targets"].shape))
    print("[INFO] valid_len:", tuple(out["valid_len"].shape))