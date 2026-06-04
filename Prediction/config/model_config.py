#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class HARPModelConfig:
    # Basic
    model_name: str
    num_history_years: int
    num_future_years: int
    num_classes: int

    # Input dimensions
    input_dim_stock: int
    input_dim_flow: int
    input_dim_joint: int

    # Model dimensions
    d_model: int
    ffn_dim: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    dropout: float
    attn_dropout: float
    ff_dropout: float

    # Historical encoder
    use_year_embedding: bool
    use_recency_embedding: bool
    max_year_tokens: int
    max_recency_tokens: int

    # Relational fusion
    fusion_num_heads: int
    fusion_dropout: float

    # Decoder
    decoder_use_cross_attention: bool
    decoder_use_selective_gate: bool
    decoder_future_year_start: int

    # Training objective defaults
    focal_gamma: float
    label_smoothing: float
    aux_stock_weight: float
    aux_flow_weight: float
    smoothness_weight: float

    # Scheduled sampling defaults
    scheduled_sampling_start: float
    scheduled_sampling_end: float

    # Optimization defaults
    lr: float
    weight_decay: float
    grad_clip: float

    def to_dict(self) -> Dict:
        return asdict(self)


def build_model_config(
    input_dim_stock: int = 256,
    input_dim_flow: int = 256,
    input_dim_joint: int = 256,
) -> HARPModelConfig:
    """
    Build the default HARP model configuration.

    These values are strong defaults for the current scholar impact
    classification task and can be overridden later in the training script.
    """
    return HARPModelConfig(
        model_name="HARP",
        num_history_years=15,
        num_future_years=6,
        num_classes=4,

        input_dim_stock=input_dim_stock,
        input_dim_flow=input_dim_flow,
        input_dim_joint=input_dim_joint,

        d_model=256,
        ffn_dim=512,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=3,
        dropout=0.10,
        attn_dropout=0.10,
        ff_dropout=0.10,

        use_year_embedding=True,
        use_recency_embedding=True,
        max_year_tokens=32,
        max_recency_tokens=32,

        fusion_num_heads=4,
        fusion_dropout=0.10,

        decoder_use_cross_attention=True,
        decoder_use_selective_gate=True,
        decoder_future_year_start=2015,

        focal_gamma=1.5,
        label_smoothing=0.0,
        aux_stock_weight=0.15,
        aux_flow_weight=0.15,
        smoothness_weight=0.02,

        scheduled_sampling_start=1.0,
        scheduled_sampling_end=0.2,

        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
    )


if __name__ == "__main__":
    cfg = build_model_config()
    print(cfg.to_dict())