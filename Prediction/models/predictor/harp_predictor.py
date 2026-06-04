#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both:
#   1) python -m models.predictor.harp_predictor
#   2) python models/predictor/harp_predictor.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.predictor.historical_encoder import HistoricalTriStreamEncoder
from models.predictor.relational_fusion import RelationalFusionModule
from models.predictor.autoregressive_decoder import HARPAutoregressiveDecoder
from models.predictor.decision_head import HARPDecisionHead


class HARPPredictor(nn.Module):
    """
    Full HARP predictor with true step-wise autoregressive rollout.

    Pipeline:
        1) Historical tri-stream encoding
        2) Relational fusion
        3) Step-wise future rollout decoder
        4) Unified shared decision head

    Key property:
        The same decision head is used for:
            - step-wise rollout feedback
            - final sequence classification
        so the predictive dynamics and supervision target stay aligned.
    """

    def __init__(
        self,
        input_dim_stock: int = 256,
        input_dim_flow: int = 256,
        input_dim_joint: int = 256,
        d_model: int = 256,
        num_classes: int = 4,
        num_history_steps: int = 15,
        num_future_steps: int = 6,
        encoder_layers: int = 2,
        num_heads: int = 4,
        fusion_num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        use_year_embedding: bool = True,
        use_recency_embedding: bool = True,
        max_year_tokens: int = 64,
        max_recency_tokens: int = 64,
        base_year: int = 2000,
        future_year_start: int = 2015,
        temporal_refine_layers: int = 1,
        pooling_hidden_dim: Optional[int] = None,
        use_selective_gate: bool = True,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim_stock = int(input_dim_stock)
        self.input_dim_flow = int(input_dim_flow)
        self.input_dim_joint = int(input_dim_joint)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_history_steps = int(num_history_steps)
        self.num_future_steps = int(num_future_steps)
        self.future_year_start = int(future_year_start)

        self.history_encoder = HistoricalTriStreamEncoder(
            input_dim_stock=input_dim_stock,
            input_dim_flow=input_dim_flow,
            input_dim_joint=input_dim_joint,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_year_embedding=use_year_embedding,
            use_recency_embedding=use_recency_embedding,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            use_pre_norm=use_pre_norm,
        )

        self.relational_fusion = RelationalFusionModule(
            d_model=d_model,
            fusion_num_heads=fusion_num_heads,
            fusion_dropout=fusion_dropout,
            temporal_refine_layers=temporal_refine_layers,
            temporal_refine_heads=num_heads,
            temporal_refine_ffn_dim=ffn_dim,
            temporal_refine_attn_dropout=attn_dropout,
            pooling_hidden_dim=pooling_hidden_dim or d_model,
            pooling_dropout=dropout,
            use_pre_norm=use_pre_norm,
        )

        self.decision_head = HARPDecisionHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=d_model,
            dropout=dropout,
            use_layernorm=True,
        )

        self.future_decoder = HARPAutoregressiveDecoder(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_classes=num_classes,
            future_steps=num_future_steps,
            future_year_start=future_year_start,
            use_selective_gate=use_selective_gate,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
        )

    @staticmethod
    def labels_to_one_hot(
        labels: torch.Tensor,
        num_classes: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Convert integer labels [B, F] into one-hot [B, F, C].
        """
        if labels.ndim != 2:
            raise ValueError(f"labels must be [B,F], got {tuple(labels.shape)}")
        if torch.any(labels < 0) or torch.any(labels >= num_classes):
            raise ValueError(
                f"labels must be within [0,{num_classes - 1}], "
                f"got min={labels.min().item()}, max={labels.max().item()}"
            )
        return F.one_hot(labels.long(), num_classes=num_classes).to(dtype=dtype)

    def build_prev_class_probs_from_labels(
        self,
        labels: torch.Tensor,
        start_mode: str = "uniform",
        smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Build shifted previous-class probability sequence from ground truth labels.

        For future step k:
            prev[:, k] = label from step k-1

        For step 0:
            controlled by start_mode.
        """
        if labels.ndim != 2:
            raise ValueError(f"labels must be [B,F], got {tuple(labels.shape)}")
        B, F_steps = labels.shape
        if F_steps != self.num_future_steps:
            raise ValueError(
                f"labels second dim must equal num_future_steps={self.num_future_steps}, got {F_steps}"
            )
        if start_mode not in {"uniform", "zeros", "first_label"}:
            raise ValueError(f"Unsupported start_mode: {start_mode}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0,1), got {smoothing}")

        one_hot = self.labels_to_one_hot(
            labels=labels,
            num_classes=self.num_classes,
            dtype=torch.float32,
        )  # [B, F, C]

        prev = torch.zeros_like(one_hot)
        if start_mode == "uniform":
            prev[:, 0, :] = 1.0 / float(self.num_classes)
        elif start_mode == "zeros":
            prev[:, 0, :] = 0.0
        else:
            prev[:, 0, :] = one_hot[:, 0, :]

        prev[:, 1:, :] = one_hot[:, :-1, :]

        if smoothing > 0.0:
            prev = prev * (1.0 - smoothing) + smoothing / float(self.num_classes)

        return prev

    def mix_prev_class_probs(
        self,
        gt_prev_probs: torch.Tensor,
        pred_prev_probs: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        """
        Mix ground-truth and predicted previous-class sequences.
        """
        if gt_prev_probs.shape != pred_prev_probs.shape:
            raise ValueError(
                f"Shape mismatch: gt_prev_probs={tuple(gt_prev_probs.shape)}, "
                f"pred_prev_probs={tuple(pred_prev_probs.shape)}"
            )
        if not (0.0 <= teacher_forcing_ratio <= 1.0):
            raise ValueError(
                f"teacher_forcing_ratio must be in [0,1], got {teacher_forcing_ratio}"
            )

        ratio = float(teacher_forcing_ratio)
        return ratio * gt_prev_probs + (1.0 - ratio) * pred_prev_probs

    @torch.no_grad()
    def build_prev_class_probs_from_prediction(
        self,
        class_probs: torch.Tensor,
        start_mode: str = "uniform",
    ) -> torch.Tensor:
        """
        Build shifted previous-class probability sequence from predicted probs.

        This is used for inference refinement or scheduled sampling support.
        """
        if class_probs.ndim != 3:
            raise ValueError(f"class_probs must be [B,F,C], got {tuple(class_probs.shape)}")

        B, F_steps, C = class_probs.shape
        if F_steps != self.num_future_steps or C != self.num_classes:
            raise ValueError(
                f"class_probs shape mismatch: expected [B,{self.num_future_steps},{self.num_classes}], "
                f"got {tuple(class_probs.shape)}"
            )

        prev = torch.zeros_like(class_probs)
        if start_mode == "uniform":
            prev[:, 0, :] = 1.0 / float(self.num_classes)
        elif start_mode == "zeros":
            prev[:, 0, :] = 0.0
        elif start_mode == "first_label":
            prev[:, 0, :] = class_probs[:, 0, :]
        else:
            raise ValueError(f"Unsupported start_mode: {start_mode}")

        prev[:, 1:, :] = class_probs[:, :-1, :]
        return prev

    def forward(
        self,
        x_stock: torch.Tensor,
        x_flow: torch.Tensor,
        x_joint: torch.Tensor,
        mask: torch.Tensor,
        year_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        prev_class_probs_override: Optional[torch.Tensor] = None,
        return_analysis: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor | Dict]:
        """
        Args:
            x_stock   : [B, T_hist, D_s]
            x_flow    : [B, T_hist, D_f]
            x_joint   : [B, T_hist, D_j]
            mask      : [B, T_hist]
            year_ids  : [B, T_hist]
            recency_ids: [B, T_hist]
            prev_class_probs_override:
                external shifted previous-class sequence [B, F, C]
                for teacher forcing / scheduled sampling.
        """
        history_out = self.history_encoder(
            x_stock=x_stock,
            x_flow=x_flow,
            x_joint=x_joint,
            mask=mask,
            year_ids=year_ids,
            recency_ids=recency_ids,
            return_intermediates=return_intermediates,
        )

        fusion_out = self.relational_fusion(
            h_stock=history_out["h_stock"],
            h_flow=history_out["h_flow"],
            h_joint=history_out["h_joint"],
            mask=mask,
            return_analysis=return_analysis,
        )

        decoder_out = self.future_decoder(
            h_rel=fusion_out["h_rel"],
            c_hist=fusion_out["c_hist"],
            history_mask=mask,
            prev_class_probs=prev_class_probs_override,
            future_year_ids=None,
            future_recency_ids=None,
            feedback_head=self.decision_head,
            return_analysis=return_analysis,
        )

        # Final sequence prediction uses the same shared decision head.
        head_out = self.decision_head.forward_sequence(
            decoder_out["future_states"],
        )

        outputs: Dict[str, torch.Tensor | Dict] = {
            "class_logits": head_out["class_logits"],
            "class_probs": head_out["class_probs"],
            "class_pred": head_out["class_pred"],
            "stock_probs": head_out["stock_probs"],
            "flow_probs": head_out["flow_probs"],
            "stock_pred": head_out["stock_pred"],
            "flow_pred": head_out["flow_pred"],

            "future_states": decoder_out["future_states"],
            "future_year_ids": decoder_out["future_year_ids"],
            "future_recency_ids": decoder_out["future_recency_ids"],

            "rollout_feedback_logits": decoder_out["feedback_logits"],
            "rollout_feedback_probs": decoder_out["feedback_probs"],

            "h_rel": fusion_out["h_rel"],
            "c_hist": fusion_out["c_hist"],
        }

        if return_analysis:
            if "cross_stream_attn" in fusion_out:
                outputs["cross_stream_attn"] = fusion_out["cross_stream_attn"]
            if "pooling_weights" in fusion_out:
                outputs["pooling_weights"] = fusion_out["pooling_weights"]
            if "step_cross_attn" in decoder_out:
                outputs["step_cross_attn"] = decoder_out["step_cross_attn"]
            if "step_gate_values" in decoder_out:
                outputs["step_gate_values"] = decoder_out["step_gate_values"]

        if return_intermediates and "stream_details" in history_out:
            outputs["stream_details"] = history_out["stream_details"]

        return outputs


if __name__ == "__main__":
    B, T_hist = 4, 15
    T_future = 6
    D_stream = 256

    x_stock = torch.randn(B, T_hist, D_stream)
    x_flow = torch.randn(B, T_hist, D_stream)
    x_joint = torch.randn(B, T_hist, D_stream)

    mask = torch.ones(B, T_hist)
    mask[0, 12:] = 0
    mask[1, 10:] = 0

    year_ids = torch.arange(2000, 2000 + T_hist).unsqueeze(0).expand(B, -1)
    recency_ids = torch.arange(T_hist).unsqueeze(0).expand(B, -1)
    labels = torch.randint(0, 4, (B, T_future))

    model = HARPPredictor(
        input_dim_stock=256,
        input_dim_flow=256,
        input_dim_joint=256,
        d_model=256,
        num_classes=4,
        num_history_steps=15,
        num_future_steps=6,
        encoder_layers=2,
        num_heads=4,
        fusion_num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        attn_dropout=0.1,
        fusion_dropout=0.1,
        use_year_embedding=True,
        use_recency_embedding=True,
        max_year_tokens=64,
        max_recency_tokens=64,
        base_year=2000,
        future_year_start=2015,
        temporal_refine_layers=1,
        pooling_hidden_dim=256,
        use_selective_gate=True,
        use_pre_norm=True,
    )

    gt_prev_probs = model.build_prev_class_probs_from_labels(
        labels=labels,
        start_mode="uniform",
        smoothing=0.0,
    )

    out = model(
        x_stock=x_stock,
        x_flow=x_flow,
        x_joint=x_joint,
        mask=mask,
        year_ids=year_ids,
        recency_ids=recency_ids,
        prev_class_probs_override=gt_prev_probs,
        return_analysis=True,
        return_intermediates=True,
    )

    print("[INFO] class_logits shape:", tuple(out["class_logits"].shape))
    print("[INFO] class_probs shape:", tuple(out["class_probs"].shape))
    print("[INFO] class_pred shape:", tuple(out["class_pred"].shape))
    print("[INFO] stock_probs shape:", tuple(out["stock_probs"].shape))
    print("[INFO] flow_probs shape:", tuple(out["flow_probs"].shape))
    print("[INFO] future_states shape:", tuple(out["future_states"].shape))
    print("[INFO] rollout_feedback_logits shape:", tuple(out["rollout_feedback_logits"].shape))
    print("[INFO] rollout_feedback_probs shape:", tuple(out["rollout_feedback_probs"].shape))
    print("[INFO] h_rel shape:", tuple(out["h_rel"].shape))
    print("[INFO] c_hist shape:", tuple(out["c_hist"].shape))
    print("[INFO] cross_stream_attn shape:", tuple(out["cross_stream_attn"].shape))
    print("[INFO] pooling_weights shape:", tuple(out["pooling_weights"].shape))
    print("[INFO] num step_cross_attn:", len(out["step_cross_attn"]))
    print("[INFO] num step_gate_values:", len(out["step_gate_values"]))
