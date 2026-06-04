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
#   1) python -m models.predictor.decision_head
#   2) python models/predictor/decision_head.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class HARPDecisionHead(nn.Module):
    """
    Unified 4-class decision head for HARP.

    This head supports two modes:

    1) Step mode:
        input  : [B, D]
        output : logits [B, 4]

    2) Sequence mode:
        input  : [B, F, D]
        output : logits [B, F, 4]

    Main task:
        four-class prediction

    Auxiliary outputs:
        stock / flow probabilities derived from the 4-class probabilities

    Class mapping:
        0 -> LL
        1 -> LH
        2 -> HL
        3 -> HH
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.num_classes = int(num_classes)
        if self.num_classes != 4:
            raise ValueError(f"HARPDecisionHead expects num_classes=4, got {num_classes}")

        self.pre_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def class_probs_to_stock_flow(class_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            class_probs: [B, F, 4] or [B, 4]

        Returns:
            stock_probs: [B, F] or [B]
            flow_probs : [B, F] or [B]
        """
        if class_probs.ndim not in {2, 3} or class_probs.shape[-1] != 4:
            raise ValueError(f"class_probs must be [B,4] or [B,F,4], got {tuple(class_probs.shape)}")

        lh = class_probs[..., 1]
        hl = class_probs[..., 2]
        hh = class_probs[..., 3]

        stock_probs = hl + hh
        flow_probs = lh + hh

        return {
            "stock_probs": stock_probs,
            "flow_probs": flow_probs,
        }

    @staticmethod
    def class_logits_to_predictions(class_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert class logits into:
            - class_probs
            - class_pred
            - stock_probs / flow_probs
            - stock_pred / flow_pred

        Supports:
            class_logits [B, 4]
            class_logits [B, F, 4]
        """
        if class_logits.ndim not in {2, 3} or class_logits.shape[-1] != 4:
            raise ValueError(f"class_logits must be [B,4] or [B,F,4], got {tuple(class_logits.shape)}")

        class_probs = F.softmax(class_logits, dim=-1)
        class_pred = torch.argmax(class_logits, dim=-1)

        aux = HARPDecisionHead.class_probs_to_stock_flow(class_probs)
        stock_probs = aux["stock_probs"]
        flow_probs = aux["flow_probs"]

        stock_pred = (class_pred >= 2).to(torch.long)
        flow_pred = torch.remainder(class_pred, 2).to(torch.long)

        return {
            "class_probs": class_probs,
            "class_pred": class_pred,
            "stock_probs": stock_probs,
            "flow_probs": flow_probs,
            "stock_pred": stock_pred,
            "flow_pred": flow_pred,
        }

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal classifier application.

        Supports:
            x [B, D]     -> logits [B, 4]
            x [B, F, D]  -> logits [B, F, 4]
        """
        if x.ndim not in {2, 3}:
            raise ValueError(f"x must be [B,D] or [B,F,D], got {tuple(x.shape)}")

        x = self.pre_norm(x)
        logits = self.classifier(x)
        return logits

    def forward_step(
        self,
        step_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step-wise classification.

        Args:
            step_state: [B, D]

        Returns:
            logits    : [B, 4]
        """
        if step_state.ndim != 2:
            raise ValueError(f"step_state must be [B,D], got {tuple(step_state.shape)}")
        logits = self._compute_logits(step_state)
        if logits.ndim != 2:
            raise RuntimeError(f"Expected step logits [B,4], got {tuple(logits.shape)}")
        return logits

    def forward_sequence(
        self,
        future_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sequence-wise classification.

        Args:
            future_states: [B, F, D]

        Returns:
            {
                "class_logits": [B, F, 4],
                "class_probs":  [B, F, 4],
                "class_pred":   [B, F],
                "stock_probs":  [B, F],
                "flow_probs":   [B, F],
                "stock_pred":   [B, F],
                "flow_pred":    [B, F],
            }
        """
        if future_states.ndim != 3:
            raise ValueError(f"future_states must be [B,F,D], got {tuple(future_states.shape)}")

        class_logits = self._compute_logits(future_states)
        pred_dict = self.class_logits_to_predictions(class_logits)
        pred_dict["class_logits"] = class_logits
        return pred_dict

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Default forward.

        If x is:
            [B, D]    -> returns step-wise outputs
            [B, F, D] -> returns sequence-wise outputs
        """
        if x.ndim == 2:
            class_logits = self.forward_step(x)
            pred_dict = self.class_logits_to_predictions(class_logits)
            pred_dict["class_logits"] = class_logits
            return pred_dict

        if x.ndim == 3:
            return self.forward_sequence(x)

        raise ValueError(f"x must be [B,D] or [B,F,D], got {tuple(x.shape)}")


if __name__ == "__main__":
    B, T_future, D = 4, 6, 256

    step_state = torch.randn(B, D)
    future_states = torch.randn(B, T_future, D)

    head = HARPDecisionHead(
        d_model=D,
        num_classes=4,
        hidden_dim=256,
        dropout=0.1,
        use_layernorm=True,
    )

    step_logits = head.forward_step(step_state)
    print("[INFO] step_logits shape:", tuple(step_logits.shape))

    seq_out = head.forward_sequence(future_states)
    for k, v in seq_out.items():
        print(f"[INFO] seq {k} shape: {tuple(v.shape)}")

    auto_step_out = head(step_state)
    auto_seq_out = head(future_states)
    print("[INFO] auto step class_logits shape:", tuple(auto_step_out["class_logits"].shape))
    print("[INFO] auto seq class_logits shape:", tuple(auto_seq_out["class_logits"].shape))
