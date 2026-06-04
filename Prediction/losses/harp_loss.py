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
#   1) python -m losses.harp_loss
#   2) python losses/harp_loss.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class FocalCrossEntropyLoss(nn.Module):
    """
    Multi-class focal cross entropy with optional class weights,
    label smoothing, and year weighting support at the caller side.

    This implementation returns per-sample / per-time-step losses
    when reduction='none'.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        class_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0,1), got {label_smoothing}")

        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.eps = float(eps)

        if class_weight is not None:
            if class_weight.ndim != 1:
                raise ValueError(
                    f"class_weight must be 1D [num_classes], got shape={tuple(class_weight.shape)}"
                )
            self.register_buffer("class_weight", class_weight.to(torch.float32))
        else:
            self.class_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits : [B, F, C] or [N, C]
            targets: [B, F] or [N]

        Returns:
            loss according to reduction
        """
        if logits.ndim == 3:
            B, F_steps, C = logits.shape
            if targets.shape != (B, F_steps):
                raise ValueError(
                    f"targets shape mismatch: expected {(B, F_steps)}, got {tuple(targets.shape)}"
                )
            logits_flat = logits.reshape(B * F_steps, C)
            targets_flat = targets.reshape(B * F_steps)
            restore_shape = (B, F_steps)
        elif logits.ndim == 2:
            N, C = logits.shape
            if targets.shape != (N,):
                raise ValueError(
                    f"targets shape mismatch: expected {(N,)}, got {tuple(targets.shape)}"
                )
            logits_flat = logits
            targets_flat = targets
            restore_shape = None
        else:
            raise ValueError(f"logits must be [B,F,C] or [N,C], got {tuple(logits.shape)}")

        if torch.any(targets_flat < 0) or torch.any(targets_flat >= C):
            raise ValueError(
                f"targets out of range [0,{C - 1}], "
                f"min={targets_flat.min().item()}, max={targets_flat.max().item()}"
            )

        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)

        if self.label_smoothing > 0.0:
            num_classes = logits_flat.shape[-1]
            smooth_loss = -log_probs.mean(dim=-1)
            ce_loss = (1.0 - self.label_smoothing) * (-target_log_probs) + self.label_smoothing * smooth_loss
        else:
            ce_loss = -target_log_probs

        focal_factor = (1.0 - target_probs).clamp_min(self.eps).pow(self.gamma)
        loss = focal_factor * ce_loss

        if self.class_weight is not None:
            weight = self.class_weight.gather(dim=0, index=targets_flat)
            loss = loss * weight

        if restore_shape is not None:
            loss = loss.reshape(*restore_shape)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal smoothness loss over future class probability trajectories.

    Encourages adjacent future-step probability distributions to avoid
    implausible oscillations, without forcing them to be identical.
    """

    def __init__(
        self,
        p: int = 1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if p not in {1, 2}:
            raise ValueError(f"p must be 1 or 2, got {p}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.p = int(p)
        self.reduction = reduction

    def forward(self, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_probs: [B, F, C]

        Returns:
            smoothness loss over adjacent steps
        """
        if class_probs.ndim != 3:
            raise ValueError(f"class_probs must be [B,F,C], got {tuple(class_probs.shape)}")

        diff = class_probs[:, 1:, :] - class_probs[:, :-1, :]  # [B, F-1, C]
        if self.p == 1:
            loss = diff.abs().mean(dim=-1)  # [B, F-1]
        else:
            loss = diff.pow(2).mean(dim=-1)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class HARPLoss(nn.Module):
    """
    Full loss module for HARP.

    Main objective:
        - focal multi-class cross entropy on 4-class predictions

    Auxiliary objectives:
        - stock BCE on probabilities derived from 4-class outputs
        - flow BCE on probabilities derived from 4-class outputs
        - temporal smoothness on class probability trajectories

    Support:
        - year weighting
        - class weighting
        - full loss breakdown for logging
    """

    def __init__(
        self,
        focal_gamma: float = 1.5,
        class_weight: Optional[torch.Tensor] = None,
        year_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        aux_stock_weight: float = 0.15,
        aux_flow_weight: float = 0.15,
        smoothness_weight: float = 0.02,
        smoothness_p: int = 1,
        bce_reduction: str = "none",
    ) -> None:
        super().__init__()

        self.main_loss_fn = FocalCrossEntropyLoss(
            gamma=focal_gamma,
            class_weight=class_weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )

        self.smooth_loss_fn = TemporalSmoothnessLoss(
            p=smoothness_p,
            reduction="mean",
        )

        self.aux_stock_weight = float(aux_stock_weight)
        self.aux_flow_weight = float(aux_flow_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.bce_reduction = bce_reduction

        if year_weight is not None:
            if year_weight.ndim != 1:
                raise ValueError(
                    f"year_weight must be 1D [future_steps], got shape={tuple(year_weight.shape)}"
                )
            self.register_buffer("year_weight", year_weight.to(torch.float32))
        else:
            self.year_weight = None

    def _apply_year_weight(
        self,
        loss_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply year-wise weights to a [B, F] loss matrix.
        """
        if loss_matrix.ndim != 2:
            raise ValueError(f"loss_matrix must be [B,F], got {tuple(loss_matrix.shape)}")

        if self.year_weight is None:
            return loss_matrix

        if loss_matrix.shape[1] != self.year_weight.shape[0]:
            raise ValueError(
                f"Year dimension mismatch: loss_matrix={tuple(loss_matrix.shape)}, "
                f"year_weight={tuple(self.year_weight.shape)}"
            )

        return loss_matrix * self.year_weight.unsqueeze(0)

    def _compute_aux_bce(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss matrix [B, F].
        """
        if probs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: probs={tuple(probs.shape)}, targets={tuple(targets.shape)}"
            )
        loss = F.binary_cross_entropy(
            probs,
            targets.to(probs.dtype),
            reduction="none",
        )
        return loss

    def forward(
        self,
        class_logits: torch.Tensor,
        class_probs: torch.Tensor,
        labels: torch.Tensor,
        stock_probs: torch.Tensor,
        flow_probs: torch.Tensor,
        stock_targets: torch.Tensor,
        flow_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            class_logits  : [B, F, 4]
            class_probs   : [B, F, 4]
            labels        : [B, F]
            stock_probs   : [B, F]
            flow_probs    : [B, F]
            stock_targets : [B, F]
            flow_targets  : [B, F]

        Returns:
            {
                "loss": scalar,
                "loss_cls": scalar,
                "loss_stock": scalar,
                "loss_flow": scalar,
                "loss_smooth": scalar,
                "loss_cls_matrix": [B, F],
                "loss_stock_matrix": [B, F],
                "loss_flow_matrix": [B, F],
            }
        """
        if class_logits.ndim != 3 or class_logits.shape[-1] != 4:
            raise ValueError(f"class_logits must be [B,F,4], got {tuple(class_logits.shape)}")
        if class_probs.ndim != 3 or class_probs.shape[-1] != 4:
            raise ValueError(f"class_probs must be [B,F,4], got {tuple(class_probs.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"labels must be [B,F], got {tuple(labels.shape)}")
        if stock_probs.ndim != 2 or flow_probs.ndim != 2:
            raise ValueError(
                f"stock_probs / flow_probs must be [B,F], got "
                f"stock_probs={tuple(stock_probs.shape)}, flow_probs={tuple(flow_probs.shape)}"
            )
        if stock_targets.ndim != 2 or flow_targets.ndim != 2:
            raise ValueError(
                f"stock_targets / flow_targets must be [B,F], got "
                f"stock_targets={tuple(stock_targets.shape)}, flow_targets={tuple(flow_targets.shape)}"
            )

        B, F_steps, C = class_logits.shape
        if class_probs.shape != (B, F_steps, C):
            raise ValueError(
                f"class_probs shape mismatch: expected {(B, F_steps, C)}, got {tuple(class_probs.shape)}"
            )
        if labels.shape != (B, F_steps):
            raise ValueError(
                f"labels shape mismatch: expected {(B, F_steps)}, got {tuple(labels.shape)}"
            )
        if stock_probs.shape != (B, F_steps) or flow_probs.shape != (B, F_steps):
            raise ValueError(
                f"aux prob shape mismatch: stock_probs={tuple(stock_probs.shape)}, "
                f"flow_probs={tuple(flow_probs.shape)}, expected {(B, F_steps)}"
            )
        if stock_targets.shape != (B, F_steps) or flow_targets.shape != (B, F_steps):
            raise ValueError(
                f"aux target shape mismatch: stock_targets={tuple(stock_targets.shape)}, "
                f"flow_targets={tuple(flow_targets.shape)}, expected {(B, F_steps)}"
            )

        # Numerical safety guard for large-scale training.
        class_probs = torch.nan_to_num(
            class_probs,
            nan=1.0 / class_probs.shape[-1],
            posinf=1.0,
            neginf=0.0,
        )
        stock_probs = torch.nan_to_num(
            stock_probs,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )
        flow_probs = torch.nan_to_num(
            flow_probs,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

        class_probs = class_probs.clamp(min=1e-6, max=1.0 - 1e-6)
        stock_probs = stock_probs.clamp(min=1e-6, max=1.0 - 1e-6)
        flow_probs = flow_probs.clamp(min=1e-6, max=1.0 - 1e-6)
        
        # Main focal CE loss [B, F]
        loss_cls_matrix = self.main_loss_fn(
            logits=class_logits,
            targets=labels,
        )
        loss_cls_matrix = self._apply_year_weight(loss_cls_matrix)
        loss_cls = loss_cls_matrix.mean()

        # Auxiliary BCE losses [B, F]
        loss_stock_matrix = self._compute_aux_bce(
            probs=stock_probs,
            targets=stock_targets,
        )
        loss_stock_matrix = self._apply_year_weight(loss_stock_matrix)
        loss_stock = loss_stock_matrix.mean()

        loss_flow_matrix = self._compute_aux_bce(
            probs=flow_probs,
            targets=flow_targets,
        )
        loss_flow_matrix = self._apply_year_weight(loss_flow_matrix)
        loss_flow = loss_flow_matrix.mean()

        # Temporal smoothness
        loss_smooth = self.smooth_loss_fn(class_probs)

        total_loss = (
            loss_cls
            + self.aux_stock_weight * loss_stock
            + self.aux_flow_weight * loss_flow
            + self.smoothness_weight * loss_smooth
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_stock": loss_stock,
            "loss_flow": loss_flow,
            "loss_smooth": loss_smooth,
            "loss_cls_matrix": loss_cls_matrix,
            "loss_stock_matrix": loss_stock_matrix,
            "loss_flow_matrix": loss_flow_matrix,
        }


if __name__ == "__main__":
    B, T_future, C = 4, 6, 4

    class_logits = torch.randn(B, T_future, C)
    class_probs = F.softmax(class_logits, dim=-1)
    labels = torch.randint(0, 4, (B, T_future))

    stock_probs = class_probs[..., 2] + class_probs[..., 3]
    flow_probs = class_probs[..., 1] + class_probs[..., 3]

    stock_targets = torch.div(labels, 2, rounding_mode="floor").to(torch.float32)
    flow_targets = torch.remainder(labels, 2).to(torch.float32)

    class_weight = torch.tensor([1.0, 1.2, 1.3, 1.8], dtype=torch.float32)
    year_weight = torch.tensor([0.85, 0.90, 1.00, 1.10, 1.20, 1.35], dtype=torch.float32)

    criterion = HARPLoss(
        focal_gamma=1.5,
        class_weight=class_weight,
        year_weight=year_weight,
        label_smoothing=0.0,
        aux_stock_weight=0.15,
        aux_flow_weight=0.15,
        smoothness_weight=0.02,
        smoothness_p=1,
    )

    out = criterion(
        class_logits=class_logits,
        class_probs=class_probs,
        labels=labels,
        stock_probs=stock_probs,
        flow_probs=flow_probs,
        stock_targets=stock_targets,
        flow_targets=flow_targets,
    )

    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"[INFO] {k}: shape={tuple(v.shape)}")
        else:
            print(f"[INFO] {k}: {v}")