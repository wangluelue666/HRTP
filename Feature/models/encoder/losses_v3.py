#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE between stock and flow embeddings.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = torch.matmul(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss12 = F.cross_entropy(logits, labels)
        loss21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss12 + loss21)


class MarginAlignedCosineLoss(nn.Module):
    """
    Encourage corresponding ring representations to align,
    but do not force hard collapse.
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        cos = torch.sum(z1 * z2, dim=-1)
        # Penalize only when similarity is below target margin band
        loss = F.relu(self.margin - cos + 1.0)
        return loss.mean()


class DropoutConsistencyLoss(nn.Module):
    """
    Self-consistency regularizer by random feature dropout.
    """

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(F.dropout(z, p=self.dropout_p, training=True), dim=-1)
        z2 = F.normalize(F.dropout(z, p=self.dropout_p, training=True), dim=-1)
        return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


class EncoderLossBundleV3(nn.Module):
    """
    Full V3 loss:
      - view contrastive loss on z_stock / z_flow
      - ring alignment loss on ring1 / ring2
      - consistency loss on z_joint
      - aux cls loss on z_joint -> 4-class cls
      - aux hs loss on z_stock -> hs
      - aux hf loss on z_flow -> hf
    """

    def __init__(
        self,
        temperature: float = 0.2,
        ring_margin: float = 0.2,
        consistency_dropout: float = 0.1,
        weight_view: float = 1.0,
        weight_ring: float = 0.3,
        weight_consistency: float = 0.2,
        weight_cls: float = 1.0,
        weight_hs: float = 0.5,
        weight_hf: float = 0.5,
    ) -> None:
        super().__init__()

        self.view_loss = SymmetricInfoNCELoss(temperature=temperature)
        self.ring_loss = MarginAlignedCosineLoss(margin=ring_margin)
        self.consistency_loss = DropoutConsistencyLoss(dropout_p=consistency_dropout)

        self.weight_view = float(weight_view)
        self.weight_ring = float(weight_ring)
        self.weight_consistency = float(weight_consistency)
        self.weight_cls = float(weight_cls)
        self.weight_hs = float(weight_hs)
        self.weight_hf = float(weight_hf)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        hist_cls: torch.Tensor,
        hist_hs: torch.Tensor,
        hist_hf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_view = self.view_loss(outputs["z_stock"], outputs["z_flow"])

        loss_ring1 = self.ring_loss(outputs["stock_ring1_repr"], outputs["flow_ring1_repr"])
        loss_ring2 = self.ring_loss(outputs["stock_ring2_repr"], outputs["flow_ring2_repr"])
        loss_ring = 0.5 * (loss_ring1 + loss_ring2)

        loss_consistency = self.consistency_loss(outputs["z_joint"])

        loss_cls = F.cross_entropy(outputs["aux_cls_logits"], hist_cls)
        loss_hs = F.cross_entropy(outputs["aux_hs_logits"], hist_hs)
        loss_hf = F.cross_entropy(outputs["aux_hf_logits"], hist_hf)

        total = (
            self.weight_view * loss_view
            + self.weight_ring * loss_ring
            + self.weight_consistency * loss_consistency
            + self.weight_cls * loss_cls
            + self.weight_hs * loss_hs
            + self.weight_hf * loss_hf
        )

        with torch.no_grad():
            cls_acc = (outputs["aux_cls_logits"].argmax(dim=-1) == hist_cls).float().mean()
            hs_acc = (outputs["aux_hs_logits"].argmax(dim=-1) == hist_hs).float().mean()
            hf_acc = (outputs["aux_hf_logits"].argmax(dim=-1) == hist_hf).float().mean()

        return {
            "loss": total,
            "loss_view": loss_view.detach(),
            "loss_ring": loss_ring.detach(),
            "loss_consistency": loss_consistency.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_hs": loss_hs.detach(),
            "loss_hf": loss_hf.detach(),
            "cls_acc": cls_acc.detach(),
            "hs_acc": hs_acc.detach(),
            "hf_acc": hf_acc.detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    bsz, dim = 4, 64
    outputs = {
        "z_stock": torch.randn(bsz, dim),
        "z_flow": torch.randn(bsz, dim),
        "z_joint": torch.randn(bsz, dim),
        "stock_ring1_repr": torch.randn(bsz, dim),
        "flow_ring1_repr": torch.randn(bsz, dim),
        "stock_ring2_repr": torch.randn(bsz, dim),
        "flow_ring2_repr": torch.randn(bsz, dim),
        "aux_cls_logits": torch.randn(bsz, 4),
        "aux_hs_logits": torch.randn(bsz, 2),
        "aux_hf_logits": torch.randn(bsz, 2),
    }
    hist_cls = torch.randint(0, 4, (bsz,))
    hist_hs = torch.randint(0, 2, (bsz,))
    hist_hf = torch.randint(0, 2, (bsz,))

    bundle = EncoderLossBundleV3()
    out = bundle(outputs, hist_cls, hist_hs, hist_hf)
    for k, v in out.items():
        print(k, float(v))