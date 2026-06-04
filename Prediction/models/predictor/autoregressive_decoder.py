#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both:
#   1) python -m models.predictor.autoregressive_decoder
#   2) python models/predictor/autoregressive_decoder.py
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.predictor.rollout_cell import RolloutCell


class FutureConditionEmbedding(nn.Module):
    """
    Build per-step future condition embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_year_tokens: int = 64,
        max_recency_tokens: int = 64,
        base_year: int = 2000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.max_year_tokens = int(max_year_tokens)
        self.max_recency_tokens = int(max_recency_tokens)
        self.base_year = int(base_year)

        self.year_embedding = nn.Embedding(self.max_year_tokens, d_model)
        self.recency_embedding = nn.Embedding(self.max_recency_tokens, d_model)

        self.year_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.recency_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.year_norm = nn.LayerNorm(d_model)
        self.recency_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.year_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.recency_embedding.weight, mean=0.0, std=0.02)

    def _normalize_year_ids(self, year_ids: torch.Tensor) -> torch.Tensor:
        year_idx = year_ids - self.base_year
        year_idx = torch.clamp(year_idx, min=0, max=self.max_year_tokens - 1)
        return year_idx

    def _normalize_recency_ids(self, recency_ids: torch.Tensor) -> torch.Tensor:
        recency_idx = torch.clamp(recency_ids, min=0, max=self.max_recency_tokens - 1)
        return recency_idx

    def forward(
        self,
        future_year_ids: torch.Tensor,
        future_recency_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if future_year_ids.ndim != 2:
            raise ValueError(f"future_year_ids must be [B,F], got {tuple(future_year_ids.shape)}")
        if future_recency_ids.ndim != 2:
            raise ValueError(
                f"future_recency_ids must be [B,F], got {tuple(future_recency_ids.shape)}"
            )
        if future_year_ids.shape != future_recency_ids.shape:
            raise ValueError(
                f"future_year_ids and future_recency_ids shape mismatch: "
                f"{tuple(future_year_ids.shape)} vs {tuple(future_recency_ids.shape)}"
            )

        year_idx = self._normalize_year_ids(future_year_ids)
        recency_idx = self._normalize_recency_ids(future_recency_ids)

        year_embed = self.year_embedding(year_idx)
        year_embed = self.year_norm(self.year_proj(year_embed))

        recency_embed = self.recency_embedding(recency_idx)
        recency_embed = self.recency_norm(self.recency_proj(recency_embed))

        return {
            "year_embed": year_embed,
            "recency_embed": recency_embed,
        }


class InternalFeedbackHead(nn.Module):
    """
    Fallback feedback head used only when no external shared decision head is supplied.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be [B,D], got {tuple(z.shape)}")
        return self.mlp(z)


class HARPAutoregressiveDecoder(nn.Module):
    """
    True step-wise autoregressive decoder for HARP.

    Important behavior:
        - If prev_class_probs is provided, it is treated as an external
          shifted previous-class sequence for teacher forcing / scheduled sampling.
        - If prev_class_probs is None, the decoder performs pure rollout:
            step 0 uses uniform start
            step k>0 uses the model's own previous prediction p_{k-1}
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 4,
        future_steps: int = 6,
        future_year_start: int = 2015,
        use_selective_gate: bool = True,
        max_year_tokens: int = 64,
        max_recency_tokens: int = 64,
        base_year: int = 2000,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.num_classes = int(num_classes)
        self.future_steps = int(future_steps)
        self.future_year_start = int(future_year_start)

        self.init_state = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.init_state_norm = nn.LayerNorm(d_model)

        self.condition_embedding = FutureConditionEmbedding(
            d_model=d_model,
            max_year_tokens=max_year_tokens,
            max_recency_tokens=max_recency_tokens,
            base_year=base_year,
            dropout=dropout,
        )

        self.rollout_cell = RolloutCell(
            d_model=d_model,
            num_classes=num_classes,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_selective_gate=use_selective_gate,
        )

        self.internal_feedback_head = InternalFeedbackHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=d_model,
            dropout=dropout,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def _build_uniform_start_probs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, self.num_classes),
            fill_value=1.0 / float(self.num_classes),
            device=device,
            dtype=dtype,
        )

    def _build_future_year_ids(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        years = torch.arange(
            self.future_year_start,
            self.future_year_start + self.future_steps,
            device=device,
            dtype=torch.long,
        )
        return years.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _build_future_recency_ids(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        recency = torch.arange(
            self.future_steps,
            device=device,
            dtype=torch.long,
        )
        return recency.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _init_z0(self, c_hist: torch.Tensor) -> torch.Tensor:
        if c_hist.ndim != 2:
            raise ValueError(f"c_hist must be [B,D], got {tuple(c_hist.shape)}")
        z0 = self.init_state_norm(self.init_state(c_hist))
        return z0

    def _predict_feedback_logits(
        self,
        z_step: torch.Tensor,
        feedback_head: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if feedback_head is None:
            return self.internal_feedback_head(z_step)

        if hasattr(feedback_head, "forward_step"):
            logits = feedback_head.forward_step(z_step)
        else:
            logits = feedback_head(z_step)

        if logits.ndim != 2:
            raise ValueError(
                f"feedback head must return [B,C] for step input, got {tuple(logits.shape)}"
            )
        return logits

    def forward(
        self,
        h_rel: torch.Tensor,
        c_hist: torch.Tensor,
        history_mask: torch.Tensor,
        prev_class_probs: Optional[torch.Tensor] = None,
        future_year_ids: Optional[torch.Tensor] = None,
        future_recency_ids: Optional[torch.Tensor] = None,
        feedback_head: Optional[nn.Module] = None,
        return_analysis: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if h_rel.ndim != 3:
            raise ValueError(f"h_rel must be [B,T,D], got {tuple(h_rel.shape)}")
        if c_hist.ndim != 2:
            raise ValueError(f"c_hist must be [B,D], got {tuple(c_hist.shape)}")
        if history_mask.ndim != 2:
            raise ValueError(f"history_mask must be [B,T], got {tuple(history_mask.shape)}")

        B, T_hist, D = h_rel.shape
        if c_hist.shape != (B, D):
            raise ValueError(
                f"c_hist shape mismatch: expected {(B, D)}, got {tuple(c_hist.shape)}"
            )
        if history_mask.shape != (B, T_hist):
            raise ValueError(
                f"history_mask shape mismatch: expected {(B, T_hist)}, got {tuple(history_mask.shape)}"
            )

        device = h_rel.device
        dtype = h_rel.dtype

        use_external_prev = prev_class_probs is not None
        if use_external_prev:
            expected_shape = (B, self.future_steps, self.num_classes)
            if prev_class_probs.shape != expected_shape:
                raise ValueError(
                    f"prev_class_probs must be {expected_shape}, got {tuple(prev_class_probs.shape)}"
                )
            prev_class_probs = prev_class_probs.to(dtype=dtype, device=device)

        if future_year_ids is None:
            future_year_ids = self._build_future_year_ids(
                batch_size=B,
                device=device,
            )
        if future_recency_ids is None:
            future_recency_ids = self._build_future_recency_ids(
                batch_size=B,
                device=device,
            )

        if future_year_ids.shape != (B, self.future_steps):
            raise ValueError(
                f"future_year_ids must be {(B, self.future_steps)}, got {tuple(future_year_ids.shape)}"
            )
        if future_recency_ids.shape != (B, self.future_steps):
            raise ValueError(
                f"future_recency_ids must be {(B, self.future_steps)}, got {tuple(future_recency_ids.shape)}"
            )

        cond = self.condition_embedding(
            future_year_ids=future_year_ids,
            future_recency_ids=future_recency_ids,
        )
        year_embed = cond["year_embed"]           # [B, F, D]
        recency_embed = cond["recency_embed"]     # [B, F, D]

        z_prev = self._init_z0(c_hist)            # [B, D]
        p_prev_model = None
        p_start = self._build_uniform_start_probs(
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        future_states: List[torch.Tensor] = []
        feedback_logits_list: List[torch.Tensor] = []
        feedback_probs_list: List[torch.Tensor] = []

        step_cross_attn: List[torch.Tensor] = []
        step_gate_values: List[torch.Tensor] = []

        for step_idx in range(self.future_steps):
            if use_external_prev:
                p_prev = prev_class_probs[:, step_idx, :]
            else:
                if step_idx == 0:
                    p_prev = p_start
                else:
                    if p_prev_model is None:
                        raise RuntimeError(
                            "Pure rollout expected p_prev_model for step_idx > 0, but got None."
                        )
                    p_prev = p_prev_model

            step_out = self.rollout_cell(
                z_prev=z_prev,
                p_prev=p_prev,
                c_hist=c_hist,
                h_rel=h_rel,
                history_mask=history_mask,
                year_embed=year_embed[:, step_idx, :],
                recency_embed=recency_embed[:, step_idx, :],
                return_analysis=return_analysis,
            )

            z_next = step_out["z_next"]                       # [B, D]
            logits_k = self._predict_feedback_logits(
                z_step=z_next,
                feedback_head=feedback_head,
            )                                                # [B, C]
            probs_k = F.softmax(logits_k, dim=-1)            # [B, C]

            future_states.append(z_next.unsqueeze(1))
            feedback_logits_list.append(logits_k.unsqueeze(1))
            feedback_probs_list.append(probs_k.unsqueeze(1))

            if return_analysis:
                if "cross_attn" in step_out:
                    step_cross_attn.append(step_out["cross_attn"])
                if "gate_values" in step_out:
                    step_gate_values.append(step_out["gate_values"])

            z_prev = z_next
            p_prev_model = probs_k

        future_states_tensor = torch.cat(future_states, dim=1)            # [B, F, D]
        future_states_tensor = self.out_norm(future_states_tensor)

        feedback_logits_tensor = torch.cat(feedback_logits_list, dim=1)   # [B, F, C]
        feedback_probs_tensor = torch.cat(feedback_probs_list, dim=1)     # [B, F, C]

        outputs: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "future_states": future_states_tensor,
            "feedback_logits": feedback_logits_tensor,
            "feedback_probs": feedback_probs_tensor,
            "future_year_ids": future_year_ids,
            "future_recency_ids": future_recency_ids,
        }

        if return_analysis:
            outputs["step_cross_attn"] = step_cross_attn
            outputs["step_gate_values"] = step_gate_values

        return outputs


if __name__ == "__main__":
    B, T_hist, D, C = 4, 15, 256, 4
    F_steps = 6

    h_rel = torch.randn(B, T_hist, D)
    c_hist = torch.randn(B, D)
    history_mask = torch.ones(B, T_hist)
    history_mask[0, 12:] = 0

    prev_class_probs = torch.softmax(torch.randn(B, F_steps, C), dim=-1)

    decoder = HARPAutoregressiveDecoder(
        d_model=D,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        num_classes=C,
        future_steps=F_steps,
        future_year_start=2015,
        use_selective_gate=True,
        max_year_tokens=64,
        max_recency_tokens=64,
        base_year=2000,
    )

    out_gt = decoder(
        h_rel=h_rel,
        c_hist=c_hist,
        history_mask=history_mask,
        prev_class_probs=prev_class_probs,
        feedback_head=None,
        return_analysis=True,
    )
    print("[INFO] GT mode future_states shape:", tuple(out_gt["future_states"].shape))
    print("[INFO] GT mode feedback_probs shape:", tuple(out_gt["feedback_probs"].shape))

    out_pure = decoder(
        h_rel=h_rel,
        c_hist=c_hist,
        history_mask=history_mask,
        prev_class_probs=None,
        feedback_head=None,
        return_analysis=True,
    )
    print("[INFO] PURE mode future_states shape:", tuple(out_pure["future_states"].shape))
    print("[INFO] PURE mode feedback_probs shape:", tuple(out_pure["feedback_probs"].shape))
