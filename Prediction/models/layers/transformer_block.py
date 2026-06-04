#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_causal_attn_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a boolean causal attention mask for MultiheadAttention.

    Returns:
        attn_mask: [L, L], True means blocked position.
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask


def _to_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert valid mask [B, L] with 1 valid / 0 invalid into
    key_padding_mask [B, L] with True for padding.
    """
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"mask must be [B,L], got shape={tuple(mask.shape)}")
    return ~(mask > 0)


class PositionwiseFeedForward(nn.Module):
    """
    Standard Transformer FFN block.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with optional key padding mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.use_pre_norm = bool(use_pre_norm)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            hidden_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,L,D], got shape={tuple(x.shape)}")

        key_padding_mask = _to_key_padding_mask(mask)

        residual = x
        x_in = self.norm1(x) if self.use_pre_norm else x
        attn_out, attn_weights = self.self_attn(
            query=x_in,
            key=x_in,
            value=x_in,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = residual + self.dropout1(attn_out)
        if not self.use_pre_norm:
            x = self.norm1(x)

        residual = x
        x_in = self.norm2(x) if self.use_pre_norm else x
        ff_out = self.ffn(x_in)
        x = residual + self.dropout2(ff_out)
        if not self.use_pre_norm:
            x = self.norm2(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        if return_attn:
            return x, attn_weights
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block with:
        - causal self-attention
        - optional cross-attention over encoder memory
        - position-wise FFN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_cross_attention: bool = True,
        use_pre_norm: bool = True,
    ) -> None:
        super().__init__()

        self.use_cross_attention = bool(use_cross_attention)
        self.use_pre_norm = bool(use_pre_norm)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        else:
            self.cross_attn = None

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) if self.use_cross_attention else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            hidden_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Args:
            x          : [B, L, D]
            self_mask  : [B, L], 1 valid, 0 invalid
            memory     : [B, S, D]
            memory_mask: [B, S], 1 valid, 0 invalid
            causal     : whether to apply causal self-attention mask
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B,L,D], got shape={tuple(x.shape)}")

        B, L, _ = x.shape
        self_key_padding_mask = _to_key_padding_mask(self_mask)
        memory_key_padding_mask = _to_key_padding_mask(memory_mask) if memory_mask is not None else None
        causal_mask = build_causal_attn_mask(L, device=x.device) if causal else None

        attn_info = {}

        # Self-attention
        residual = x
        x_in = self.norm1(x) if self.use_pre_norm else x
        self_attn_out, self_attn_weights = self.self_attn(
            query=x_in,
            key=x_in,
            value=x_in,
            attn_mask=causal_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = residual + self.dropout1(self_attn_out)
        if not self.use_pre_norm:
            x = self.norm1(x)

        if return_attn:
            attn_info["self_attn"] = self_attn_weights

        # Cross-attention
        if self.use_cross_attention:
            if memory is None:
                raise ValueError("memory is required when use_cross_attention=True")
            if memory.ndim != 3:
                raise ValueError(f"memory must be [B,S,D], got shape={tuple(memory.shape)}")

            residual = x
            x_in = self.norm2(x) if self.use_pre_norm else x
            cross_attn_out, cross_attn_weights = self.cross_attn(
                query=x_in,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=return_attn,
                average_attn_weights=True,
            )
            x = residual + self.dropout2(cross_attn_out)
            if not self.use_pre_norm:
                x = self.norm2(x)

            if return_attn:
                attn_info["cross_attn"] = cross_attn_weights

        # FFN
        residual = x
        x_in = self.norm3(x) if self.use_pre_norm else x
        ff_out = self.ffn(x_in)
        x = residual + self.dropout3(ff_out)
        if not self.use_pre_norm:
            x = self.norm3(x)

        if self_mask is not None:
            x = x * self_mask.unsqueeze(-1).to(x.dtype)

        if return_attn:
            return x, attn_info
        return x


if __name__ == "__main__":
    B, S, L, D = 4, 15, 6, 256
    memory = torch.randn(B, S, D)
    x = torch.randn(B, L, D)

    memory_mask = torch.ones(B, S)
    memory_mask[0, 12:] = 0
    self_mask = torch.ones(B, L)

    enc = TransformerEncoderBlock(
        d_model=D,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        attn_dropout=0.1,
        use_pre_norm=True,
    )
    dec = TransformerDecoderBlock(
        d_model=D,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        attn_dropout=0.1,
        use_cross_attention=True,
        use_pre_norm=True,
    )

    mem_out, enc_attn = enc(memory, mask=memory_mask, return_attn=True)
    dec_out, dec_attn = dec(
        x,
        self_mask=self_mask,
        memory=mem_out,
        memory_mask=memory_mask,
        causal=True,
        return_attn=True,
    )

    print("[INFO] encoder output shape:", tuple(mem_out.shape))
    print("[INFO] decoder output shape:", tuple(dec_out.shape))
    print("[INFO] encoder attn shape:", None if enc_attn is None else tuple(enc_attn.shape))
    print("[INFO] decoder self attn shape:", None if dec_attn["self_attn"] is None else tuple(dec_attn["self_attn"].shape))
    print("[INFO] decoder cross attn shape:", None if dec_attn["cross_attn"] is None else tuple(dec_attn["cross_attn"].shape))