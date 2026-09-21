"""Phase 4 stretch goal: Transformer fraud head with causal self-attention.

Built from a hand-rolled encoder layer (rather than nn.TransformerEncoder)
specifically so attention weights can be returned -- Phase 7's attention
heatmap ("which past transactions drove a given prediction") needs them,
and PyTorch's built-in TransformerEncoder discards them internally.

Causality: a square subsequent mask stops position t from attending to
t+1..T-1, so, combined with the right-padding convention from sequences.py,
a real (unpadded) step's output never depends on padded steps after it --
the same guarantee the LSTM gets for free from being a recurrent left-to-
right scan.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sfraud.models.layers import InputEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CausalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=True
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x, attn_weights


class TransformerFraudModel(nn.Module):
    def __init__(self, n_numeric: int, vocab_sizes: list[int], cfg: dict, max_seq_len: int):
        super().__init__()
        self.encoder = InputEncoder(n_numeric, vocab_sizes, cfg["embedding_dim"])
        self.input_proj = nn.Linear(self.encoder.output_dim, cfg["d_model"])
        self.pos_encoding = PositionalEncoding(cfg["d_model"], max_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                CausalEncoderLayer(cfg["d_model"], cfg["n_heads"], cfg["dim_feedforward"], cfg["dropout"])
                for _ in range(cfg["n_layers"])
            ]
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["d_model"], 1)
        self._causal_mask_cache: dict[int, torch.Tensor] = {}

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T not in self._causal_mask_cache:
            mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            self._causal_mask_cache[T] = mask
        return self._causal_mask_cache[T].to(device)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor, return_attn: bool = False
    ):
        x = self.encoder(x_num, x_cat)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        attn_mask = self._causal_mask(x.size(1), x.device)
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask)
            attn_weights_all.append(attn_weights)

        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)
        if return_attn:
            return logits, attn_weights_all
        return logits
