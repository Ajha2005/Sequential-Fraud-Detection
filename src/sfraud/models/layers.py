"""Shared building block: turns the per-step categorical index tensor into
learned embeddings and concatenates them with the numeric features. Used by
the LSTM, Transformer, and autoencoder so all three consume identical
per-step representations (Section 6: "Embedding layers -- inside the main
model")."""

from __future__ import annotations

import torch
import torch.nn as nn


class CategoricalEmbedder(nn.Module):
    def __init__(self, vocab_sizes: list[int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim, padding_idx=0) for vocab_size in vocab_sizes]
        )
        self.output_dim = len(vocab_sizes) * embedding_dim

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, T, n_cat) -> (B, T, n_cat * embedding_dim)
        embedded = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=-1)


class InputEncoder(nn.Module):
    """Concatenates numeric features with categorical embeddings at each step."""

    def __init__(self, n_numeric: int, vocab_sizes: list[int], embedding_dim: int):
        super().__init__()
        self.cat_embedder = CategoricalEmbedder(vocab_sizes, embedding_dim)
        self.output_dim = n_numeric + self.cat_embedder.output_dim

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_emb = self.cat_embedder(x_cat)
        return torch.cat([x_num, cat_emb], dim=-1)
