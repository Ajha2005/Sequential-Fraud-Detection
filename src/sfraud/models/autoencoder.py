"""Phase 5: autoencoder anomaly head.

Operates per-transaction (not per-sequence) on the same engineered features
as the baselines -- numeric fields already include the causal amt_dev /
amt_zscore / time_delta "deviation from personal baseline" signal from
engineering.py, so a compact autoencoder over these plus categorical fields
still captures personalized behavioral drift without needing its own
recurrent state.

Mixed reconstruction target: numeric features are reconstructed with MSE,
each categorical field is reconstructed as a classification head (softmax
over that field's vocabulary) trained with cross-entropy. This matters for
the fault/fraud story in Section 3 -- a "new device" or "switched email
domain" is a categorical surprise, not a numeric one, and cross-entropy on
the categorical head is what actually penalizes it. Reconstructing raw
embeddings with MSE would not.

Trained on normal-only (isFraud == 0) rows; the combined per-row loss at
inference time is the anomaly score used by fault_fraud.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfraud.models.layers import CategoricalEmbedder


class TransactionAutoencoder(nn.Module):
    def __init__(self, n_numeric: int, vocab_sizes: list[int], cfg: dict):
        super().__init__()
        self.n_numeric = n_numeric
        self.vocab_sizes = vocab_sizes
        embedding_dim = cfg.get("embedding_dim", 8)

        self.cat_embedder = CategoricalEmbedder(vocab_sizes, embedding_dim)
        input_dim = n_numeric + self.cat_embedder.output_dim

        hidden = cfg["hidden_size"]
        latent = cfg["latent_size"]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
        )
        self.numeric_head = nn.Linear(hidden, n_numeric)
        self.categorical_heads = nn.ModuleList([nn.Linear(hidden, v) for v in vocab_sizes])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, cat_emb], dim=-1)
        z = self.encoder(x)
        h = self.decoder_trunk(z)
        numeric_recon = self.numeric_head(h)
        categorical_logits = [head(h) for head in self.categorical_heads]
        return numeric_recon, categorical_logits

    def per_row_loss(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Combined reconstruction loss per row (no reduction) -- this is the anomaly score."""
        numeric_recon, categorical_logits = self.forward(x_num, x_cat)
        numeric_loss = ((numeric_recon - x_num) ** 2).mean(dim=-1)
        cat_loss = torch.zeros_like(numeric_loss)
        for i, logits in enumerate(categorical_logits):
            cat_loss = cat_loss + F.cross_entropy(logits, x_cat[..., i], reduction="none")
        cat_loss = cat_loss / max(len(categorical_logits), 1)
        return numeric_loss + cat_loss
