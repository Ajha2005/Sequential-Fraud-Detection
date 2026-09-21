"""Phase 4: LSTM fraud head. Encodes each cardholder's transaction history
and outputs a fraud probability at *every* step (many-to-many), so that
Phase 6's accuracy-vs-N curve can measure performance as a function of how
many transactions the model has seen so far."""

from __future__ import annotations

import torch
import torch.nn as nn

from sfraud.models.layers import InputEncoder


class LSTMFraudModel(nn.Module):
    def __init__(self, n_numeric: int, vocab_sizes: list[int], cfg: dict):
        super().__init__()
        self.encoder = InputEncoder(n_numeric, vocab_sizes, cfg["embedding_dim"])
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            bidirectional=cfg["bidirectional"],
            batch_first=True,
        )
        out_dim = cfg["hidden_size"] * (2 if cfg["bidirectional"] else 1)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_num, x_cat)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.head(out).squeeze(-1)  # (B, T)
        return logits
