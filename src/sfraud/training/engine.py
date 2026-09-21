"""Shared train/eval loop for the sequence models (LSTM + Transformer).
Both output per-step logits of shape (B, T); loss and metrics are always
computed only over valid (non-padded) positions via `mask`."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def masked_bce_with_logits(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    per_elem = loss_fn(logits, y)
    per_elem = per_elem * mask
    return per_elem.sum() / mask.sum().clamp(min=1.0)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
    pos_weight: torch.Tensor,
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = masked_bce_with_logits(logits, y, mask, pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        n = mask.sum().item()
        total_loss += loss.item() * n
        total_count += n
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_flat(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (y_true, y_prob) over valid (non-padded) positions only."""
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits)

        valid = mask.bool()
        all_probs.append(probs[valid].cpu().numpy())
        all_y.append(y[valid].cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_probs)


def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    y_true, y_prob = predict_flat(model, loader, device)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {"auc_roc": auc, "pr_auc": pr_auc, "n": int(len(y_true))}


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    n_pos, n_total = 0.0, 0.0
    for batch in loader:
        y, mask = batch["y"], batch["mask"]
        n_pos += (y * mask).sum().item()
        n_total += mask.sum().item()
    n_neg = n_total - n_pos
    return torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
