"""Runs a trained sequence model over a full split and returns both the raw
padded (B, T) arrays and a flattened one-row-per-real-transaction DataFrame.
This is the shared backbone for Phase 5 (fault/fraud quadrant), Phase 6
(accuracy-vs-N, error analysis) and Phase 7 (per-user traces, quadrant
scatter, confusion matrix) -- every one of them needs model output joined
back to TransactionID/card_id/position, so it is computed once, here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sfraud.datasets.torch_dataset import SequenceDataset, load_feature_meta
from sfraud.models.lstm import LSTMFraudModel
from sfraud.models.transformer import TransformerFraudModel
from sfraud.utils.config import checkpoints_dir


def load_trained_model(cfg: dict, model_type: str, device: torch.device) -> torch.nn.Module:
    meta = load_feature_meta(cfg)
    vocab_sizes = [meta["categorical_vocab_sizes"][c] for c in meta["categorical_features"]]
    n_numeric = len(meta["numeric_features"])

    if model_type == "lstm":
        model = LSTMFraudModel(n_numeric, vocab_sizes, cfg["lstm"])
        ckpt = checkpoints_dir() / "lstm_best.pt"
    elif model_type == "transformer":
        model = TransformerFraudModel(n_numeric, vocab_sizes, cfg["transformer"], meta["max_seq_len"])
        ckpt = checkpoints_dir() / "transformer_best.pt"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()  # a "load trained model for inference" helper must hand back eval mode --
    return model  # forgetting this silently leaves dropout active (bit us in the attention heatmap: see tests)


@torch.no_grad()
def get_full_predictions(model: torch.nn.Module, dataset: SequenceDataset, device: torch.device, batch_size: int = 128) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    probs_chunks = []
    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_cat = batch["x_cat"].to(device)
        logits = model(x_num, x_cat)
        probs_chunks.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(probs_chunks, axis=0)  # (n_cards, T)
    return {
        "probs": probs,
        "y": dataset.y.numpy(),
        "mask": dataset.mask.numpy(),
        "fault": dataset.fault.numpy(),
        "txn_ids": dataset.txn_ids.numpy(),
        "txn_dt": dataset.txn_dt.numpy(),
        "amt_raw": dataset.amt_raw.numpy(),
        "card_ids": dataset.card_ids,
    }


def flatten_predictions(preds: dict) -> pd.DataFrame:
    n_cards, T = preds["mask"].shape
    card_id_grid = np.repeat(preds["card_ids"][:, None], T, axis=1)
    position_grid = np.tile(np.arange(T), (n_cards, 1))  # 0-based -> N = position + 1

    mask = preds["mask"].astype(bool)
    df = pd.DataFrame(
        {
            "TransactionID": preds["txn_ids"][mask],
            "card_id": card_id_grid[mask],
            "position": position_grid[mask],
            "n_seen": position_grid[mask] + 1,
            "TransactionDT": preds["txn_dt"][mask],
            "TransactionAmt": preds["amt_raw"][mask],
            "isFraud": preds["y"][mask].astype(int),
            "synthetic_fault_flag": preds["fault"][mask].astype(int),
            "fraud_prob": preds["probs"][mask],
        }
    )
    return df
