"""PyTorch Dataset/DataLoader over the padded sequences built by
sfraud.data.sequences. Each item is one cardholder's full (padded) history."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sfraud.utils.config import load_config, resolve_path


class SequenceDataset(Dataset):
    def __init__(self, split: str, cfg: dict | None = None):
        cfg = cfg or load_config()
        processed_dir = resolve_path(cfg["data"]["processed_dir"])
        npz = np.load(processed_dir / f"sequences_{split}.npz", allow_pickle=True)
        self.X_num = torch.tensor(npz["X_num"], dtype=torch.float32)
        self.X_cat = torch.tensor(npz["X_cat"], dtype=torch.long)
        self.y = torch.tensor(npz["y"], dtype=torch.float32)
        self.fault = torch.tensor(npz["fault"], dtype=torch.float32)
        self.mask = torch.tensor(npz["mask"], dtype=torch.float32)
        self.amt_raw = torch.tensor(npz["amt_raw"], dtype=torch.float32)
        self.txn_ids = torch.tensor(npz["txn_ids"], dtype=torch.long)
        self.txn_dt = torch.tensor(npz["txn_dt"], dtype=torch.long)
        self.card_ids = npz["card_ids"]

    def __len__(self) -> int:
        return self.X_num.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_num": self.X_num[idx],
            "x_cat": self.X_cat[idx],
            "y": self.y[idx],
            "fault": self.fault[idx],
            "mask": self.mask[idx],
            "amt_raw": self.amt_raw[idx],
            "txn_ids": self.txn_ids[idx],
            "txn_dt": self.txn_dt[idx],
            "card_idx": idx,
        }


def load_feature_meta(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    with open(processed_dir / "feature_meta.json") as f:
        return json.load(f)


class FlatFeatureDataset(Dataset):
    """Per-transaction (non-sequential) view of the engineered features, used
    by the autoencoder anomaly head. Rows come straight from
    features_flat.parquet, so it is trivial to filter (e.g. normal-only for
    training) or keep unfiltered (for scoring every transaction)."""

    def __init__(self, df, numeric_cols: list[str], categorical_cols: list[str]):
        cat_idx_cols = [f"{c}_idx" for c in categorical_cols]
        self.transaction_ids = df["TransactionID"].to_numpy()
        self.X_num = torch.tensor(df[numeric_cols].to_numpy(), dtype=torch.float32)
        self.X_cat = torch.tensor(df[cat_idx_cols].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return self.X_num.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x_num": self.X_num[idx], "x_cat": self.X_cat[idx], "transaction_id": self.transaction_ids[idx]}


def get_dataloaders(cfg: dict | None = None, batch_size: int | None = None) -> dict[str, DataLoader]:
    cfg = cfg or load_config()
    bs = batch_size or cfg["lstm"]["batch_size"]
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = SequenceDataset(split, cfg)
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=(split == "train"), drop_last=False)
    return loaders
