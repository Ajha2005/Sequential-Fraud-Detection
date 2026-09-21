"""Phase 2: sequence construction pipeline.

Turns the flat, per-transaction feature table into fixed-length, padded
per-card sequences ready for a PyTorch DataLoader, plus a flat table for the
tabular baselines (Phase 3) that shares the exact same engineered features
and the exact same train/val/test card membership, so the baseline-vs-deep-
model comparison in Phase 6 is apples to apples.

Padding/truncation convention (documented once, here, since every later
plot and metric depends on it):
  - a card's transactions are sorted chronologically and RIGHT-padded: real
    transactions occupy positions 0..len-1, padding fills len..max_seq_len-1.
    Sequence position therefore equals "number of transactions seen so far"
    (N = position + 1), which is exactly what Phase 6's accuracy-vs-N curve
    and Phase 4's per-step fraud head need.
  - a card with more than max_seq_len transactions is truncated to its
    *first* max_seq_len transactions (evaluation only ever looks at
    positions up to max_seq_len anyway; dropping the tail keeps "position
    N" meaning the same thing for every card).

Split convention: cards (not rows) are assigned to train/val/test, ordered
by each card's first transaction time and cut into contiguous blocks. This
satisfies the blueprint's "split by user/time, not randomly per-row" in one
pass -- no card's rows are split across sets, and train broadly precedes
val/test in time.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from sfraud.features.engineering import engineer_features
from sfraud.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    scfg = cfg["sequence"]
    first_seen = df.groupby("card_id")["TransactionDT"].min().sort_values()
    card_order = first_seen.index.to_numpy()
    n = len(card_order)

    n_train = int(round(n * scfg["train_frac"]))
    n_val = int(round(n * scfg["val_frac"]))
    train_cards = set(card_order[:n_train])
    val_cards = set(card_order[n_train:n_train + n_val])
    test_cards = set(card_order[n_train + n_val:])

    def split_of(cid):
        if cid in train_cards:
            return "train"
        if cid in val_cards:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["card_id"].map(split_of)
    return df


def _fit_scaler(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    train_vals = df.loc[df["split"] == "train", numeric_cols]
    mean = train_vals.mean()
    std = train_vals.std().replace(0, 1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict()}


def _apply_scaler(df: pd.DataFrame, numeric_cols: list[str], scaler: dict) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        df[col] = (df[col] - scaler["mean"][col]) / scaler["std"][col]
    return df


def _build_split_arrays(
    df_split: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], max_seq_len: int
) -> dict[str, np.ndarray]:
    cat_idx_cols = [f"{c}_idx" for c in categorical_cols]
    card_ids = df_split["card_id"].unique()
    n_cards = len(card_ids)
    n_num = len(numeric_cols)
    n_cat = len(cat_idx_cols)

    X_num = np.zeros((n_cards, max_seq_len, n_num), dtype=np.float32)
    X_cat = np.zeros((n_cards, max_seq_len, n_cat), dtype=np.int64)
    y = np.zeros((n_cards, max_seq_len), dtype=np.float32)
    fault = np.zeros((n_cards, max_seq_len), dtype=np.float32)
    mask = np.zeros((n_cards, max_seq_len), dtype=np.float32)
    amt_raw = np.zeros((n_cards, max_seq_len), dtype=np.float32)
    txn_ids = np.full((n_cards, max_seq_len), -1, dtype=np.int64)
    txn_dt = np.full((n_cards, max_seq_len), -1, dtype=np.int64)

    grouped = df_split.sort_values("TransactionDT").groupby("card_id", sort=False)
    for i, card_id in enumerate(card_ids):
        g = grouped.get_group(card_id).iloc[:max_seq_len]
        L = len(g)
        X_num[i, :L] = g[numeric_cols].to_numpy(dtype=np.float32)
        X_cat[i, :L] = g[cat_idx_cols].to_numpy(dtype=np.int64)
        y[i, :L] = g["isFraud"].to_numpy(dtype=np.float32)
        fault[i, :L] = g["synthetic_fault_flag"].to_numpy(dtype=np.float32)
        mask[i, :L] = 1.0
        amt_raw[i, :L] = g["TransactionAmt"].to_numpy(dtype=np.float32)
        txn_ids[i, :L] = g["TransactionID"].to_numpy(dtype=np.int64)
        txn_dt[i, :L] = g["TransactionDT"].to_numpy(dtype=np.int64)

    return {
        "X_num": X_num, "X_cat": X_cat, "y": y, "fault": fault, "mask": mask,
        "amt_raw": amt_raw, "txn_ids": txn_ids, "txn_dt": txn_dt,
        "card_ids": np.array(card_ids, dtype=object),
    }


def build_sequences(cfg: dict | None = None) -> dict:
    from sfraud.data.merge import load_merged_dataset

    cfg = cfg or load_config()
    numeric_cols = cfg["numeric_features"]
    categorical_cols = cfg["categorical_features"]
    max_seq_len = cfg["sequence"]["max_seq_len"]

    df = load_merged_dataset(cfg)
    df, vocabs = engineer_features(df, categorical_cols)
    df = assign_splits(df, cfg)

    scaler = _fit_scaler(df, numeric_cols)
    df = _apply_scaler(df, numeric_cols, scaler)

    out_dir = resolve_path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_cols = [
        "TransactionID", "card_id", "TransactionDT", "TransactionAmt", "isFraud",
        "synthetic_fault_flag", "is_synthetic", "split", "txn_index_in_card",
        *numeric_cols, *categorical_cols, *[f"{c}_idx" for c in categorical_cols],
    ]
    df[flat_cols].to_parquet(out_dir / "features_flat.parquet", index=False)

    split_summary = {}
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split]
        arrays = _build_split_arrays(df_split, numeric_cols, categorical_cols, max_seq_len)
        np.savez_compressed(out_dir / f"sequences_{split}.npz", **arrays)
        split_summary[split] = {
            "n_cards": int(arrays["mask"].shape[0]),
            "n_transactions": int(arrays["mask"].sum()),
            "fraud_rate": float(arrays["y"][arrays["mask"] == 1].mean()),
        }
        logger.info("split=%s -> %s", split, split_summary[split])

    meta = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "categorical_vocab_sizes": {c: len(vocabs[c]) for c in categorical_cols},
        "vocabs": vocabs,
        "max_seq_len": max_seq_len,
        "numeric_scaler": scaler,
        "split_summary": split_summary,
    }
    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved sequences + feature_meta.json to %s", out_dir)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_sequences()
