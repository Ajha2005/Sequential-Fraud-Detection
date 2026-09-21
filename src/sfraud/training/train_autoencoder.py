"""Phase 5 entry point: train the autoencoder anomaly head on normal-only
(isFraud == 0) transactions, then score every transaction (all splits, all
labels) so fault_fraud.py can combine the anomaly score with the fraud head's
probability."""

from __future__ import annotations

import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sfraud.datasets.torch_dataset import FlatFeatureDataset, load_feature_meta
from sfraud.models.autoencoder import TransactionAutoencoder
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir, resolve_path
from sfraud.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acfg = cfg["autoencoder"]
    numeric_cols = cfg["numeric_features"]
    categorical_cols = cfg["categorical_features"]
    meta = load_feature_meta(cfg)
    vocab_sizes = [meta["categorical_vocab_sizes"][c] for c in categorical_cols]

    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "features_flat.parquet")

    train_normal = df[(df["split"] == "train") & (df["isFraud"] == 0)]
    train_ds = FlatFeatureDataset(train_normal, numeric_cols, categorical_cols)
    train_loader = DataLoader(train_ds, batch_size=acfg["batch_size"], shuffle=True)

    val_normal = df[(df["split"] == "val") & (df["isFraud"] == 0)]
    val_ds = FlatFeatureDataset(val_normal, numeric_cols, categorical_cols)
    val_loader = DataLoader(val_ds, batch_size=acfg["batch_size"], shuffle=False)

    model = TransactionAutoencoder(len(numeric_cols), vocab_sizes, acfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=acfg["learning_rate"])

    logger.info(
        "Autoencoder training on %d normal rows (train), validating on %d normal rows",
        len(train_ds), len(val_ds),
    )

    best_val_loss = float("inf")
    ckpt_path = checkpoints_dir() / "autoencoder_best.pt"
    history = []

    for epoch in range(1, acfg["epochs"] + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            x_num, x_cat = batch["x_num"].to(device), batch["x_cat"].to(device)
            optimizer.zero_grad()
            loss = model.per_row_loss(x_num, x_cat).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_num)
            total_n += len(x_num)
        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                x_num, x_cat = batch["x_num"].to(device), batch["x_cat"].to(device)
                val_losses.append(model.per_row_loss(x_num, x_cat))
            val_loss = torch.cat(val_losses).mean().item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("epoch %02d/%d  train_loss=%.4f  val_loss=%.4f", epoch, acfg["epochs"], train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Score every transaction in every split (fraud included) for the
    # fault/fraud quadrant logic downstream.
    full_ds = FlatFeatureDataset(df, numeric_cols, categorical_cols)
    full_loader = DataLoader(full_ds, batch_size=acfg["batch_size"], shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in full_loader:
            x_num, x_cat = batch["x_num"].to(device), batch["x_cat"].to(device)
            loss = model.per_row_loss(x_num, x_cat)
            scores.append(loss.cpu())
    anomaly_score = torch.cat(scores).numpy()

    out = df[["TransactionID", "card_id", "split", "isFraud", "synthetic_fault_flag"]].copy()
    out["anomaly_score"] = anomaly_score
    out.to_parquet(processed_dir / "anomaly_scores.parquet", index=False)

    with open(metrics_dir() / "autoencoder_metrics.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss}, f, indent=2)

    logger.info("Saved anomaly scores for %d transactions to %s", len(out), processed_dir / "anomaly_scores.parquet")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
