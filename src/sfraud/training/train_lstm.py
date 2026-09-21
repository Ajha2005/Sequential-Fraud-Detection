"""Phase 4 entry point: train the LSTM fraud head."""

from __future__ import annotations

import json
import logging

import torch

from sfraud.datasets.torch_dataset import get_dataloaders, load_feature_meta
from sfraud.evaluation.metrics import binary_classification_metrics
from sfraud.models.lstm import LSTMFraudModel
from sfraud.training.engine import compute_pos_weight, evaluate_epoch, predict_flat, train_one_epoch
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir
from sfraud.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lcfg = cfg["lstm"]
    loaders = get_dataloaders(cfg, batch_size=lcfg["batch_size"])
    meta = load_feature_meta(cfg)
    vocab_sizes = [meta["categorical_vocab_sizes"][c] for c in cfg["categorical_features"]]
    n_numeric = len(cfg["numeric_features"])

    model = LSTMFraudModel(n_numeric, vocab_sizes, lcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["learning_rate"], weight_decay=lcfg["weight_decay"])
    pos_weight = compute_pos_weight(loaders["train"]).to(device)
    logger.info("LSTM pos_weight=%.2f, params=%d", pos_weight.item(), sum(p.numel() for p in model.parameters()))

    best_val_pr_auc = -1.0
    history = []
    ckpt_path = checkpoints_dir() / "lstm_best.pt"

    for epoch in range(1, lcfg["epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, pos_weight)
        val_metrics = evaluate_epoch(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        logger.info(
            "epoch %02d/%d  train_loss=%.4f  val_auc=%.4f  val_pr_auc=%.4f",
            epoch, lcfg["epochs"], train_loss, val_metrics["auc_roc"], val_metrics["pr_auc"],
        )
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    results = {"history": history}
    for split in ["val", "test"]:
        y_true, y_prob = predict_flat(model, loaders[split], device)
        results[split] = binary_classification_metrics(y_true, y_prob)
        logger.info(
            "FINAL lstm %-5s AUC-ROC=%.4f  PR-AUC=%.4f  precision=%.3f  recall=%.3f",
            split, results[split]["auc_roc"], results[split]["pr_auc"],
            results[split]["precision"], results[split]["recall"],
        )

    with open(metrics_dir() / "lstm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
