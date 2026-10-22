"""Phase 6: compare every model on the same held-out test set, against the
published-benchmark reference points from Section 2 of the blueprint, and
compute the headline "accuracy/AUC as a function of N transactions seen"
confidence curve for every model (sequence models get this for free per-step;
baselines get it too by joining their per-row predictions back to each
transaction's position in its card's history, via `txn_index_in_card`)."""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from sfraud.datasets.torch_dataset import SequenceDataset, load_feature_meta
from sfraud.evaluation.metrics import binary_classification_metrics
from sfraud.evaluation.predict import flatten_predictions, get_full_predictions, load_trained_model
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir, resolve_path

logger = logging.getLogger(__name__)

# Section 2 of the blueprint: published/external reference points. These are
# not computed from our run -- they anchor the comparison chart so our
# numbers can be read in context.
PUBLISHED_BENCHMARKS = {
    "Random / majority-class baseline": {"auc_roc": 0.50, "source": "sanity floor"},
    "Strong supervised baseline (published, low)": {"auc_roc": 0.90, "source": "published range 0.90-0.93"},
    "Strong supervised baseline (published, high)": {"auc_roc": 0.93, "source": "published range 0.90-0.93"},
    "Kaggle competition winners (ceiling)": {"auc_roc": 0.945, "source": "published range 0.945-0.950, large ensembles"},
}


def _baseline_predictions_with_position(cfg: dict, model_name: str) -> pd.DataFrame:
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "features_flat.parquet")
    test_df = df[df["split"] == "test"].copy()

    meta = load_feature_meta(cfg)
    numeric_cols = meta["numeric_features"]
    categorical_idx_cols = [f"{c}_idx" for c in meta["categorical_features"]]
    feature_cols = numeric_cols + categorical_idx_cols

    model = joblib.load(checkpoints_dir() / f"{model_name}.joblib")
    test_df["fraud_prob"] = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["n_seen"] = test_df["txn_index_in_card"] + 1
    return test_df[["TransactionID", "card_id", "n_seen", "isFraud", "synthetic_fault_flag", "fraud_prob"]]


def _sequence_predictions_with_position(cfg: dict, model_type: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, model_type, device)
    ds = SequenceDataset("test", cfg)
    preds = get_full_predictions(model, ds, device)
    return flatten_predictions(preds)


def get_test_predictions(cfg: dict) -> dict[str, pd.DataFrame]:
    return {
        "xgboost": _baseline_predictions_with_position(cfg, "xgboost"),
        "logreg": _baseline_predictions_with_position(cfg, "logreg"),
        "lstm": _sequence_predictions_with_position(cfg, "lstm"),
        "transformer": _sequence_predictions_with_position(cfg, "transformer"),
    }


def confidence_curve(
    preds: dict[str, pd.DataFrame], max_seq_len: int, min_samples: int = 20, min_positives: int = 3
) -> pd.DataFrame:
    """Capped at max_seq_len for every model: the sequence models never see a
    position beyond it (sequences.py truncates there), so leaving it uncapped
    for the baselines (which use the untruncated flat table) would compare
    them over different populations at the tail."""
    rows = []
    for model_name, df in preds.items():
        df = df[df["n_seen"] <= max_seq_len]
        for n_seen, g in df.groupby("n_seen"):
            if len(g) < min_samples:
                continue
            y_true, y_prob = g["isFraud"].to_numpy(), g["fraud_prob"].to_numpy()
            auc = np.nan
            if y_true.sum() >= min_positives and y_true.sum() < len(y_true):
                from sklearn.metrics import roc_auc_score

                auc = roc_auc_score(y_true, y_prob)
            acc = ((y_prob >= 0.5).astype(int) == y_true).mean()
            rows.append(
                {"model": model_name, "n_seen": int(n_seen), "n_samples": len(g), "auc_roc": auc, "accuracy": acc}
            )
    return pd.DataFrame(rows)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    preds = get_test_predictions(cfg)

    comparison = {}
    for model_name, df in preds.items():
        comparison[model_name] = binary_classification_metrics(df["isFraud"], df["fraud_prob"])
    comparison["published_benchmarks"] = PUBLISHED_BENCHMARKS

    with open(metrics_dir() / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    meta_path = resolve_path(cfg["data"]["processed_dir"]) / "feature_meta.json"
    with open(meta_path) as f:
        max_seq_len = json.load(f)["max_seq_len"]
    curve_df = confidence_curve(preds, max_seq_len)
    curve_df.to_csv(metrics_dir() / "confidence_curve.csv", index=False)

    for model_name in preds:
        m = comparison[model_name]
        logger.info("%-12s test AUC-ROC=%.4f  PR-AUC=%.4f", model_name, m["auc_roc"], m["pr_auc"])

    return {"comparison": comparison, "confidence_curve": curve_df, "predictions": preds}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
