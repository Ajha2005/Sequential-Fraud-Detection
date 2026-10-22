"""Phase 3 entry point: train + evaluate the XGBoost and Logistic Regression
baselines on the flat (non-sequential) engineered features, using the same
card-level train/val/test split as the sequence models."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from sfraud.datasets.torch_dataset import load_feature_meta
from sfraud.evaluation.metrics import binary_classification_metrics
from sfraud.models.baseline import build_logreg_pipeline, build_xgboost
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir, resolve_path
from sfraud.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["seed"])

    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(processed_dir / "features_flat.parquet")

    meta = load_feature_meta(cfg)
    numeric_cols = meta["numeric_features"]
    categorical_idx_cols = [f"{c}_idx" for c in meta["categorical_features"]]
    feature_cols = numeric_cols + categorical_idx_cols

    splits = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    X = {s: splits[s][feature_cols] for s in splits}
    y = {s: splits[s]["isFraud"].to_numpy() for s in splits}

    n_pos, n_neg = y["train"].sum(), len(y["train"]) - y["train"].sum()
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    logger.info("Train rows=%d, fraud=%d, scale_pos_weight=%.2f", len(y["train"]), n_pos, scale_pos_weight)

    results = {}

    # --- XGBoost ---
    xgb = build_xgboost(cfg, scale_pos_weight)
    xgb.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    results["xgboost"] = {
        split: binary_classification_metrics(y[split], xgb.predict_proba(X[split])[:, 1])
        for split in ["val", "test"]
    }
    joblib.dump(xgb, checkpoints_dir() / "xgboost.joblib")

    # --- Logistic Regression ---
    logreg = build_logreg_pipeline(cfg, numeric_cols, categorical_idx_cols)
    logreg.fit(X["train"], y["train"])
    results["logreg"] = {
        split: binary_classification_metrics(y[split], logreg.predict_proba(X[split])[:, 1])
        for split in ["val", "test"]
    }
    joblib.dump(logreg, checkpoints_dir() / "logreg.joblib")

    with open(metrics_dir() / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    for model_name, split_metrics in results.items():
        for split, m in split_metrics.items():
            logger.info(
                "%-8s %-5s AUC-ROC=%.4f  PR-AUC=%.4f  precision=%.3f  recall=%.3f  acc=%.4f",
                model_name, split, m["auc_roc"], m["pr_auc"], m["precision"], m["recall"], m["accuracy"],
            )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
