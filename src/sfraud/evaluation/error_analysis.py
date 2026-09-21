"""Phase 6: stratify LSTM performance by how much history a user had, and
characterize false positives/negatives -- including how many of them land in
the 'Fault' quadrant from Phase 5 (the blueprint's specific hypothesis: a
chunk of "errors" are actually benign anomalies the fraud head reasonably
got confused by, not model failures)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sfraud.evaluation.compare import _sequence_predictions_with_position
from sfraud.utils.config import load_config, metrics_dir, resolve_path

logger = logging.getLogger(__name__)


def stratify_by_history_length(df: pd.DataFrame, bins: list[int]) -> pd.DataFrame:
    """df must have one row per (card, transaction) with columns card_id, n_seen, isFraud, fraud_prob.
    A card's "history length" = the most transactions it reaches in this split (capped by truncation)."""
    card_length = df.groupby("card_id")["n_seen"].max().rename("history_length")
    df = df.join(card_length, on="card_id")

    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    df["length_bucket"] = pd.cut(df["history_length"], bins=bins, labels=labels, right=False)

    rows = []
    for bucket, g in df.groupby("length_bucket", observed=True):
        y_true, y_prob = g["isFraud"].to_numpy(), g["fraud_prob"].to_numpy()
        auc = roc_auc_score(y_true, y_prob) if 0 < y_true.sum() < len(y_true) else np.nan
        acc = ((y_prob >= 0.5).astype(int) == y_true).mean()
        rows.append(
            {
                "length_bucket": str(bucket),
                "n_cards": g["card_id"].nunique(),
                "n_transactions": len(g),
                "fraud_rate": float(y_true.mean()),
                "auc_roc": auc,
                "accuracy": float(acc),
            }
        )
    return pd.DataFrame(rows)


def characterize_errors(df: pd.DataFrame, quadrant_df: pd.DataFrame | None, threshold: float = 0.5) -> dict:
    df = df.copy()
    df["pred"] = (df["fraud_prob"] >= threshold).astype(int)
    fp = df[(df["isFraud"] == 0) & (df["pred"] == 1)]
    fn = df[(df["isFraud"] == 1) & (df["pred"] == 0)]
    tp = df[(df["isFraud"] == 1) & (df["pred"] == 1)]
    tn = df[(df["isFraud"] == 0) & (df["pred"] == 0)]

    def profile(g: pd.DataFrame) -> dict:
        if len(g) == 0:
            return {"n": 0}
        out = {
            "n": int(len(g)),
            "mean_n_seen": float(g["n_seen"].mean()),
            "mean_amount": float(g["TransactionAmt"].mean()) if "TransactionAmt" in g else None,
            "fraction_synthetic_fault": float(g["synthetic_fault_flag"].mean()),
        }
        if quadrant_df is not None:
            merged = g.merge(quadrant_df[["TransactionID", "quadrant"]], on="TransactionID", how="left")
            out["quadrant_breakdown"] = merged["quadrant"].value_counts(dropna=False).to_dict()
        return out

    return {
        "false_positive": profile(fp),
        "false_negative": profile(fn),
        "true_positive": profile(tp),
        "true_negative": profile(tn),
        "note": "quadrant_breakdown shows where each error group lands in the Phase 5 fault/fraud quadrant "
                "(e.g. false positives concentrated in 'Fault' would support: the model isn't wrong so much as "
                "flagging real benign anomalies that happen to share isFraud=0).",
    }


def run(cfg: dict | None = None, model_type: str = "lstm") -> dict:
    cfg = cfg or load_config()
    processed_dir = resolve_path(cfg["data"]["processed_dir"])

    df = _sequence_predictions_with_position(cfg, model_type)

    strat = stratify_by_history_length(df, cfg["evaluation"]["history_length_bins"])
    strat.to_csv(metrics_dir() / "stratify_by_history_length.csv", index=False)
    logger.info("Stratification by history length:\n%s", strat.to_string(index=False))

    quadrant_path = processed_dir / "fault_fraud_quadrants.parquet"
    quadrant_df = pd.read_parquet(quadrant_path) if quadrant_path.exists() else None

    error_profile = characterize_errors(df, quadrant_df)
    with open(metrics_dir() / "error_analysis.json", "w") as f:
        json.dump(error_profile, f, indent=2, default=str)
    logger.info("Error analysis: %s", json.dumps(error_profile, indent=2, default=str))

    return {"stratification": strat, "error_profile": error_profile}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
