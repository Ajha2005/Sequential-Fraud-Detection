"""Phase 3 design / Phase 5 delivery: combine the fraud head's probability
with the autoencoder's anomaly score into the fault/fraud/normal/edge-case
quadrant from Section 3 of the blueprint, then pull a sample from the
'Fault' quadrant for qualitative review.

Uses only the val+test (held-out) splits: the fraud head has seen train
labels directly, so its train-split probabilities are optimistic and would
make the quadrant assignment look better than it really is.

`synthetic_fault_flag` is available here only because the underlying data is
our synthetic stand-in (see synthetic.py) -- it lets us additionally report
an empirical recovery rate (what fraction of known injected faults the
anomaly score actually flags as high-anomaly) as a sanity check on the
*method*. With real Kaggle data this ground truth would not exist and the
qualitative sample would be reviewed by hand instead, exactly as the
blueprint specifies.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import torch

from sfraud.datasets.torch_dataset import SequenceDataset
from sfraud.evaluation.predict import flatten_predictions, get_full_predictions, load_trained_model
from sfraud.utils.config import load_config, metrics_dir, resolve_path, samples_dir

logger = logging.getLogger(__name__)

QUADRANT_LOW_LOW = "Normal"
QUADRANT_LOW_HIGH = "Fault"
QUADRANT_HIGH_HIGH = "Fraud"
QUADRANT_HIGH_LOW = "Edge case"


def categorize(row: pd.Series) -> str:
    if row["fraud_high"] and row["anomaly_high"]:
        return QUADRANT_HIGH_HIGH
    if row["fraud_high"] and not row["anomaly_high"]:
        return QUADRANT_HIGH_LOW
    if not row["fraud_high"] and row["anomaly_high"]:
        return QUADRANT_LOW_HIGH
    return QUADRANT_LOW_LOW


def build_fraud_prob_table(cfg: dict, model_type: str = "lstm") -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, model_type, device)

    frames = []
    for split in ["val", "test"]:
        ds = SequenceDataset(split, cfg)
        preds = get_full_predictions(model, ds, device)
        flat = flatten_predictions(preds)
        flat["split"] = split
        frames.append(flat)
    return pd.concat(frames, ignore_index=True)


def run(cfg: dict | None = None, model_type: str = "lstm") -> pd.DataFrame:
    cfg = cfg or load_config()
    ffcfg = cfg["fault_fraud"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])

    fraud_df = build_fraud_prob_table(cfg, model_type)
    anomaly_df = pd.read_parquet(processed_dir / "anomaly_scores.parquet")
    anomaly_df = anomaly_df[anomaly_df["split"].isin(["val", "test"])]

    df = fraud_df.merge(
        anomaly_df[["TransactionID", "anomaly_score"]], on="TransactionID", how="inner"
    )

    # "High anomaly" is relative to what NORMAL transactions look like in
    # this holdout set (Section 3 / config comment), not an arbitrary global cut.
    normal_scores = df.loc[df["isFraud"] == 0, "anomaly_score"]
    anomaly_threshold = float(normal_scores.quantile(ffcfg["anomaly_score_percentile"] / 100.0))
    fraud_threshold = ffcfg["fraud_prob_threshold"]

    df["fraud_high"] = df["fraud_prob"] >= fraud_threshold
    df["anomaly_high"] = df["anomaly_score"] >= anomaly_threshold
    df["quadrant"] = df.apply(categorize, axis=1)

    quadrant_counts = df["quadrant"].value_counts().to_dict()

    # Qualitative validation (only possible because this is synthetic data --
    # see module docstring): does the "Fault" quadrant actually recover the
    # injected benign anomalies, and does "Fraud" recover real fraud?
    fault_recall = float(
        df.loc[df["synthetic_fault_flag"] == 1, "quadrant"].eq(QUADRANT_LOW_HIGH).mean()
    ) if (df["synthetic_fault_flag"] == 1).any() else float("nan")
    fraud_recall = float(
        df.loc[df["isFraud"] == 1, "quadrant"].eq(QUADRANT_HIGH_HIGH).mean()
    ) if (df["isFraud"] == 1).any() else float("nan")
    normal_precision = float(
        df.loc[(df["isFraud"] == 0) & (df["synthetic_fault_flag"] == 0), "quadrant"].eq(QUADRANT_LOW_LOW).mean()
    )

    summary = {
        "model_used_for_fraud_prob": model_type,
        "fraud_prob_threshold": fraud_threshold,
        "anomaly_score_threshold": anomaly_threshold,
        "anomaly_percentile_basis": ffcfg["anomaly_score_percentile"],
        "quadrant_counts": quadrant_counts,
        "qualitative_validation_on_synthetic_ground_truth": {
            "note": "Only measurable because the underlying data is synthetic (see synthetic.py). "
                    "With real Kaggle data, 'Fault' quadrant samples must be reviewed manually instead.",
            "fraction_of_true_synthetic_faults_landing_in_Fault_quadrant": fault_recall,
            "fraction_of_true_fraud_landing_in_Fraud_quadrant": fraud_recall,
            "fraction_of_true_normal_landing_in_Normal_quadrant": normal_precision,
        },
    }

    with open(metrics_dir() / "fault_fraud_quadrant.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Fault/fraud quadrant summary: %s", json.dumps(summary, indent=2))

    # Sample the 'Fault' quadrant for the qualitative write-up (Phase 5).
    fault_sample = (
        df[df["quadrant"] == QUADRANT_LOW_HIGH]
        .sort_values("anomaly_score", ascending=False)
        .head(20)
        [["TransactionID", "card_id", "n_seen", "TransactionAmt", "fraud_prob", "anomaly_score",
          "isFraud", "synthetic_fault_flag"]]
    )
    fault_sample.to_csv(samples_dir() / "fault_quadrant_review_sample.csv", index=False)

    df.to_parquet(processed_dir / "fault_fraud_quadrants.parquet", index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
