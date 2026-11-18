"""Review of the Fault quadrant on real data (no "fault" ground truth exists).

Phase 5 labels a transaction "Fault" when the fraud model gives it a low fraud
probability but the autoencoder gives it a high anomaly score. On real data
nothing says whether those transactions are benign, so this module does two
things that do not need a label:

  1. MEASURES, over *all* Fault rows, whether the anomaly scores look driven by
     extreme feature values (the largest standardized feature value per row,
     versus Normal rows), which features most often drive them, how the product
     / amount mix differs from Normal, how concentrated the rows are on a few
     cards, and how the fraud rate changes with the score.
  2. DRAWS a stratified random sample across the whole score range (several rows
     from each of five bands) with the context a reviewer needs, plus blank
     `reviewer_label` / `reviewer_note` columns for a manual look.

It cannot prove the Fault quadrant is benign; it characterizes it.

`draft_labels` adds a rule-based DRAFT reading of the sampled rows (benign-looking
/ suspicious / unclear) so a reviewer has a starting point. The labels are
heuristics over the fields available -- they are not ground truth and are meant
to be confirmed or overridden by a person.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sfraud.utils.config import load_config, metrics_dir, resolve_path, samples_dir

logger = logging.getLogger(__name__)

BAND_EDGES = [0.0, 0.5, 0.8, 0.95, 0.99, 1.0]  # quantile edges of the Fault rows' anomaly scores
BAND_LABELS = ["0-50%", "50-80%", "80-95%", "95-99%", "99-100%"]
CONTEXT_COLUMNS = ["ProductCD", "email_domain", "device_type", "card_type", "card_category"]
SAMPLE_COLUMNS = [
    "band", "TransactionID", "card_id", "n_seen", "TransactionAmt", "ProductCD", "email_domain", "device_type",
    "card_type", "fraud_prob", "anomaly_score", "isFraud", "max_abs_z",
    "top1_feature", "top1_z", "top2_feature", "top2_z", "top3_feature", "top3_z",
]


def extremeness(flat: pd.DataFrame, numeric_cols: list[str], ids: pd.Series, index: pd.Index) -> pd.DataFrame:
    """For each transaction: the largest |z| among the (train-standardized)
    numeric features, and its three most extreme features with their signed z.
    `flat` must be indexed by TransactionID."""
    X = flat.loc[ids, numeric_cols].to_numpy(dtype=np.float32)
    A = np.abs(X)
    top = np.argsort(-A, axis=1)[:, :3]
    names = np.array(numeric_cols)
    out = pd.DataFrame({"max_abs_z": A.max(axis=1)}, index=index)
    for k in range(min(3, top.shape[1])):
        out[f"top{k + 1}_feature"] = names[top[:, k]]
        out[f"top{k + 1}_z"] = np.take_along_axis(X, top[:, [k]], axis=1)[:, 0].round(1)
    return out


def assign_bands(scores: pd.Series) -> pd.Series:
    """Five bands by the rank of the anomaly score within the Fault rows
    (rank-based, so ties cannot make the band edges collide)."""
    pct = scores.rank(pct=True, method="first")
    return pd.cut(pct, bins=BAND_EDGES, labels=BAND_LABELS, include_lowest=True)


# Behavioural features: if one of these is the top-driving feature, the row is unusual *for its card*.
BEHAVIOURAL_PREFIXES = ("amt_", "log_amt", "log_time_delta", "stab_", "txn_")
NEAR_THRESHOLD_MARGIN = 0.035  # "just under the fraud threshold" = within this margin of it


def draft_labels(sample: pd.DataFrame, fraud_threshold: float) -> pd.DataFrame:
    """Rule-based DRAFT labels for a sampled set of Fault rows. Heuristics, not ground truth.

    suspicious      labelled fraud; OR the top-driving feature is behavioural (unusual amount /
                    timing / category for this card); OR the fraud probability is within
                    NEAR_THRESHOLD_MARGIN of the fraud threshold (a near-miss of the fraud model).
    benign-looking  not suspicious, AND an established card (n_seen >= 4), AND a low fraud
                    probability (< 0.15), AND the extremeness sits only in anonymised columns.
                    Means "no red flag in the available fields", NOT "verified benign".
    unclear         everything else (notably first/second transactions, which have no history).
    """
    out = sample.copy()
    sig = out.groupby(["top1_feature", "top1_z"])["TransactionID"].transform("size")
    behavioural = out["top1_feature"].astype(str).str.startswith(BEHAVIOURAL_PREFIXES)
    near = out["fraud_prob"] >= fraud_threshold - NEAR_THRESHOLD_MARGIN
    suspicious = (out["isFraud"] == 1) | behavioural | near
    benign = (~suspicious) & (out["n_seen"] >= 4) & (out["fraud_prob"] < 0.15)
    out["reviewer_label"] = np.where(suspicious, "suspicious", np.where(benign, "benign-looking", "unclear"))

    notes = []
    for i, r in out.iterrows():
        parts = [f"{r['ProductCD']}, ${r['TransactionAmt']:,.2f}".replace(".00", ""), f"n_seen {int(r['n_seen'])}",
                 f"driver {r['top1_feature']} (z {r['top1_z']})"]
        if sig.loc[i] > 1:
            parts.append(f"same extreme {r['top1_feature']} value as {int(sig.loc[i]) - 1} other sampled row(s)")
        if r["isFraud"] == 1:
            parts.append("labelled fraud")
        if behavioural.loc[i]:
            parts.append("behavioural driver (unusual for this card)")
        if near.loc[i]:
            parts.append(f"fraud prob {r['fraud_prob']:.3f} just under threshold {fraud_threshold:.3f}")
        if int(r["n_seen"]) == 1:
            parts.append("first transaction on card (no history)")
        if r["max_abs_z"] < 3:
            parts.append("no extreme numeric value (score likely from categorical fields)")
        notes.append("; ".join(parts))
    out["reviewer_note"] = notes
    out["label_source"] = "draft (rule-based; not ground truth)"
    return out


def run(
    cfg: dict | None = None,
    processed_dir: Path | None = None,
    out_sample: Path | None = None,
    out_summary: Path | None = None,
    per_band: int = 12,
    seed: int = 0,
    normal_n: int = 20000,
) -> dict:
    cfg = cfg or load_config()
    processed_dir = Path(processed_dir) if processed_dir else resolve_path(cfg["data"]["processed_dir"])
    out_sample = Path(out_sample) if out_sample else samples_dir() / "fault_review_sample.csv"
    out_summary = Path(out_summary) if out_summary else metrics_dir() / "fault_review_summary.json"

    q = pd.read_parquet(processed_dir / "fault_fraud_quadrants.parquet")
    flat = pd.read_parquet(processed_dir / "features_flat.parquet").set_index("TransactionID")
    numeric_cols = json.load(open(processed_dir / "feature_meta.json"))["numeric_features"]
    ctx_cols = ["TransactionID", *CONTEXT_COLUMNS]
    ctx = pd.read_parquet(processed_dir / "merged.parquet", columns=ctx_cols).set_index("TransactionID")

    fault = q[q["quadrant"] == "Fault"].copy()
    if fault.empty:
        raise ValueError("No transactions in the Fault quadrant; run Phase 5 first")
    normal_all = q[q["quadrant"] == "Normal"]
    normal = normal_all.sample(min(normal_n, len(normal_all)), random_state=seed).copy()

    fault = fault.join(extremeness(flat, numeric_cols, fault["TransactionID"], fault.index)).join(ctx, on="TransactionID")
    normal = normal.join(extremeness(flat, numeric_cols, normal["TransactionID"], normal.index)).join(ctx, on="TransactionID")

    def share(s: pd.Series, t: float) -> float:
        return float((s > t).mean())

    summary = {
        "n_fault_rows": int(len(fault)),
        "n_distinct_cards_in_fault": int(fault["card_id"].nunique()),
        "top10_cards_share_of_fault_rows": float(fault["card_id"].value_counts().head(10).sum() / len(fault)),
        "fraud_rate_fault": float(fault["isFraud"].mean()),
        "fraud_rate_normal_sample": float(normal["isFraud"].mean()),
        "max_abs_z": {
            "fault_median": float(fault["max_abs_z"].median()), "normal_median": float(normal["max_abs_z"].median()),
            "fault_p90": float(fault["max_abs_z"].quantile(0.9)), "normal_p90": float(normal["max_abs_z"].quantile(0.9)),
            **{f"share_fault_gt_{t}": share(fault["max_abs_z"], t) for t in (5, 10, 50)},
            **{f"share_normal_gt_{t}": share(normal["max_abs_z"], t) for t in (5, 10, 50)},
        },
        "amount": {
            "fault_median": float(fault["TransactionAmt"].median()), "normal_median": float(normal["TransactionAmt"].median()),
            "fault_p90": float(fault["TransactionAmt"].quantile(0.9)), "normal_p90": float(normal["TransactionAmt"].quantile(0.9)),
        },
        "most_common_top_feature_in_fault": fault["top1_feature"].value_counts().head(10).to_dict(),
        "ProductCD_share_fault": fault["ProductCD"].value_counts(normalize=True).round(3).to_dict(),
        "ProductCD_share_normal": normal["ProductCD"].value_counts(normalize=True).round(3).to_dict(),
    }

    fault["band"] = assign_bands(fault["anomaly_score"])
    bands = fault.groupby("band", observed=True).agg(
        rows=("TransactionID", "size"), score_min=("anomaly_score", "min"), score_max=("anomaly_score", "max"),
        fraud_rate=("isFraud", "mean"), median_amount=("TransactionAmt", "median"),
        median_max_abs_z=("max_abs_z", "median"), share_maxz_gt10=("max_abs_z", lambda s: (s > 10).mean()),
    ).round(3)
    summary["bands"] = bands.reset_index().astype({"band": str}).to_dict(orient="records")

    # Stratified random sample (explicit per-band sampling: groupby.apply drops the group column under pandas 3).
    parts = [g.sample(min(per_band, len(g)), random_state=seed) for _, g in fault.groupby("band", observed=True)]
    sample = pd.concat(parts).sort_values("anomaly_score", ascending=False)
    sample = sample[[c for c in SAMPLE_COLUMNS if c in sample.columns]].copy()
    sample["reviewer_label"] = ""  # fill in: benign-looking / suspicious / unclear
    sample["reviewer_note"] = ""

    out_sample.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_sample, index=False)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    logger.info("Fault-quadrant review summary:\n%s", json.dumps({k: v for k, v in summary.items() if k != "bands"}, indent=2, default=float))
    logger.info("By anomaly-score band inside the Fault quadrant:\n%s", bands.to_string())
    logger.info("Wrote %d sampled rows to %s and the summary to %s", len(sample), out_sample, out_summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
