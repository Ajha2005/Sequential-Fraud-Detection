"""Bootstrap confidence intervals for differences between models.

Resamples test CARDS with replacement (a card's transactions are correlated, so
they are resampled together) and recomputes AUC-ROC and PR-AUC for each model
on every resample. The reported interval for "a minus b" is the 2.5th-97.5th
percentile of the paired differences.

What the interval covers: sampling noise in the test set only. The models are
fixed -- training randomness (seed, early-stopping point) is NOT resampled -- so
it understates total uncertainty. Reads data/processed/common_predictions.parquet
written by Phase 6b (every model scored on the same rows).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sfraud.utils.config import load_config, metrics_dir, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_PAIRS = [
    ("xgboost_plus", "xgboost_plus_nohist"),
    ("xgboost_plus", "xgboost"),
    ("blend", "xgboost_plus"),
    ("xgboost_plus", "lstm"),
]


def bootstrap_differences(
    test: pd.DataFrame, pairs: list[tuple[str, str]] = DEFAULT_PAIRS, n_resamples: int = 200, seed: int = 0
) -> list[dict]:
    """test: one row per transaction with `card_id`, `isFraud` and one score column per model."""
    test = test.reset_index(drop=True)
    groups = test.groupby("card_id").indices
    cards = np.array(list(groups))
    rng = np.random.default_rng(seed)
    diffs = {p: {"auc": [], "pr": []} for p in pairs}
    for _ in range(n_resamples):
        picked = rng.choice(cards, len(cards), replace=True)
        s = test.iloc[np.concatenate([groups[c] for c in picked])]
        y = s["isFraud"].to_numpy()
        for a, b in pairs:
            diffs[(a, b)]["auc"].append(roc_auc_score(y, s[a]) - roc_auc_score(y, s[b]))
            diffs[(a, b)]["pr"].append(average_precision_score(y, s[a]) - average_precision_score(y, s[b]))
    out = []
    for (a, b), v in diffs.items():
        row = {"a": a, "b": b}
        for m in ("auc", "pr"):
            lo, hi = np.percentile(v[m], [2.5, 97.5])
            row[f"{m}_mean"] = round(float(np.mean(v[m])), 4)
            row[f"{m}_ci"] = [round(float(lo), 4), round(float(hi), 4)]
        out.append(row)
    return out


def run(cfg: dict | None = None, predictions: Path | None = None, out: Path | None = None,
        n_resamples: int = 200, seed: int = 0) -> dict:
    cfg = cfg or load_config()
    predictions = Path(predictions) if predictions else resolve_path(cfg["data"]["processed_dir"]) / "common_predictions.parquet"
    out = Path(out) if out else metrics_dir() / "bootstrap_differences.json"
    d = pd.read_parquet(predictions)
    test = d[d["split"] == "test"]
    result = {
        "note": "Bootstrap over test CARDS (a card's transactions are resampled together) on the common test rows. "
                "The interval reflects test-sampling noise only (models fixed; no retraining or seed variation).",
        "n_resamples": n_resamples,
        "differences": bootstrap_differences(test, n_resamples=n_resamples, seed=seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    for r in result["differences"]:
        logger.info("%s minus %s: AUC %+.4f %s | PR-AUC %+.4f %s", r["a"], r["b"], r["auc_mean"], r["auc_ci"], r["pr_mean"], r["pr_ci"])
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
