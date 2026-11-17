"""Phase 6b: fair model comparison, validation-chosen blend and thresholds.

Why this exists
  * The sequence models only see each card's first `max_seq_len` transactions,
    while the tabular models score every row, so Phase 6's per-model numbers
    are computed over slightly different transaction sets. Here every model is
    scored on the SAME validation and test rows (the intersection).
  * Probabilities from differently trained models are not on one scale, so
    each is Platt-calibrated on validation before two models are averaged.
    The blend weight is chosen on validation PR-AUC.
  * A fixed 0.5 cut-off is arbitrary. For every model (and the blend) this
    picks thresholds on VALIDATION -- the F1-optimal one, plus the
    highest-precision one reaching each target recall -- and reports what
    they do on TEST.
  * Test AUC / PR-AUC broken down by how much history each prediction had.

Nothing here looks at test labels to choose anything.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from sfraud.datasets.torch_dataset import SequenceDataset, load_feature_meta
from sfraud.evaluation.metrics import binary_classification_metrics
from sfraud.evaluation.predict import flatten_predictions, get_full_predictions, load_trained_model
from sfraud.evaluation.thresholds import (
    apply_platt, best_f1_threshold, fit_platt, operating_point_metrics, threshold_for_recall,
)
from sfraud.features.xgb_features import load_xgb_table
from sfraud.training.train_xgb_plus import predict_proba_best
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir, resolve_path

logger = logging.getLogger(__name__)

MODEL_ORDER = ["logreg", "xgboost", "xgboost_plus_nohist", "xgboost_plus", "lstm", "transformer"]
EVAL_SPLITS = ["val", "test"]


def _xgb_plus_scores(model_name: str, table: pd.DataFrame) -> pd.Series | None:
    path = checkpoints_dir() / f"{model_name}.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)
    cols = model.get_booster().feature_names
    return pd.Series(predict_proba_best(model, table[cols]), index=table.index, name=model_name)


def _flat_baseline_scores(cfg: dict, model_name: str) -> pd.DataFrame | None:
    path = checkpoints_dir() / f"{model_name}.joblib"
    if not path.exists():
        return None
    flat = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "features_flat.parquet")
    flat = flat[flat["split"].isin(EVAL_SPLITS)]
    meta = load_feature_meta(cfg)
    cols = meta["numeric_features"] + [f"{c}_idx" for c in meta["categorical_features"]]
    model = joblib.load(path)
    return pd.DataFrame({"TransactionID": flat["TransactionID"].to_numpy(),
                         model_name: model.predict_proba(flat[cols])[:, 1]})


def _sequence_scores(cfg: dict, model_type: str) -> pd.DataFrame | None:
    ckpt = checkpoints_dir() / f"{model_type}_best.pt"
    if not ckpt.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(cfg, model_type, device)
    frames = []
    for split in EVAL_SPLITS:
        flat = flatten_predictions(get_full_predictions(model, SequenceDataset(split, cfg), device))
        frames.append(flat[["TransactionID", "fraud_prob"]].rename(columns={"fraud_prob": model_type}))
    return pd.concat(frames, ignore_index=True)


def build_common_frame(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """One row per validation/test transaction scored by EVERY available model."""
    table, _ = load_xgb_table(cfg)
    base = table[table["split"].isin(EVAL_SPLITS)].copy()
    base["n_seen"] = (base["txn_index_in_card"] + 1).astype(int)
    frame = base[["TransactionID", "card_id", "split", "isFraud", "n_seen"]].reset_index(drop=True)

    models = []
    for name in MODEL_ORDER:
        if name.startswith("xgboost_plus"):
            s = _xgb_plus_scores(name, base)
            part = None if s is None else pd.DataFrame({"TransactionID": base["TransactionID"].to_numpy(), name: s.to_numpy()})
        elif name in ("logreg", "xgboost"):
            part = _flat_baseline_scores(cfg, name)
        else:
            part = _sequence_scores(cfg, name)
        if part is None:
            logger.info("skipping %s (no trained checkpoint found)", name)
            continue
        frame = frame.merge(part, on="TransactionID", how="inner")
        models.append(name)
    return frame, models


def _rank_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc_roc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def choose_blend(y_val: np.ndarray, cal_a: np.ndarray, cal_b: np.ndarray, step: float) -> dict:
    """Weight w in [0, 1] on model A (1 - w on B) maximizing validation PR-AUC."""
    best_w, best_pr = 0.0, -1.0
    for w in np.round(np.arange(0.0, 1.0 + 1e-9, step), 6):
        pr = average_precision_score(y_val, w * cal_a + (1 - w) * cal_b)
        if pr > best_pr:
            best_w, best_pr = float(w), float(pr)
    return {"weight_first_model": best_w, "val_pr_auc": best_pr}


def history_breakdown(test: pd.DataFrame, models: list[str], bins: list[int]) -> pd.DataFrame:
    """Test AUC-ROC / PR-AUC by how many transactions the card had up to and
    including this one (n_seen), per model."""
    edges = list(bins)
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" if edges[i + 1] < 1000 else f"{edges[i]}+" for i in range(len(edges) - 1)]
    bucket = pd.cut(test["n_seen"], bins=edges[:-1] + [np.inf], labels=labels, right=False)
    rows = []
    for label in labels:
        g = test[bucket == label]
        row = {"n_seen_bucket": label, "n_transactions": int(len(g)), "n_fraud": int(g["isFraud"].sum())}
        for m in models:
            if g["isFraud"].nunique() == 2 and g["isFraud"].sum() >= 5:
                row[f"{m}_auc_roc"] = float(roc_auc_score(g["isFraud"], g[m]))
                row[f"{m}_pr_auc"] = float(average_precision_score(g["isFraud"], g[m]))
            else:
                row[f"{m}_auc_roc"] = row[f"{m}_pr_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ecfg = cfg["evaluation"]
    frame, models = build_common_frame(cfg)
    blend_a, blend_b = ecfg["blend_models"]
    for m in (blend_a, blend_b):
        if m not in models:
            raise RuntimeError(f"Blend model '{m}' has no trained checkpoint; available: {models}")

    val, test = frame[frame["split"] == "val"].copy(), frame[frame["split"] == "test"].copy()
    y_val, y_test = val["isFraud"].to_numpy(), test["isFraud"].to_numpy()
    logger.info("Common population: val=%d rows (%d fraud), test=%d rows (%d fraud)",
                len(val), y_val.sum(), len(test), y_test.sum())

    # --- calibrate on validation, blend ----------------------------------------
    platt = {m: fit_platt(y_val, val[m].to_numpy()) for m in (blend_a, blend_b)}
    cal = {s: {m: apply_platt(d[m].to_numpy(), platt[m]) for m in (blend_a, blend_b)} for s, d in (("val", val), ("test", test))}
    choice = choose_blend(y_val, cal["val"][blend_a], cal["val"][blend_b], ecfg["blend_weight_step"])
    w = choice["weight_first_model"]
    val["blend"] = w * cal["val"][blend_a] + (1 - w) * cal["val"][blend_b]
    test["blend"] = w * cal["test"][blend_a] + (1 - w) * cal["test"][blend_b]
    all_models = models + ["blend"]
    logger.info("Blend = %.2f*%s + %.2f*%s (weights chosen on validation PR-AUC=%.4f)",
                w, blend_a, 1 - w, blend_b, choice["val_pr_auc"])

    # --- every model on the identical rows -------------------------------------
    models_on_common = {
        split: {m: _rank_metrics(d["isFraud"].to_numpy(), d[m].to_numpy()) for m in all_models}
        for split, d in (("val", val), ("test", test))
    }
    for m in all_models:
        t = models_on_common["test"][m]
        logger.info("%-20s test AUC-ROC=%.4f  PR-AUC=%.4f  (same %d rows for every model)",
                    m, t["auc_roc"], t["pr_auc"], len(test))

    # --- validation-chosen operating points ------------------------------------
    operating_points, rows = {}, []
    for m in all_models:
        strategies = {"f1": best_f1_threshold(y_val, val[m].to_numpy())}
        for target in ecfg["recall_targets"]:
            strategies[f"recall_{int(round(target * 100))}"] = threshold_for_recall(y_val, val[m].to_numpy(), target)
        operating_points[m] = {}
        for name, thr in strategies.items():
            v = operating_point_metrics(y_val, val[m].to_numpy(), thr)
            t = operating_point_metrics(y_test, test[m].to_numpy(), thr)
            operating_points[m][name] = {"val": v, "test": t}
            (tn, fp), (fn, tp) = t["confusion_matrix"]
            rows.append({
                "model": m, "strategy": name, "threshold": thr,
                "val_precision": v["precision"], "val_recall": v["recall"], "val_f1": v["f1"],
                "test_precision": t["precision"], "test_recall": t["recall"], "test_f1": t["f1"],
                "test_alert_rate": t["alert_rate"], "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
            })
    threshold_table = pd.DataFrame(rows)
    threshold_table.to_csv(metrics_dir() / "threshold_table.csv", index=False)

    by_history = history_breakdown(test, all_models, ecfg["history_length_bins"])
    by_history.to_csv(metrics_dir() / "by_history_length_common.csv", index=False)

    result = {
        "common_population": {
            "val": {"n": int(len(val)), "n_fraud": int(y_val.sum())},
            "test": {"n": int(len(test)), "n_fraud": int(y_test.sum())},
            "note": "every model below is scored on exactly these rows",
        },
        "models_on_common": models_on_common,
        "models_on_common_at_0.5": {
            m: binary_classification_metrics(y_test, test[m].to_numpy()) for m in all_models
        },
        "calibration": platt,
        "blend": {"models": [blend_a, blend_b], **choice,
                  "test": _rank_metrics(y_test, test["blend"].to_numpy())},
        "operating_points": operating_points,
    }
    with open(metrics_dir() / "blend_and_thresholds.json", "w") as f:
        json.dump(result, f, indent=2)

    out = pd.concat([val, test], ignore_index=True)
    out.to_parquet(resolve_path(cfg["data"]["processed_dir"]) / "common_predictions.parquet", index=False)

    logger.info("Threshold table (thresholds chosen on validation, metrics on test):\n%s",
                threshold_table[["model", "strategy", "threshold", "test_precision", "test_recall", "test_f1"]]
                .round(4).to_string(index=False))
    logger.info("Test AUC-ROC by history available (n_seen):\n%s", by_history.round(4).to_string(index=False))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
