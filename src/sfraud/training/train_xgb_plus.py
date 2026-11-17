"""Phase 3b: the stronger gradient-boosted model, and its history ablation.

  xgboost_plus          every feature in features_xgb.parquet (NaN-preserving
                        C/D/V, extra raw columns, frequency encodings, causal
                        per-card history).
  xgboost_plus_nohist   the SAME features minus every history-derived column,
                        trained with the SAME hyperparameters -- the only
                        difference is what history information it can see.
                        Comparing the two (and both against the LSTM) is the
                        fair test of whether card history helps.

Early stopping and the small hyperparameter grid are scored on the validation
split; the test split is only evaluated once, at the end.
"""

from __future__ import annotations

import itertools
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from sfraud.evaluation.metrics import binary_classification_metrics
from sfraud.features.xgb_features import load_xgb_table
from sfraud.utils.config import checkpoints_dir, load_config, metrics_dir
from sfraud.utils.seed import set_seed

logger = logging.getLogger(__name__)


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency of the project
        return "cpu"


def build_model(xcfg: dict, seed: int, device: str, scale_pos_weight: float, *, max_depth: int,
                colsample_bytree: float, learning_rate: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=xcfg["n_estimators"],
        early_stopping_rounds=xcfg["early_stopping_rounds"],
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        subsample=xcfg["subsample"],
        min_child_weight=xcfg["min_child_weight"],
        reg_lambda=xcfg["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
    )


def predict_proba_best(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Probabilities from the early-stopped (best) iteration, explicitly."""
    best = getattr(model, "best_iteration", None)
    if best is not None:
        return model.predict_proba(X, iteration_range=(0, int(best) + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def _fit(xcfg, seed, device, spw, features, data, **params) -> tuple[XGBClassifier, float]:
    model = build_model(xcfg, seed, device, spw, **params)
    model.fit(data["X_train"][features], data["y_train"],
              eval_set=[(data["X_val"][features], data["y_val"])], verbose=False)
    val_pr = float(average_precision_score(data["y_val"], predict_proba_best(model, data["X_val"][features])))
    return model, val_pr


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    xcfg = cfg["xgboost_plus"]
    device = resolve_device(xcfg.get("device", "auto"))

    table, meta = load_xgb_table(cfg)
    features = meta["features"]
    history = set(meta["history_features"])
    features_nohist = [f for f in features if f not in history]
    logger.info(
        "XGBoost-plus: %d features (%d history-derived, %d without), device=%s",
        len(features), len(history), len(features_nohist), device,
    )

    data = {}
    for split in ["train", "val", "test"]:
        part = table[table["split"] == split]
        data[f"X_{split}"], data[f"y_{split}"] = part, part["isFraud"].to_numpy()
    n_pos = int(data["y_train"].sum())
    n_neg = len(data["y_train"]) - n_pos
    spw_setting = xcfg["scale_pos_weight"]
    spw = float(n_neg / max(n_pos, 1)) if spw_setting == "balanced" else float(spw_setting)

    # --- small hyperparameter search, scored on validation PR-AUC -------------
    best = {"max_depth": xcfg["max_depth"], "colsample_bytree": xcfg["colsample_bytree"]}
    search_log = []
    scfg = xcfg.get("search", {})
    if scfg.get("enabled", False):
        best_val = -1.0
        for depth, colsample in itertools.product(scfg["max_depth"], scfg["colsample_bytree"]):
            model, val_pr = _fit(xcfg, cfg["seed"], device, spw, features, data,
                                 max_depth=depth, colsample_bytree=colsample, learning_rate=scfg["learning_rate"])
            search_log.append({"max_depth": depth, "colsample_bytree": colsample,
                               "best_iteration": int(model.best_iteration), "val_pr_auc": val_pr})
            logger.info("search max_depth=%d colsample=%.2f -> val PR-AUC=%.4f (best_iter=%d)",
                        depth, colsample, val_pr, model.best_iteration)
            if val_pr > best_val:
                best_val, best = val_pr, {"max_depth": depth, "colsample_bytree": colsample}
        logger.info("search picked %s", best)

    # --- final models at the full learning rate -------------------------------
    results = {"device": device, "chosen_params": best, "search": search_log, "scale_pos_weight": spw}
    importance = {}
    for name, cols in (("xgboost_plus", features), ("xgboost_plus_nohist", features_nohist)):
        model, val_pr = _fit(xcfg, cfg["seed"], device, spw, cols, data,
                             learning_rate=xcfg["learning_rate"], **best)
        results[name] = {
            "n_features": len(cols),
            "best_iteration": int(model.best_iteration),
            **{split: binary_classification_metrics(data[f"y_{split}"], predict_proba_best(model, data[f"X_{split}"][cols]))
               for split in ["val", "test"]},
        }
        gain = model.get_booster().get_score(importance_type="gain")
        importance[name] = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:30])
        joblib.dump(model, checkpoints_dir() / f"{name}.joblib")
        for split in ["val", "test"]:
            m = results[name][split]
            logger.info("%-20s %-5s AUC-ROC=%.4f  PR-AUC=%.4f  (best_iter=%d)",
                        name, split, m["auc_roc"], m["pr_auc"], model.best_iteration)

    results["top_features_by_gain"] = importance
    with open(metrics_dir() / "xgboost_plus_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    delta = results["xgboost_plus"]["test"]["pr_auc"] - results["xgboost_plus_nohist"]["test"]["pr_auc"]
    logger.info("History ablation (test PR-AUC, with minus without history features): %+.4f", delta)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
