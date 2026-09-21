"""Shared metric computation used by every model (baselines, LSTM,
Transformer). AUC-ROC and PR-AUC are the headline metrics throughout this
project (per Section 2 of the blueprint) since raw accuracy is misleading
under ~96%+ class imbalance."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) < 2:
        out["auc_roc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    return out


def roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    return {
        "fpr": fpr, "tpr": tpr,
        "precision": precision, "recall": recall,
    }
